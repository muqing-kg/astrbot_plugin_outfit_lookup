"""AstrBot 剑网3外观截图识别插件。

命令（带不带 / 均可）：
- 外观识别 [萝莉|正太|成女|成男] + 图片（支持同消息图片、引用消息中的图片）
- 命令不带图片时进入等待状态，60 秒内仅接收同一用户发送的图片
- 不指定体型时，若配置了多模态模型则自动判断体型

回复方式：参考 astrbot_plugin_jx3box 直发消息链，
不经 AstrBot 结果装饰（无 @ / 引用 / 前缀等任何附加），处理完即 stop_event。
"""

from __future__ import annotations

import base64
import re
import time

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

BODY_MAP = {
    "萝莉": "luoli", "成女": "chengnv", "成男": "chengnan", "正太": "zhengtai",
    "f1": "luoli", "f2": "chengnv", "m1": "zhengtai", "m2": "chengnan",
    "loli": "luoli",
}
MAX_CONCURRENT = 3
WAIT_SECONDS = 60
CANCEL_WORD = "撤销"
LOW_CONFIDENCE = 60.0


@register(
    "astrbot_plugin_outfit_lookup",
    "沐倾",
    "剑网3外观截图识别：发送外观截图，返回外观名称。",
    "1.2.2",
    "https://github.com/muqing-kg/astrbot_plugin_outfit_lookup",
)
class OutfitLookupPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.api_base = str(config.get("api_base_url", "")).rstrip("/")
        self.top_k = int(config.get("top_k", 5))
        self.multimodal_model = str(config.get("multimodal_model_id", "")).strip()
        self.low_conf = float(config.get("low_confidence_threshold", 60.0))
        self._active = 0
        self._waiters: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}

    # ==================== 直发回复 ====================

    async def _reply(self, event: AstrMessageEvent, text: str) -> None:
        await self._reply_chain(event, [Comp.Plain(text)])

    async def _reply_chain(self, event: AstrMessageEvent, comps: list) -> None:
        """直发消息：不经 AstrBot 结果装饰（无 @ / 引用 / 前缀等任何附加）。"""
        chain = MessageChain(chain=comps)
        try:
            await event.send(chain)
        except Exception:
            logger.exception("direct send failed, fallback to context.send_message")
            try:
                await self.context.send_message(str(event.unified_msg_origin), chain)
            except Exception:
                logger.exception("fallback send failed")
        try:
            event.stop_event()
        except Exception:
            pass

    # ==================== 命令入口 ====================

    @filter.regex(r"^/?外观识别(?:\s|$|\[)")
    async def outfit_lookup(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        chain = event.get_messages()
        raw_text = "".join(c.text for c in chain if getattr(c, "text", None))
        raw_text = re.sub(r"^/?外观识别", "", raw_text).strip()
        parts = raw_text.split() if raw_text else []
        body_key = self.BODY_MAP.get(parts[0].strip().lower(), "") if parts else ""

        image_comp = self._find_image(chain)
        if image_comp is None:
            self._waiters[user_id] = {
                "expire": time.time() + WAIT_SECONDS,
                "body_key": body_key,
            }
            await self._reply(
                event,
                f"请在 {WAIT_SECONDS} 秒内发送要识别的图片（发送「{CANCEL_WORD}」取消）",
            )
            return
        await self._recognize_and_reply(event, image_comp, body_key, user_id)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        waiter = self._waiters.get(user_id)
        pending = self._pending.get(user_id)
        if not waiter and not pending:
            return
        text = (event.message_str or "").strip()
        if text.startswith("/") or re.match(r"^/?外观识别", text):
            return

        if waiter:
            if time.time() > waiter["expire"]:
                del self._waiters[user_id]
                await self._reply(event, "等待已超时，请重新发送「外观识别」。")
                return
            if text == CANCEL_WORD:
                del self._waiters[user_id]
                await self._reply(event, "已取消识别。")
                return
            image_comp = self._find_image(event.get_messages())
            if image_comp is not None:
                del self._waiters[user_id]
                body_key = waiter.get("body_key") or ""
                await self._recognize_and_reply(event, image_comp, body_key, user_id)
            elif text:
                await self._reply(event, "请发送图片，或发送「撤销」取消。")
            return

        if pending and text:
            del self._pending[user_id]
            correct = self._resolve_label(text, pending["results"])
            if correct is None:
                await self._reply(
                    event,
                    "编号超出范围，请发送列表中的编号，或直接发送正确的外观名称。",
                )
                self._pending[user_id] = pending
                return
            ok = await self._annotate(pending["archive_file"], correct)
            if ok:
                await self._reply(event, f"已记录：{correct}\n感谢反馈，这将帮助改进识别～")
            else:
                await self._reply(event, "记录失败，请稍后再试。")
                self._pending[user_id] = pending

    # ==================== 识别主流程 ====================

    async def _recognize_and_reply(self, event, image_comp, body_key, user_id):
        if self._active >= MAX_CONCURRENT:
            await self._reply(event, "当前识别任务较多，请稍后再试～")
            return
        self._active += 1
        try:
            await self._reply(event, "正在识别中...请耐心等待")
            image_bytes = await self._read_image(image_comp)
            if image_bytes is None:
                await self._reply(event, "图片读取失败，请重新发送。")
                return

            if not body_key and self._get_multimodal_provider() is not None:
                data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                body_key = await self._detect_body_by_llm(data_url) or ""

            payload = {"top_k": self.top_k}
            if body_key:
                payload["body_type"] = body_key
            form = aiohttp.FormData()
            form.add_field("image", image_bytes, filename="query.jpg", content_type="image/jpeg")
            timeout = aiohttp.ClientTimeout(total=180)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_base}/recognize", params=payload, data=form
                ) as resp:
                    if resp.status != 200:
                        await self._reply(event, "识别服务暂时不可用，请稍后再试。")
                        return
                    result = await resp.json()
        finally:
            self._active -= 1

        results = result.get("results") or []
        if not results:
            await self._reply(event, "未识别到外观，请确认截图内容。")
            return

        archive_file = result.get("archive_file") or ""
        if archive_file:
            self._pending[user_id] = {"archive_file": archive_file, "results": results}

        await self._reply(event, self._format(results))

    # ==================== 多模态 ====================

    def _get_multimodal_provider(self):
        if not self.multimodal_model:
            return None
        for provider in self.context.get_all_providers():
            if self.multimodal_model in (provider.provider_config.get("id", ""), provider.provider_config.get("model_config", {}).get("model", "")):
                return provider
        return self.context.get_using_provider() if self.context.get_using_provider() else None

    async def _detect_body_by_llm(self, image_data_url: str) -> str | None:
        provider = self._get_multimodal_provider()
        if provider is None:
            return None
        try:
            reply = await provider.text_chat(
                prompt=(
                    "这是剑网3游戏的角色截图。请判断角色体型，只回答以下四个词之一："
                    "萝莉、正太、成女、成男。"
                ),
                session_id=None,
                image_urls=[image_data_url],
            )
            text = reply.completion_text or ""
        except Exception:
            logger.exception("多模态调用失败")
            return None
        match = re.search(r"萝莉|正太|成女|成男", text)
        return BODY_MAP[match.group()] if match else None

    # ==================== 工具方法 ====================

    @staticmethod
    def _find_image(chain):
        for comp in chain:
            if isinstance(comp, Comp.Image):
                return comp
        for comp in chain:
            if not isinstance(comp, Comp.Reply):
                continue
            inner_chain = getattr(comp, "chain", None)
            if not inner_chain:
                continue
            for sub in inner_chain:
                if isinstance(sub, Comp.Image):
                    return sub
        return None

    @staticmethod
    async def _read_image(comp) -> bytes | None:
        try:
            b64 = await comp.convert_to_base64()
            return base64.b64decode(b64)
        except Exception:
            logger.exception("图片读取失败")
            return None

    async def _annotate(self, archive_file: str, correct: str) -> bool:
        if not self.api_base:
            return False
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_base}/annotate",
                    json={"file": archive_file, "name": correct},
                ) as resp:
                    return resp.status == 200
        except Exception:
            logger.exception("标注请求失败")
            return False

    @staticmethod
    def _resolve_label(text: str, results: list[dict]) -> str | None:
        text = text.strip()
        if text.isdigit():
            idx = int(text)
            return results[idx - 1].get("name") if 1 <= idx <= len(results) else None
        return text if text else None

    @staticmethod
    def _format(results: list[dict]) -> str:
        if not results:
            return "未识别到外观，请确认截图内容。"
        top = results[0]
        percent = top.get("probability_percent") or 0
        lines = ["识别完成", "", f"1.{top.get('name', '未知')} 相似度{percent}%"]
        for i, item in enumerate(results[1:4], 2):
            p = item.get("probability_percent")
            lines.append(f"{i}.{item.get('name', '未知')} 相似度{p}%")
        if percent < LOW_CONFIDENCE:
            lines.append("")
            lines.append("置信度较低，结果仅供参考。")
        lines.append("")
        lines.append(
            "如以上有正确的外观，可发送对应编号（如 1 或 2），"
            "帮助改进识别；如果都不对，也可以直接发送正确的外观名称。"
        )
        return chr(10).join(lines)


if __name__ == "__main__":
    pass
