import asyncio
import base64
import re
import time
from pathlib import Path

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image, Reply

BODY_MAP = {
    "萝莉": "luoli", "成女": "chengnv", "成男": "chengnan", "正太": "zhengtai",
    "f1": "luoli", "f2": "chengnv", "m1": "zhengtai", "m2": "chengnan",
    "loli": "luoli",
}
BODY_PROMPT = {"luoli": "萝莉", "zhengtai": "正太", "chengnv": "成女", "chengnan": "成男"}
MAX_CONCURRENT = 3
WAIT_SECONDS = 60
CANCEL_WORD = "撤销"
LOW_CONFIDENCE = 60.0


@register(
    "astrbot_plugin_outfit_lookup",
    "沐倾",
    "剑网3外观截图识别：发送外观截图，返回外观名称。",
    "1.1.0",
    "https://github.com/muqing-kg/astrbot_plugin_outfit_lookup",
)
class OutfitLookupPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.api_base = str(config.get("api_base_url", "")).rstrip("/")
        self.top_k = int(config.get("top_k", 5))
        self.enable_llm = bool(config.get("enable_llm_body_detect", True))
        self.low_conf = float(config.get("low_confidence_threshold", 60.0))
        self._active = 0
        self._waiters: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}

    @filter.command("外观识别")
    async def outfit_lookup(self, event: AstrMessageEvent, body_type: str = ""):
        user_id = event.get_sender_id()
        body_key = self.BODY_MAP.get(body_type.strip().lower(), "") if body_type else ""

        image_comp = self._find_image(event)
        if image_comp is None:
            self._waiters[user_id] = {
                "expire": time.time() + WAIT_SECONDS,
                "body_key": body_key,
            }
            yield event.plain_result(
                f"请在 {WAIT_SECONDS} 秒内发送要识别的图片（发送「{CANCEL_WORD}」取消）"
            )
            return

        async for msg in self._run_recognition(event, image_comp, body_key, user_id):
            yield msg

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        text = (event.message_str or "").strip()

        waiter = self._waiters.get(user_id)
        pending = self._pending.get(user_id)
        if not waiter and not pending:
            return
        if text.startswith("外观识别"):
            return

        if waiter:
            if time.time() > waiter["expire"]:
                del self._waiters[user_id]
                yield event.plain_result("等待已超时，请重新发送「外观识别」。")
                return
            if text == CANCEL_WORD:
                del self._waiters[user_id]
                yield event.plain_result("已取消识别。")
                return
            image_comp = self._find_image(event)
            if image_comp is not None:
                del self._waiters[user_id]
                body_key = waiter.get("body_key") or ""
                async for msg in self._run_recognition(event, image_comp, body_key, user_id):
                    yield msg
            elif text:
                yield event.plain_result("请发送图片，或发送「撤销」取消。")
            return

        if pending and text:
            del self._pending[user_id]
            correct = self._resolve_label(text, pending["results"])
            if correct is None:
                yield event.plain_result(
                    "编号超出范围，请发送列表中的编号，或直接发送正确的外观名称。"
                )
                self._pending[user_id] = pending
                return
            ok = await self._annotate(pending["archive_file"], correct)
            if ok:
                yield event.plain_result(f"已记录：{correct}\n感谢反馈，这将帮助改进识别～")
            else:
                yield event.plain_result("记录失败，请稍后再试。")
                self._pending[user_id] = pending

    async def _run_recognition(self, event, image_comp, body_key, user_id):
        if self._active >= MAX_CONCURRENT:
            yield event.plain_result("当前识别任务较多，请稍后再试～")
            return
        self._active += 1
        try:
            yield event.plain_result("正在识别中...请耐心等待")
            image_bytes = await self._read_image(image_comp)
            if image_bytes is None:
                yield event.plain_result("图片读取失败，请重新发送。")
                return

            if not body_key and self.enable_llm:
                body_key = await self._detect_body_by_llm(image_bytes) or ""

            payload = {"top_k": self.top_k, "with_images": "1"}
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
                        yield event.plain_result("识别服务暂时不可用，请稍后再试。")
                        return
                    result = await resp.json()
        finally:
            self._active -= 1

        results = result.get("results") or []
        if not results:
            yield event.plain_result("未识别到外观，请确认截图内容。")
            return

        # 多模态最终对比
        if self.enable_llm and results[0].get("candidate_image"):
            candidates = [
                {"name": r.get("name"), "candidate_image": r.get("candidate_image")}
                for r in results
                if r.get("candidate_image")
            ]
            pick = await self._recheck_by_llm(image_bytes, candidates)
            if pick is not None and 0 < pick < len(results):
                results.insert(0, results.pop(pick))

        archive_file = result.get("archive_file") or ""
        if archive_file:
            self._pending[user_id] = {"archive_file": archive_file, "results": results}

        yield event.plain_result(self._format(results))

    async def _ask_multimodal(self, prompt: str, image_bytes_list: list[bytes]) -> str | None:
        provider = self.context.get_using_provider()
        if provider is None:
            return None
        data_urls = [
            "data:image/jpeg;base64," + base64.b64encode(b).decode() for b in image_bytes_list
        ]
        try:
            reply = await provider.text_chat(prompt=prompt, session_id=None, image_urls=data_urls)
            return reply.completion_text or ""
        except Exception:
            logger.exception("多模态调用失败")
            return None

    async def _detect_body_by_llm(self, image_bytes: bytes) -> str | None:
        text = await self._ask_multimodal(
            "这是剑网3游戏的角色截图。请判断角色体型，只回答以下四个词之一："
            "萝莉、正太、成女、成男。",
            [image_bytes],
        )
        if not text:
            return None
        match = re.search(r"萝莉|正太|成女|成男", text)
        return BODY_MAP[match.group()] if match else None

    async def _recheck_by_llm(self, image_bytes: bytes, candidates: list[dict]) -> int | None:
        prompt = (
            "第 1 张图是查询截图，后面的图是候选外观参考图（按顺序为第 2、3…张）。"
            "请找出与查询截图穿着同一套服装（相同设计和颜色）的候选图，"
            "只回答该候选图的序号数字。"
        )
        images = [image_bytes] + [
            base64.b64decode(c["candidate_image"]) for c in candidates
        ]
        text = await self._ask_multimodal(prompt, images)
        if not text:
            return None
        match = re.search(r"\d+", text)
        if not match:
            return None
        best = int(match.group())
        return best - 2 if 2 <= best <= len(candidates) + 1 else None

    @staticmethod
    def _find_image(event: AstrMessageEvent):
        chain = event.get_messages().content
        for comp in chain:
            if isinstance(comp, Image):
                return comp
        for comp in chain:
            if not isinstance(comp, Reply):
                continue
            for attr in ("chain", "messages", "content"):
                inner = getattr(comp, attr, None)
                if not inner:
                    continue
                try:
                    for sub in inner:
                        if isinstance(sub, Image):
                            return sub
                except TypeError:
                    pass
        return None

    @staticmethod
    async def _read_image(comp: Image) -> bytes | None:
        try:
            if getattr(comp, "base64", None):
                payload = comp.base64
                if "," in payload:
                    payload = payload.split(",", 1)[1]
                return base64.b64decode(payload)
            if getattr(comp, "file", None) and str(comp.file).startswith("file://"):
                return Path(str(comp.file)[7:]).read_bytes()
            if getattr(comp, "url", None) and str(comp.url).startswith("http"):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(str(comp.url), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            return await resp.read()
            return None
        except Exception:
            logger.exception("图片读取失败")
            return None

    async def _annotate(self, archive_file: str, correct: str) -> bool:
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
