import base64
import json
import re
from pathlib import Path

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image

BODY_MAP = {
    "萝莉": "luoli", "成女": "chengnv", "成男": "chengnan", "正太": "zhengtai",
    "f1": "luoli", "f2": "chengnv", "m1": "zhengtai", "m2": "chengnan",
    "loli": "luoli",
}
BODY_PROMPT = {
    "luoli": "萝莉", "zhengtai": "正太", "chengnv": "成女", "chengnan": "成男",
}


@register(
    "astrbot_plugin_outfit_lookup",
    "muqing",
    "剑网3外观截图识别：发送外观截图，返回外观名称。",
    "1.0.0",
    "https://github.com/muqing-kg/astrbot_plugin_outfit_lookup",
)
class OutfitLookupPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.api_base = str(config.get("api_base_url", "http://192.168.1.4:8390")).rstrip("/")
        self.top_k = int(config.get("top_k", 5))
        self.enable_llm = bool(config.get("enable_llm_body_detect", True))

    @filter.command("外观识别")
    async def outfit_lookup(self, event: AstrMessageEvent, body_type: str = ""):
        image_comp = self._find_image(event)
        if image_comp is None:
            yield event.plain_result(
                "请在发送命令的同一消息里附带一张游戏外观截图（近景/半身最佳）。\n"
                "用法：外观识别 [萝莉|正太|成女|成男] + 图片"
            )
            return

        body_key = BODY_MAP.get(body_type.strip().lower(), "")
        if body_key:
            yield event.plain_result(f"正在识别（{BODY_PROMPT[body_key]}）…")
        elif self.enable_llm:
            yield event.plain_result("正在识别（体型判断中）…")
        else:
            yield event.plain_result("正在识别…")

        image_bytes = await self._read_image(image_comp)
        if image_bytes is None:
            yield event.plain_result("图片读取失败，请重新发送。")
            return

        if not body_key and self.enable_llm:
            body_key = await self._detect_body_by_llm(image_bytes)

        result = await self._call_api(image_bytes, body_key)
        if result is None:
            yield event.plain_result("识别服务暂时不可用，请稍后再试。")
            return

        yield event.plain_result(self._format(result))

    # ---------- helpers ----------

    @staticmethod
    def _find_image(event: AstrMessageEvent):
        for comp in event.get_messages().content:
            if isinstance(comp, Image):
                return comp
        return None

    @staticmethod
    async def _read_image(comp: Image) -> bytes | None:
        """Extract image bytes from an astrbot Image component."""
        try:
            if getattr(comp, "base64", None):
                payload = comp.base64
                if "," in payload:
                    payload = payload.split(",", 1)[1]
                return base64.b64decode(payload)
            if getattr(comp, "file", None) and str(comp.file).startswith("file://"):
                return Path(str(comp.file)[7:]).read_bytes()
            if getattr(comp, "url", None) and str(comp.url).startswith("http"):
                async with aiohttp.ClientSession() as session:
                    async with session.get(str(comp.url), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            return await resp.read()
            return None
        except Exception:
            logger.exception("图片读取失败")
            return None

    async def _detect_body_by_llm(self, image_bytes: bytes) -> str | None:
        provider = self.context.get_using_provider()
        if provider is None:
            return None
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        prompt = (
            "这是剑网3游戏的角色截图。请判断角色体型，只回答以下四个词之一："
            "萝莉、正太、成女、成男。"
        )
        try:
            reply = await provider.text_chat(
                prompt=prompt, session_id=None, image_urls=[data_url],
            )
            text = reply.completion_text or ""
            match = re.search(r"萝莉|正太|成女|成男", text)
            return BODY_MAP[match.group()] if match else None
        except Exception:
            logger.exception("体型判断失败，退回混搜模式")
            return None

    async def _call_api(self, image_bytes: bytes, body_key: str) -> dict | None:
        params = {"top_k": self.top_k}
        if body_key:
            params["body_type"] = body_key
        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename="query.jpg", content_type="image/jpeg")
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_base}/recognize", params=params, data=form
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"识别 API 返回 {resp.status}: {await resp.text()}")
                        return None
                    return await resp.json()
        except Exception:
            logger.exception("识别 API 调用失败")
            return None

    @staticmethod
    def _format(result: dict) -> str:
        results = result.get("results") or []
        if not results:
            return "未识别到外观，请确认截图内容。"
        top = results[0]
        lines = ["🏮 外观识别结果", f"外观：{top.get('name', '未知')}"]
        percent = top.get("probability_percent")
        if percent is not None:
            lines.append(f"匹配度：{percent}%")
        if len(results) > 1:
            lines.append("—— 其他候选 ——")
            for i, item in enumerate(results[1:4], 2):
                p = item.get("probability_percent")
                p_text = f" {p}%" if p is not None else ""
                lines.append(f"{i}. {item.get('name', '未知')}{p_text}")
        return "\n".join(lines)
