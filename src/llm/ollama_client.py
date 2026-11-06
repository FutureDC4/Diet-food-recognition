"""Ollama HTTP 客户端：流式生成 + 降级兜底。

不依赖 ollama 官方 SDK，直接用 HTTP（/api/tags、/api/generate），
好处是超时、流式、错误处理都可控；即使没装 ollama 包也能用。
"""

from __future__ import annotations

import json
from typing import Generator, List, Optional

import requests

from ..utils.config_loader import get
from ..utils.logger import get_logger

log = get_logger("LLM")

FALLBACK_TEXT = (
    "【注意】未能连接本地 Ollama 服务，本次不生成 AI 膳食建议。\n"
    "  排查步骤：\n"
    "    1) 启动服务：ollama serve\n"
    "    2) 拉取模型：ollama pull qwen2.5:7b（或 qwen2.5:1.5b）\n"
    "    3) 检查 config.yaml 里 ollama.host 与 ollama.model 是否正确\n"
    "  以上检测与营养计算结果依然有效，可先据此自行安排饮食。"
)


class OllamaClient:
    """极简 Ollama 客户端。"""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        num_gpu: Optional[int] = None,
    ):
        self.host = str(host or get("ollama.host", "http://localhost:11434")).rstrip("/")
        self.model = str(model or get("ollama.model", "qwen2.5:1.5b"))
        self.timeout = int(timeout if timeout is not None else get("ollama.timeout", 180))
        self.temperature = float(
            temperature if temperature is not None else get("ollama.temperature", 0.7)
        )
        self.num_predict = int(num_predict if num_predict is not None else get("ollama.num_predict", 900))
        self.use_stream = bool(stream if stream is not None else get("ollama.stream", True))
        # 0 = 纯 CPU 推理；-1 = 交给 Ollama 自己决定（有 GPU 就用 GPU）。
        # 老驱动配新版 Ollama 时 GPU 内核会崩（CUDA error: device kernel image is invalid），
        # 这时把 config.yaml 的 ollama.num_gpu 设为 0 即可稳定运行。
        raw_gpu = num_gpu if num_gpu is not None else get("ollama.num_gpu", -1)
        self.num_gpu = int(raw_gpu) if raw_gpu is not None else -1

    # ------------------------------------------------------------------
    def is_available(self, timeout: float = 5.0) -> bool:
        """Ollama 服务是否在线（GET /api/tags）。"""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException as exc:
            log.warning(f"Ollama 不可用（{self.host}）：{exc.__class__.__name__}")
            return False

    def list_models(self, timeout: float = 5.0) -> List[str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            log.warning(f"获取 Ollama 模型列表失败：{exc}")
            return []

    def has_model(self, model: Optional[str] = None) -> bool:
        target = (model or self.model).strip()
        names = self.list_models()
        if not names:
            return False
        base = target.split(":")[0]
        return any(n == target or n.split(":")[0] == base for n in names)

    # ------------------------------------------------------------------
    def _payload(self, prompt: str, system: Optional[str], stream: bool) -> dict:
        options = {"temperature": self.temperature, "num_predict": self.num_predict}
        if self.num_gpu is not None and self.num_gpu >= 0:
            options["num_gpu"] = self.num_gpu
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }
        if system:
            payload["system"] = system
        return payload

    def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        fallback: bool = True,
    ) -> Generator[str, None, None]:
        """流式生成，逐块 yield 文本。Ollama 不可用时（fallback=True）yield 降级提示。"""
        if not self.is_available():
            log.warning("Ollama 离线，返回降级提示")
            if fallback:
                yield FALLBACK_TEXT
            return

        if not self.has_model():
            names = self.list_models()
            msg = (
                f"【注意】本地 Ollama 没有模型 `{self.model}`（已安装：{', '.join(names) or '无'}）。\n"
                f"  请先执行：ollama pull {self.model}\n"
                f"  或修改 config.yaml 中 ollama.model 为已有模型。"
            )
            log.warning(msg.replace("\n", " "))
            if fallback:
                yield msg
            return

        try:
            with requests.post(
                f"{self.host}/api/generate",
                json=self._payload(prompt, system, True),
                stream=True,
                timeout=(10, self.timeout),
            ) as resp:
                if resp.status_code != 200:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        log.debug(f"跳过无法解析的流式块：{raw[:100]}")
                        continue
                    if chunk.get("error"):
                        raise requests.HTTPError(str(chunk["error"]))
                    piece = chunk.get("response", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except (requests.RequestException, OSError) as exc:
            detail = str(exc)
            log.error(f"Ollama 生成失败：{exc.__class__.__name__}: {detail[:300]}")
            if fallback:
                hint = ""
                if "CUDA" in detail or "kernel image" in detail or "llama-server" in detail:
                    hint = ("\n  这通常是显卡驱动版本过低、与 Ollama 的 CUDA 内核不匹配。\n"
                            "  临时解决：把 config.yaml 里 ollama.num_gpu 设为 0（走 CPU 推理）。\n"
                            "  根本解决：升级 NVIDIA 驱动后把 num_gpu 改回 -1。")
                yield (f"\n\n【注意】生成过程中断（{exc.__class__.__name__}），以上内容可能不完整。"
                       f"{hint}\n  详情：{detail[:200]}")

    # ------------------------------------------------------------------
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """非流式生成（内部聚合流式结果，调用方只关心完整文本时用）。"""
        return "".join(self.stream_generate(prompt, system=system))

    def describe(self) -> str:
        status = "在线" if self.is_available() else "离线"
        return f"Ollama[{self.host} | 模型 {self.model} | {status}]"


if __name__ == "__main__":
    client = OllamaClient()
    print(client.describe())
    print("可用模型:", client.list_models())
    print("目标模型是否存在:", client.has_model())
    for chunk in client.stream_generate("用一句话介绍米饭的营养价值。"):
        print(chunk, end="", flush=True)
    print()
