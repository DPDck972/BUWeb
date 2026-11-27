"""Juiz LLM via API do Groq (padrão: openai/gpt-oss-120b) ou Hugging Face, conectado ao servidor MCP em modo somente leitura."""
from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)

from .config import get_servidor_config

# ── Provedor padrão: Groq (API compatível com OpenAI) ─────────────────────────
JUDGE_PROVIDER_DEFAULT = "groq"
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_JUDGE_MODEL_DEFAULT = "openai/gpt-oss-120b"
GROQ_JUDGE_FALLBACK_MODEL = "llama-3.3-70b-versatile"

# ── Provedor alternativo: Hugging Face Inference Router ───────────────────────
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
HF_JUDGE_MODEL_DEFAULT = "Qwen/Qwen2.5-72B-Instruct"
HF_JUDGE_FALLBACK_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def get_judge_provider(provider: str | None = None) -> str:
    """Resolve o provedor do juiz ('groq' ou 'huggingface')."""
    raw = (provider or os.getenv("JUDGE_PROVIDER") or JUDGE_PROVIDER_DEFAULT).strip().lower()
    if raw in ("hf", "huggingface"):
        return "huggingface"
    if raw == "groq":
        return "groq"
    raise ValueError(
        f"Provedor de juiz '{raw}' inválido. Use 'groq' ou 'huggingface'."
    )


def get_judge_model(model: str | None = None, provider: str | None = None) -> str:
    """Resolve o modelo do juiz conforme argumento, variáveis de ambiente e provedor."""
    if model:
        return model
    if os.getenv("JUDGE_MODEL"):
        return os.environ["JUDGE_MODEL"]
    prov = get_judge_provider(provider)
    if prov == "groq":
        return os.getenv("GROQ_JUDGE_MODEL", GROQ_JUDGE_MODEL_DEFAULT)
    return os.getenv("HF_JUDGE_MODEL", HF_JUDGE_MODEL_DEFAULT)


JUDGE_SYSTEM_PROMPT = """\
Você é um juiz auditor rigoroso avaliando respostas de um assistente de auditoria eleitoral (BUWeb RN).
Você tem acesso SOMENTE LEITURA às ferramentas do servidor MCP oficial dos Boletins de Urna.

REGRAS DE AVALIAÇÃO:
1. Sempre que a pergunta envolver números, rankings, comparações, somas, biometria ou existência de seções,
   CONSULTE as ferramentas MCP disponíveis para conferir os fatos reais antes de emitir o veredito.
2. Uma resposta é CORRETA (`veredito_juiz: true`) se:
   - Todos os números, zonas, seções, municípios e conclusões (ex.: maior/menor) concordam com os dados reais do MCP;
   - Em rankings ("quais seções têm mais..."), o 1º colocado e seu valor batem com o MCP e os demais itens listados são consistentes;
   - Em perguntas fora de escopo ou sobre seções/atributos inexistentes, o assistente recusa claramente sem inventar dados nem usar conhecimento externo.
3. Uma resposta é INCORRETA (`veredito_juiz: false`) se inventa números, erra zona/seção, inverte comparações ("mais" vs "menos"),
   omite a resposta solicitada ou responde perguntas fora de escopo com conhecimento externo.
4. Quando terminar a auditoria, NÃO chame nenhuma ferramenta adicional (nunca chame uma ferramenta chamada "json").
   Responda diretamente em texto puro contendo APENAS um objeto JSON válido no formato exato:
   {"veredito_juiz": true, "justificativa_juiz": "explicação curta e objetiva em português"}
"""


def mcp_tools_to_openai_schema(mcp_tools) -> list[dict]:
    """Converte ferramentas listadas pelo servidor MCP para o formato tools da API OpenAI."""
    schemas = []
    for tool in mcp_tools:
        input_schema = getattr(tool, "inputSchema", None) or {
            "type": "object",
            "properties": {},
        }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": input_schema,
                },
            }
        )
    return schemas


def _extract_mcp_text(call_result) -> str:
    if isinstance(call_result, str):
        return call_result
    if isinstance(call_result, dict):
        return json.dumps(call_result, ensure_ascii=False)
    content = getattr(call_result, "content", None)
    if content is None:
        return str(call_result)
    parts = [getattr(block, "text", str(block)) for block in content]
    return "".join(parts).strip()


def _parse_judge_json(content: str) -> tuple[bool, str]:
    """Extrai (veredito_juiz, justificativa_juiz) da saída textual/JSON do juiz."""
    texto = (content or "").strip()
    candidatos = [texto]

    bloco = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.S | re.I)
    if bloco:
        candidatos.insert(0, bloco.group(1))

    chaves = re.search(r"\{.*\}", texto, re.S)
    if chaves:
        candidatos.append(chaves.group(0))

    for cand in candidatos:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            veredito_raw = obj.get("veredito_juiz")
            if veredito_raw is None:
                veredito_raw = obj.get("correta")
            if isinstance(veredito_raw, str):
                veredito = veredito_raw.strip().lower() in ("true", "sim", "correta", "1")
            else:
                veredito = bool(veredito_raw)
            justificativa = str(
                obj.get("justificativa_juiz")
                or obj.get("motivo")
                or obj.get("justificativa")
                or texto[:300]
            ).strip()
            return veredito, justificativa

    baixa = texto.lower()
    veredito = '"veredito_juiz": true' in baixa or '"correta": true' in baixa
    return veredito, texto[:300] or "Sem justificativa retornada pelo juiz."


def _build_default_client(
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> AsyncOpenAI:
    prov = get_judge_provider(provider)
    if prov == "groq":
        resolved_url = base_url or os.getenv("GROQ_BASE_URL", GROQ_BASE_URL)
        token = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        if not token:
            raise EnvironmentError(
                "Variável de ambiente GROQ_API_KEY não definida. "
                "Configure GROQ_API_KEY com sua chave da API do Groq para usar o juiz "
                f"({GROQ_JUDGE_MODEL_DEFAULT})."
            )
        return AsyncOpenAI(base_url=resolved_url, api_key=token)

    resolved_url = base_url or os.getenv("HF_BASE_URL", HF_BASE_URL)
    token = (api_key or os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise EnvironmentError(
            "Variável de ambiente HF_TOKEN não definida. "
            "Configure HF_TOKEN com seu token da Hugging Face Inference API para usar o juiz."
        )
    return AsyncOpenAI(base_url=resolved_url, api_key=token)


def _extract_verdict_from_tool_use_failed(exc: BadRequestError) -> tuple[bool, str] | None:
    """Recupera veredito quando provedores como Groq emitem tool_use_failed ao gerar JSON final."""
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict) or err.get("code") != "tool_use_failed":
        return None
    failed_gen = err.get("failed_generation")
    if not isinstance(failed_gen, str) or "veredito_juiz" not in failed_gen:
        return None
    try:
        parsed = json.loads(failed_gen)
    except (json.JSONDecodeError, TypeError):
        return _parse_judge_json(failed_gen)
    if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), dict):
        args = parsed["arguments"]
        if "veredito_juiz" in args:
            return bool(args["veredito_juiz"]), str(
                args.get("justificativa_juiz") or args.get("motivo") or ""
            ).strip()
    if isinstance(parsed, dict) and "veredito_juiz" in parsed:
        return bool(parsed["veredito_juiz"]), str(
            parsed.get("justificativa_juiz") or parsed.get("motivo") or ""
        ).strip()
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


class MCPJudge:
    """Juiz LLM (Groq openai/gpt-oss-120b por padrão, ou HF) com acesso somente leitura ao MCP."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tool_rounds: int = 5,
        max_retries: int = 4,
        retry_delay_s: float = 2.0,
        request_delay_s: float = 0.0,
        data_path: str | None = None,
        tools_schema: list[dict] | None = None,
        tool_executor: Callable[[str, dict], Any] | None = None,
    ):
        self.provider = get_judge_provider(provider)
        self.model = get_judge_model(model, provider=self.provider)
        self._client = client
        self._base_url = base_url
        self._api_key = api_key
        self.max_tool_rounds = max_tool_rounds
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.request_delay_s = request_delay_s
        self._data_path = data_path
        self._tools_schema = tools_schema
        self._tool_executor = tool_executor
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_default_client(
                provider=self.provider,
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    async def __aenter__(self) -> "MCPJudge":
        if self._tool_executor is None:
            cfg = get_servidor_config(data_path=self._data_path)["eleicao"]
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg["args"],
                env=cfg["env"],
            )
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            listed = await self._session.list_tools()
            self._tools_schema = mcp_tools_to_openai_schema(listed.tools)
        elif self._tools_schema is None:
            self._tools_schema = []
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None
        return False

    async def _execute_mcp_tool(self, name: str, args: dict) -> str:
        if self._tool_executor is not None:
            res = self._tool_executor(name, args)
            if asyncio.iscoroutine(res):
                res = await res
            return _extract_mcp_text(res)
        if self._session is None:
            raise RuntimeError(
                "Sessão MCP não inicializada. Use `async with MCPJudge(...) as judge:`."
            )
        result = await self._session.call_tool(name, args)
        return _extract_mcp_text(result)

    async def _call_llm_with_retry(self, messages: list[dict], use_tools: bool):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if use_tools and self._tools_schema:
            kwargs["tools"] = self._tools_schema
            kwargs["tool_choice"] = "auto"

        last_exc = None
        for attempt in range(self.max_retries + 1):
            if self.request_delay_s > 0 and attempt == 0:
                await asyncio.sleep(self.request_delay_s)
            try:
                return await self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                if getattr(exc, "status_code", None) in (429, 500, 502, 503, 504):
                    last_exc = exc
                else:
                    raise

            if attempt < self.max_retries:
                retry_after = _retry_after_seconds(last_exc)
                backoff = (
                    retry_after
                    if retry_after is not None
                    else self.retry_delay_s * (2**attempt)
                )
                await asyncio.sleep(backoff)

        raise last_exc

    async def avaliar(
        self,
        pergunta: str,
        resposta: str,
        categoria: str | None = None,
        esperado: str | None = None,
    ) -> dict:
        """Avalia uma resposta consultando o servidor MCP quando necessário."""
        bloco_user = [f"PERGUNTA DO USUÁRIO:\n{pergunta}"]
        if categoria:
            bloco_user.append(f"CATEGORIA:\n{categoria}")
        if esperado:
            bloco_user.append(
                f"REFERÊNCIA AUXILIAR (confirme via MCP quando aplicável):\n{esperado}"
            )
        bloco_user.append(f"RESPOSTA DO ASSISTENTE:\n{resposta}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(bloco_user)},
        ]
        consultas_mcp: list[dict[str, Any]] = []
        final_content = ""

        for round_idx in range(self.max_tool_rounds + 1):
            allow_tools = round_idx < self.max_tool_rounds
            try:
                completion = await self._call_llm_with_retry(messages, use_tools=allow_tools)
            except BadRequestError as exc:
                # Groq (ex.: openai/gpt-oss-120b) pode emitir tool_use_failed ao devolver JSON final
                recovered = _extract_verdict_from_tool_use_failed(exc)
                if recovered is not None:
                    return {
                        "veredito_juiz": bool(recovered[0]),
                        "justificativa_juiz": recovered[1] or "Veredito recuperado da resposta final.",
                        "consultas_juiz_mcp": consultas_mcp,
                    }
                if allow_tools:
                    completion = await self._call_llm_with_retry(messages, use_tools=False)
                else:
                    raise
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls and allow_tools:
                messages.append(
                    {
                        "role": "assistant",
                        "content": getattr(msg, "content", None) or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    nome = tc.function.name
                    raw_args = tc.function.arguments
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args) if raw_args.strip() else {}
                        except json.JSONDecodeError:
                            args = {"_raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = {}

                    resultado_mcp = await self._execute_mcp_tool(nome, args)
                    consultas_mcp.append({"ferramenta": nome, "argumentos": args})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": resultado_mcp,
                        }
                    )
                continue

            final_content = getattr(msg, "content", None) or ""
            break

        veredito, justificativa = _parse_judge_json(final_content)
        return {
            "veredito_juiz": bool(veredito),
            "justificativa_juiz": justificativa,
            "consultas_juiz_mcp": consultas_mcp,
        }


async def avaliar_resposta(
    pergunta: str,
    resposta: str,
    *,
    categoria: str | None = None,
    esperado: str | None = None,
    judge: MCPJudge | None = None,
    **judge_kwargs,
) -> dict:
    """Função de conveniência para avaliar uma resposta com o MCPJudge."""
    if judge is not None:
        return await judge.avaliar(
            pergunta=pergunta,
            resposta=resposta,
            categoria=categoria,
            esperado=esperado,
        )
    async with MCPJudge(**judge_kwargs) as temp_judge:
        return await temp_judge.avaliar(
            pergunta=pergunta,
            resposta=resposta,
            categoria=categoria,
            esperado=esperado,
        )
