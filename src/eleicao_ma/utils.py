"""Utilitários de exibição, execução, extração de tokens e monitoramento de recursos."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from contextvars import ContextVar

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

try:
    import psutil
except ImportError:  # pragma: no cover - fallback defensivo
    psutil = None

_METRICAS_EXECUCAO: ContextVar[dict | None] = ContextVar(
    "metricas_execucao", default=None
)


def iniciar_metricas():
    """Inicia contadores isolados para a execução assíncrona atual."""
    estado = {
        "chamadas_modelo": 0,
        "tokens_entrada": 0,
        "tokens_saida": 0,
    }
    token = _METRICAS_EXECUCAO.set(estado)
    return estado, token


def finalizar_metricas(token):
    """Restaura o ContextVar de métricas."""
    _METRICAS_EXECUCAO.reset(token)


def _to_int(valor) -> int:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return 0


def extrair_tokens_mensagem(obj) -> tuple[int, int]:
    """Extrai (tokens_entrada, tokens_saida) de mensagens LangChain, Ollama ou objetos de uso."""
    if obj is None:
        return 0, 0

    # 1) Dict direto (ex.: usage_metadata ou token_usage)
    if isinstance(obj, dict):
        if "raw" in obj and obj["raw"] is not None:
            return extrair_tokens_mensagem(obj["raw"])
        inp = _to_int(
            obj.get("input_tokens")
            or obj.get("prompt_tokens")
            or obj.get("prompt_eval_count")
        )
        out = _to_int(
            obj.get("output_tokens")
            or obj.get("completion_tokens")
            or obj.get("eval_count")
        )
        if inp or out:
            return inp, out
        for chave in ("usage_metadata", "token_usage", "usage", "response_metadata"):
            sub = obj.get(chave)
            if sub:
                si, so = extrair_tokens_mensagem(sub)
                if si or so:
                    return si, so
        return 0, 0

    # 2) Atributo usage_metadata (LangChain AIMessage padrão)
    usage_meta = getattr(obj, "usage_metadata", None)
    if usage_meta:
        inp, out = extrair_tokens_mensagem(usage_meta)
        if inp or out:
            return inp, out

    # 3) Atributo response_metadata (Ollama / OpenAI / Anthropic via LangChain)
    resp_meta = getattr(obj, "response_metadata", None)
    if resp_meta:
        inp, out = extrair_tokens_mensagem(resp_meta)
        if inp or out:
            return inp, out

    # 4) Atributo token_usage (CrewAI CrewOutput / UsageMetrics)
    token_usage = getattr(obj, "token_usage", None)
    if token_usage is not None and token_usage is not obj:
        inp, out = extrair_tokens_mensagem(token_usage)
        if inp or out:
            return inp, out

    # 5) Objeto com atributos diretos (ex.: CrewAI UsageMetrics)
    inp = _to_int(
        getattr(obj, "input_tokens", None)
        or getattr(obj, "prompt_tokens", None)
        or getattr(obj, "prompt_eval_count", None)
    )
    out = _to_int(
        getattr(obj, "output_tokens", None)
        or getattr(obj, "completion_tokens", None)
        or getattr(obj, "eval_count", None)
    )
    return inp, out


def registrar_chamada_modelo(resposta=None):
    """Conta uma chamada ao modelo e acumula tokens da execução assíncrona atual."""
    estado = _METRICAS_EXECUCAO.get()
    if estado is None:
        return
    estado["chamadas_modelo"] += 1
    if resposta is not None:
        inp, out = extrair_tokens_mensagem(resposta)
        estado["tokens_entrada"] += inp
        estado["tokens_saida"] += out


def _ler_gpu_mem_mb() -> float | None:
    """Lê memória de GPU usada em MB (nvidia-smi ou PyTorch CUDA), com fallback para None."""
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=True,
            )
            valores = [
                float(linha.strip())
                for linha in proc.stdout.splitlines()
                if linha.strip()
            ]
            if valores:
                return round(max(valores), 2)
        except Exception:
            pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass

    return None


class ResourceMonitor:
    """Monitor leve em background para pico de CPU (%), RAM (MB) e GPU (MB)."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process() if psutil is not None else None
        self.cpu_percent_max: float = 0.0
        self.ram_mb_max: float = 0.0
        self.gpu_mem_mb_max: float | None = None

    def _sample(self):
        if psutil is not None:
            try:
                cpu = float(psutil.cpu_percent(interval=None))
                if cpu > self.cpu_percent_max:
                    self.cpu_percent_max = cpu
            except Exception:
                pass
            if self._proc is not None:
                try:
                    rss = self._proc.memory_info().rss
                    for child in self._proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                    ram_mb = rss / (1024 * 1024)
                    if ram_mb > self.ram_mb_max:
                        self.ram_mb_max = ram_mb
                except Exception:
                    pass

        gpu_mb = _ler_gpu_mem_mb()
        if gpu_mb is not None:
            if self.gpu_mem_mb_max is None or gpu_mb > self.gpu_mem_mb_max:
                self.gpu_mem_mb_max = gpu_mb

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # priming
            except Exception:
                pass
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        return False

    def snapshot(self) -> dict:
        return {
            "cpu_percent_max": round(float(self.cpu_percent_max), 2),
            "ram_mb_max": round(float(self.ram_mb_max), 2),
            "gpu_mem_mb_max": (
                round(float(self.gpu_mem_mb_max), 2)
                if self.gpu_mem_mb_max is not None
                else None
            ),
        }


def texto(conteudo):
    """Extrai o texto de um resultado de ferramenta (pode ser str ou lista de blocos)."""
    if isinstance(conteudo, str):
        return conteudo
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in conteudo
    )


def resumo(resultado):
    """Uma linha sobre o resultado de uma ferramenta, para o trace."""
    try:
        d = json.loads(resultado)
    except (json.JSONDecodeError, TypeError):
        return "erro: " + str(resultado).splitlines()[0][:90]
    if isinstance(d, dict) and "error" in d:
        return "erro: " + d["error"]
    if isinstance(d, dict) and "secao" in d and "zona" in d:
        return (
            f"Zona {d['zona']}, Seção {d['secao']}, "
            f"município {d.get('municipio', '')} · {d.get('comparecimento', 0)} compareceram"
        )
    if isinstance(d, dict) and "total_found" in d:
        secoes = [(s.get("ZONA_N"), s.get("SECAO_N")) for s in d.get("secoes", [])]
        return f'{d["total_found"]} seções; top: {secoes[:3]}'
    if isinstance(d, dict) and "best" in d:
        return "melhor: " + ", ".join(f"{k}={v}" for k, v in d["best"].items())
    if isinstance(d, dict) and "total_votos" in d:
        return f'total_votos={d["total_votos"]} tipo={d.get("tipo_voto")} cargo={d.get("cargo")}'
    if isinstance(d, dict) and "total_sem_biometria" in d:
        return f'sem_biometria={d["total_sem_biometria"]} ({d.get("percentual")}%)'
    if isinstance(d, dict) and "total" in d and "detalhe" in d:
        return d.get("resposta", f'total={d["total"]}')
    return str(resultado)[:120]


def trajetoria(msgs):
    """Chamadas de ferramenta e resultados em texto — para o verificador ler."""
    linhas = []
    for m in msgs:
        if isinstance(m, AIMessage):
            linhas += [
                f"CALL {c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                for c in m.tool_calls
            ]
        elif isinstance(m, ToolMessage):
            linhas.append(f"RESULT {texto(m.content)}")
    return "\n".join(linhas)


def tem_evidencia(evidencia, resultados):
    """
    Valida objeção do verificador: cada número e zona/seção citados na evidência
    precisam aparecer nos resultados das ferramentas, exceto em evidências estruturais.
    """
    baixa = str(evidencia).lower()
    if (
        "não encontrada" in baixa
        or "error" in baixa
        or "pergunta cita" in baixa
        or "pergunta pede" in baixa
    ):
        return True
    fatos = re.findall(r"\d+(?:\.\d+)?", str(evidencia))
    return bool(fatos) and all(f in resultados.replace(" ", "") for f in fatos)


def extrair_resposta(msgs) -> str:
    """Extrai a resposta final: última AIMessage sem tool_calls (preferindo gerador/redator)."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and not m.tool_calls and m.name is None:
            return m.content if isinstance(m.content, str) else texto(m.content)
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and not m.tool_calls and m.name == "verificador":
            return m.content if isinstance(m.content, str) else texto(m.content)
    return ""


def mostrar(r):
    """Imprime o trace de uma execução."""
    for m in r["messages"][1:]:
        nome = getattr(m, "name", None)
        if isinstance(m, AIMessage) and m.tool_calls:
            for c in m.tool_calls:
                print(
                    f"   🤖 pede        → {c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                )
        elif isinstance(m, ToolMessage):
            print(f"   📡 servidor    → {resumo(texto(m.content))}")
        elif nome == "verificador":
            print(f"   🔍 verificador → {str(m.content)[:300]}")
        elif nome == "supervisor":
            print(f"   🧭 supervisor  → {str(m.content)[:160]}")
        elif nome in ("buscador", "comparador"):
            print(f"   📋 {nome:<11} → {str(m.content).replace(chr(10), ' ')[:160]}…")
    print(f"\n💬 {r['resposta']}\n")
    print(f"⏱ {r['chamadas']} chamadas ao modelo · {r['segundos']} s")


async def rodar(app, pergunta, ver=True):
    """Executa uma arquitetura LangGraph e devolve resposta, trajetória e métricas."""
    metricas, token = iniciar_metricas()
    t0 = time.perf_counter()
    try:
        with ResourceMonitor() as monitor:
            out = await app.ainvoke(
                {"messages": [HumanMessage(pergunta)]}, {"recursion_limit": 40}
            )
        latencia = round(time.perf_counter() - t0, 3)
        msgs = out["messages"]
        ferramentas_chamadas = [
            c["name"]
            for m in msgs
            if isinstance(m, AIMessage)
            for c in (m.tool_calls or [])
        ]
        tokens_in = metricas["tokens_entrada"]
        tokens_out = metricas["tokens_saida"]
        hw = monitor.snapshot()
        r = {
            "resposta": extrair_resposta(msgs),
            "ferramentas": ferramentas_chamadas,
            "ferramentas_chamadas": ferramentas_chamadas,
            "chamadas": metricas["chamadas_modelo"],
            "chamadas_modelo": metricas["chamadas_modelo"],
            "chamadas_ferramentas": len(ferramentas_chamadas),
            "segundos": round(latencia, 1),
            "latencia_s": latencia,
            "tokens_entrada": tokens_in,
            "tokens_saida": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "cpu_percent_max": hw["cpu_percent_max"],
            "ram_mb_max": hw["ram_mb_max"],
            "gpu_mem_mb_max": hw["gpu_mem_mb_max"],
            "messages": msgs,
        }
        if ver:
            print(f"👤 {pergunta}\n")
            mostrar(r)
        return r
    finally:
        finalizar_metricas(token)
