"""Utilitários de exibição e execução — adaptados da Lesson 12 para o domínio eleitoral."""
import json
import re
import time
from contextvars import ContextVar

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_CHAMADAS_MODELO: ContextVar[list[int] | None] = ContextVar(
    "chamadas_modelo", default=None
)


def registrar_chamada_modelo():
    """Conta uma chamada ao modelo na execução assíncrona atual."""
    contador = _CHAMADAS_MODELO.get()
    if contador is not None:
        contador[0] += 1


def texto(conteudo):
    """Extrai o texto de um resultado de ferramenta (pode ser str ou lista de blocos)."""
    if isinstance(conteudo, str):
        return conteudo
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in conteudo)


def resumo(resultado):
    """Uma linha sobre o resultado de uma ferramenta, para o trace."""
    try:
        d = json.loads(resultado)
    except (json.JSONDecodeError, TypeError):
        return "erro: " + str(resultado).splitlines()[0][:90]
    if "error" in d:
        return "erro: " + d["error"]
    if "total_found" in d:
        secoes = [(s.get("ZONA_N"), s.get("SECAO_N")) for s in d.get("secoes", [])]
        return f'{d["total_found"]} seções; top: {secoes[:3]}'
    if "best" in d:
        return "melhor: " + ", ".join(f"{k}={v}" for k, v in d["best"].items())
    if "total_votos" in d:
        return f'total_votos={d["total_votos"]} tipo={d.get("tipo_voto")} cargo={d.get("cargo")}'
    if "total_sem_biometria" in d:
        return f'sem_biometria={d["total_sem_biometria"]} ({d.get("percentual")}%)'
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
    precisam aparecer nos resultados das ferramentas.
    """
    fatos = re.findall(r"\d+(?:\.\d+)?", evidencia)
    return bool(fatos) and all(f in resultados.replace(" ", "") for f in fatos)


def mostrar(r):
    """Imprime o trace de uma execução."""
    for m in r["messages"][1:]:
        nome = getattr(m, "name", None)
        if isinstance(m, AIMessage) and m.tool_calls:
            for c in m.tool_calls:
                print(f"   🤖 pede        → {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
        elif isinstance(m, ToolMessage):
            print(f"   📡 servidor    → {resumo(texto(m.content))}")
        elif nome == "verificador":
            print(f"   🔍 verificador → {m.content[:300]}")
        elif nome == "supervisor":
            print(f"   🧭 supervisor  → {m.content[:160]}")
        elif nome in ("buscador", "comparador", "anomalista"):
            print(f"   📋 {nome:<11} → {m.content.replace(chr(10), ' ')[:160]}…")
    print(f"\n💬 {r['resposta']}\n")
    print(f"⏱ {r['chamadas']} chamadas ao modelo · {r['segundos']} s")


async def rodar(app, pergunta, ver=True):
    """Executa uma arquitetura e devolve resposta, trajetória, chamadas e tempo."""
    contador = [0]
    token = _CHAMADAS_MODELO.set(contador)
    t = time.time()
    try:
        out = await app.ainvoke(
            {"messages": [HumanMessage(pergunta)]}, {"recursion_limit": 40}
        )
        msgs = out["messages"]
        resposta = next(
            m.content for m in reversed(msgs)
            if isinstance(m, AIMessage) and not m.tool_calls and m.name is None
        )
        r = {
            "resposta":    resposta,
            "ferramentas": [c["name"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls],
            "chamadas":    contador[0],
            "segundos":    round(time.time() - t, 1),
            "messages":    msgs,
        }
        if ver:
            print(f"👤 {pergunta}\n")
            mostrar(r)
        return r
    finally:
        _CHAMADAS_MODELO.reset(token)
