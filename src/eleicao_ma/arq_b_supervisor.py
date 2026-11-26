"""
Arquitetura B — Supervisor + Especialistas (Lesson 12, Parte 3).

Alinhada ao notebook para o domínio eleitoral (BUWeb RN):
- Supervisor: roteia entre buscador e comparador (2 especialistas)
- Especialistas: cada um recebe só sua tarefa e devolve um relatório
- Redator: sintetiza os relatórios em resposta final

Fluxo:
    START → supervisor → buscador   ──┐ → relatório → supervisor
                       → comparador ──┘
                       → "answer"   → redator → END
"""
from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .arq_a_gerador_verificador import (
    _ainvoke_structured,
    _build_structured_runner,
    agente_simples,
)
from .prompts import BUSCADOR_SYS, COMPARADOR_SYS, REDATOR_SYS, SUPERVISOR
from .utils import registrar_chamada_modelo


# ── Esquema de roteamento estruturado ─────────────────────────────────────────


class Rota(BaseModel):
    proximo: Literal["buscador", "comparador", "answer"] = Field(
        description="quem age a seguir"
    )
    tarefa: str = Field(
        description=(
            "instrução precisa para o especialista, com zona/seção e filtros; "
            "vazio para 'answer'"
        )
    )


# ── Estado ────────────────────────────────────────────────────────────────────


class EstadoSupervisor(TypedDict):
    messages: Annotated[list, add_messages]
    relatorios: list  # relatórios acumulados dos especialistas
    proximo: str
    tarefa: str


# ── Arquitetura B ─────────────────────────────────────────────────────────────


def supervisor_especialistas(ferramentas, llm, max_passos: int = 4):
    """
    Arquitetura B — Supervisor + 2 Especialistas (buscador e comparador) + Redator.

    Parâmetros
    ----------
    ferramentas : list
        Ferramentas LangChain obtidas via MultiServerMCPClient.
    llm : ChatOllama
        Modelo base (qwen3:8b, temperature=0).
    max_passos : int
        Número máximo de especialistas invocados antes de forçar "answer".
    """
    t = {f.name: f for f in ferramentas}

    buscador_tools = [
        t[nome]
        for nome in ("search_secoes", "total_por_tipo", "total_sem_biometria")
        if nome in t
    ]
    comparador_tools = [
        t[nome]
        for nome in ("compare_secoes", "secao_details", "soma_secoes")
        if nome in t
    ]

    especialistas = {
        "buscador": agente_simples(buscador_tools, llm, BUSCADOR_SYS),
        "comparador": agente_simples(comparador_tools, llm, COMPARADOR_SYS),
    }

    roteador = _build_structured_runner(llm, Rota)
    redator = llm

    def briefing(estado):
        """O que o supervisor e o redator leem."""
        rel = "\n".join(estado.get("relatorios", [])) or "(nenhum ainda)"
        return (
            f"PEDIDO DO USUÁRIO:\n{estado['messages'][0].content}\n\n"
            f"RELATÓRIOS DOS ESPECIALISTAS:\n{rel}"
        )

    async def no_supervisor(estado):
        if len(estado.get("relatorios", [])) >= max_passos:
            return {"proximo": "answer", "tarefa": ""}
        rota = await _ainvoke_structured(
            roteador,
            [
                SystemMessage(SUPERVISOR),
                HumanMessage(briefing(estado)),
            ],
        )
        return {
            "proximo": rota.proximo,
            "tarefa": rota.tarefa,
            "messages": [
                AIMessage(f"{rota.proximo}: {rota.tarefa}", name="supervisor")
            ],
        }

    def especialista(nome):
        async def no(estado):
            out = await especialistas[nome].ainvoke(
                {"messages": [HumanMessage(estado["tarefa"])]}
            )
            relatorio = f"[{nome}] {out['messages'][-1].content}"
            return {
                "messages": out["messages"][1:-1] + [AIMessage(relatorio, name=nome)],
                "relatorios": estado.get("relatorios", []) + [relatorio],
            }

        return no

    async def no_redator(estado):
        resp = await redator.ainvoke(
            [
                SystemMessage(REDATOR_SYS),
                HumanMessage(briefing(estado)),
            ]
        )
        registrar_chamada_modelo(resp)
        return {"messages": [resp]}

    g = StateGraph(EstadoSupervisor)
    g.add_node("supervisor", no_supervisor)
    for nome in especialistas:
        g.add_node(nome, especialista(nome))
        g.add_edge(nome, "supervisor")
    g.add_node("redator", no_redator)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda e: e["proximo"],
        {
            "buscador": "buscador",
            "comparador": "comparador",
            "answer": "redator",
        },
    )
    g.add_edge("redator", END)
    return g.compile()
