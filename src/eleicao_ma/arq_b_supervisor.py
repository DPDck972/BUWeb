"""
Arquitetura B — Supervisor + Especialistas (Lesson 12, Parte 3).

Adaptação para o domínio eleitoral (BUWeb RN):
- Supervisor: roteia entre buscador, comparador e anomalista
- Especialistas: cada um recebe só sua tarefa e devolve um relatório
- Redator: sintetiza os relatórios em resposta final

Fluxo:
    START → supervisor → buscador   ──┐
                       → comparador ──┤ → relatório → supervisor
                       → anomalista ──┘
                       → "answer"   → redator → END
"""
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .arq_a_gerador_verificador import agente_simples
from .prompts import (
    ANOMALISTA_SYS,
    BUSCADOR_SYS,
    COMPARADOR_SYS,
    REDATOR_SYS,
    SUPERVISOR,
)
from .utils import registrar_chamada_modelo


# ── Esquema de roteamento estruturado ─────────────────────────────────────────

class Rota(BaseModel):
    proximo: Literal["buscador", "comparador", "anomalista", "answer"] = Field(
        description="quem age a seguir"
    )
    tarefa: str = Field(
        description="instrução precisa para o especialista, com zona/seção e filtros; "
                    "vazio para 'answer'"
    )


# ── Estado ────────────────────────────────────────────────────────────────────

class EstadoSupervisor(TypedDict):
    messages:  Annotated[list, add_messages]
    relatorios: list        # relatórios acumulados dos especialistas
    proximo:   str
    tarefa:    str


# ── Arquitetura B ─────────────────────────────────────────────────────────────

def supervisor_especialistas(ferramentas, llm, max_passos: int = 5):
    """
    Arquitetura B — Supervisor + Especialistas.

    Parâmetros
    ----------
    ferramentas : list
        Ferramentas LangChain obtidas via MultiServerMCPClient.
    llm : ChatOllama
        Modelo base (qwen2.5:7b, temperature=0).
    max_passos : int
        Número máximo de especialistas invocados antes de forçar "answer".
    """
    # mapeia nome → ferramenta
    t = {f.name: f for f in ferramentas}

    # cria os agentes especialistas (cada um com subconjunto de ferramentas)
    especialistas = {
        "buscador":   agente_simples(
            [t["search_secoes"], t["secao_details"]], llm, BUSCADOR_SYS
        ),
        "comparador": agente_simples(
            [t["compare_secoes"], t["total_por_tipo"]], llm, COMPARADOR_SYS
        ),
        "anomalista": agente_simples(
            [t["total_sem_biometria"], t["search_secoes"]], llm, ANOMALISTA_SYS
        ),
    }

    roteador = llm.with_structured_output(Rota)
    redator  = llm

    # ── helpers ───────────────────────────────────────────────────────────────

    def briefing(estado):
        """O que o supervisor e o redator leem."""
        rel = "\n".join(estado.get("relatorios", [])) or "(nenhum ainda)"
        return (
            f"PEDIDO DO USUÁRIO:\n{estado['messages'][0].content}\n\n"
            f"RELATÓRIOS DOS ESPECIALISTAS:\n{rel}"
        )

    # ── nós ──────────────────────────────────────────────────────────────────

    async def no_supervisor(estado):
        if len(estado.get("relatorios", [])) >= max_passos:
            return {"proximo": "answer", "tarefa": ""}
        registrar_chamada_modelo()
        rota = await roteador.ainvoke([
            SystemMessage(SUPERVISOR),
            HumanMessage(briefing(estado)),
        ])
        return {
            "proximo":   rota.proximo,
            "tarefa":    rota.tarefa,
            "messages":  [AIMessage(f"{rota.proximo}: {rota.tarefa}", name="supervisor")],
        }

    def especialista(nome):
        async def no(estado):
            out      = await especialistas[nome].ainvoke({"messages": [HumanMessage(estado["tarefa"])]})
            relatorio = f"[{nome}] {out['messages'][-1].content}"
            return {
                # inclui chamadas internas + mensagem final do especialista
                "messages":   out["messages"][1:-1] + [AIMessage(relatorio, name=nome)],
                "relatorios": estado.get("relatorios", []) + [relatorio],
            }
        return no

    async def no_redator(estado):
        registrar_chamada_modelo()
        return {
            "messages": [await redator.ainvoke([
                SystemMessage(REDATOR_SYS),
                HumanMessage(briefing(estado)),
            ])]
        }

    # ── grafo ─────────────────────────────────────────────────────────────────

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
        {"buscador": "buscador", "comparador": "comparador",
         "anomalista": "anomalista", "answer": "redator"},
    )
    g.add_edge("redator", END)
    return g.compile()
