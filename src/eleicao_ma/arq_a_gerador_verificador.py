"""
Arquitetura A — Gerador + Verificador (Lesson 12, Parte 2).

Adaptação para o domínio eleitoral (BUWeb RN):
- Gerador: ReAct loop com as 5 ferramentas do mcp_server.py
- Verificador: confere afirmações numéricas contra os resultados das ferramentas;
  só objeta com evidência; usa modo pensar=True (thinking) do qwen

Fluxo:
    START → gerador ⇄ ferramentas
                │
                ▼ respondeu
            verificador ── ok ──► END
                │
                └── erro + evidência ──► gerador  (máx. max_revisoes vezes)
"""
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .prompts import SYSTEM, VERIFICADOR
from .utils import registrar_chamada_modelo, tem_evidencia, texto, trajetoria


# ── Esquema de saída estruturada do verificador ───────────────────────────────

class Problema(BaseModel):
    afirmacao: str = Field(description="a afirmação errada, copiada da resposta")
    evidencia: str = Field(
        description="o trecho do resultado da ferramenta que contradiz a afirmação, "
                    "citando os números e zona/seção"
    )


class Revisao(BaseModel):
    problemas: list[Problema] = Field(
        description="afirmações erradas; lista vazia se a resposta estiver correta"
    )


# ── Estado ────────────────────────────────────────────────────────────────────

class Estado(TypedDict):
    messages: Annotated[list, add_messages]


# ── Fábrica do agente único (base da Arq. A e B) ─────────────────────────────

def agente_simples(ferramentas, llm, sistema=SYSTEM):
    """
    O agente da Lesson 12 Parte 1: loop ReAct como grafo LangGraph.
    modelo → ferramentas → modelo ... até responder com texto.
    """
    llm_tools = llm.bind_tools(ferramentas)

    async def no_modelo(estado):
        registrar_chamada_modelo()
        return {"messages": [await llm_tools.ainvoke(
            [SystemMessage(sistema)] + estado["messages"]
        )]}

    g = StateGraph(Estado)
    g.add_node("modelo",      no_modelo)
    g.add_node("ferramentas", ToolNode(ferramentas, handle_tool_errors=True))
    g.add_edge(START, "modelo")
    g.add_conditional_edges("modelo", tools_condition,
                             {"tools": "ferramentas", END: END})
    g.add_edge("ferramentas", "modelo")
    return g.compile()


# ── Arquitetura A ─────────────────────────────────────────────────────────────

def gerador_verificador(ferramentas, llm, max_revisoes: int = 2):
    """
    Arquitetura A — Gerador + Verificador.

    Parâmetros
    ----------
    ferramentas : list
        Ferramentas LangChain obtidas via MultiServerMCPClient.
    llm : ChatOllama
        Modelo base (qwen2.5:7b, temperature=0).
    max_revisoes : int
        Número máximo de devoluções ao gerador antes de forçar o fim.
    """
    # o verificador usa thinking=True para ser mais cuidadoso com rankings
    llm_verificador = ChatOllama(
        model=llm.model,
        temperature=0,
        num_ctx=8192,
        num_predict=4000,
        think=True,          # modo raciocínio (disponível no qwen2.5:7b via Ollama)
    ).with_structured_output(Revisao)

    gerador_llm = llm.bind_tools(ferramentas)

    # ── nós ──────────────────────────────────────────────────────────────────

    async def no_gerador(estado):
        registrar_chamada_modelo()
        return {"messages": [await gerador_llm.ainvoke(
            [SystemMessage(SYSTEM)] + estado["messages"]
        )]}

    async def no_verificador(estado):
        msgs     = estado["messages"]
        resposta = msgs[-1].content

        # coleta IDs eleitorais citados: zona/seção no padrão "zona X seção Y"
        # e chama secao_details para cada par encontrado
        ferr_map  = {f.name: f for f in ferramentas}
        detalhes  = ferr_map.get("secao_details")

        traj      = trajetoria(msgs)
        consultas = []
        if detalhes:
            pares = set(re.findall(r"zona\s*(\d+)[,\s]+se[çc][ãa]o\s*(\d+)", resposta, re.IGNORECASE))
            for zona, secao in pares:
                res = texto(await detalhes.ainvoke({"zona": int(zona), "secao": int(secao)}))
                consultas.append(f"CALL secao_details(zona={zona}, secao={secao})\nRESULT {res}")

        resultados = traj + "\n" + "\n".join(consultas)

        registrar_chamada_modelo()
        rev = await llm_verificador.ainvoke([
            SystemMessage(VERIFICADOR),
            HumanMessage(
                f"PEDIDO DO USUÁRIO:\n{msgs[0].content}\n\n"
                f"CHAMADAS E RESULTADOS DAS FERRAMENTAS:\n{resultados}\n\n"
                f"RESPOSTA DO ASSISTENTE:\n{resposta}"
            ),
        ])

        validos     = [p for p in rev.problemas if tem_evidencia(p.evidencia, resultados)]
        descartados = len(rev.problemas) - len(validos)

        if not validos:
            msg = f"consultou dados; ok"
            if descartados:
                msg += f" ({descartados} objeção sem evidência descartada)"
            return {"messages": [AIMessage(msg, name="verificador")]}

        retorno  = "O verificador encontrou problemas. Corrija-os, chamando ferramentas se necessário:\n"
        retorno += "\n".join(
            f'- "{p.afirmacao}" é contradito por: {p.evidencia}'
            for p in validos
        )
        return {"messages": [HumanMessage(retorno, name="verificador")]}

    # ── arestas condicionais ──────────────────────────────────────────────────

    def depois_do_gerador(estado):
        return "ferramentas" if estado["messages"][-1].tool_calls else "verificador"

    def depois_do_verificador(estado):
        revisoes = sum(
            1 for m in estado["messages"]
            if isinstance(m, HumanMessage) and m.name == "verificador"
        )
        aprovado = isinstance(estado["messages"][-1], AIMessage)
        return END if aprovado or revisoes > max_revisoes else "gerador"

    # ── grafo ─────────────────────────────────────────────────────────────────

    g = StateGraph(Estado)
    g.add_node("gerador",      no_gerador)
    g.add_node("ferramentas",  ToolNode(ferramentas, handle_tool_errors=True))
    g.add_node("verificador",  no_verificador)
    g.add_edge(START, "gerador")
    g.add_conditional_edges("gerador",     depois_do_gerador,     ["ferramentas", "verificador"])
    g.add_edge("ferramentas", "gerador")
    g.add_conditional_edges("verificador", depois_do_verificador, ["gerador", END])
    return g.compile()


# import necessário dentro do módulo (usado em no_verificador)
import re   # noqa: E402
