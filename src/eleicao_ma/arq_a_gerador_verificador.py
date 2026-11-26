"""
Arquitetura A — Gerador + Verificador (Lesson 12, Parte 2).

Adaptação para o domínio eleitoral (BUWeb RN):
- Gerador: ReAct loop com as ferramentas do mcp_server.py
- Verificador: confere afirmações numéricas e estruturais contra os resultados das
  ferramentas; só objeta com evidência; usa modo reasoning=True do qwen3:8b.

Fluxo:
    START → gerador ⇄ ferramentas
                │
                ▼ respondeu
            verificador ── ok ──► END
                │
                └── erro + evidência ──► gerador  (máx. max_revisoes vezes)
"""
from __future__ import annotations

import json
import re
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
        description=(
            "o trecho do resultado da ferramenta que contradiz a afirmação, "
            "citando os números e zona/seção"
        )
    )


class Revisao(BaseModel):
    problemas: list[Problema] = Field(
        description="afirmações erradas; lista vazia se a resposta estiver correta"
    )


# ── Estado ────────────────────────────────────────────────────────────────────


class Estado(TypedDict):
    messages: Annotated[list, add_messages]


# ── Helpers de saída estruturada e validação proativa ─────────────────────────


def _build_structured_runner(llm, schema):
    """Usa include_raw=True quando suportado para capturar metadados de tokens."""
    try:
        return llm.with_structured_output(schema, include_raw=True)
    except TypeError:
        return llm.with_structured_output(schema)


async def _ainvoke_structured(runnable, messages):
    """Invoca saída estruturada, contabiliza chamada/tokens e devolve o objeto parseado."""
    res = await runnable.ainvoke(messages)
    if isinstance(res, dict) and "parsed" in res:
        registrar_chamada_modelo(res.get("raw"))
        return res["parsed"]
    registrar_chamada_modelo(res)
    return res


def secoes_citadas(conteudo: str) -> set[tuple[str, str]]:
    """Extrai pares (zona, secao) em qualquer ordem: 'Zona 3, Seção 504', 'seção 505 da zona 3', '3/504'."""
    pares: set[tuple[str, str]] = set()
    pares |= set(re.findall(r"zona\s*(\d+)[^\d]*se[çc][ãa]o\s*(\d+)", conteudo, re.I))
    pares |= {
        (z, s)
        for s, z in re.findall(r"se[çc][ãa]o\s*(\d+)[^\d]*zona\s*(\d+)", conteudo, re.I)
    }
    pares |= set(re.findall(r"\b(\d{1,3})/(\d{1,4})\b", conteudo))
    for m in re.finditer(r"se[çc][õo]es?\s+([\d\s,e]+?)\s+da\s+zona\s+(\d+)", conteudo, re.I):
        zona = m.group(2)
        for secao in re.findall(r"\d+", m.group(1)):
            pares.add((zona, secao))
    return pares


def operacao_pedida(pergunta: str) -> str | None:
    """Detecta a operação que a pergunta exige: soma, minimo, maximo ou None."""
    p = pergunta.lower()
    if re.search(r"\bmenos\b|\bmenor\b|\bmínimo\b|\bminimo\b", p):
        return "minimo"
    if re.search(r"\bmais\b|\bmaior\b|\bmáximo\b|\bmaximo\b", p):
        return "maximo"
    if re.search(r"\bsom(a|ando|ar|e)\b|\bjunt[oa]s?\b|\bcombinad[oa]s?\b", p):
        return "soma"
    return None


# ── Fábrica do agente único (base da Arq. A e B) ─────────────────────────────


def agente_simples(ferramentas, llm, sistema=SYSTEM):
    """
    O agente da Lesson 12 Parte 1: loop ReAct como grafo LangGraph.
    modelo → ferramentas → modelo ... até responder com texto.
    """
    llm_tools = llm.bind_tools(ferramentas)

    async def no_modelo(estado):
        resp = await llm_tools.ainvoke([SystemMessage(sistema)] + estado["messages"])
        registrar_chamada_modelo(resp)
        return {"messages": [resp]}

    g = StateGraph(Estado)
    g.add_node("modelo", no_modelo)
    g.add_node("ferramentas", ToolNode(ferramentas, handle_tool_errors=True))
    g.add_edge(START, "modelo")
    g.add_conditional_edges(
        "modelo", tools_condition, {"tools": "ferramentas", END: END}
    )
    g.add_edge("ferramentas", "modelo")
    return g.compile()


# ── Arquitetura A ─────────────────────────────────────────────────────────────


def gerador_verificador(
    ferramentas,
    llm,
    max_revisoes: int = 2,
    llm_verificador=None,
):
    """
    Arquitetura A — Gerador + Verificador (sincronizada com o notebook).

    Parâmetros
    ----------
    ferramentas : list
        Ferramentas LangChain obtidas via MultiServerMCPClient.
    llm : ChatOllama
        Modelo base (qwen3:8b, temperature=0).
    max_revisoes : int
        Número máximo de devoluções ao gerador antes de forçar o fim.
    llm_verificador : optional
        Modelo opcional para o verificador (útil em testes unitários).
    """
    base_verificador = llm_verificador or ChatOllama(
        model=llm.model,
        temperature=0,
        num_ctx=8192,
        num_predict=4000,
        reasoning=True,
    )
    verificador_runnable = _build_structured_runner(base_verificador, Revisao)
    gerador_llm = llm.bind_tools(ferramentas)

    async def no_gerador(estado):
        resp = await gerador_llm.ainvoke([SystemMessage(SYSTEM)] + estado["messages"])
        registrar_chamada_modelo(resp)
        return {"messages": [resp]}

    async def no_verificador(estado):
        msgs = estado["messages"]
        pergunta = msgs[0].content if isinstance(msgs[0].content, str) else texto(msgs[0].content)
        resposta = msgs[-1].content if isinstance(msgs[-1].content, str) else texto(msgs[-1].content)

        secoes_pergunta = secoes_citadas(pergunta)
        secoes_resposta = secoes_citadas(resposta)
        op = operacao_pedida(pergunta)
        ferramentas_usadas = {
            c["name"]
            for m in msgs
            if isinstance(m, AIMessage)
            for c in (m.tool_calls or [])
        }

        problemas_proativos: list[Problema] = []

        if secoes_pergunta and not secoes_pergunta.issubset(secoes_resposta):
            problemas_proativos.append(
                Problema(
                    afirmacao="A resposta não trata das seções pedidas na pergunta",
                    evidencia=(
                        f"Pergunta cita {sorted(secoes_pergunta)}; "
                        f"resposta cita {sorted(secoes_resposta)}"
                    ),
                )
            )

        ferr_map = {f.name: f for f in ferramentas}
        if op == "soma" and "soma_secoes" in ferr_map and "soma_secoes" not in ferramentas_usadas:
            problemas_proativos.append(
                Problema(
                    afirmacao="A pergunta pede uma soma, mas a resposta não somou os votos",
                    evidencia=(
                        f"Pergunta pede soma; ferramentas chamadas: {sorted(ferramentas_usadas)}"
                    ),
                )
            )

        detalhes = ferr_map.get("secao_details")
        consultas: list[str] = []
        if detalhes is not None:
            for zona, secao in sorted(secoes_resposta):
                res = texto(await detalhes.ainvoke({"zona": int(zona), "secao": int(secao)}))
                consultas.append(f"CALL secao_details(zona={zona}, secao={secao})\nRESULT {res}")
                try:
                    dados_json = json.loads(res)
                    if isinstance(dados_json, dict) and "error" in dados_json:
                        problemas_proativos.append(
                            Problema(
                                afirmacao=f"Menção à seção {zona}/{secao} na resposta",
                                evidencia=f"Erro retornado pelo servidor: {dados_json['error']}",
                            )
                        )
                except Exception:
                    if "error" in res.lower() or "não encontrada" in res.lower():
                        problemas_proativos.append(
                            Problema(
                                afirmacao=f"Menção à seção {zona}/{secao} na resposta",
                                evidencia=f"Erro retornado pelo servidor: {res}",
                            )
                        )

        resultados = trajetoria(msgs) + ("\n" + "\n".join(consultas) if consultas else "")
        rev = await _ainvoke_structured(
            verificador_runnable,
            [
                SystemMessage(VERIFICADOR),
                HumanMessage(
                    f"PEDIDO DO USUÁRIO:\n{pergunta}\n\n"
                    f"CHAMADAS E RESULTADOS DAS FERRAMENTAS:\n{resultados}\n\n"
                    f"RESPOSTA DO ASSISTENTE:\n{resposta}"
                ),
            ],
        )

        todos_problemas = list(rev.problemas) + problemas_proativos
        validos = [p for p in todos_problemas if tem_evidencia(p.evidencia, resultados)]

        anteriores: set[str] = set()
        for m in msgs:
            if isinstance(m, HumanMessage) and m.name == "verificador":
                anteriores |= set(re.findall(r'- "([^"]+)"', str(m.content)))
        novas = [p for p in validos if p.afirmacao not in anteriores]

        if not validos or not novas:
            return {"messages": [AIMessage(resposta, name="verificador")]}

        retorno = (
            "O verificador encontrou problemas. Corrija-os, chamando ferramentas se necessário:\n"
        )
        retorno += "\n".join(
            f'- "{p.afirmacao}" é contradito por: {p.evidencia}' for p in validos
        )
        return {"messages": [HumanMessage(retorno, name="verificador")]}

    def depois_do_gerador(estado):
        return "ferramentas" if estado["messages"][-1].tool_calls else "verificador"

    def depois_do_verificador(estado):
        revisoes = sum(
            1
            for m in estado["messages"]
            if isinstance(m, HumanMessage) and m.name == "verificador"
        )
        aprovado = isinstance(estado["messages"][-1], AIMessage)
        return END if aprovado or revisoes > max_revisoes else "gerador"

    g = StateGraph(Estado)
    g.add_node("gerador", no_gerador)
    g.add_node("ferramentas", ToolNode(ferramentas, handle_tool_errors=True))
    g.add_node("verificador", no_verificador)
    g.add_edge(START, "gerador")
    g.add_conditional_edges(
        "gerador", depois_do_gerador, ["ferramentas", "verificador"]
    )
    g.add_edge("ferramentas", "gerador")
    g.add_conditional_edges("verificador", depois_do_verificador, ["gerador", END])
    return g.compile()
