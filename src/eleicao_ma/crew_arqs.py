"""As três arquiteturas CrewAI extraídas do notebook e alinhadas ao domínio eleitoral."""
from __future__ import annotations

import json
import time

from .config import MODEL, OLLAMA_URL
from .utils import ResourceMonitor, extrair_tokens_mensagem, resumo

RESPOSTA_PADRAO = (
    "Responda em portugues, com fatos do BU e citando zona e secao quando houver."
)

ASSISTENTE_CFG = dict(
    role="Assistente de auditoria eleitoral RN",
    goal="Responder perguntas sobre BUWeb com base exclusiva nos dados das ferramentas",
    backstory="Voce usa apenas as ferramentas de eleicoes e nunca inventa numeros ou secoes.",
)

GERENTE_BACKSTORY = """\
Voce coordena dois especialistas para auditoria de eleicoes no RN.
- buscador: encontra rankings e totais gerais (search_secoes, total_por_tipo, total_sem_biometria).
- comparador: detalha e compara secoes especificas (secao_details, compare_secoes, soma_secoes).
Delegue uma tarefa precisa por vez e responda ao usuario apenas com fatos dos relatorios.
Nunca invente secao, municipio ou numero."""


def build_crew_llm(
    model: str = MODEL,
    base_url: str = OLLAMA_URL,
    temperature: float = 0,
    context_window_size: int = 8192,
    max_tokens: int = 800,
    frequency_penalty: float = 0.3,
):
    """Cria o objeto LLM do CrewAI apontando para o Ollama."""
    from crewai import LLM

    return LLM(
        model=f"ollama_chat/{model}",
        base_url=base_url,
        temperature=temperature,
        reasoning_effort="none",
        context_window_size=context_window_size,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )


def _selecionar(tools_map: dict, nomes: tuple[str, ...]) -> list:
    return [tools_map[n] for n in nomes if n in tools_map]


def crew_unico(tools_map: dict, llm):
    """Arquitetura 1 (CrewAI): Agente único com todas as ferramentas MCP."""
    from crewai import Agent, Crew, Task

    assistente = Agent(
        **ASSISTENTE_CFG,
        tools=list(tools_map.values()),
        llm=llm,
        max_iter=6,
    )
    return Crew(
        agents=[assistente],
        tasks=[
            Task(
                description="Pergunta do usuario: {pedido}",
                expected_output=RESPOSTA_PADRAO,
                agent=assistente,
            )
        ],
    )


def crew_gerador_verificador(tools_map: dict, llm):
    """Arquitetura 2 (CrewAI): Gerador + Verificador factual."""
    from crewai import Agent, Crew, Task

    assistente = Agent(
        **ASSISTENTE_CFG,
        tools=list(tools_map.values()),
        llm=llm,
        max_iter=6,
    )
    verificador = Agent(
        role="Verificador factual eleitoral",
        goal="Conferir se zona, secao e numeros da resposta batem com as ferramentas",
        backstory="Voce revisa fatos e corrige respostas com base em evidencias do BU.",
        tools=_selecionar(
            tools_map,
            ("secao_details", "compare_secoes", "total_por_tipo", "soma_secoes"),
        ),
        llm=llm,
        max_iter=6,
    )
    responder = Task(
        description="Pergunta do usuario: {pedido}",
        expected_output=RESPOSTA_PADRAO,
        agent=assistente,
    )
    conferir = Task(
        description=(
            "Pergunta do usuario: {pedido}\n"
            "Confira a resposta contra os dados das ferramentas e devolva a resposta final corrigida."
        ),
        expected_output=RESPOSTA_PADRAO,
        agent=verificador,
        context=[responder],
    )
    return Crew(agents=[assistente, verificador], tasks=[responder, conferir])


crew_dupla = crew_gerador_verificador


def crew_supervisor(tools_map: dict, llm):
    """Arquitetura 3 (CrewAI): Supervisor hierárquico + 2 especialistas (buscador e comparador)."""
    from crewai import Agent, Crew, Process, Task

    buscador = Agent(
        role="buscador",
        goal="Buscar rankings e totais de votos no BU",
        backstory="Especialista em busca e agregacao de secoes.",
        tools=_selecionar(
            tools_map,
            ("search_secoes", "total_por_tipo", "total_sem_biometria", "soma_secoes"),
        ),
        llm=llm,
        max_iter=6,
    )
    comparador = Agent(
        role="comparador",
        goal="Comparar e detalhar secoes especificas",
        backstory="Especialista em comparacao de secoes e conferencia numerica.",
        tools=_selecionar(
            tools_map,
            ("secao_details", "compare_secoes", "soma_secoes"),
        ),
        llm=llm,
        max_iter=6,
    )
    gerente = Agent(
        role="supervisor",
        goal="Responder ao usuario delegando para o especialista correto",
        backstory=GERENTE_BACKSTORY,
        llm=llm,
        allow_delegation=True,
        max_iter=8,
    )
    return Crew(
        agents=[buscador, comparador],
        tasks=[
            Task(
                description="Pergunta do usuario: {pedido}",
                expected_output=RESPOSTA_PADRAO,
            )
        ],
        process=Process.hierarchical,
        manager_agent=gerente,
    )


def extrair_tokens_crew(saida, passos: list[dict] | None = None) -> tuple[int, int]:
    """Extrai (tokens_entrada, tokens_saida) do CrewOutput ou dos eventos coletados."""
    inp, out = extrair_tokens_mensagem(saida)
    if inp or out:
        return inp, out
    if passos:
        total_in = 0
        total_out = 0
        for p in passos:
            if p.get("tipo") == "modelo":
                total_in += int(p.get("tokens_entrada", 0) or 0)
                total_out += int(p.get("tokens_saida", 0) or 0)
        return total_in, total_out
    return 0, 0


def mostrar_crew(r: dict, tools_names: set[str] | None = None):
    """Imprime o trace de uma execução CrewAI."""
    estoque = tools_names or set(r.get("ferramentas_chamadas", []))
    for p in r.get("passos", []):
        args = p.get("args") if isinstance(p.get("args"), dict) else {}
        quando = f"[{p['t']:5.1f} s]"
        if p["tipo"] == "delegou":
            print(
                f"{quando} 🧭 supervisor → {args.get('coworker', '?')}: "
                f"{str(args.get('task') or args.get('question', ''))[:140]}"
            )
        elif p["tipo"] == "ferramenta" and p["nome"] in estoque:
            print(
                f"{quando} 🤖 {p['agente']:<10} pede → "
                f"{p['nome']}({json.dumps(args, ensure_ascii=False)})"
            )
            print(f"{'':10}📡 servidor   → {resumo(p['saida'])}")
        elif p["tipo"] == "ferramenta":
            print(
                f"{quando} 📋 {args.get('coworker', '?'):<10} → "
                f"{p['saida'].replace(chr(10), ' ')[:560]}…"
            )
    print(f"\n💬 {r['resposta']}\n")
    print(f"⏱ {r['chamadas_modelo']} chamadas ao modelo · {r['segundos']} s")


async def rodar_crew(
    montar,
    pergunta: str,
    tools_names: set[str] | None = None,
    ver: bool = False,
) -> dict:
    """Executa uma factory de CrewAI e devolve resposta, trajetória e métricas."""
    passos: list[dict] = []
    t0 = time.perf_counter()

    try:
        from crewai.events import crewai_event_bus
        from crewai.events.types.llm_events import LLMCallCompletedEvent
        from crewai.events.types.tool_usage_events import (
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        usar_eventos = True
    except (ImportError, OSError):
        usar_eventos = False

    with ResourceMonitor() as monitor:
        if usar_eventos:
            with crewai_event_bus.scoped_handlers():

                @crewai_event_bus.on(LLMCallCompletedEvent)
                def conta_chamada(fonte, ev):
                    ev_in, ev_out = extrair_tokens_mensagem(ev)
                    passos.append(
                        dict(
                            tipo="modelo",
                            agente=getattr(ev, "agent_role", "?"),
                            t=time.perf_counter() - t0,
                            tokens_entrada=ev_in,
                            tokens_saida=ev_out,
                        )
                    )

                @crewai_event_bus.on(ToolUsageStartedEvent)
                def anota_delegacao(fonte, ev):
                    if tools_names is not None and ev.tool_name not in tools_names:
                        passos.append(
                            dict(
                                tipo="delegou",
                                agente=getattr(ev, "agent_role", "?"),
                                t=time.perf_counter() - t0,
                                args=ev.tool_args,
                            )
                        )

                @crewai_event_bus.on(ToolUsageFinishedEvent)
                def anota_ferramenta(fonte, ev):
                    passos.append(
                        dict(
                            tipo="ferramenta",
                            agente=getattr(ev, "agent_role", "?"),
                            t=time.perf_counter() - t0,
                            nome=ev.tool_name,
                            args=ev.tool_args,
                            saida=str(ev.output),
                        )
                    )

                saida = await montar().kickoff_async(inputs={"pedido": pergunta})
                if hasattr(crewai_event_bus, "flush"):
                    crewai_event_bus.flush()
        else:
            saida = await montar().kickoff_async(inputs={"pedido": pergunta})

    latencia = round(time.perf_counter() - t0, 3)

    if tools_names is not None:
        ferramentas_chamadas = [
            p["nome"]
            for p in passos
            if p["tipo"] == "ferramenta" and p["nome"] in tools_names
        ]
    else:
        ferramentas_chamadas = [
            p["nome"] for p in passos if p["tipo"] == "ferramenta"
        ]

    # Fallback para mocks/execuções sem event bus que anexem metadados na saída
    if not ferramentas_chamadas and hasattr(saida, "ferramentas_chamadas"):
        ferramentas_chamadas = list(getattr(saida, "ferramentas_chamadas") or [])

    chamadas_modelo = sum(p["tipo"] == "modelo" for p in passos)
    if chamadas_modelo == 0:
        usage = getattr(saida, "token_usage", None)
        if usage is not None:
            chamadas_modelo = int(
                (usage.get("successful_requests") if isinstance(usage, dict) else getattr(usage, "successful_requests", 0))
                or 0
            )
        if chamadas_modelo == 0 and hasattr(saida, "chamadas_modelo"):
            chamadas_modelo = int(getattr(saida, "chamadas_modelo") or 0)

    tokens_in, tokens_out = extrair_tokens_crew(saida, passos)
    hw = monitor.snapshot()
    resposta = getattr(saida, "raw", None)
    if resposta is None:
        resposta = str(saida)

    r = {
        "resposta": resposta,
        "ferramentas": ferramentas_chamadas,
        "ferramentas_chamadas": ferramentas_chamadas,
        "chamadas": chamadas_modelo,
        "chamadas_modelo": chamadas_modelo,
        "chamadas_ferramentas": len(ferramentas_chamadas),
        "segundos": round(latencia, 1),
        "latencia_s": latencia,
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "cpu_percent_max": hw["cpu_percent_max"],
        "ram_mb_max": hw["ram_mb_max"],
        "gpu_mem_mb_max": hw["gpu_mem_mb_max"],
        "passos": passos,
    }
    if ver:
        print(f"👤 {pergunta}\n")
        mostrar_crew(r, tools_names=tools_names)
    return r
