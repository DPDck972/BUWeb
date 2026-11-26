"""Testes diretos do servidor FastMCP (sem LLM/Ollama) via protocolo MCP stdio."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from eleicao_ma import config, tools

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "bweb_2t_RN_261020260742.csv"
SERVER_LIBS = ROOT / "servidor_libs"
SERVER_PY = ROOT / "src" / "eleicao_ma" / "mcp_server.py"

EXPECTED_TOOLS = {
    "search_secoes",
    "secao_details",
    "compare_secoes",
    "total_por_tipo",
    "total_sem_biometria",
    "soma_secoes",
}


def _mcp_server_params() -> StdioServerParameters:
    if not CSV_PATH.exists():
        pytest.skip("CSV de dados não encontrado")
    env = os.environ.copy()
    if (SERVER_LIBS / "fastmcp").exists():
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{SERVER_LIBS}{os.pathsep}{existing_pp}"
            if existing_pp
            else str(SERVER_LIBS)
        )
    env["BU_DATA_PATH"] = str(CSV_PATH)
    env["FASTMCP_LOG_LEVEL"] = "ERROR"
    env["FASTMCP_SHOW_SERVER_BANNER"] = "false"
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PY)],
        env=env,
    )


def _extract_json(call_result) -> dict:
    assert not call_result.isError, f"MCP retornou isError=True: {call_result}"
    text_parts = [
        getattr(block, "text", str(block)) for block in (call_result.content or [])
    ]
    raw = "".join(text_parts).strip()
    assert raw, "Resposta vazia do servidor MCP"
    return json.loads(raw)


def _run_with_session(coro_fn):
    async def runner():
        params = _mcp_server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await coro_fn(session)

    return asyncio.run(runner())


def test_mcp_server_tools_valid_cases():
    """Valida todas as ferramentas via servidor MCP real com o CSV do TSE."""

    async def check(session: ClientSession):
        listed = await session.list_tools()
        names = {t.name for t in listed.tools}
        assert EXPECTED_TOOLS.issubset(names)

        # 1. total_por_tipo
        r_total = _extract_json(
            await session.call_tool(
                "total_por_tipo", {"tipo_voto": "Nulo", "cargo": "Presidente"}
            )
        )
        assert r_total["total_votos"] > 0
        assert r_total["tipo_voto"] == "Nulo"
        assert r_total["cargo"] == "Presidente"

        # 2. total_sem_biometria (total geral)
        r_bio = _extract_json(await session.call_tool("total_sem_biometria", {}))
        assert r_bio["total_sem_biometria"] > 0
        assert r_bio["secoes"] > 0

        # 3. secao_details
        r_det = _extract_json(
            await session.call_tool("secao_details", {"zona": 3, "secao": 505})
        )
        assert r_det["zona"] == 3
        assert r_det["secao"] == 505
        assert r_det["municipio"] == "NATAL"
        assert r_det["aptos"] == 460
        assert r_det["comparecimento"] == 432

        # 4. compare_secoes
        r_cmp = _extract_json(
            await session.call_tool(
                "compare_secoes",
                {
                    "zona_a": 3,
                    "secao_a": 505,
                    "zona_b": 3,
                    "secao_b": 504,
                    "tipo_voto": "Nulo",
                },
            )
        )
        assert "best" in r_cmp
        assert r_cmp["best"]["mais_votos"] == ["3/504"]
        assert "17" in r_cmp["resposta"] and "14" in r_cmp["resposta"]

        # 5. search_secoes
        r_search = _extract_json(
            await session.call_tool(
                "search_secoes",
                {"tipo_voto": "Nulo", "cargo": "Presidente", "top_n": 3},
            )
        )
        assert len(r_search["secoes"]) == 3
        assert r_search["secoes"][0]["ZONA_N"] == 25
        assert r_search["secoes"][0]["SECAO_N"] == 105

        # 6. soma_secoes
        r_soma = _extract_json(
            await session.call_tool(
                "soma_secoes",
                {
                    "zona_a": 3,
                    "secao_a": 505,
                    "zona_b": 3,
                    "secao_b": 504,
                    "tipo_voto": "Nulo",
                },
            )
        )
        assert r_soma["total"] == 31

    _run_with_session(check)


def test_mcp_server_invalid_params_return_actionable_errors():
    """Parâmetros inválidos chegam ao tools.py e devolvem erro claro sem exceção não tratada."""

    async def check(session: ClientSession):
        # search_secoes: top_n inválido / <= 0 / zona 0 / tipo_voto inválido / cargo inválido
        r_top_abc = _extract_json(
            await session.call_tool("search_secoes", {"top_n": "abc"})
        )
        assert "error" in r_top_abc and "top_n" in r_top_abc["error"]

        r_top_zero = _extract_json(
            await session.call_tool("search_secoes", {"top_n": 0})
        )
        assert "error" in r_top_zero and "maior que zero" in r_top_zero["error"]

        r_top_neg = _extract_json(
            await session.call_tool("search_secoes", {"top_n": -2})
        )
        assert "error" in r_top_neg

        r_zona0 = _extract_json(
            await session.call_tool("search_secoes", {"zona": 0})
        )
        assert "error" in r_zona0 and "zona 0" in r_zona0["error"]

        r_tipo_inv = _extract_json(
            await session.call_tool("total_por_tipo", {"tipo_voto": "invalido"})
        )
        assert "error" in r_tipo_inv
        for esperado in ("Nulo", "Branco", "Nominal"):
            assert esperado in r_tipo_inv["error"]

        r_cargo_inv = _extract_json(
            await session.call_tool("total_por_tipo", {"cargo": "Senador"})
        )
        assert "error" in r_cargo_inv
        assert "Governador" in r_cargo_inv["error"] and "Presidente" in r_cargo_inv["error"]

        # secao_details: seção inexistente e parâmetro não numérico
        r_sec_missing = _extract_json(
            await session.call_tool("secao_details", {"zona": 3, "secao": 999})
        )
        assert "error" in r_sec_missing and "não encontrada" in r_sec_missing["error"]

        r_sec_abc = _extract_json(
            await session.call_tool("secao_details", {"zona": "abc", "secao": 505})
        )
        assert "error" in r_sec_abc and "inteiro" in r_sec_abc["error"]

        # compare_secoes: tipo_voto inválido deve listar opções aceitas
        r_cmp_inv = _extract_json(
            await session.call_tool(
                "compare_secoes",
                {
                    "zona_a": 3,
                    "secao_a": 505,
                    "zona_b": 3,
                    "secao_b": 504,
                    "tipo_voto": "invalido",
                },
            )
        )
        assert "error" in r_cmp_inv
        for esperado in ("Nulo", "Branco", "Nominal"):
            assert esperado in r_cmp_inv["error"]

        # compare_secoes: seção inexistente
        r_cmp_missing = _extract_json(
            await session.call_tool(
                "compare_secoes",
                {"zona_a": 3, "secao_a": 505, "zona_b": 3, "secao_b": 500},
            )
        )
        assert "error" in r_cmp_missing and "3/500" in r_cmp_missing["error"]

        # total_sem_biometria: zona 0
        r_bio_z0 = _extract_json(
            await session.call_tool("total_sem_biometria", {"zona": 0})
        )
        assert "error" in r_bio_z0 and "zona 0" in r_bio_z0["error"]

        # soma_secoes: tipo_voto inválido e seção inexistente
        r_soma_inv = _extract_json(
            await session.call_tool(
                "soma_secoes",
                {
                    "zona_a": 3,
                    "secao_a": 505,
                    "zona_b": 3,
                    "secao_b": 504,
                    "tipo_voto": "invalido",
                },
            )
        )
        assert "error" in r_soma_inv
        for esperado in ("Nulo", "Branco", "Nominal"):
            assert esperado in r_soma_inv["error"]

        r_soma_missing = _extract_json(
            await session.call_tool(
                "soma_secoes",
                {"zona_a": 3, "secao_a": 505, "zona_b": 3, "secao_b": 500},
            )
        )
        assert "error" in r_soma_missing and "3/500" in r_soma_missing["error"]

    _run_with_session(check)


def test_mcp_server_empty_optional_strings_treated_as_omitted():
    """Strings vazias ou com espaços em filtros opcionais não devem falhar."""

    async def check(session: ClientSession):
        r_bio_geral = _extract_json(await session.call_tool("total_sem_biometria", {}))
        r_bio_empty = _extract_json(
            await session.call_tool(
                "total_sem_biometria", {"zona": "   ", "municipio": ""}
            )
        )
        assert r_bio_empty["total_sem_biometria"] == r_bio_geral["total_sem_biometria"]
        assert r_bio_empty["secoes"] == r_bio_geral["secoes"]

        r_total_empty = _extract_json(
            await session.call_tool("total_por_tipo", {"tipo_voto": "", "cargo": "  "})
        )
        assert r_total_empty["total_votos"] > 0

        r_search_empty = _extract_json(
            await session.call_tool(
                "search_secoes",
                {"tipo_voto": "", "zona": " ", "cargo": "", "min_votos": "", "top_n": ""},
            )
        )
        assert len(r_search_empty["secoes"]) == 5

        r_cmp_empty = _extract_json(
            await session.call_tool(
                "compare_secoes",
                {
                    "zona_a": 3,
                    "secao_a": 505,
                    "zona_b": 3,
                    "secao_b": 504,
                    "tipo_voto": "  ",
                },
            )
        )
        assert "best" in r_cmp_empty
        assert r_cmp_empty["tipo_voto_aplicado"] is None

        r_soma_empty = _extract_json(
            await session.call_tool(
                "soma_secoes",
                {
                    "zona_a": 3,
                    "secao_a": 505,
                    "zona_b": 3,
                    "secao_b": 504,
                    "tipo_voto": "",
                },
            )
        )
        assert r_soma_empty["total"] > 0
        assert r_soma_empty["tipo_voto"] is None

    _run_with_session(check)


def test_mcp_server_resource_schema_and_prompt():
    """Valida leitura do resource eleicao-RN://schema e renderização do prompt recomendar_analise."""

    async def check(session: ClientSession):
        # Resource eleicao-RN://schema (URI scheme normalizado para minúsculas por RFC 3986)
        resources = await session.list_resources()
        uris = {str(r.uri).lower() for r in resources.resources}
        assert "eleicao-rn://schema" in uris

        schema_res = await session.read_resource("eleicao-rn://schema")
        schema_text = "".join(
            getattr(c, "text", "") for c in (schema_res.contents or [])
        )
        for col in (
            "DS_CARGO_PERGUNTA",
            "DS_TIPO_VOTAVEL",
            "QT_VOTOS",
            "QT_APTOS",
            "QT_COMPARECIMENTO",
            "QT_ABSTENCOES",
            "QT_ELEI_BIOM_SEM_HABILITACAO",
        ):
            assert col in schema_text

        # Prompt recomendar_analise
        prompts = await session.list_prompts()
        prompt_names = {p.name for p in prompts.prompts}
        assert "recomendar_analise" in prompt_names

        rendered = await session.get_prompt(
            "recomendar_analise",
            {"cargo": "Presidente", "tipo_voto": "Nulo", "top_n": "3"},
        )
        assert rendered.messages
        prompt_text = " ".join(
            getattr(m.content, "text", str(m.content)) for m in rendered.messages
        )
        assert "Presidente" in prompt_text
        assert "Nulo" in prompt_text
        assert "search_secoes" in prompt_text
        assert "top_n=3" in prompt_text

    _run_with_session(check)


def test_tools_direct_edge_cases():
    """Garante comportamento direto em tools.py para tolerância de cargo e top_n."""
    assert "error" in tools.search_secoes(top_n="abc")
    assert "error" in tools.search_secoes(top_n=0)
    assert "error" in tools.compare_secoes(3, 505, 3, 504, tipo_voto="invalido")
    assert "Nulo" in tools.compare_secoes(3, 505, 3, 504, tipo_voto="invalido")["error"]
    # Tolerância: passar tipo de voto no campo cargo redireciona para tipo_voto
    r = tools.search_secoes(cargo="Nulo", top_n=2)
    assert "secoes" in r and len(r["secoes"]) == 2
    assert r["filtros_aplicados"]["tipo_voto"] == "Nulo"
    assert config.SERVER_PY.endswith("mcp_server.py")
