"""Pipeline de avaliação: ambiente, trajetória, juiz MCP, CSV por execução e verificação manual."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_PATH, MODEL, OLLAMA_URL, ROOT, SERVER_LIBS, get_servidor_config
from .judge import MCPJudge, get_judge_model, get_judge_provider

PERGUNTAS_PATH = ROOT / "data" / "perguntas.json"
DEFAULT_RESULTS_DIR = ROOT / "results"

CATEGORIAS_VALIDAS = {
    "item_especifico",
    "busca_filtro",
    "comparacao",
    "calculo",
    "capciosa",
    "fora_escopo",
}

GATILHOS_RECUSA = (
    "não existe",
    "não está",
    "não tenho",
    "não posso",
    "não fornece",
    "não disponível",
    "não consta",
    "fora do escopo",
    "não faz parte",
    "não há",
    "só respondo",
    "apenas sobre",
    "não contempla",
    "não encontrad",
    "não foi encontrado",
)

COLUNAS_RESULTADOS: tuple[str, ...] = (
    "sistema",
    "framework",
    "arquitetura",
    "pergunta_id",
    "categoria",
    "ferramenta_esperada",
    "veredito_juiz",
    "justificativa_juiz",
    "consultas_juiz_mcp",
    "trajetoria_ok",
    "ferramentas_chamadas",
    "latencia_s",
    "tokens_entrada",
    "tokens_saida",
    "tokens_total",
    "chamadas_modelo",
    "chamadas_ferramentas",
    "cpu_percent_max",
    "ram_mb_max",
    "gpu_mem_mb_max",
    "resposta",
)

COLUNAS_VERIFICACAO_MANUAL: tuple[str, ...] = (
    "sistema",
    "pergunta_id",
    "categoria",
    "pergunta",
    "resposta",
    "veredito_juiz",
    "justificativa_juiz",
    "veredito_manual",
)


def carregar_perguntas(path: str | Path | None = None) -> list[dict]:
    """Carrega e valida minimamente o conjunto de perguntas estruturadas em JSON."""
    target = Path(path) if path is not None else PERGUNTAS_PATH
    with target.open("r", encoding="utf-8") as f:
        perguntas = json.load(f)
    if not isinstance(perguntas, list) or not perguntas:
        raise ValueError(f"Arquivo de perguntas inválido ou vazio: {target}")
    return perguntas


def avaliar_trajetoria(
    pergunta_item: dict,
    ferramentas_chamadas: list[str],
    resposta: str = "",
) -> bool:
    """
    Calcula `trajetoria_ok` automaticamente para uma execução:

    - Quando `ferramenta_esperada` é uma string (nome da ferramenta MCP):
      retorna True se `ferramenta_esperada` foi chamada ao menos uma vez.
    - Quando `ferramenta_esperada` é `None` (pergunta fora de escopo):
      retorna True se o agente recusou explicitamente (algum gatilho em
      `GATILHOS_RECUSA` ou em `checar`) e não violou nenhum termo em `proibido`.
    """
    esperada = pergunta_item.get("ferramenta_esperada")
    if esperada is not None:
        return esperada in (ferramentas_chamadas or [])

    resp_low = (resposta or "").lower()
    gatilhos = tuple(GATILHOS_RECUSA) + tuple(
        str(t).lower() for t in pergunta_item.get("checar", [])
    )
    recusou = any(g in resp_low for g in gatilhos)
    violou_proibido = any(
        re.search(re.escape(str(t).lower()), resp_low)
        for t in pergunta_item.get("proibido", [])
    )
    return bool(recusou and not violou_proibido)


def _versao_pacote(nome: str, extra_dir: str | None = None) -> str | None:
    try:
        return pkg_version(nome)
    except PackageNotFoundError:
        pass
    if extra_dir and Path(extra_dir).is_dir():
        for dist_info in Path(extra_dir).glob(f"{nome.replace('-', '_')}-*.dist-info"):
            meta = dist_info / "METADATA"
            if meta.is_file():
                for line in meta.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
    return None


def _detectar_quantizacao_ollama(modelo: str) -> str | None:
    if not shutil.which("ollama"):
        return None
    try:
        proc = subprocess.run(
            ["ollama", "show", modelo],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        for line in proc.stdout.splitlines():
            if "quantization" in line.lower():
                partes = line.split()
                if len(partes) >= 2:
                    return partes[-1].strip()
    except Exception:
        return None
    return None


def _detectar_gpu_info() -> str | None:
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            linhas = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if linhas:
                return "; ".join(linhas)
        except Exception:
            pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def coletar_ambiente(
    modelo_local: str = MODEL,
    modelo_juiz: str | None = None,
    provedor_juiz: str | None = None,
    server_libs: str | None = SERVER_LIBS,
) -> dict[str, Any]:
    """Coleta metadados reprodutíveis de hardware, modelos e versões de bibliotecas."""
    try:
        import psutil

        cpu_fisicos = psutil.cpu_count(logical=False)
        cpu_logicos = psutil.cpu_count(logical=True)
        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        cpu_fisicos = None
        cpu_logicos = os.cpu_count()
        ram_total_gb = None

    cpu_modelo = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_modelo = line.split(":", 1)[1].strip()
                break

    prov_resolvido = get_judge_provider(provedor_juiz)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sistema_operacional": platform.platform(),
        "cpu": {
            "modelo": cpu_modelo,
            "nucleos_fisicos": cpu_fisicos,
            "nucleos_logicos": cpu_logicos,
        },
        "ram_total_gb": ram_total_gb,
        "gpu": _detectar_gpu_info(),
        "modelo_local": modelo_local,
        "quantizacao_modelo_local": _detectar_quantizacao_ollama(modelo_local),
        "provedor_juiz": prov_resolvido,
        "modelo_juiz": get_judge_model(modelo_juiz, provider=prov_resolvido),
        "versoes": {
            "python": platform.python_version(),
            "langgraph": _versao_pacote("langgraph"),
            "crewai": _versao_pacote("crewai"),
            "mcp": _versao_pacote("mcp"),
            "fastmcp": _versao_pacote("fastmcp", extra_dir=server_libs),
            "pandas": _versao_pacote("pandas"),
        },
    }


def salvar_ambiente(
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    modelo_local: str = MODEL,
    modelo_juiz: str | None = None,
    provedor_juiz: str | None = None,
) -> Path:
    """Grava `results/ambiente.json` e devolve o caminho gerado."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ambiente = coletar_ambiente(
        modelo_local=modelo_local,
        modelo_juiz=modelo_juiz,
        provedor_juiz=provedor_juiz,
    )
    dest = out / "ambiente.json"
    dest.write_text(json.dumps(ambiente, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def resumo_metricas_por_sistema(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega acerto do juiz, trajetória, latência (média/mediana/p95), tokens e chamadas."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["sistema", "framework", "arquitetura"], sort=False, as_index=False)
        .agg(
            perguntas=("pergunta_id", "count"),
            acertos_juiz=("veredito_juiz", "sum"),
            taxa_acerto_juiz=("veredito_juiz", "mean"),
            taxa_trajetoria_ok=("trajetoria_ok", "mean"),
            latencia_media_s=("latencia_s", "mean"),
            latencia_mediana_s=("latencia_s", "median"),
            latencia_p95_s=("latencia_s", lambda s: float(s.quantile(0.95))),
            latencia_total_s=("latencia_s", "sum"),
            tokens_entrada_total=("tokens_entrada", "sum"),
            tokens_saida_total=("tokens_saida", "sum"),
            tokens_total=("tokens_total", "sum"),
            chamadas_modelo_total=("chamadas_modelo", "sum"),
            chamadas_ferramentas_total=("chamadas_ferramentas", "sum"),
            cpu_percent_pico=("cpu_percent_max", "max"),
            ram_mb_pico=("ram_mb_max", "max"),
        )
    )


async def executar_benchmark(
    sistemas: dict[str, Any],
    *,
    perguntas: list[dict] | None = None,
    perguntas_path: str | Path | None = None,
    judge: MCPJudge | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_delay_s: float = 1.0,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    modelo_local: str = MODEL,
    salvar_copia_timestamp: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """
    Executa o benchmark completo para os sistemas fornecidos via `answer(question)`.

    Retorna `(df_resultados, df_verificacao_manual, caminhos_arquivos)`.
    """
    lista_perguntas = (
        perguntas if perguntas is not None else carregar_perguntas(perguntas_path)
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ambiente_path = salvar_ambiente(
        output_dir=out,
        modelo_local=modelo_local,
        modelo_juiz=judge.model if judge is not None else judge_model,
        provedor_juiz=judge.provider if judge is not None else judge_provider,
    )

    linhas_resultados: list[dict[str, Any]] = []
    linhas_manual: list[dict[str, Any]] = []

    async def _rodar_com_juiz(active_judge: MCPJudge):
        for nome_sistema, sistema_obj in sistemas.items():
            if verbose:
                print(f"\n=== Executando sistema: {nome_sistema} ===")
            for p in lista_perguntas:
                pid = p["id"]
                pergunta_txt = p["pergunta"]
                categoria = p["categoria"]
                ferr_esp = p.get("ferramenta_esperada")

                try:
                    res = await sistema_obj.answer(pergunta_txt)
                except Exception as exc:
                    res = {
                        "sistema": nome_sistema,
                        "framework": getattr(sistema_obj, "framework", "?"),
                        "arquitetura": getattr(sistema_obj, "arquitetura", "?"),
                        "resposta": f"ERRO DE EXECUÇÃO: {exc}",
                        "ferramentas_chamadas": [],
                        "chamadas_modelo": 0,
                        "chamadas_ferramentas": 0,
                        "latencia_s": 0.0,
                        "tokens_entrada": 0,
                        "tokens_saida": 0,
                        "tokens_total": 0,
                        "cpu_percent_max": 0.0,
                        "ram_mb_max": 0.0,
                        "gpu_mem_mb_max": None,
                    }

                traj_ok = avaliar_trajetoria(
                    p,
                    res["ferramentas_chamadas"],
                    res["resposta"],
                )

                aval = await active_judge.avaliar(
                    pergunta=pergunta_txt,
                    resposta=res["resposta"],
                    categoria=categoria,
                    esperado=p.get("esperado"),
                )

                linhas_resultados.append(
                    {
                        "sistema": res["sistema"],
                        "framework": res["framework"],
                        "arquitetura": res["arquitetura"],
                        "pergunta_id": pid,
                        "categoria": categoria,
                        "ferramenta_esperada": ferr_esp,
                        "veredito_juiz": bool(aval["veredito_juiz"]),
                        "justificativa_juiz": aval["justificativa_juiz"],
                        "consultas_juiz_mcp": json.dumps(
                            aval["consultas_juiz_mcp"], ensure_ascii=False
                        ),
                        "trajetoria_ok": bool(traj_ok),
                        "ferramentas_chamadas": json.dumps(
                            res["ferramentas_chamadas"], ensure_ascii=False
                        ),
                        "latencia_s": res["latencia_s"],
                        "tokens_entrada": res["tokens_entrada"],
                        "tokens_saida": res["tokens_saida"],
                        "tokens_total": res["tokens_total"],
                        "chamadas_modelo": res["chamadas_modelo"],
                        "chamadas_ferramentas": res["chamadas_ferramentas"],
                        "cpu_percent_max": res["cpu_percent_max"],
                        "ram_mb_max": res["ram_mb_max"],
                        "gpu_mem_mb_max": res["gpu_mem_mb_max"],
                        "resposta": res["resposta"],
                    }
                )

                linhas_manual.append(
                    {
                        "sistema": res["sistema"],
                        "pergunta_id": pid,
                        "categoria": categoria,
                        "pergunta": pergunta_txt,
                        "resposta": res["resposta"],
                        "veredito_juiz": bool(aval["veredito_juiz"]),
                        "justificativa_juiz": aval["justificativa_juiz"],
                        "veredito_manual": "",
                    }
                )

                if verbose:
                    st_j = "✅" if aval["veredito_juiz"] else "❌"
                    st_t = "✅" if traj_ok else "❌"
                    print(
                        f"  [{pid} · {categoria}] Juiz: {st_j} | Traj: {st_t} | "
                        f"{res['latencia_s']:.2f}s | {res['chamadas_modelo']} calls"
                    )

    if judge is not None:
        await _rodar_com_juiz(judge)
    else:
        async with MCPJudge(
            model=judge_model,
            provider=judge_provider,
            request_delay_s=judge_delay_s,
        ) as active_judge:
            await _rodar_com_juiz(active_judge)

    df_resultados = pd.DataFrame(linhas_resultados, columns=list(COLUNAS_RESULTADOS))
    df_manual = pd.DataFrame(linhas_manual, columns=list(COLUNAS_VERIFICACAO_MANUAL))

    csv_resultados = out / "resultados_execucoes.csv"
    csv_manual = out / "verificacao_manual.csv"
    df_resultados.to_csv(csv_resultados, index=False)
    df_manual.to_csv(csv_manual, index=False)

    arquivos: dict[str, Path] = {
        "ambiente": ambiente_path,
        "resultados": csv_resultados,
        "verificacao_manual": csv_manual,
    }

    if salvar_copia_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        copia_path = out / f"resultados_execucoes_{ts}.csv"
        df_resultados.to_csv(copia_path, index=False)
        arquivos["resultados_timestamp"] = copia_path

    return df_resultados, df_manual, arquivos


async def executar_benchmark_completo_cli(
    *,
    perguntas_path: str | Path = PERGUNTAS_PATH,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    modelo_local: str = MODEL,
    ollama_url: str = OLLAMA_URL,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_delay_s: float = 1.0,
    sistemas_filtro: list[str] | None = None,
    data_path: str | Path | None = None,
):
    """Inicializa LangGraph + CrewAI com o servidor MCP unificado e executa o benchmark."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_ollama import ChatOllama

    from .crew_arqs import build_crew_llm
    from .systems import SYSTEM_SPECS, criar_sistemas_crewai, criar_sistemas_langgraph

    selecionados = set(sistemas_filtro) if sistemas_filtro else set(SYSTEM_SPECS)
    desconhecidos = selecionados - set(SYSTEM_SPECS)
    if desconhecidos:
        raise ValueError(f"Sistemas desconhecidos: {sorted(desconhecidos)}")

    servidor_cfg = get_servidor_config(data_path=data_path or DATA_PATH)
    sistemas: dict[str, Any] = {}
    mcp_crew = None

    try:
        if any(s.startswith("langgraph_") for s in selecionados):
            lg_tools = await MultiServerMCPClient(servidor_cfg).get_tools()
            lg_llm = ChatOllama(
                model=modelo_local,
                base_url=ollama_url,
                temperature=0,
                num_ctx=8192,
                num_predict=600,
                reasoning=False,
            )
            for nome, obj in criar_sistemas_langgraph(lg_tools, lg_llm).items():
                if nome in selecionados:
                    sistemas[nome] = obj

        if any(s.startswith("crewai_") for s in selecionados):
            from crewai_tools import MCPServerAdapter
            from mcp import StdioServerParameters

            eleicao_cfg = servidor_cfg["eleicao"]
            mcp_crew = MCPServerAdapter(
                StdioServerParameters(
                    command=eleicao_cfg["command"],
                    args=eleicao_cfg["args"],
                    env=eleicao_cfg["env"],
                )
            )
            crew_tools_map = {t.name: t for t in mcp_crew.tools}
            crew_llm = build_crew_llm(model=modelo_local, base_url=ollama_url)
            for nome, obj in criar_sistemas_crewai(crew_tools_map, crew_llm).items():
                if nome in selecionados:
                    sistemas[nome] = obj

        return await executar_benchmark(
            sistemas,
            perguntas_path=perguntas_path,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_delay_s=judge_delay_s,
            output_dir=output_dir,
            modelo_local=modelo_local,
        )
    finally:
        if mcp_crew is not None:
            try:
                mcp_crew.stop()
            except Exception:
                pass


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Executa o benchmark multiagente BUWeb com Juiz LLM (Groq openai/gpt-oss-120b ou HF) + MCP."
    )
    parser.add_argument(
        "--perguntas",
        default=str(PERGUNTAS_PATH),
        help="Caminho para data/perguntas.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Diretório onde salvar ambiente.json e CSVs",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Modelo local no Ollama (padrão: qwen3:8b)",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=["groq", "huggingface"],
        help="Provedor do juiz (padrão: JUDGE_PROVIDER ou groq)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Modelo do juiz (padrão Groq: openai/gpt-oss-120b; HF: Qwen/Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--delay-judge",
        type=float,
        default=1.0,
        help="Atraso em segundos entre chamadas ao juiz para evitar rate limit",
    )
    parser.add_argument(
        "--sistemas",
        nargs="+",
        default=None,
        help="Subconjunto opcional de sistemas (ex.: langgraph_unico crewai_unico)",
    )
    args = parser.parse_args(argv)

    df_res, _df_manual, arquivos = asyncio.run(
        executar_benchmark_completo_cli(
            perguntas_path=args.perguntas,
            output_dir=args.output_dir,
            modelo_local=args.model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_delay_s=args.delay_judge,
            sistemas_filtro=args.sistemas,
        )
    )
    print("\n=== RESUMO POR SISTEMA ===")
    print(resumo_metricas_por_sistema(df_res).to_string(index=False))
    print("\nArquivos gerados:")
    for chave, caminho in arquivos.items():
        print(f"  - {chave}: {caminho}")


if __name__ == "__main__":
    main()
