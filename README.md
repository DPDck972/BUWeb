# Análise de Boletins de Urna (BUWeb) — agentes LangGraph (Eleições RN)

Projeto acadêmico de análise e auditoria de **Boletins de Urna (BUWeb)** do
Tribunal Superior Eleitoral (TSE), com dados do Rio Grande do Norte. O notebook
compara três formas de responder a perguntas eleitorais: agente único,
Gerador–Verificador e Supervisor–Especialistas.
Na Parte 5, o notebook repete as mesmas arquiteturas no CrewAI para
comparativo direto com o LangGraph, usando as mesmas perguntas e métricas.

As arquiteturas usam **LangGraph** para orquestrar os fluxos e **MCP (Model
Context Protocol)** para conectar os agentes às ferramentas de consulta. O
servidor processa os dados com pandas; o modelo de linguagem interpreta a
pergunta, escolhe ferramentas e redige a resposta em português.

**DCA0305 · Machine Learning-Based Systems Design · UFRN**
Baseado na *Lesson 12 — AI Agents III · Multi-agent*.

---

## 🏛️ O que o projeto faz

O agente consulta o arquivo de BU para responder perguntas sobre:

- **Votos por cargo e tipo**: Governador ou Presidente; votos nominais, brancos
  ou nulos;
- **Seções eleitorais**: busca e ranking por filtros, com detalhes de zona,
  seção e município;
- **Comparação de seções**: aptos, comparecimento, abstenções e votos;
- **Eleitorado sem biometria válida**, no total, por zona ou município;
- **Verificação factual**: na arquitetura Gerador–Verificador, afirmações
  numéricas são conferidas com os resultados das ferramentas.

Os totais e comparações são calculados pelas ferramentas sobre os dados, não
estimados pelo modelo. O modelo não tem uma ferramenta de acesso à internet.

---

## 🧠 Arquiteturas

| Arquitetura | Fluxo | Propósito |
|---|---|---|
| **Agente único** | modelo ⇄ ferramentas | Linha de base ReAct: decide quais consultas executar e responde. |
| **Gerador + Verificador** | gerador ⇄ ferramentas → verificador | Confere a resposta contra os resultados das ferramentas e pode pedir uma revisão limitada. |
| **Supervisor + Especialistas** | supervisor → especialistas → redator | Divide a tarefa entre buscador, comparador e analista de biometria antes de redigir a resposta. |

### Fluxo MCP

```text
┌──────────────────────────────┐       stdio / JSON-RPC       ┌───────────────────────┐
│ Agente LangGraph + Ollama    │ ◄──────────────────────────► │ Servidor FastMCP      │
│ agente único / gerador /     │                               │ ferramentas eleitorais│
│ supervisor e especialistas  │                               └───────────┬───────────┘
└──────────────────────────────┘                                           │
                                                                           ▼
                                                              ┌────────────────────────┐
                                                              │ pandas + CSV BUWeb TSE │
                                                              └────────────────────────┘
```

O cliente LangGraph usa a versão MCP 1.x e o servidor FastMCP usa MCP 2.x.
Por isso, as dependências do servidor são instaladas separadamente em
`servidor_libs/`.

### Ferramentas disponíveis

| Ferramenta | O que consulta |
|---|---|
| `search_secoes` | Busca e ordena seções por tipo de voto, zona, cargo e mínimo de votos. |
| `secao_details` | Detalha aptos, comparecimento, abstenções, biometria e votos de uma seção. |
| `compare_secoes` | Compara duas seções, opcionalmente filtrando por tipo de voto. |
| `total_por_tipo` | Soma votos nominais, brancos ou nulos, com filtro opcional por cargo. |
| `total_sem_biometria` | Soma eleitores sem biometria válida, com filtro opcional por zona ou município. |

O servidor também publica o recurso `eleicao-RN://schema` com as colunas do
arquivo e o prompt `recomendar_analise`.

---

## 📁 Estrutura do projeto

```text
pj_eleicao_multiagente/
├── data/
│   └── bweb_2t_RN_261020260742.csv    # Dados BUWeb do TSE usados pelo projeto
├── notebooks/
│   └── Projeto_AI_Agents_III_Agentes_eleicao.ipynb
├── src/eleicao_ma/
│   ├── arq_a_gerador_verificador.py   # Agente único e arquitetura Gerador–Verificador
│   ├── arq_b_supervisor.py            # Supervisor, especialistas e redator
│   ├── config.py                      # Modelo, caminhos e configuração do MCP
│   ├── data.py                        # Leitura e normalização do CSV
│   ├── mcp_server.py                  # Servidor FastMCP
│   ├── prompts.py                     # Instruções dos agentes
│   ├── tools.py                       # Ferramentas de consulta eleitoral
│   └── utils.py                       # Execução, métricas e exibição
├── tests/
│   ├── test_agent_call_metrics.py     # Métricas e isolamento entre execuções
│   └── test_arqs.py                   # Testes das ferramentas e arquiteturas
├── requirement.txt                    # Dependências do cliente LangGraph
├── requirements-servidor.txt          # Dependências isoladas do FastMCP
└── README.md
```

### Arquivos principais

- <a href="https://github.com/arleswasb/Projeto_AI_Agents_III_Multiagent_eleicao/blob/main/notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb"><img src="https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter" height="20" alt="Notebook" /></a>
  <a href="https://colab.research.google.com/github/arleswasb/Projeto_AI_Agents_III_Multiagent_eleicao/blob/main/notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="20" alt="Open in Colab" /></a>
  [**`notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb`**](./notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb):
  notebook de preparação, experimentos, comparação das arquiteturas e avaliação.
- [**`data/bweb_2t_RN_261020260742.csv`**](./data/bweb_2t_RN_261020260742.csv):
  arquivo BUWeb usado pelas ferramentas e pelos testes.
- [`src/eleicao_ma/`](./src/eleicao_ma/): implementação Python das ferramentas,
  do servidor MCP e das arquiteturas.

---

## ⚙️ Instalação

### Requisitos

- Python 3.10 ou superior;
- [Ollama](https://ollama.com/) instalado e em execução para executar o modelo;
- o CSV em `data/` (já incluído no repositório) ou um caminho definido por
  `BU_DATA_PATH`.

Clone o repositório e crie um ambiente virtual:

```bash
git clone https://github.com/arleswasb/Projeto_AI_Agents_III_Multiagent_eleicao.git
cd Projeto_AI_Agents_III_Multiagent_eleicao

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirement.txt
```

Instale as dependências do servidor MCP isoladamente; não as misture com as do
cliente LangGraph:

```bash
python -m pip install --target servidor_libs -r requirements-servidor.txt
```

O notebook também prepara essas dependências quando executado, se necessário.
Para executar o experimento validado no Colab, o notebook configura o modelo
`qwen3:8b`. Inicie o Ollama e baixe esse modelo:

```bash
ollama serve
```

Em outro terminal:

```bash
ollama pull qwen3:8b
```

---

## 📓 Executar o notebook

- **Google Colab:** use o botão **Open in Colab** acima.
- **Localmente:** abra
  [`notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb`](./notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb)
  no VS Code ou Jupyter e execute as células em ordem, com o ambiente preparado
  e o Ollama ativo.

O notebook organiza o experimento em etapas:

1. prepara e valida os dados BUWeb;
2. configura o servidor MCP, o modelo e as ferramentas;
3. executa o agente único, o Gerador–Verificador e o Supervisor–Especialistas;
4. compara respostas com um gabarito calculado sobre o CSV e com a avaliação de
   um juiz LLM.

---

## 🧪 Testes

Com as dependências instaladas e o arquivo de dados disponível:

```bash
python -m pytest -q
```

Os testes de métricas usam modelos falsos, sem precisar iniciar o Ollama. A
avaliação do notebook e os benchmarks com o modelo real requerem o Ollama e o
CSV.

---

## 📊 Resultados registrados

O notebook registra uma avaliação de **20 perguntas**, com checagem automática,
juiz LLM, quantidade de chamadas ao modelo e tempo:

| Arquitetura | Checagem automática | Juiz LLM | Chamadas | Tempo |
|---|---:|---:|---:|---:|
| Agente único | 16/20 | 19/20 | 37 | 128 s |
| Gerador + Verificador | 17/20 | 19/20 | 73 | 657 s |
| Supervisor | 9/20 | 11/20 | 145 | 356 s |

Esses números correspondem à execução registrada no Colab com `qwen3:8b`.
Representam aquele conjunto de perguntas e aquela execução, não uma garantia de
desempenho geral. O notebook também discute divergências entre a checagem
automática e o juiz LLM.

---

## 🛡️ Fonte dos dados

O arquivo do projeto é baseado nos Boletins de Urna disponibilizados pelo TSE.
Consulte o [Portal de Dados Abertos do TSE](https://dadosabertos.tse.jus.br/)
para informações sobre os dados eleitorais e suas atualizações.
