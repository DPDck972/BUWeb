# Projeto AI Agents III — eleição multiagente

Projeto de análise do BUWeb do Rio Grande do Norte com três arquiteturas
LangGraph: agente único, Gerador–Verificador e Supervisor–Especialistas. O
notebook principal, validado no Colab e sem a seção CrewAI, é
`notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb`.

## Preparar o ambiente

Use Python e instale as dependências do cliente LangGraph:

```bash
python -m pip install -r requirement.txt
```

O servidor MCP usa FastMCP em um diretório isolado para não misturar suas
dependências com as do cliente LangGraph:

```bash
python -m pip install --target servidor_libs -r requirements-servidor.txt
```

O notebook também instala essa dependência isolada se ela ainda não existir.
Mantenha `data/bweb_2t_RN_261020260742.csv` no projeto ou defina `BU_DATA_PATH`.

Instale e inicie o Ollama, baixe o modelo configurado e mantenha o servidor ativo:

```bash
ollama pull qwen3:8b
ollama serve
```

Abra `notebooks/Projeto_AI_Agents_III_Agentes_eleicao.ipynb`, selecione o
ambiente Python preparado e execute as células em ordem. O notebook baixa e
valida o CSV oficial do TSE, prepara as dependências do cliente e do servidor
MCP e verifica o Ollama e o modelo `qwen3:8b`.

## Testes

```bash
python -m pytest -q
```

Os testes de métricas usam modelos falsos e não precisam iniciar Ollama. A
execução das arquiteturas e a repetição do benchmark dependem do Ollama e do
CSV; os resultados registrados no notebook correspondem à execução validada
no Colab.
