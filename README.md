# 🎓 Gestão Acadêmica EPTNM — CEFET-MG

Aplicação web (Streamlit) para os coordenadores dos cursos técnicos de nível
médio (**EPTNM**) do CEFET-MG. O coordenador informa seu **e-mail institucional**,
envia o *Mapa de Turma* (`.xls`) e recebe o **relatório de acompanhamento
acadêmico (PDF) por e-mail**.

Evolução dos scripts originais que rodavam no Google Colab (no histórico do
repositório), agora como uma página acessível pelo navegador e aberta a qualquer
coordenador do EPTNM — não só Trânsito/Estradas.

## ✨ Como funciona

1. O coordenador informa o **e-mail `@cefetmg.br`** (o serviço só envia para esse
   domínio — é assim que se garante o uso restrito a professores do CEFET).
2. Envia os **Mapas de Turma** (`.xls`) de **um ou mais bimestres** (até 4 arquivos por curso, um para cada bimestre). O nome do curso, a turma e o bimestre de cada arquivo são lidos automaticamente de seus cabeçalhos.
3. O app processa notas e faltas, calcula estatísticas (com o **limiar de
   aprovação ajustado ao bimestre** — 60% de 20 ou 30 pontos), gera gráficos,
   destaca os alunos com mais disciplinas críticas e os com mais faltas, e
   monta o relatório em PDF com **sumário (TOC)**.
4. O PDF é **enviado por e-mail** ao coordenador. A tela mostra apenas a
   confirmação — nada de download nem dados expostos na página.

> **Arquivos individuais**: o app só processa arquivos cujos cabeçalhos indicam um
> bimestre individual (1º, 2º, 3º ou 4º Bimestre). Mapas agregados (contendo múltiplos bimestres em um único arquivo) são
> rejeitados, mas você pode enviar até 4 arquivos individuais simultaneamente.

### Pontuação por bimestre (CEFET-MG)

| Bimestre | Pontuação máxima | Limiar de aprovação parcial (60%) |
|---|---|---|
| 1º | 20 | 12,0 |
| 2º | 30 | 18,0 |
| 3º | 20 | 12,0 |
| 4º | 30 | 18,0 |

O limiar é aplicado em todas as estatísticas (taxa de aprovação geral, alunos
críticos, disciplina mais crítica etc.).

### Análise de faltas

O relatório inclui uma seção de **frequência por sinal estatístico**: para cada
disciplina mostra média, mediana, P90, desvio padrão e o número de alunos acima
de P90 / acima de média+2σ. Também lista os 10 alunos com mais faltas no
bimestre e gráficos de distribuição. Esse sinal **não substitui** o limite legal
de 25% da carga horária — ele apenas aponta, sem depender do calendário, quem
merece atenção.

### Estimativa de risco de repetência

O relatório apresenta um índice heurístico (0% a 100%) com o risco estimado de repetência dos 10 alunos mais críticos. Ele é calculado por meio de uma média ponderada entre a proporção de disciplinas abaixo do limiar de aprovação (peso de 70%) e a frequência de faltas em relação ao P90 da turma (peso de 30%). Caso os dados de faltas não estejam disponíveis, o cálculo é baseado unicamente no componente de disciplinas com notas (peso de 100%).

> **Limitação importante**: Este índice é uma estimativa de sinalização baseada apenas nas notas e faltas do bimestre atual. Como a aplicação é *stateless* e processa cada bimestre de forma isolada, ela não realiza um consolidado anual nem possui histórico de retenções anteriores. Portanto, o indicador é puramente informativo e não substitui ou prediz os critérios oficiais de retenção do CEFET-MG (frequência global anual mínima de 75% e média anual).

### Análise multibimestral

Ao enviar mais de um arquivo do mesmo curso/turma (até 4 arquivos no máximo, um para cada bimestre), o aplicativo ativa automaticamente a **Análise Multibimestral** no relatório. Esta análise consolida as informações dos bimestres fornecidos e traz novas métricas por aluno:

- **Evolução das Médias**: histórico temporal da média geral do aluno ao longo dos bimestres enviados.
- **Média Anual**: estimativa da média anual baseada nos pesos oficiais do CEFET-MG (20/30/20/30). É rotulada como **parcial** se faltarem bimestres no envio, ou **completa** quando os 4 bimestres são fornecidos.
- **Faltas Acumuladas**: soma das faltas do aluno ao longo de todos os bimestres enviados (com aviso de "parcial" se menor que 4 bimestres).
- **Tendência**: tendência de desempenho de notas comparando os dois últimos bimestres enviados, permitindo destacar os 10 alunos com maior queda.

> **Comportamento com 1 arquivo**: caso seja enviado apenas 1 arquivo (ou 1 par, no caso de Trânsito + Estradas), o comportamento do aplicativo é **idêntico ao atual**, sem a inclusão de novas seções multibimestrais no relatório PDF.

### Caso especial: Trânsito + Estradas (1ª série)

Há um **seletor de modo no corpo da página**, logo acima dos campos de envio,
com as opções "Curso único (qualquer curso do EPTNM)" e "Curso integrado
Trânsito + Estradas (1ª série)". No modo integrado, o app pede **de 1 a 4
arquivos de cada lado, pareados por bimestre** (mapas de Trânsito e mapas de
Estradas, porque o de Estradas traz o ensino médio compartilhado) e produz
**2 relatórios** — um para cada curso — anexados no mesmo e-mail. No modo
"Curso único", basta enviar os arquivos no campo único e as disciplinas são
extraídas dinamicamente do próprio mapa.

## 🗂️ Estrutura

```
.
├── app.py                  # Interface Streamlit (formulário, validação, envio)
├── core/
│   ├── disciplinas.py      # Catálogo de nomes amigáveis de disciplinas
│   ├── manipulacao.py      # Leitura/processamento dos .xls -> DataFrames
│   ├── relatorios.py       # Estatísticas, gráficos, IA e geração do PDF
│   └── email_sender.py     # Validação de e-mail e envio SMTP (Gmail)
├── assets/                 # Logo institucional opcional (logo_cefet.png)
├── tests/                  # Testes unitários (conftest.py, test_transito_estradas.py)
├── .streamlit/
│   ├── config.toml         # Tema
│   └── secrets.toml.example
└── requirements.txt
```

## ⚙️ Configuração (segredos)

Copie `.streamlit/secrets.toml.example` para `.streamlit/secrets.toml` (local) ou
cole o conteúdo em **Settings → Secrets** no Streamlit Cloud:

| Segredo | Obrigatório | Para quê |
|---|---|---|
| `GMAIL_USER` | sim | Conta Gmail que **envia** os relatórios |
| `GMAIL_APP_PASSWORD` | sim | **Senha de app** do Gmail (não a senha normal) |
| `OPENAI_API_KEY` | não | Comentário analítico por IA (opcional) |

### Gerando a "Senha de app" do Gmail

1. Ative a **verificação em 2 etapas** na conta Google.
2. Acesse <https://myaccount.google.com/apppasswords> e gere uma senha de app.
3. Use essa senha (16 caracteres) em `GMAIL_APP_PASSWORD`.

> Recomenda-se uma conta Google Workspace institucional como remetente, para que
> os e-mails saiam de um endereço oficial.

## ▶️ Rodar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501`.

> No Ubuntu, se faltar o módulo de ambiente virtual: `sudo apt install python3-venv`.

## ☁️ Publicar no Streamlit Community Cloud (gratuito)

1. Suba este repositório para o GitHub.
2. Em <https://share.streamlit.io>, conecte o repo e aponte o *main file* para `app.py`.
3. Em **Settings → Secrets**, cole `GMAIL_USER`, `GMAIL_APP_PASSWORD` (e
   opcionalmente `OPENAI_API_KEY`).
4. *Deploy* — o Streamlit instala o `requirements.txt` automaticamente.

## 🔒 Privacidade e limitações

- Os mapas de turma e o PDF são processados **em memória** e não ficam salvos no
  servidor. O `.gitignore` impede o commit de `*.xls`, `*.csv` e `*.pdf`.
- A restrição por domínio `@cefetmg.br` é uma barreira simples: o relatório
  **só é entregue na caixa institucional** informada. Ela não verifica a posse da
  conta (qualquer um poderia digitar um endereço `@cefetmg.br` de terceiros, mas o
  resultado iria para a caixa daquela pessoa, não para quem enviou). Se for
  necessário garantir a posse, dá para adicionar um **código de verificação** por
  e-mail antes de processar.

---
Desenvolvido pelo Professor Diego Camargo — `diegocamargo@cefetmg.br`.
