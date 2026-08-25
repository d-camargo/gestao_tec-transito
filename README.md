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
   destaca os alunos com média **abaixo do limiar de aprovação parcial** e os
   com faltas acima da média da turma, e monta o relatório em PDF com
   **sumário (TOC)** e **logo institucional em todas as páginas**.
4. O PDF é **enviado por e-mail** ao coordenador. A tela mostra apenas a
   confirmação — nada de download nem dados expostos na página. Se configurado,
   o registro de uso é gravado de forma silenciosa no Google Sheets.

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

O limiar é um **corte absoluto** (não depende do desempenho da turma) e é
aplicado em todas as estatísticas: taxa de aprovação geral, disciplina mais
crítica, a lista de alunos abaixo do limiar, o boxplot/faixas de STEM, o
gráfico de dispersão notas × faltas e a probabilidade de reprovação.

## 🗂️ Estrutura do relatório

```
Capa · Sumário
1. Glossário
2. Estatísticas Gerais da Turma (notas + frequência)
   painel de indicadores (notas + frequência, quando há faltas lançadas)
   2.1 Desempenho e Frequência por Aluno   — tabela completa, em paisagem
   2.2 Alunos com Média Abaixo do Limiar
   2.3 Alunos com Faltas Acima da Média
   2.4 Resumo de Faltas por Disciplina
   2.5 Visualizações Gráficas              — todos os gráficos, depois das tabelas
3. Resumo Estatístico por Disciplina
   3.1 Disciplinas sem Notas Lançadas
4. Análise de Matemática, Física e Química
5. Análise Multibimestral                  (só com 2 ou mais bimestres enviados)
   5.1 Estimativa de Média Anual
   5.2 IDA — Índice de Desempenho do Aluno
   5.3 Alunos com Maior Queda de Desempenho
   5.4 Probabilidade de Reprovação
6. Análise e Comentários (IA)
```

Com **1 bimestre só**, a seção 5 (e suas subseções) simplesmente não entra no
relatório — o resto da estrutura é idêntico.

### 2.1 — Tabela completa de desempenho por aluno (paisagem)

Uma linha por aluno, uma coluna por disciplina com notas, mais as colunas
**Média** e **Faltas**. Cada célula de disciplina mostra `"nota / faltas"`
(ex.: `14,5 / 3`); sem faltas lançadas para aquela disciplina, mostra só a
nota; nota ausente vira `—`. Os cabeçalhos das colunas são abreviados
(**D1…Dn**), com uma legenda logo abaixo mapeando cada abreviação ao nome
completo da disciplina — isso garante que a tabela caiba independentemente do
comprimento dos nomes vindos do SIGAA. A página desta seção é impressa em
**paisagem** (as demais páginas continuam em retrato).

### 2.2/2.3 — Critério de entrada das listas

- **Alunos com Média Abaixo do Limiar (2.2)**: corte **absoluto** — entra todo
  aluno cuja média própria é menor que o limiar de aprovação parcial (60% da
  pontuação do bimestre). Aluno **exatamente** no limiar não entra. Alunos com
  média igual ou acima do limiar que ainda têm alguma disciplina abaixo dele
  não entram na lista, mas são contados à parte na legenda.
- **Alunos com Faltas Acima da Média (2.3)**: corte **relativo** à turma —
  entra todo aluno cujo total de faltas supera a média de faltas totais da
  turma. A coluna **Sinal** destaca, dentro dessa lista, quem também ultrapassa
  o P90 e a média + 2σ. Esses limites estatísticos **não substituem** o limite
  legal de 25% da carga horária — apontam, sem depender do calendário, quem
  merece um olhar atento.

Nenhuma das duas listas carrega mais uma coluna de IDA (o IDA passou a ser
exclusivo da análise multibimestral — ver seção 5.2).

### Asterisco — suspeita de lançamento incompleto (seção 3)

A tabela de resumo por disciplina marca com `*` toda disciplina em que ao
menos um de quatro sinais dispara, calculados sobre a nota máxima observada
(`max_observado`) e a cobertura (fração de alunos com nota lançada):

| Sinal | Condição | Leitura |
|---|---|---|
| **A — forte** | nota máxima observada ≤ 50% da pontuação do bimestre | "provavelmente não lançada" — texto próprio, mais forte |
| **B — absoluto** | nota máxima observada < 90% da pontuação do bimestre | "a conferir" |
| **C — relativo** | nota máxima observada < 80% da mediana dos máximos das demais disciplinas | "a conferir" |
| **D — cobertura** | menos de 90% dos alunos da turma têm nota lançada nessa disciplina | "a conferir" |

B e C se cobrem mutuamente: B pega o mapa inteiro parcial (upload no meio do
bimestre, com todas as disciplinas baixas — caso em que a mediana de C também
fica baixa, cegando o sinal relativo); C pega a disciplina que destoa das
demais num bimestre genuinamente difícil, caso em que B marcaria a turma
inteira injustamente. D é o único sinal independente dos outros três: pega a
disciplina em que só uma fração dos alunos tem nota lançada, mesmo que quem
lançou tenha tirado nota máxima — caso que passava batido antes.

### Disciplinas STEM (seção 4)

Quando o mapa contém Matemática, Física ou Química — identificadas pelo
**nome na legenda**, pois os códigos do SIGAA mudam a cada série (Educação
Física fica de fora) —, o relatório ganha uma seção própria com uma tabela
resumo (média, mediana, desvio padrão, quantos e qual % de alunos abaixo do
limiar, média de faltas) e **quatro gráficos**, nesta ordem:

1. **Boxplot** das notas nas três disciplinas, com linha no limiar.
2. **Faixas de aproveitamento**: barras horizontais empilhadas, uma por
   disciplina, com o percentual de alunos em cada faixa (`< 40%`, `40–60%`,
   `60–80%`, `≥ 80%` da pontuação do bimestre) — paleta semáforo. Responde
   "onde está a massa crítica de cada disciplina", que o boxplot não responde.
3. **Sobreposição de baixo desempenho**: barras com a contagem de alunos que
   estão abaixo do limiar em 0, 1, 2 ou 3 das STEM ao mesmo tempo — identifica
   o núcleo duro que precisa de intervenção diferente de quem está mal em uma
   disciplina só.
4. **Comparação de médias**: a média de cada disciplina STEM contra a média
   geral da turma.

Elas ganharam seção própria por serem as que mais reprovam no EPTNM e
ficarem diluídas no boxplot geral.

### Dispersão de Notas × Faltas (última figura da seção 2.5)

Um ponto por aluno: eixo X = total de faltas, eixo Y = média (pontos do
bimestre). Quatro quadrantes pintados por baixo dos pontos, a partir dos
mesmos dois cortes das seções 2.2/2.3 — o **limiar de aprovação** (horizontal)
e a **média de faltas da turma** (vertical; o P90 entra só como linha
tracejada de referência, sem definir quadrante):

| Quadrante | Nota | Faltas | Rótulo |
|---|---|---|---|
| superior-esquerdo | ≥ limiar | ≤ média | **Em dia** |
| superior-direito | ≥ limiar | > média | **Risco de frequência** |
| inferior-esquerdo | < limiar | ≤ média | **Risco de desempenho** |
| inferior-direito | < limiar | > média | **Risco duplo** |

Aluno exatamente no limiar conta como "≥ limiar"; aluno com faltas
exatamente na média conta como "≤ média". Só os alunos do quadrante **Risco
duplo** são numerados no gráfico (por gravidade — menor média primeiro,
desempate por mais faltas), com a lista "número → nome" logo abaixo da
legenda — nome escrito em cada ponto não escala para turmas grandes. Esses
rótulos são deliberadamente diferentes das faixas do IDA
(Confortável/Regular/Atenção/Crítico): são eixos de leitura distintos.

### Análise de faltas

Além da lista da seção 2.3, o relatório traz na 2.4 um **resumo por
disciplina** (média, mediana, P90, desvio padrão, alunos acima de P90 e acima
de média + 2σ), e no painel de indicadores gerais da seção 2, quando há faltas
lançadas: **Média de Faltas por Aluno**, **Mediana**, **P90** e **Total de
Faltas da Turma**.

### IDA — Índice de Desempenho do Aluno (seção 5.2, exclusivo da análise multibimestral)

Desde a reestruturação, o IDA deixou de ser calculado por bimestre isolado e
passou a resumir a situação do aluno no **acumulado dos bimestres enviados**
— por isso só aparece com 2 ou mais bimestres. Escala de −1 a +1, cujo zero é
o limiar de aprovação parcial **acumulado** (60% da soma dos pontos dos
bimestres já enviados): IDA negativo significa estar abaixo do necessário para
aprovação no acumulado até aqui.

Fórmula: `S` = soma das médias do aluno nos bimestres em que tem nota; `P` =
soma da pontuação desses bimestres; `L = 0,60·P`. A componente de notas é
`(S−L)/(P−L)` para `S ≥ L` e `(S−L)/L` para `S < L` — sempre entre −1 e +1. A
componente de faltas é `1 − (faltas acumuladas ÷ P90 acumulado de faltas da
turma)`, limitada entre −1 e +1. O IDA combina as duas pelos pesos 0,7 (notas)
e 0,3 (faltas); sem faltas lançadas (ou com P90 igual a zero), usa só as
notas. Aluno sem nenhuma média lançada fica de fora (IDA indefinido). Faixas
de leitura: IDA ≥ +0,30 Confortável · 0 a +0,30 Regular · −0,30 a 0 Atenção ·
abaixo de −0,30 Crítico.

O relatório traz o gráfico de barras dos alunos abaixo da média de IDA da
turma e uma tabela com todos os alunos de IDA negativo.

> A antiga **estimativa heurística de risco de repetência** (0% a 100%, sobre
> os 10 alunos mais críticos) foi **removida** — o lugar dela na seção 5 foi
> tomado pela probabilidade de reprovação abaixo, que é auditável (mostra a
> conta) em vez de uma pontuação heurística.

### Probabilidade de Reprovação (seção 5.4, exclusiva da análise multibimestral)

Estimativa de acompanhamento da chance de reprovação anual por nota no
CEFET-MG (aprovação = 60 de 100 pontos: 20/30/20/30), considerando **apenas
notas** dos bimestres regulares — nem a exigência de **75% de frequência
anual** (precisa da carga horária, que o Mapa de Turma não traz) nem a
**recuperação/exame final** entram na conta.

**Parte determinística** (por aluno): `S` = pontos obtidos nos bimestres em
que o aluno tem nota; `R` = 100 − a pontuação **desses mesmos bimestres**
(por aluno, não global — assim um bimestre ainda sem lançamento nunca é
descontado duas vezes); `N = 60 − S` = pontos que faltam.

| Condição | Situação | Probabilidade |
|---|---|---|
| `N ≤ 0` | Aprovado por nota | 0% |
| `N > R` | Não alcança 60 pontos nas notas regulares | não estimada (exibida como "—") |
| resto | Depende — segue para a parte probabilística | calculada abaixo |

A linha "Não alcança 60 pontos nas notas regulares" **não é 100% de
reprovação**: o EPTNM tem recuperação/exame final, via de aprovação que este
modelo (só notas dos bimestres regulares) não enxerga — por isso a
probabilidade fica em branco em vez de mostrar um número sem lastro.

**Parte probabilística** (só no caso "Depende"): a partir do aproveitamento do
próprio aluno nos bimestres já lançados, um modelo normal estima a chance de
não fechar os pontos que faltam nos bimestres restantes, com **encolhimento
estatístico** (dados de poucos bimestres puxados em direção à variância da
turma, para não superestimar a confiança a partir de 1 ou 2 pontos) e um
**piso de desvio-padrão**. O resultado é sempre limitado entre **1% e 99%** —
o modelo nunca afirma certeza.

A tabela traz todos os alunos (não um recorte), ordenados por gravidade:
primeiro quem não alcança 60 pontos (por pontos que faltam, decrescente),
depois quem "depende" (por probabilidade decrescente), por último os
aprovados por nota.

## 📊 Análise multibimestral (seção 5)

Ao enviar mais de um arquivo do mesmo curso/turma (até 4 arquivos no máximo, um para cada bimestre), o aplicativo ativa automaticamente a **Análise Multibimestral** no relatório. Esta análise consolida as informações dos bimestres fornecidos e traz:

- **Média Anual (5.1)**: estimativa da média anual baseada nos pesos oficiais do CEFET-MG (20/30/20/30). É rotulada como **parcial** se faltarem bimestres no envio, ou **completa** quando os 4 bimestres são fornecidos.
- **IDA — Índice de Desempenho do Aluno (5.2)**: ver seção própria acima.
- **Alunos com Maior Queda de Desempenho (5.3)**: comparação dos dois últimos bimestres enviados, com **todos** os alunos em queda (sem recorte de Top 10).
- **Probabilidade de Reprovação (5.4)**: ver seção própria acima.

> **Comportamento com 1 arquivo**: caso seja enviado apenas 1 arquivo (ou 1 par, no caso de Trânsito + Estradas), o comportamento do aplicativo é **idêntico ao de um bimestre único**, sem a seção 5 no relatório PDF.

### Caso especial: Trânsito + Estradas (1ª série)

Há um **seletor de modo no corpo da página**, logo acima dos campos de envio,
com as opções "Curso único (qualquer curso do EPTNM)" e "Curso integrado
Trânsito + Estradas (1ª série)". No modo integrado, o app pede **de 1 a 4
arquivos de cada lado, pareados por bimestre** (mapas de Trânsito e mapas de
Estradas, porque o de Estradas traz o ensino médio compartilhado) e produz
**2 relatórios** — um para cada curso — anexados no mesmo e-mail. No modo
"Curso único", basta enviar os arquivos no campo único e as disciplinas são
extraídas dinamicamente do próprio mapa.

## 🗂️ Estrutura do projeto

```
.
├── app.py                  # Interface Streamlit (formulário, validação, envio)
├── core/
│   ├── disciplinas.py      # Catálogo de nomes amigáveis de disciplinas
│   ├── email_sender.py     # Validação de e-mail e envio SMTP (Gmail)
│   ├── manipulacao.py      # Leitura/processamento dos .xls -> DataFrames
│   ├── relatorios.py       # Estatísticas, gráficos, IA e geração do PDF
│   └── usage_tracker.py    # Registro de uso em Google Sheets (opcional)
├── assets/                 # Logo institucional opcional (logo_cefet.png)
├── tests/                  # Testes unitários (conftest.py + suítes por tema)
├── .streamlit/
│   ├── config.toml         # Tema
│   └── secrets.toml.example
├── DEPLOY.md               # Guia detalhado de deploy no Streamlit Cloud
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
| `GOOGLE_SHEETS_ID` | não | ID da planilha para registro de uso (opcional) |
| `[gcp_service_account]` | não | Credenciais da conta de serviço Google Cloud (opcional) |

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
