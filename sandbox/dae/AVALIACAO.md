# AVALIACAO — Planilha de frequência da DAE (sandbox)

Avaliação da planilha da DAE (aba `2026 - geral`, export de 2026-09-27) quanto às duas
perguntas do Diego: (1) o que ela resolve no cálculo de frequência do relatório atual e
(2) como destacar Pé-de-Meia e bolsistas nos relatórios de gestão.

> **Números agregados apenas.** Nenhum nome, matrícula, CPF ou e-mail aparece neste
> arquivo. As métricas abaixo vêm do levantamento feito sobre o export real citado;
> ver §6 sobre a re-execução da rotina.

## 1. O que a planilha resolve na frequência

- **Denominador de carga horária (HA).** O relatório atual do app afirma, em 3 pontos de
  `core/relatorios.py`, que a exigência legal de 75% de frequência "precisa de carga
  horária, que não está disponível". A coluna `HA ofertadas` da DAE é exatamente esse
  denominador, por aluno e por mês (total do mês, não por disciplina). Com ela, a regra
  de 75% passa a ser aplicável: `Σ HA presenciadas / Σ HA ofertadas ≥ 75%`.
- **Agregação por bimestre e período de apuração (decisão (a)).** O protótipo
  (`sandbox/dae/frequencia.py`) agrega os meses em bimestres, acumula a razão ponderada
  e informa o mês de referência do snapshot e, por bimestre, os meses do calendário, os
  meses efetivamente lançados e o selo de bimestre **parcial** (ex.: 3º bimestre só com
  agosto lançado → "parcial: agosto de ago–set").
- **Ponderada vs. `Acumulado` da DAE.** O `Acumulado` da DAE é **média simples** dos
  percentuais mensais, não a razão ponderada exigida pela regra legal. Em **1.882
  alunos** a diferença entre as duas ultrapassa 1 p.p., chegando a **~20 p.p.** —
  fevereiro (≈20–30 HA) pesa o mesmo que março (≈150 HA) na média simples. Para o
  limiar de 75% da carga horária, usar a razão ponderada (`diff_vs_dae_pp` no
  `tabela_frequencia`). **619 alunos** têm `Acumulado` vazio (nenhum mês lançado).
- **Cobertura por matrícula.** 5.753 dos **5.959** alunos casam com o mesmo filtro do
  app (`^20\d{9}$`, `core/manipulacao.py::extrair_dataframes`); as ~200 restantes são
  matrículas antigas/fora do padrão (7, 9, 12, 13 dígitos). A junção é por matrícula.
- **Aderência faltas do mapa × HA da DAE.** O cruzamento por bimestre
  (`sandbox/dae/cruzamento.py`) compara as faltas do mapa de turma com
  `Σ(HA ofertadas − HA presenciadas)` por bimestre. **Pendente**: não havia mapas
  `.xls` de Trânsito/Estradas disponíveis na execução desta avaliação; rodar o CLI do
  cruzamento quando houver (ver §6).
- **Limitação.** `HA ofertadas` é o total do mês por aluno, **não por disciplina** — o
  destaque por disciplina do relatório atual não é coberto pela planilha.

## 2. Destaque de Pé-de-Meia e bolsas DAE

- **Trânsito + Estradas: 122 alunos** (59 Trânsito, 61 Estradas, 2 variantes de nome),
  dos quais **Pé-de-Meia `Elegível` 31**, **bolsa BA/BP 23** e **BCE 13**.
- **Total da planilha:** `Pé de meia` ∈ {`Elegível` 901, `Não elegível` 2.849, `N/C`
  2.208, `Elegibilidade indefinida` 1}; bolsas ∈ {`Sim`, `Não`} (BA/BP 594 Sim; BCE 95
  Sim).
- **Proposta (D8), prototipada em `sandbox/dae/prototipo_pdf.py`** (saída em
  `saida/<AAAA-MM>/prototipo_destaque.pdf`; com dados sintéticos, em
  `saida/sintetico/`): coluna curta "Prog." com as siglas na tabela 2.1 + seção própria
  "Estudantes acompanhados pela Assistência Estudantil" listando só quem tem programa,
  com frequência ponderada por bimestre e acumulada e **alerta destacado quando <
  75%** (a regra do Pé-de-Meia exige frequência mínima). Relatório exclusivamente
  interno (decisão (b)); minimização: só a sigla do programa, nunca CPF, e-mail, renda
  ou motivo do benefício.

## 3. LGPD e uso interno (D8/D10)

- **Exibido:** nome, matrícula, frequência, desempenho, vínculo (sigla) a programa.
- **Minimizado desde a leitura (D4):** o carregador descarta `CPF` na entrada e só
  abre a aba `2026 - geral`; as abas `Acompanhamento PdM`, `Acompanhamento Bolsistas
  AE` e `Emails` (com e-mails pessoais) nunca são lidas. Condição socioeconômica só
  aparece como sigla de programa.
- **Rodapé em todas as páginas** do protótipo: "Documento de uso interno — contém
  dados pessoais de estudantes (LGPD)" e **quadro "Tratamento de dados pessoais
  (LGPD)"** antes da seção de destaque, com finalidade, base legal (Lei nº 13.709/2018,
  execução de políticas públicas — art. 7º, III, e art. 23), dados exibidos/não
  exibidos e obrigações de quem recebe. O texto vive na constante `NOTA_LGPD` e é
  **RASCUNHO pendente de validação do Diego** (e, se ele quiser, do encarregado de
  dados do CEFET-MG) — não é parecer jurídico.
- **Ao integrar no app:** não gravar a planilha em disco nem em log; manter a
  minimização de D4.

## 4. Decisões adotadas e pressupostos

| # | Decisão |
|---|---|
| (a) | Unidade de calendário = **bimestre**; o relatório informa **os meses usados**. |
| (b) | Relatórios **exclusivamente internos** (coordenação e órgãos do CEFET-MG). |
| (c) | Nota/lembrete de LGPD no relatório; texto do protótipo é **rascunho** do Diego validar. |
| (d) | `N/C` em `Pé de meia` = **"Nada consta"** — interpretação adotada, **não confirmada pela DAE** (nem a DAE soube dizer). Cai nela **2.208 alunos** no total (contagem em Trânsito+Estradas fica para a re-execução, §6). Não marca PdM. |
| (e) | A DAE atualiza a planilha **mês a mês**; cada execução do sandbox **refaz o snapshot** do mês a partir do arquivo corrente — sem histórico, sem estado. |
| (f) | Integração = **arquivo exportado/upload manual** (xlsx ou csv). Sem gspread/API/download automático. |

- **Rotina mensal (D11):** mês de referência = último mês com `HA ofertadas` > 0;
  saídas em `saida/<AAAA-MM>/`, sobrescritas a cada execução do mês.
- **Única pendência de conteúdo:** confirmar a tabela mês→bimestre do calendário 2026
  do CEFET-MG (D6 mantém `MESES_POR_BIMESTRE` como parâmetro por isso).

## 5. Questões futuras (fora do escopo do sandbox)

- **Integração via API:** o app já tem uma service account `gspread` em
  `core/usage_tracker.py`; avaliar usá-la para ler a planilha da DAE diretamente vs.
  manter upload manual no Streamlit (planilha com ~6 mil linhas).
- **Plano de integração real no app:** sugerido nível de versão **MINOR**
  (funcionalidade nova, sem quebra). Próximos passos: (i) confirmar calendário
  bimestre/mês; (ii) validar o texto LGPD; (iii) confirmar com a DAE o significado do
  `N/C`; (iv) decidir API vs. upload; (v) integrar frequência ponderada e seção de
  destaque ao relatório, sem gravar dado pessoal em disco/log.

## 6. Estado de execução e re-execução

- Os agregados de §1–§2 vêm do levantamento de 2026-09-27 sobre o export da aba
  `2026 - geral` (preenchido até agosto). Na confecção deste arquivo, `dados/` estava
  vazio e não havia mapas `.xls` locais — **cruzamento com mapas não executado**.
- **Rotina:** baixar a planilha (Arquivo → Fazer download; pasta `.xlsx` ou aba
  `2026 - geral` `.csv`), salvar em `sandbox/dae/dados/` e rodar
  `cruzamento.py`/`prototipo_pdf.py` (comandos no `README.md`). Cada execução
  recalcula tudo e sobrescreve `saida/<AAAA-MM>/`.
- Ressalva de formato: `acumulado_dae` é lido como fração ([0–1], percentuais com `%`
  são convertidos); se a DAE exportar percentual como número cru (ex.: 95,5), o
  `diff_vs_dae_pp` sai com magnitude absurda (~10 mil p.p.) — sinal visível de
  escala trocada, ajustar caso ocorra.
