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
  `Σ(HA ofertadas − HA presenciadas)` por bimestre. Mapa real de Estradas recebido e
  lado do mapa validado (passos 12–13); **pendência única do cruzamento: o export da DAE** (ver §6).
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
| (d) | `N/C` em `Pé de meia` = **"Nada consta"** — interpretação adotada, **não confirmada pela DAE** (nem a DAE soube dizer). Cai nela **2.208 alunos** no total (contagem em Trânsito+Estradas: rodar `.venv/bin/python3 sandbox/dae/cruzamento.py` com o export da DAE, §6). Não marca PdM. |
| (e) | A DAE atualiza a planilha **mês a mês**; cada execução do sandbox **refaz o snapshot** do mês a partir do arquivo corrente — sem histórico, sem estado. |
| (f) | Integração = **arquivo exportado/upload manual** (xlsx ou csv). Sem gspread/API/download automático. |

- **Rotina mensal (D11):** mês de referência = último mês com `HA ofertadas` > 0;
  saídas em `saida/<AAAA-MM>/`, sobrescritas a cada execução do mês.
- **Confirmação do calendário (D6):** a tabela mês→bimestre foi **confirmada pelo calendário oficial 2026** (Deliberação CEPE nº 17/2025 e nº 1/2026), com a inclusão de dezembro (4 dias letivos) no 4º bimestre — ver §7.

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
  `2026 - geral` (preenchido até agosto).
- **Mapa real de Estradas:** recebido em 2026-09-27 (via Discord, em `dados/`, não
  versionado) com os agregados: 1 turma, 2ª série, 1º BI/2026, 45 alunos, 17 disciplinas,
  1.887 faltas, mediana 28/aluno.
- **Validação do mapa:** lado do mapa validado (passos 12–13), com o comportamento de
  leitura, contagens e cruzamento testados tanto de forma isolada quanto integrada.
- **Pendência única do cruzamento: o export da DAE.** Com o mapa de turma homologado,
  a **pendência única do cruzamento: o export da DAE** atualizado na pasta `dados/` para
  a conciliação final de faltas e frequência.
- **`N/C` em Estradas:** continua pendente porque o mapa não tem a coluna de benefícios
  ou situação de Pé-de-Meia (informação controlada pela DAE).
- **Guarda de escala do acumulado (C12):** a guarda implementada em `sandbox/dae/carregar.py`
  detecta e normaliza automaticamente o `acumulado_dae` para fração ([0–1]) e registra em
  `attrs["escala_acumulado"]`, aceitando frações ou percentuais crus (ex.: 95,5). O mapa
  de turma, por sua vez, não informa a escala, registrando estritamente as faltas absolutas.
- **Rotina:** quando o export da DAE estiver disponível (Arquivo → Fazer download; pasta
  `.xlsx` ou aba `2026 - geral` `.csv`), salvar em `sandbox/dae/dados/` e rodar
  `cruzamento.py`/`prototipo_pdf.py` (comandos no `README.md`). Cada execução
  recalcula tudo e sobrescreve `saida/<AAAA-MM>/`.

## 7. Calendário acadêmico e carga horária efetiva

### Tabela oficial de dias letivos por dia da semana
Fonte: Deliberação CEPE/CEFET-MG nº 17, de 30/09/2025, alterada pela Deliberação CEPE/CEFET-MG nº 1, de 27/02/2026 (versão de maio/2026).

| Bimestre | Período | SEG | TER | QUA | QUI | SEX | SÁB | Total | Acumulado | Diários até |
|---|---|---|---|---|---|---|---|---|---|---|
| 1º BI | 23/02 a 08/05 | 10 | 10 | 11 | 10 | 9 | 3 | 53 | 53 | 22/05 |
| 2º BI | 11/05 a 17/07 | 10 | 10 | 10 | 9 | 9 | 6 | 54 | 107 | 14/08 |
| 3º BI | 03/08 a 03/10 | 8 | 9 | 9 | 9 | 9 | 5 | 49 | 156 | 09/10 |
| 4º BI | 05/10 a 04/12 | 7 | 9 | 8 | 9 | 8 | 3 | 44 | 200 | 07/12 |
| **Soma** | **Ano letivo** | **35** | **38** | **38** | **37** | **35** | **17** | **200** | — | — |

Total de dias úteis regulares (segunda a sexta-feira): 35 + 38 + 38 + 37 + 35 = 183 dias. Total com sábados letivos: 200 dias.

### Dias por mês × bimestre e fronteiras
A distribuição oficial dos 200 dias letivos entre os meses e sua partição bimestral (`dias_por_mes_bimestre`):
- **1º Bimestre (53 dias):** fevereiro (5 dias), março (23 dias), abril (20 dias), maio (5 dias).
- **2º Bimestre (54 dias):** maio (17 dias), junho (23 dias), julho (14 dias).
- **3º Bimestre (49 dias):** agosto (23 dias), setembro (23 dias), outubro (3 dias).
- **4º Bimestre (44 dias):** outubro (19 dias), novembro (21 dias), dezembro (4 dias).

Partição dos meses de fronteira:
- **Maio (22 dias totais):** 5 dias letivos no 1º BI (até o encerramento em 08/05) e 17 dias letivos no 2º BI (a partir de 11/05).
- **Outubro (22 dias totais):** 3 dias letivos no 3º BI (até o término em 03/10) e 19 dias letivos no 4º BI (a partir de 05/10).
- **Dezembro (4 dias totais):** 4 dias letivos no 4º BI (até o encerramento do ano letivo em 04/12). A planilha da DAE cobria apurações mensais até novembro, mas o calendário oficial delimita esses 4 dias finais do ano letivo.

### Confirmação do `MESES_POR_BIMESTRE`
O mapeamento de meses por bimestre adotado provisoriamente em D6 foi **confirmado pelo calendário oficial 2026**:
- 1º BI: fevereiro, março, abril, maio
- 2º BI: maio, junho, julho
- 3º BI: agosto, setembro, outubro
- 4º BI: outubro, novembro, dezembro (com dezembro, 4 dias letivos, fora da planilha mensal da DAE)

### Faixas de carga horária efetiva: Cenário A vs. REAL Estradas
A carga horária efetivamente lecionada varia conforme os dias da semana em que as aulas são alocadas:

| Aulas semanais | CH Nominal (40 sem.) | Cenário A (Piso dias úteis) | Cenário REAL (Estradas / Trânsito) |
|---|---|---|---|
| 1 aula/sem | 40 h/a | 35 a 38 h/a (87,5% a 95,0%) | 35 a 39 h/a |
| 2 aulas/sem | 80 h/a | 70 a 76 h/a (87,5% a 95,0%) | 70 a 78 h/a |
| 3 aulas/sem | 120 h/a | 105 a 114 h/a (87,5% a 95,0%) | 105 a 116 h/a |
| 4 aulas/sem | 160 h/a | 140 a 152 h/a (87,5% a 95,0%) | 140 a 154 h/a |

No Cenário A, o piso ocorre nas disciplinas com aulas exclusivamente às segundas ou sextas-feiras (35 dias × aulas), e o teto nas alocadas às terças ou quartas-feiras (38 dias × aulas). No Cenário REAL para Estradas e Trânsito, soma-se o sábado letivo de 23/05 (reproduzindo um dia útil), elevando os tetos máximos para 78 h/a (2 aulas), 116 h/a (3 aulas) e 154 h/a (4 aulas).

> **Aviso metodológico — Papel das faixas desde a rev. 4:**
> Desde a revisão 4 do projeto, as faixas teóricas acima atuam estritamente como **verificação cruzada e fallback**. A **fonte primária da carga horária efetiva por disciplina é a planilha de horários** (`CH_Efetiva_*.xlsx` — C15/C17 / passo 17), onde cada oferta de turma e subgrupo tem sua distribuição semanal exata mapeada (ex.: 2 aulas na segunda-feira = 70 h/a; 2 aulas na terça-feira = 76 h/a; 1 aula na terça e 1 na quinta = 75 h/a). Apenas quando uma disciplina não possui horário mapeado na planilha adota-se a estimativa pelas faixas teóricas ou a classificação como sem horário.

### Conclusão prática: Limite de faltas para 75%
Pela legislação educacional (LDB / CEFET-MG), a frequência mínima para aprovação é de 75% sobre a carga horária efetivamente ministrada, isto é, `faltas ≤ floor(0,25 × CH_efetiva)`.
- Se a carga horária considerada fosse a **nominal** (80 h/a para 2 aulas semanais), o aluno poderia ter até `floor(80 × 0,25) = 20` faltas.
- Porém, na **carga horária real lecionada**:
  - Em disciplina com aulas às **segundas-feiras** ou **sextas-feiras** (CH efetiva = 70 h/a no Cenário A):
    `70 × 0,25 = 17,5` → o limite máximo permitido é de **17 faltas**. Com 18 faltas, a frequência cai para `52 / 70 = 74,29%`, e o aluno é **reprovado por infrequência**.
  - Em disciplina com aulas às **terças-feiras** ou **quartas-feiras** (CH efetiva = 76 h/a no Cenário A):
    `76 × 0,25 = 19,0` → o limite máximo permitido é de **19 faltas**. Com 20 faltas, a frequência cai para `56 / 76 = 73,68%`.
- **Conclusão:** Para 2 aulas semanais, o limite real de faltas oscila entre **17 e 19 faltas**, e **nunca atinge as 20 faltas** que decorreriam da carga horária nominal. O estudante que faltar 18 vezes a uma aula de segunda-feira já estará reprovado, embora pudesse supor estar dentro da margem caso consultasse apenas a carga nominal.

### Revogação do Cenário B (rev. 2)
Na revisão 2 das discussões metodológicas, havia sido cogitado um "Cenário B", que distribuía uniformemente todos os 17 sábados letivos entre todas as disciplinas para simular um teto de 200 dias de aula para qualquer componente curricular.
Esse Cenário B foi **formalmente revogado**:
- Os 17 sábados letivos têm destinação temática e departamental específica pela Deliberação CEPE nº 17/2025 (ex.: mostras, testes de proficiência, sábados temáticos de áreas).
- Distribuir sábados indiscriminadamente inflacionava o denominador da carga horária de disciplinas que não tinham aula aos sábados, mascarando a infrequência de alunos faltosos e violando o princípio pedagógico e legal da frequência sobre aulas efetivamente ministradas.
- Permanecem vigentes apenas os cenários **A** (piso conservador de dias úteis) e **REAL** (A + sábados da coordenação do curso).

### Pendências de C8
1. **C8(i) — Quinta-feira de abril (não reproduz no documento em uso):** O levantamento da rev. 4 do plano previa uma quinta-feira de abril "não listada" em *Dias sem aula* do 1º BI (reconstrução dando QUI 11 × 10 e abril 21 × 20). A verificação contra o `.md` oficial versionado (versão de maio/2026) mostra que a quinta **02/04 está listada** em *Dias sem aula* e que a reconstrução dia a dia fecha **exatamente** com as tabelas-resumo: QUI = 11 quintas no período − 02/04 = **10**; abril = 22 dias úteis − 4 sem aula (02, 03, 20 e 21/04) + 2 sábados letivos (11 e 25/04) = **20**. Por isso `divergencias()` devolve hoje **lista vazia** e permanece como guarda: qualquer lacuna que uma revisão futura do `.md` introduzir volta a ser exposta por essa função.
2. **C8(ii) — Atribuição dos sábados por área acadêmica:** A maioria dos 17 sábados letivos é alocada a áreas temáticas (DELTEC, Matemática, Física, etc.), sem especificação de quais séries e turmas participam. Permanece pendente para ciclos futuros verificar se tais eventos contabilizam carga horária curricular para disciplinas específicas ou se funcionam como atividades extracurriculares.
3. **C8(iv) — Educação Física fora da grade de salas teóricas:** Disciplina ministrada no complexo poliesportivo/quadras, ausente da planilha de horários das salas 305–437.
4. **C8(v) — Laboratórios práticos fora da grade de salas teóricas:** Laboratório de Solos, Laboratório de Desenho Topográfico e Laboratório de Topografia ministrados em laboratórios específicos do departamento, sem registro de horário na planilha de salas teóricas 305–437.

---

## 8. CH efetiva real (planilha de horários)

### Fonte e método
- **Fonte:** Arquivo `CH_Efetiva_Disciplinas_Integrado_2026.xlsx` localizado em `sandbox/dae/dados/` (não versionado por conter identificação de docentes na coluna `Professor(a)`).
- **Escopo:** Mapeamento da matriz curricular lecionada do Ensino Técnico Integrado nas salas 305 a 437 do Campus Nova Suíça.
- **Método:** Leitura da distribuição semanal de aulas (segunda a sexta-feira) por turma e componente curricular, multiplicada pelos dias letivos úteis de cada bimestre registrados na aba `Calendário` e lastreados nas Deliberações CEPE nº 17/2025 e nº 1/2026.

### Universo mapeado
- **309 linhas** de oferta de disciplinas por turma e subgrupo.
- **45 turmas** distintas cobrindo as três séries do Ensino Médio Integrado (todas estritamente aderentes ao padrão `<CURSO>-<SERIE><LETRA>`).
- **14 cursos** técnicos representados.
- **Subgrupos:** 270 ofertas em turma inteira (`""`), 20 com divisão `T1` e 19 com divisão `T2`.
- **Aulas semanais:** 240 ofertas com 2 aulas/semana, 32 com 3 aulas/semana, 29 com 4 aulas/semana e 8 com 1 aula/semana.

### Distribuição em relação à CH nominal (94 / 94 / 121)
A distribuição agregada das 309 disciplinas conforme a razão entre a carga horária lecionada e a carga horária nominal anual (40 semanas de referência) resultou em:
- **Abaixo de 90% (< 90% do nominal):** **94** disciplinas (30,42%).
- **90% a 95% (90% a 95% do nominal):** **94** disciplinas (30,42%).
- **95% ou mais (≥ 95% do nominal):** **121** disciplinas (39,16%).
- **Total:** **309** disciplinas (100,0%).

### Conformidade com o calendário oficial: 0 divergências
- **Aba `Calendário`:** **0 divergências** em relação ao calendário oficial em `sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md`. Os dias letivos úteis bimestrais (50 no 1º BI, 48 no 2º BI, 44 no 3º BI e 41 no 4º BI, totalizando 183 dias úteis de segunda a sexta-feira) são idênticos.
- **Linhas recalculadas:** Todas as 309 linhas recalculadas a partir da matriz semanal e do calendário oficial apresentam **0 divergências** com os valores da planilha.
- **Faixas do anexo:** Todas as 309 disciplinas têm sua CH efetiva anual rigorosamente dentro das faixas teóricas calculadas por `faixa_ch`:
  - 1 aula/sem: todas entre 35 e 38 h/a.
  - 2 aulas/sem: todas entre 70 e 76 h/a.
  - 3 aulas/sem: todas entre 105 e 114 h/a.
  - 4 aulas/sem: todas entre 140 e 152 h/a.
  - Extremos observados na base: mínimo de **35 h/a** e máximo de **152 h/a**.

### Cenário A e herança de pendências
- **Planilha = Cenário A:** A apuração das aulas da planilha adota exclusivamente os dias úteis regulares (segunda a sexta-feira). Nenhum dos 17 sábados letivos entra na grade regular das disciplinas. Em particular, o sábado letivo de 23/05/2026 atribuído às coordenações de Estradas e Trânsito fica fora do cômputo da planilha, mantendo a postura de piso conservador e a pendência de C8(ii).
- **Coerência com C8(i):** a planilha fixa 10 quintas-feiras letivas no 1º bimestre e 20 dias letivos em abril — exatamente o que a reconstrução dia a dia do `.md` oficial produz (0 divergências; ver "Pendências de C8" na seção do calendário).

### Estudo de caso: Estradas 2ª série (Cruzamento com Mapa Real)
No cruzamento da turma de Estradas 2ª série (turmas `EST-2A` e `EST/TT-2A`, com 14 linhas na planilha de horários) contra o mapa real de turma (`Estradas_2025-2026.xls`, com 17 disciplinas na legenda):
- **13 disciplinas com CH real:** Casadas deterministicamente com a planilha de horários.
- **4 disciplinas sem horário:** `EDUCAÇÃO FÍSICA - 2ª SÉRIE`, `LABORATÓRIO DE SOLOS`, `LABORATÓRIO DE DESENHO TOPOGRÁFICO` e `LABORATÓRIO DE TOPOGRAFIA`.
- **Causa da ausência (C8(iv) e C8(v)):** Limitação de salas no mapeamento original de horários, que abrangeu estritamente as 36 salas de aula teóricas (305 a 437). Componentes ministrados em quadras/espaços esportivos (C8(iv)) ou laboratórios práticos especializados do departamento (C8(v)) não constam da grade de salas teóricas.

#### Tabela de disciplinas de Estradas 2ª série (1º Bimestre)

| Disciplina | CH 1º BI | Limite de faltas (≤ 25%) |
|---|---|---|
| BIOLOGIA | 18 h/a | 4 faltas |
| FILOSOFIA | 20 h/a | 5 faltas |
| FÍSICA | 30 h/a | 7 faltas |
| GEOGRAFIA | 31 h/a | 7 faltas |
| HISTÓRIA | 20 h/a | 5 faltas |
| INGLÊS | 18 h/a | 4 faltas |
| LÍNGUA PORTUGUESA | 22 h/a | 5 faltas |
| MATEMÁTICA | 30 h/a | 7 faltas |
| MÁQUINAS E EQUIPAMENTOS | 18 h/a | 4 faltas |
| QUÍMICA | 22 h/a | 5 faltas |
| REDAÇÃO | 20 h/a | 5 faltas |
| SOLOS | 20 h/a | 5 faltas |
| TOPOGRAFIA | 22 h/a | 5 faltas |

*Disciplinas sem horário na planilha de salas teóricas (limite de faltas e frequência não calculados):*
- EDUCAÇÃO FÍSICA - 2ª SÉRIE
- LABORATÓRIO DE DESENHO TOPOGRÁFICO
- LABORATÓRIO DE SOLOS
- LABORATÓRIO DE TOPOGRAFIA

### Contagem de infrequência no 1º Bimestre (Passo 15)
- **Número agregado de pares aluno × disciplina abaixo de 75% no 1º BI:** **87 pares**.
- Apurado na execução do passo 15 com os dados reais de faltas do mapa sobre a turma de 45 estudantes:
  - Nas 13 disciplinas com CH real (585 pares aluno × disciplina avaliados), registraram-se **87 ocorrências** em que as faltas do aluno excederam o limite legal de 25% da carga horária lecionada no bimestre.
  - As 4 disciplinas sem horário geram valores `NaN` (180 pares indefinidos), não pontuando no cálculo de infrequência.

