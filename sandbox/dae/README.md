# Sandbox DAE — Acompanhamento Discente

Espaço de trabalho isolado para exploração, processamento e geração de relatórios a partir das planilhas de acompanhamento discente fornecidas pela DAE (Diretoria de Assuntos Estudantis).

> **Atenção (D2):** Nada aqui altera o app. O sandbox é estritamente independente e não afeta o funcionamento ou código principal (`app.py` e `core/`).

---

## 1. Proteção de Dados e LGPD (D1)
As planilhas da DAE contêm informações pessoais e sensíveis de estudantes protegidas pela LGPD.
- **Nunca comitar dados de estudantes no repositório:** as pastas `dados/` e `saida/` estão ignoradas pelo `.gitignore` e não devem ser versionadas.
- **Armazenamento e processamento estritamente locais:** mantenha os arquivos reais restritos ao seu ambiente de trabalho.

## 2. Como obter o arquivo (D3)
1. Abra a planilha da DAE no Google Sheets.
2. Pelo menu do Google Sheets, faça o download:
   - Baixar a pasta completa como `.xlsx` (`Arquivo > Fazer download > Planilha do Microsoft Excel (.xlsx)`); ou
   - Baixar a aba `2026 - geral` como `.csv` (`Arquivo > Fazer download > Valores separados por vírgula (.csv)`).
3. Salve o arquivo baixado em `sandbox/dae/dados/`.
4. *Nota:* Sem uso de API/gspread no momento — isso é questão futura.

## 3. Rotina mensal (D11)
A DAE atualiza a planilha todo mês:
1. Mensalmente, baixar o arquivo atualizado da DAE no Google Sheets.
2. Salvar em `sandbox/dae/dados/` substituindo a versão anterior.
3. Rodar os scripts de processamento.
4. Os relatórios/arquivos de saída são gerados em `saida/<AAAA-MM>/` e sobrescritos a cada execução do período correspondente.

## 4. Premissas e Validações
- **Pressuposto `N/C` (D7):** Assume-se que `N/C` significa "Nada consta". Este pressuposto foi adotado na análise, mas não foi confirmado formalmente pela DAE.
- **Aviso sobre `NOTA_LGPD` (D10):** A nota `NOTA_LGPD` utilizada nas saídas é um rascunho preliminar que ainda precisa ser validado com os setores competentes.

## 5. Como rodar os scripts e os testes (D9)
Utilize o ambiente virtual do projeto (`.venv`):

- **Cruzamento DAE × mapas de turma (`sandbox/dae/cruzamento.py`):**
  - **Execução sem argumentos (descoberta automática):**
    ```bash
    .venv/bin/python sandbox/dae/cruzamento.py
    ```
    Descobre automaticamente arquivos em `sandbox/dae/dados/`: carrega todos os mapas de turma (`.xls`) e localiza a planilha da DAE (`.xlsx` ou `.csv`). **Atenção:** arquivos no padrão `CH_Efetiva_*.xlsx` (carga horária efetiva) **não são tomados por DAE**.
  - **Modo só-mapa e linha `PENDENTE`:**
    Quando mapas de turma são encontrados em `dados/` (ou informados via `--mapas`), mas a planilha da DAE está ausente, o script executa em modo só-mapa (retorno 0), exibindo os dados do mapa e a linha literal:
    ```text
    PENDENTE: arquivo da DAE (.xlsx/.csv) ausente em sandbox/dae/dados/ — cruzamento não executado.
    ```
  - **Execução com argumentos explícitos:**
    ```bash
    .venv/bin/python sandbox/dae/cruzamento.py --dae sandbox/dae/dados/arquivo.xlsx --mapas mapa1.xls
    ```

- **Protótipo de relatório PDF (`sandbox/dae/prototipo_pdf.py`):**
  - **Uso de `--mapas` (exemplo do passo 14):**
    Permite incorporar o bloco "Cruzamento com o mapa de turma" ao PDF:
    ```bash
    .venv/bin/python sandbox/dae/prototipo_pdf.py --curso "TÉCNICO EM ESTRADAS" --cenario REAL --mapas sandbox/dae/dados/Estradas_2025-2026.xls
    ```
  - **Execução padrão:**
    ```bash
    .venv/bin/python sandbox/dae/prototipo_pdf.py --dae sandbox/dae/dados/arquivo.xlsx --curso "TÉCNICO EM TRÂNSITO" --bimestres 1,2,3
    ```

- **Quando o export da DAE chegar:**
  Quando a planilha atualizada da DAE for fornecida:
  1. Salvar o arquivo da DAE em `sandbox/dae/dados/`.
  2. Rodar `cruzamento.py`:
     ```bash
     .venv/bin/python sandbox/dae/cruzamento.py
     ```
  3. Rodar `prototipo_pdf.py` com DAE e mapas:
     ```bash
     .venv/bin/python sandbox/dae/prototipo_pdf.py --dae sandbox/dae/dados/arquivo.xlsx --mapas sandbox/dae/dados/Estradas_2025-2026.xls --curso "TÉCNICO EM ESTRADAS"
     ```

- **Executar os testes** (a partir da raiz do repositório — o `python -m` coloca a raiz no `sys.path`, permitindo `import core`):
  ```bash
  .venv/bin/python3 -m pytest -q sandbox/dae/tests
  ```

---

## 6. Calendário acadêmico

### Fonte única da verdade
O arquivo [`sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md`](file:///home/diego/projects/gestao-tec-transito/sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md) é a fonte única oficial dos dados do Calendário Escolar da Educação Profissional Técnica de Nível Médio (Integrado) do CEFET-MG (Campi Nova Suíça e Nova Gameleira). O documento é público e versionado no repositório, lastreado na Deliberação CEPE/CEFET-MG nº 17, de 30/09/2025, alterada pela Deliberação CEPE/CEFET-MG nº 1, de 27/02/2026 (versão de maio/2026).

### Tabelas e títulos lidos pelo parser (C1)
O parser `carregar_calendario` em [`sandbox/dae/calendario.py`](file:///home/diego/projects/gestao-tec-transito/sandbox/dae/calendario.py) lê de forma determinística as seções estruturadas do Markdown:
1. **Título principal (`# Calendário Escolar 2026...`) e Deliberação do CEPE.**
2. **`## Visão geral`:** tabela com colunas `Bimestre`, `Início`, `Término`, `Dias letivos`, `Acumulado` e `Data-limite dos diários`.
3. **`### Dias letivos por dia da semana`:** tabela com colunas `Bimestre`, `SEG`, `TER`, `QUA`, `QUI`, `SEX`, `SÁB`, `Total` e linha de rodapé `Soma`.
4. **`### Dias letivos por mês`:** tabela mensal com colunas `Fev` a `Dez` e `Total`.
5. **`## 1º Bimestre` a `## 4º Bimestre`:** cada bimestre contendo as subseções `**Dias sem aula (dias úteis)**` (tabela `Data`, `Motivo`) e `**Sábados letivos**` (tabela `Data`, `Responsável`).

> **Atenção — Integridade estrita:** O parser realiza validação estrutural e aritmética estrita de todas as seções e tabelas. **Mudar título ou cabeçalho quebra a carga com `ValueError` apontando a seção** ausente ou malformatada (por exemplo, `ValueError: Seção obrigatória ausente: 'Visão geral'` ou `ValueError: Linha 'Soma' ausente na seção 'Dias letivos por dia da semana'`).

### Como corrigir ou atualizar o calendário
Caso ocorram novas deliberações do CEPE, alterações no calendário ou correções pontuais:
1. Edite diretamente o próprio arquivo Markdown oficial (`sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md`).
2. Mantenha obrigatoriamente a consistência e a coerência de todos os totais:
   - **Acumulado:** a coluna `Acumulado` na Visão Geral deve ser igual à soma cumulativa dos `Dias letivos` de cada bimestre (53, 107, 156, 200).
   - **Dias da semana e Soma:** em cada bimestre, a soma horizontal (`SEG` + `TER` + `QUA` + `QUI` + `SEX` + `SÁB`) deve igualar o `Total` da linha. A linha `Soma` deve conferir com a soma vertical exata de cada coluna, totalizando 200 dias no ano.
   - **Dias por mês:** a soma de todos os meses na tabela mensal deve fechar exatamente em 200 dias anuais.
   - **Sábados letivos:** a quantidade de linhas da tabela `Sábados letivos` de cada bimestre deve bater rigorosamente com o número registrado na coluna `SÁB` daquele bimestre (3, 6, 5, 3), e toda data listada deve corresponder a um sábado real no calendário civil.

### Como pôr o calendário de outro ano
Para configurar o calendário de outro ano (por exemplo, 2027):
1. Crie um novo arquivo Markdown estruturado (ex.: `Calendario_Escolar_2027_EPTNM_Integrado_BH.md`) na pasta `sandbox/dae/calendario/`, seguindo exatamente os mesmos títulos, subtítulos e cabeçalhos de tabela do modelo 2026.
2. Na API Python, forneça o argumento de ano e caminho: `carregar_calendario(ano=2027, caminho="sandbox/dae/calendario/Calendario_Escolar_2027_EPTNM_Integrado_BH.md")`.
3. Na CLI, informe o arquivo através do parâmetro `--calendario`.

### Cenários de apuração: A vs. REAL
A apuração da carga horária lecionada e dos dias letivos comporta dois cenários:
- **Cenário A (Padrão / Piso):**
  - Considera **apenas as aulas ministradas de segunda a sexta-feira** (183 dias úteis anuais), desconsiderando quaisquer sábados letivos.
  - *Por que é o default:* Os 17 sábados letivos previstos no calendário anual são alocados tematicamente por curso ou área acadêmica, sem reproduzir um dia útil regular da grade semanal. Adotar o Cenário A é a postura institucional mais conservadora e protetiva para o discente: ao não inflar o denominador com sábados, gera uma carga horária menor ou igual à lecionada, disparando alertas precoces quando a frequência do estudante se aproxima do limiar legal de 75%.
- **Cenário REAL:**
  - Soma ao Cenário A apenas os sábados letivos sob responsabilidade direta da coordenação do curso do estudante (campo `Responsável` na tabela de sábados letivos).
  - Para os Cursos Técnicos em **Estradas** e **Trânsito**, há 1 sábado letivo no ano: **23/05/2026** (2º bimestre), sob responsabilidade conjunta das coordenações de Trânsito e Estradas.
  - Sábados de áreas acadêmicas específicas (Matemática, DELTEC/Inglês, Física, Educação Física, etc.) não entram no Cenário REAL por ausência de mapeamento determinístico de quais turmas e disciplinas participam desses eventos.

### Reposição de sábados (`--sabado-reproduz`)
No Cenário REAL, os sábados letivos atribuídos ao curso funcionam na prática como reposição da grade horária de um determinado dia útil da semana (por exemplo, reposição das aulas de uma sexta-feira ou de uma segunda-feira afetada por feriados). O argumento de linha de comando `--sabado-reproduz` (`SEG`, `TER`, `QUA`, `QUI` ou `SEX`) define qual dia útil o sábado letivo do curso reproduz. Ele é obrigatório ao executar com `--cenario REAL`.

### Exemplos de CLI
```bash
# Cenário A (padrão): apuração com piso de dias úteis regulares (sem sábados)
.venv/bin/python sandbox/dae/prototipo_pdf.py --dae sandbox/dae/dados/arquivo.xlsx --curso "TÉCNICO EM TRÂNSITO" --bimestres 1,2,3

# Cenário REAL: inclui o sábado 23/05 reproduzindo as aulas de sexta-feira
.venv/bin/python sandbox/dae/prototipo_pdf.py --dae sandbox/dae/dados/arquivo.xlsx --curso "TÉCNICO EM ESTRADAS" --cenario REAL --sabado-reproduz SEX --bimestres 1,2

# Utilizando arquivo de calendário alternativo ou de outro ano
.venv/bin/python sandbox/dae/prototipo_pdf.py --curso "TÉCNICO EM TRÂNSITO" --cenario A --calendario sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md
```

---

## 7. CH efetiva por disciplina

### Onde fica e proteção de dados
A planilha de carga horária efetiva fica na pasta local de dados:
- `sandbox/dae/dados/CH_Efetiva_Disciplinas_Integrado_2026.xlsx`

> **Não versionada no Git:** A pasta `dados/` é ignorada pelo `.gitignore` e o arquivo **não deve ser versionado**, pois contém nomes de docentes na coluna `Professor(a)`.

### Schema resumido da planilha
A pasta de trabalho possui 4 abas estruturadas:
1. **`Leia-me`:** Metadados da planilha, fontes de dados (horários das salas 305–437 e Deliberações do CEPE nº 17/2025 e nº 1/2026), premissas de cálculo e explicitação das limitações operacionais.
2. **`Calendário`:** Matriz de dias letivos úteis de segunda a sexta-feira por bimestre (1º BI: 50, 2º BI: 48, 3º BI: 44, 4º BI: 41; Total: 183 dias úteis) e sábados letivos (17 dias; Total com sábados: 200 dias).
3. **`CH por disciplina`:** Tabela detalhada de 20 colunas mapeando cada oferta de componente curricular por turma e subgrupo:
   - **Colunas de entrada (dados brutos informados manualmente):**
     - `Curso`: nome por extenso do curso técnico.
     - `Turma`: identificador no padrão estrito `<CURSO>-<SERIE><LETRA>` (ex.: `EST-2A`, `EST/TT-2A`, `EDI-1B`).
     - `Subgrupo`: divisão prática (`T1`, `T2`) ou `—` (travessão) para turma inteira / sem subgrupos.
     - `Sigla`: sigla da disciplina na grade horária.
     - `Disciplina`: nome descritivo do componente curricular.
     - `Professor(a)`: nome do(a) docente ministrante.
     - `SEG`, `TER`, `QUA`, `QUI`, `SEX`: quantidade inteira de aulas semanais alocadas em cada dia útil.
   - **Colunas de fórmula (calculadas ou derivadas de `Calendário`):**
     - `Aulas/sem`: total de aulas semanais (`=SUM(SEG:SEX)`).
     - `CH nominal`: carga horária nominal anual de referência (`=Aulas/sem * 40`).
     - `1º BI`, `2º BI`, `3º BI`, `4º BI`: horas-aula lecionadas em cada bimestre (`=SUMPRODUCT(SEG:SEX, Calendário!XºBI)`).
     - `CH efetiva`: carga lecionada no ano (`=SUM(1ºBI:4ºBI)`).
     - `Diferença`: saldo em relação ao nominal (`=CH efetiva - CH nominal`).
     - `% do nominal`: razão da carga lecionada sobre o nominal (`=CH efetiva / CH nominal`).
4. **`Resumo por carga`:** Tabela de referência da CH efetiva por aula semanal e consolidação da distribuição percentual das disciplinas sobre a carga nominal em 3 faixas: `< 90%`, `90% a 95%` e `≥ 95%`.

### O que o loader descarta e normaliza
A função `carregar_ch_efetiva()` (`sandbox/dae/ch_efetiva.py`):
- **Descarta `Professor(a)`:** Por diretriz de minimização de dados e LGPD, a coluna de docentes é descartada no ato da carga. O DataFrame retornado contém apenas atributos institucionais e curriculares, sem nomes de professores.
- **Normaliza subgrupo:** Substitui `—` ou `-` por string vazia (`""`).
- **Decompõe a turma:** Extrai as colunas `serie` (inteiro) e `letra` (string) a partir do identificador da turma.

### Atributo `ch_origem`
O DataFrame resultante carrega o metadado `df.attrs["ch_origem"]`:
- `"planilha"`: quando o arquivo foi lido com os valores numéricos pré-calculados em cache (típico de arquivos salvos pelo Excel).
- `"recalculada"`: quando as células contêm fórmulas não avaliadas (salvas sem cache de valores) ou quando invocado com `forcar_recalculo=True`. A função refaz os cálculos bimestrais aplicando os dias letivos da aba `Calendário`, assegurando resultados idênticos aos da planilha.

### Precedência no cálculo de frequência: Planilha > Faixa do calendário
A apuração da carga horária lecionada e dos limites de faltas (`frequencia_por_disciplina()` e `resumo_frequencia_por_disciplina()`) segue a ordem de precedência:
1. **Planilha (`fonte="planilha"`):** Carga horária real bimestral da planilha de horários (`ch_bim_<n>`). Prevalece sobre quaisquer estimativas teóricas, pois reflete os horários reais em sala de aula.
2. **Faixa do calendário (`fonte="estimada"`):** Quando a disciplina não está mapeada na planilha de horários, mas a coordenação informa a quantidade de aulas semanais (`aulas_sem_estimadas`), apura-se a faixa de CH lecionada do calendário oficial (`faixa_ch`), adotando o piso conservador (`min_ch`) como denominador.
3. **Sem horário (`fonte="sem horário"`):** Disciplinas sem registro de horário na planilha nem estimativa (ex.: laboratórios especializados ou Educação Física) ficam com frequência `NaN` e limites não calculados.

### Parâmetro CLI `--ch-efetiva`
O protótipo do relatório PDF aceita o caminho explícito para a planilha de carga horária efetiva através do argumento `--ch-efetiva`:
```bash
# Geração do relatório com mapa de turma e planilha de CH efetiva explícita
.venv/bin/python sandbox/dae/prototipo_pdf.py --curso "TÉCNICO EM ESTRADAS" --ch-efetiva sandbox/dae/dados/CH_Efetiva_Disciplinas_Integrado_2026.xlsx --mapas sandbox/dae/dados/Estradas_2025-2026.xls
```

### Como atualizar a planilha
Caso haja alterações de horário ou alocação de novas turmas:
1. O Diego edita a distribuição de aulas/dias diretamente no arquivo `.xlsx` em `sandbox/dae/dados/`.
2. Ao rodar os scripts ou a suíte de testes, o sandbox relê a planilha e valida sua integridade aritmética e estrutural.
3. O módulo cruza as informações com o calendário oficial (`sandbox/dae/calendario/Calendario_Escolar_2026_EPTNM_Integrado_BH.md`) via `divergencias_calendario()`, acusando qualquer divergência entre os dias letivos da planilha e as definições do `.md`.

---

## 8. Estradas + Trânsito (DET)

### União por matrícula, não soma
O Departamento de Engenharia de Transportes (DET) congrega os cursos técnicos em **Estradas** e **Trânsito**. No Ensino Médio Integrado, as turmas desses dois cursos compartilham estudantes nas disciplinas do núcleo comum da Base Nacional Comum Curricular (BNCC). Por essa razão, a apuração do conjunto discente do departamento é realizada estritamente pela **união dos estudantes únicos por matrícula**, e **não pela soma aritmética simples** dos mapas brutos (o que geraria dupla contagem indevida de estudantes).

### Fonte autoritativa do app
A consolidação discente utiliza diretamente a função de negócio oficial do app:
[`core.manipulacao.processar_multiplos_bimestres_transito_estradas`](file:///home/diego/projects/gestao-tec-transito/core/manipulacao.py).
Essa função autoritativa:
1. Identifica os discentes de Trânsito matriculados no mapa de turmas de Estradas;
2. Transfere suas notas e faltas das disciplinas do núcleo comum para a estrutura de Trânsito;
3. Remove essas matrículas da relação de Estradas.
Dessa forma, os conjuntos finais de Estradas e Trânsito resultam com **interseção estritamente nula** (`intersecao = 0`), eliminando qualquer duplicidade e mantendo conformidade integral com a lógica do app.

### Descoberta automática
Os utilitários de integração do DET possuem descoberta automática dos arquivos situados em `sandbox/dae/dados/`:
- `classificar_mapas()`: inspeciona todos os mapas `.xls` na pasta e os agrupa de acordo com o atributo `curso_amigavel` dos metadados extraídos do cabeçalho;
- `eh_det()`: valida se o conjunto de mapas representa estritamente o DET (presença simultânea de mapas de Estradas e de Trânsito);
- Os pontos de entrada (`det.carregar_det()`, `preview_relatorio.gerar_previa()` e `cruzamento.py`) ativam a descoberta automática quando executados sem argumentos de caminhos, localizando os mapas oficiais em `dados/`.

### Aliases de disciplina e como acrescentar um
Eventuais divergências entre a nomenclatura de disciplinas adotada no mapa de turma (.xls) e na grade de horários da carga horária efetiva (.xlsx) são tratadas pelo dicionário determinístico `ALIASES_DISCIPLINA` em [`sandbox/dae/ch_efetiva.py`](file:///home/diego/projects/gestao-tec-transito/sandbox/dae/ch_efetiva.py). O casamento entre legendas opera por comparação exata após normalização, sem emprego de algoritmos probabilísticos ou busca aproximada.

Para acrescentar um novo alias de disciplina:
1. Abra o arquivo [`sandbox/dae/ch_efetiva.py`](file:///home/diego/projects/gestao-tec-transito/sandbox/dae/ch_efetiva.py);
2. Localize a definição de `ALIASES_DISCIPLINA`;
3. Adicione uma nova entrada mapeando a versão normalizada do mapa para a versão normalizada da planilha (ambas em letras maiúsculas, sem acentuação e sem pontuação excedente, conforme padronizado por `normalizar_disciplina()`):
   ```python
   ALIASES_DISCIPLINA: dict[str, str] = {
       "LABORATORIO DE DE PESQUISA DE TRANSPORTES E TRANSITO": "L. DE PESQUISA DE TRANSPORTE E TRANSITO",
       "LABORATORIO DE TOPOGRAFIA URBANA": "L. DE TOPOGRAFIA URBANA",
       "NOVA DENOMINACAO NO MAPA": "DENOMINACAO NA PLANILHA",
   }
   ```
4. Execute os testes (`.venv/bin/python3 -m pytest -q sandbox/dae/tests/test_ch_efetiva.py`) para confirmar o casamento determinístico.

---

## 9. Prévia do relatório do app com a seção DAE

### Comando e localização de saída
Para gerar a prévia do relatório do app integrada com a seção da DAE:
```bash
.venv/bin/python sandbox/dae/preview_relatorio.py
```
Opções disponíveis via CLI:
- `--mapas`: lista de arquivos de mapas de turma (padrão: descoberta automática em `dados/`).
- `--dae`: arquivo da DAE a utilizar (padrão: base sintética de demonstração).
- `--ch-efetiva`: caminho da planilha de carga horária efetiva.
- `--saida`: pasta de destino dos PDFs gerados.
- `--cenario`: cenário de apuração (`A` ou `REAL`, padrão: `A`).
- `--sabado-reproduz`: dia útil reproduzido no Cenário REAL (`SEG` a `SEX`).

Os relatórios são salvos em `sandbox/dae/saida/preview/` utilizando a nomenclatura padrão do app:
- `sandbox/dae/saida/preview/relatorio_trânsito_2aserie_bim1_previa_dae.pdf`
- `sandbox/dae/saida/preview/relatorio_estradas_2aserie_bim1_previa_dae.pdf`

### Gerador do app como biblioteca por mock.patch
O script [`sandbox/dae/preview_relatorio.py`](file:///home/diego/projects/gestao-tec-transito/sandbox/dae/preview_relatorio.py) consome o gerador de relatórios do app (`core.relatorios.criar_relatorio_pdf`) como uma biblioteca externa, sem alterar o código-fonte de produção em `core/` ou `app.py`. A injeção da seção DAE é executada através de um monkeypatch temporário com `unittest.mock.patch` sobre o método `core.relatorios._DocComSumario.multiBuild`. O patch intercepta o fluxo de construção (`story`), anexa os flowables da DAE como uma nova seção numerada no sumário executivo e restaura o estado original da classe imediatamente ao término do context manager `injetar_secao_dae`.

### Faixa de prévia
Todas as páginas dos documentos gerados trazem no rodapé uma faixa discreta (fundo rosado claro, texto pequeno em Times-Roman vermelho-escuro) com os dizeres `"PRÉVIA gerada no sandbox/dae — não é relatório oficial. A seção DAE usa dados fictícios de demonstração (export da DAE pendente)."`, desenhada no canvas do ReportLab pelo hook `onPage`. Essa marcação visual evidencia o caráter não oficial do arquivo e o uso de dados fictícios na seção DAE enquanto o export da DAE não chega.

### Dados fictícios de demonstração
Na ausência de um arquivo real da DAE, a rotina recorre à base fictícia gerada por `prototipo_pdf.obter_dados_sinteticos()`. Nesses casos, o documento inclui no início da seção DAE o quadro de aviso `"DEMONSTRAÇÃO"` e o selo `"DADOS SINTÉTICOS / FICTÍCIOS"`. Essas informações simuladas têm finalidade puramente ilustrativa de layout e nunca são vinculadas a alunos reais.

### Uso interno restrito e proteção de dados
> **Atenção — O PDF é interno — contém nomes reais de alunos na parte do app; enviar só ao Diego, nunca em canal público.**
> Como a prévia invoca o pipeline oficial do app processando os mapas de turma reais de 2ª série, os relatórios contêm dados pessoais reais de estudantes (nomes e notas nas seções acadêmicas 1 a 4). Em conformidade com a LGPD e as normas institucionais, o arquivo gerado destina-se única e exclusivamente ao uso interno e restrito da coordenação. É expressamente vedado o envio ou compartilhamento em canais públicos ou grupos abertos.


