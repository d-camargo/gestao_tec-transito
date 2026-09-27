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

- **Executar scripts** (rodar a partir da raiz do repositório):
  - `sandbox/dae/cruzamento.py` — cruzamento DAE × mapas de turma:
    ```bash
    .venv/bin/python sandbox/dae/cruzamento.py --dae sandbox/dae/dados/arquivo.xlsx --mapas mapa1.xls
    ```
  - `sandbox/dae/prototipo_pdf.py` — gera o PDF de destaque:
    ```bash
    .venv/bin/python sandbox/dae/prototipo_pdf.py --dae sandbox/dae/dados/arquivo.xlsx --curso "TÉCNICO EM TRÂNSITO" --bimestres 1,2,3
    ```

- **Executar os testes** (a partir da raiz do repositório — o `python -m` coloca a raiz no `sys.path`, permitindo `import core`):
  ```bash
  .venv/bin/python3 -m pytest -q sandbox/dae/tests
  ```
