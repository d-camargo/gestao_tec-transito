"""Testes de estilo do cabeçalho das tabelas do relatório PDF (Passo 2).

Intercepta `Table.setStyle` via `monkeypatch` para inspecionar os comandos
aplicados a cada tabela gerada nos cenários de 1 bimestre e 3 bimestres.

Verifica:
(a) Todo comando BACKGROUND na linha 0 inteira ((0, 0) -> (-1, 0)) usa COR_CABECALHO_TABELA.
(b) Toda tabela com esse BACKGROUND tem também TEXTCOLOR na linha 0 igual a COR_TEXTO_CABECALHO_TABELA.
(c) Cada cenário registra pelo menos uma tabela com cabeçalho, e o de 3 bimestres registra
    mais tabelas com cabeçalho que o de 1 bimestre.
"""

from reportlab.platypus import Table, TableStyle

from core.relatorios import (
    COR_CABECALHO_TABELA,
    COR_TEXTO_CABECALHO_TABELA,
    calcular_estatisticas,
    calcular_estatisticas_multibimestre,
    criar_relatorio_pdf,
    gerar_todos_graficos,
)
from tests.test_pdf_ponta_a_ponta import _turma


def _gerar_pdf_um_bimestre():
    """Gera o relatório PDF de 1 bimestre seguindo o roteiro de test_pdf_ponta_a_ponta."""
    df_notas, df_faltas, disciplinas, metadados = _turma(1)
    stats = calcular_estatisticas(
        df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados
    )
    figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
    return criar_relatorio_pdf('Trânsito', stats, figuras)


def _gerar_pdf_tres_bimestres():
    """Gera o relatório PDF de 3 bimestres seguindo o roteiro de test_pdf_ponta_a_ponta."""
    conjuntos = [_turma(1, seed=0), _turma(2, seed=1), _turma(3, seed=2)]
    estatisticas_multibimestre = calcular_estatisticas_multibimestre(conjuntos)
    df_notas, df_faltas, disciplinas, metadados = conjuntos[-1]
    stats = calcular_estatisticas(
        df_notas, disciplinas, df_faltas=df_faltas, metadados=metadados
    )
    figuras = gerar_todos_graficos(df_notas, 'Trânsito', disciplinas, stats, df_faltas)
    return criar_relatorio_pdf(
        'Trânsito',
        stats,
        figuras,
        estatisticas_multibimestre=estatisticas_multibimestre,
    )


def _verificar_comandos_tabelas(lista_comandos_tabelas):
    """Valida as asserções (a) e (b) sobre os comandos de estilo das tabelas interceptadas.

    Retorna o número de tabelas que possuem cabeçalho padronizado.
    """
    total_com_cabecalho = 0

    for cmds in lista_comandos_tabelas:
        bg_cabecalho_cmds = [
            c for c in cmds
            if c[0] == 'BACKGROUND' and c[1] == (0, 0) and c[2] == (-1, 0)
        ]

        # (a) Todo comando BACKGROUND cujo intervalo é a linha 0 inteira usa COR_CABECALHO_TABELA
        for cmd in bg_cabecalho_cmds:
            cor_bg = cmd[3]
            assert cor_bg == COR_CABECALHO_TABELA, (
                f"Comando BACKGROUND na linha 0 usou {cor_bg}, esperado {COR_CABECALHO_TABELA}"
            )

        # (b) Toda tabela com esse BACKGROUND tem também TEXTCOLOR na linha 0 igual a COR_TEXTO_CABECALHO_TABELA
        if bg_cabecalho_cmds:
            total_com_cabecalho += 1
            tc_cabecalho_cmds = [
                c for c in cmds
                if c[0] == 'TEXTCOLOR' and c[1] == (0, 0) and c[2] == (-1, 0)
            ]
            assert any(cmd[3] == COR_TEXTO_CABECALHO_TABELA for cmd in tc_cabecalho_cmds), (
                f"Tabela com BACKGROUND de cabeçalho não tem TEXTCOLOR={COR_TEXTO_CABECALHO_TABELA} "
                f"na linha 0. Comandos TEXTCOLOR encontrados: {tc_cabecalho_cmds}"
            )

    return total_com_cabecalho


def test_cabecalho_tabelas_um_e_tres_bimestres(monkeypatch):
    """Intercepta Table.setStyle para verificar as cores do cabeçalho em 1 e 3 bimestres."""
    comandos_interceptados = []
    original_set_style = Table.setStyle

    def fake_set_style(self, tblstyle):
        if isinstance(tblstyle, TableStyle):
            cmds = tblstyle.getCommands()
        elif isinstance(tblstyle, (list, tuple)):
            cmds = TableStyle(tblstyle).getCommands()
        else:
            cmds = []
        comandos_interceptados.append(list(cmds))
        return original_set_style(self, tblstyle)

    monkeypatch.setattr(Table, 'setStyle', fake_set_style)

    # Cenário 1: 1 bimestre
    comandos_interceptados.clear()
    _gerar_pdf_um_bimestre()
    tabelas_1_bim = list(comandos_interceptados)
    qtd_cabecalhos_1 = _verificar_comandos_tabelas(tabelas_1_bim)

    # Cenário 2: 3 bimestres
    comandos_interceptados.clear()
    _gerar_pdf_tres_bimestres()
    tabelas_3_bim = list(comandos_interceptados)
    qtd_cabecalhos_3 = _verificar_comandos_tabelas(tabelas_3_bim)

    # (c) Cada cenário registra pelo menos uma tabela com cabeçalho,
    # e o de 3 bimestres mais que o de 1 (as 5.x só existem nele)
    assert qtd_cabecalhos_1 >= 1, (
        f"Cenário de 1 bimestre não registrou nenhuma tabela com cabeçalho (total={qtd_cabecalhos_1})"
    )
    assert qtd_cabecalhos_3 >= 1, (
        f"Cenário de 3 bimestres não registrou nenhuma tabela com cabeçalho (total={qtd_cabecalhos_3})"
    )
    assert qtd_cabecalhos_3 > qtd_cabecalhos_1, (
        f"Cenário de 3 bimestres ({qtd_cabecalhos_3}) deveria registrar mais tabelas com cabeçalho "
        f"do que o cenário de 1 bimestre ({qtd_cabecalhos_1})"
    )
