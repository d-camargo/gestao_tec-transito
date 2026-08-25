"""Testes da seção de Disciplinas STEM (Matemática, Física e Química)."""

import unittest

import pandas as pd

from core.relatorios import (
    calcular_estatisticas,
    gerar_todos_graficos,
    grafico_stem_boxplot,
    grafico_stem_faixas,
    grafico_stem_media_comparativa,
    grafico_stem_sobreposicao,
)

DISCIPLINAS = {
    'MAT': 'MATEMÁTICA - 2ª SÉRIE',
    'FIS': 'FÍSICA - 2ª SÉRIE',
    'QUI': 'QUÍMICA - 2ª SÉRIE',
    'POR': 'LÍNGUA PORTUGUESA - 2ª SÉRIE',
}
METADADOS = {'bimestre_num': 1}  # 20 pts, limiar 12.0


def _turma_com_stem():
    df_notas = pd.DataFrame({
        'nome': ['Ana', 'Beto', 'Caio', 'Duda'],
        'MAT': [20.0, 16.0, 10.0, 8.0],
        'FIS': [18.0, 14.0, 12.0, 6.0],
        'QUI': [16.0, 12.0, 8.0, 4.0],
        'POR': [19.0, 15.0, 11.0, 7.0],
    })
    df_faltas = pd.DataFrame({
        'nome': ['Ana', 'Beto', 'Caio', 'Duda'],
        'MAT': [0, 2, 4, 6],
        'FIS': [1, 1, 3, 5],
        'QUI': [0, 2, 2, 4],
        'POR': [0, 0, 1, 1],
    })
    return df_notas, df_faltas


def _turma_so_tecnicas():
    disciplinas = {
        'TT1': 'LABORATÓRIO DE SOLOS',
        'TT2': 'TOPOGRAFIA',
    }
    df_notas = pd.DataFrame({
        'nome': ['Ana', 'Beto', 'Caio'],
        'TT1': [18.0, 14.0, 10.0],
        'TT2': [16.0, 12.0, 8.0],
    })
    return df_notas, disciplinas


class TestEstatisticasStem(unittest.TestCase):
    """Passo 5: resumo estatístico das disciplinas STEM."""

    def test_turma_com_stem_gera_3_linhas_com_medias_corretas(self):
        df_notas, df_faltas = _turma_com_stem()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        self.assertTrue(stats['stem_disponivel'])
        resumo = stats['stem_summary_df']
        self.assertEqual(len(resumo), 3)
        self.assertEqual(set(resumo['Disciplina']),
                         {'Matemática - 2ª Série', 'Física - 2ª Série', 'Química - 2ª Série'})
        media_mat = resumo.loc[resumo['Disciplina'] == 'Matemática - 2ª Série', 'Média'].iloc[0]
        self.assertEqual(media_mat, '13,50')  # (20+16+10+8)/4
        # Portuguesa não entra na seção.
        self.assertNotIn('Língua Portuguesa - 2ª Série', set(resumo['Disciplina']))
        self.assertEqual(set(stats['stem_codigos']), {'MAT', 'FIS', 'QUI'})
        # Com faltas disponíveis, a coluna Média de Faltas vem preenchida.
        self.assertTrue((resumo['Média de Faltas'] != '').all())

    def test_turma_so_com_tecnicas_nao_tem_stem(self):
        df_notas, disciplinas = _turma_so_tecnicas()
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        self.assertFalse(stats['stem_disponivel'])
        self.assertTrue(stats['stem_summary_df'].empty)


class TestGraficosStem(unittest.TestCase):
    """Passo 6: boxplot e média comparativa das STEM."""

    def test_graficos_devolvem_figure_com_dados_stem(self):
        df_notas, _ = _turma_com_stem()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        cols_stem = stats['stem_codigos']
        limiar = stats['limiar_aprovacao']
        max_pts = stats['max_pontos_bimestre']
        fig_box = grafico_stem_boxplot(
            df_notas, 'Trânsito', DISCIPLINAS, cols_stem, max_pts, limiar)
        fig_media = grafico_stem_media_comparativa(
            df_notas, 'Trânsito', DISCIPLINAS, cols_stem,
            stats['media_geral_turma'], max_pts, limiar)
        self.assertIsNotNone(fig_box)
        self.assertIsNotNone(fig_media)
        self.assertEqual(len(fig_box.axes[0].get_yticklabels()), 3)

    def test_graficos_devolvem_none_sem_stem(self):
        df_notas, disciplinas = _turma_so_tecnicas()
        self.assertIsNone(grafico_stem_boxplot(df_notas, 'Trânsito', disciplinas, [], 20, 12.0))
        self.assertIsNone(
            grafico_stem_media_comparativa(df_notas, 'Trânsito', disciplinas, [], 0.0, 20, 12.0))

    def test_gerar_todos_graficos_inclui_chaves_stem(self):
        df_notas, df_faltas = _turma_com_stem()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', DISCIPLINAS, stats, df_faltas)
        self.assertIn('stem_boxplot', figuras)
        self.assertIn('stem_media_comparativa', figuras)
        self.assertIsNotNone(figuras['stem_boxplot'])
        self.assertIsNotNone(figuras['stem_media_comparativa'])

    def test_gerar_todos_graficos_sem_stem_nao_tem_chaves(self):
        df_notas, disciplinas = _turma_so_tecnicas()
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        figuras = gerar_todos_graficos(df_notas, 'Estradas', disciplinas, stats)
        self.assertNotIn('stem_boxplot', figuras)
        self.assertNotIn('stem_media_comparativa', figuras)


class TestGraficoStemFaixas(unittest.TestCase):
    """Passo 6: barras horizontais empilhadas por faixa de aproveitamento
    (< 40%, 40–60%, 60–80%, ≥ 80%), uma barra por disciplina STEM."""

    def test_uma_barra_por_disciplina_com_quatro_faixas_empilhadas(self):
        df_notas, _ = _turma_com_stem()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        cols_stem = stats['stem_codigos']
        max_pts = stats['max_pontos_bimestre']
        fig = grafico_stem_faixas(df_notas, 'Trânsito', DISCIPLINAS, cols_stem, max_pts)
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_yticklabels()), 3)  # MAT, FIS, QUI
        # 4 faixas empilhadas por disciplina = 4 chamadas de barh, 3 barras
        # cada uma → 12 retângulos no eixo.
        self.assertEqual(len(ax.patches), 4 * 3)

    def test_devolve_none_sem_stem(self):
        df_notas, disciplinas = _turma_so_tecnicas()
        self.assertIsNone(
            grafico_stem_faixas(df_notas, 'Estradas', disciplinas, [], 20))


class TestGraficoStemSobreposicao(unittest.TestCase):
    """Passo 6: contagem de alunos abaixo do limiar em 0, 1, 2 ou 3 STEM
    simultaneamente — identifica o núcleo duro de baixo desempenho."""

    def test_quatro_barras_somando_o_total_de_alunos(self):
        df_notas, _ = _turma_com_stem()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        cols_stem = stats['stem_codigos']
        limiar = stats['limiar_aprovacao']
        fig = grafico_stem_sobreposicao(df_notas, 'Trânsito', cols_stem, limiar)
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 4)  # barras para 0, 1, 2 e 3
        total_nas_barras = sum(p.get_height() for p in ax.patches)
        self.assertEqual(total_nas_barras, len(df_notas))

    def test_devolve_none_sem_stem(self):
        df_notas, disciplinas = _turma_so_tecnicas()
        self.assertIsNone(
            grafico_stem_sobreposicao(df_notas, 'Estradas', [], 12.0))


class TestGerarTodosGraficosIncluiFaixasESobreposicao(unittest.TestCase):
    """`gerar_todos_graficos` publica as duas figuras novas junto das STEM
    já existentes."""

    def test_chaves_stem_faixas_e_sobreposicao_presentes(self):
        df_notas, df_faltas = _turma_com_stem()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        figuras = gerar_todos_graficos(df_notas, 'Trânsito', DISCIPLINAS, stats, df_faltas)
        self.assertIn('stem_faixas', figuras)
        self.assertIn('stem_sobreposicao', figuras)
        self.assertIsNotNone(figuras['stem_faixas'])
        self.assertIsNotNone(figuras['stem_sobreposicao'])


if __name__ == "__main__":
    unittest.main()
