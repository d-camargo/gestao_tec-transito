"""Testes das listas por critério de notas (limiar) e de faltas (média da turma).

D1 (2026-08-24): o corte da lista de notas deixou de ser relativo à média da
turma e passou a ser o **limiar de aprovação parcial** (60% da pontuação do
bimestre) — corte absoluto, ligado à aprovação. O corte de faltas continua
relativo à média da turma (não mudou). D4: o IDA deixou de ser calculado no
bimestre único, então nenhuma das duas listas carrega mais a coluna `ida`.
"""

import unittest

import pandas as pd

from core.relatorios import calcular_estatisticas

DISCIPLINAS = {
    'MAT': 'MATEMÁTICA - 2ª SÉRIE',
    'FIS': 'FÍSICA - 2ª SÉRIE',
}
METADADOS = {'bimestre_num': 1}  # 20 pts, limiar 12.0


def _turma_limiar():
    """Turma pequena com médias desenhadas em torno do limiar (12.0):

    - Ana e Bruno empatam em média 9,0 (abaixo do limiar) — testa o
      desempate por nome.
    - Carla fica em 10,5 (abaixo do limiar).
    - Beto fica EXATAMENTE no limiar (12,0) — não deve entrar na lista.
    - Duda fica acima do limiar (14,0) mas tem uma disciplina (FIS = 8,0)
      abaixo do limiar — não entra na lista, mas conta na legenda.
    - Elis fica bem acima do limiar (17,0) em ambas as disciplinas.
    """
    df_notas = pd.DataFrame({
        'nome': ['Ana', 'Bruno', 'Carla', 'Beto', 'Duda', 'Elis'],
        'MAT': [8.0, 9.0, 10.0, 12.0, 20.0, 18.0],
        'FIS': [10.0, 9.0, 11.0, 12.0, 8.0, 16.0],
    })
    return df_notas


def _turma_20_com_faltas():
    """Turma de 20 alunos: 13 com média 10 e 7 com média 13 (média geral 11,05).

    Faltas: 12 alunos sem falta, 7 com total 10 e 1 com total 60 — média de
    faltas totais 6,5, então 8 alunos ficam acima dela.
    """
    alunos, faltas_rows = [], []
    for i in range(1, 21):
        nome = f'Aluno {i:02d}'
        if i <= 13:  # abaixo do limiar
            alunos.append((nome, 10.0, 10.0))
        else:  # acima do limiar, mas com Física abaixo do limiar
            alunos.append((nome, 18.0, 8.0))
        if i == 20:
            total = 60
        elif i > 12:
            total = 10
        else:
            total = 0
        metade = total // 2
        faltas_rows.append((nome, metade, total - metade))

    df_notas = pd.DataFrame(
        [{'nome': n, 'MAT': m, 'FIS': f} for n, m, f in alunos])
    df_faltas = pd.DataFrame(
        [{'nome': n, 'MAT': m, 'FIS': f} for n, m, f in faltas_rows])
    return df_notas, df_faltas


class TestAlunosAbaixoLimiar(unittest.TestCase):
    """Passo 1: lista de notas pelo limiar de aprovação parcial (60%)."""

    def test_aluno_exatamente_no_limiar_nao_entra(self):
        df_notas = _turma_limiar()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        nomes_na_lista = stats['alunos_abaixo_limiar']['nome'].tolist()
        self.assertNotIn('Beto', nomes_na_lista)

    def test_entram_so_os_alunos_com_media_abaixo_do_limiar(self):
        df_notas = _turma_limiar()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        lista = stats['alunos_abaixo_limiar']
        self.assertEqual(set(lista['nome']), {'Ana', 'Bruno', 'Carla'})
        self.assertEqual(len(lista), 3)
        self.assertEqual(stats['n_abaixo_limiar'], 3)

    def test_ordenacao_por_media_ascendente_e_desempate_por_nome(self):
        df_notas = _turma_limiar()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        lista = stats['alunos_abaixo_limiar']
        # Ana e Bruno empatam em 9,0 — desempate alfabético (Ana antes de Bruno);
        # Carla (10,5) vem por último.
        self.assertEqual(lista['nome'].tolist(), ['Ana', 'Bruno', 'Carla'])
        medias = [float(m.replace(',', '.')) for m in lista['Média']]
        self.assertEqual(medias, sorted(medias))

    def test_colunas_sem_ida(self):
        """D4: bimestre único não calcula mais IDA — a lista não carrega a coluna."""
        df_notas = _turma_limiar()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        self.assertEqual(
            list(stats['alunos_abaixo_limiar'].columns),
            ['nome', 'Média', 'disciplinas_abaixo_limiar'])
        self.assertNotIn('ida', stats['alunos_abaixo_limiar'].columns)

    def test_conta_acima_do_limiar_com_disciplina_abaixo_do_limiar(self):
        """Duda tem média acima do limiar, mas FIS abaixo — não entra na
        lista, mas é contado na legenda (buraco conhecido do corte por
        média própria)."""
        df_notas = _turma_limiar()
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        self.assertNotIn('Duda', stats['alunos_abaixo_limiar']['nome'].tolist())
        self.assertEqual(stats['n_acima_limiar_com_disciplina_abaixo'], 1)

    def test_turma_de_um_aluno_devolve_vazio(self):
        df_notas = pd.DataFrame({'nome': ['Ana'], 'MAT': [15.0], 'FIS': [15.0]})
        stats = calcular_estatisticas(df_notas, DISCIPLINAS, metadados=METADADOS)
        self.assertTrue(stats['alunos_abaixo_limiar'].empty)
        self.assertEqual(stats['n_abaixo_limiar'], 0)


class TestAlunosAcimaMediaFaltas(unittest.TestCase):
    """Passo 4: lista de faltas por média da turma (corte relativo, sem mudança)."""

    def test_turma_de_20_com_8_acima_gera_8_linhas(self):
        df_notas, df_faltas = _turma_20_com_faltas()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        lista = stats['alunos_acima_media_faltas']
        self.assertEqual(len(lista), 8)
        totais = lista['Total Faltas'].tolist()
        self.assertEqual(totais, sorted(totais, reverse=True))

    def test_aluno_com_faltas_absurdas_recebe_sinal_p90_e_sigma(self):
        df_notas, df_faltas = _turma_20_com_faltas()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        lista = stats['alunos_acima_media_faltas']
        absurdos = lista[lista['nome'] == 'Aluno 20']
        self.assertEqual(len(absurdos), 1)
        self.assertEqual(absurdos.iloc[0]['Sinal'], '> P90 e > μ+2σ')
        # Dentro da lista, quem está acima da média mas não do P90 fica sem sinal.
        demais = lista[lista['nome'] != 'Aluno 20']
        self.assertTrue((demais['Sinal'] == '').all())

    def test_legenda_usa_media_como_porta_de_entrada(self):
        df_notas, df_faltas = _turma_20_com_faltas()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        self.assertAlmostEqual(stats['media_faltas_total'], 6.5)
        self.assertIn('p90_faltas_total', stats)
        self.assertIn('sigma2_faltas_total', stats)

    def test_colunas_sem_ida(self):
        """D4: a lista de faltas também nunca teve/tem a coluna `ida`."""
        df_notas, df_faltas = _turma_20_com_faltas()
        stats = calcular_estatisticas(
            df_notas, DISCIPLINAS, df_faltas=df_faltas, metadados=METADADOS)
        self.assertNotIn('ida', stats['alunos_acima_media_faltas'].columns)


if __name__ == "__main__":
    unittest.main()
