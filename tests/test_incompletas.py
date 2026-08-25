"""Testes dos quatro sinais de suspeita de lançamento incompleto (Passo 2-bis).

`calcular_estatisticas` publica `disciplinas_incompletas`: lista de
dicionários (`codigo`, `nome`, `max_observado`, `cobertura`, `motivos`), um
por disciplina em que ao menos um dos quatro sinais A/B/C/D disparou.

- A — forte: nota máxima observada ≤ 50% da pontuação do bimestre.
- B — absoluto: nota máxima observada < 90% da pontuação do bimestre.
- C — relativo: nota máxima observada < 80% da mediana dos máximos das
  demais disciplinas.
- D — cobertura: menos de 90% dos alunos da turma têm nota lançada.

B e C se cobrem mutuamente (mapa inteiro parcial vs. disciplina destoante
num bimestre difícil) e, por isso, sempre que C dispara, B também dispara —
o corte de C (80% da mediana das demais) nunca é mais permissivo que o de B
(90% da pontuação do bimestre), já que a mediana das demais nunca ultrapassa
a própria pontuação do bimestre. D é o único sinal genuinamente independente
dos outros três.
"""

import unittest

import pandas as pd

from core.relatorios import calcular_estatisticas, get_simplified_name

METADADOS = {'bimestre_num': 1}  # 20 pts, limiar 12.0


def _por_codigo(incompletas, codigo):
    for d in incompletas:
        if d['codigo'] == codigo:
            return d
    return None


class TestSinalA(unittest.TestCase):
    """A — forte: nota máxima observada ≤ 50% da pontuação do bimestre."""

    def test_sinal_a_dispara_com_maximo_na_metade_da_pontuacao(self):
        disciplinas = {'MAT': 'MATEMÁTICA - 2ª SÉRIE', 'FIS': 'FÍSICA - 2ª SÉRIE'}
        # MAT: máximo observado = 10 (exatamente 50% de 20) — dispara A.
        # FIS: máximo observado = 20 (pontuação cheia) — nenhum sinal.
        df_notas = pd.DataFrame({
            'nome': [f'Aluno {i}' for i in range(1, 6)],
            'MAT': [10.0, 8.0, 6.0, 9.0, 10.0],
            'FIS': [20.0, 18.0, 15.0, 12.0, 20.0],
        })
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        incompletas = stats['disciplinas_incompletas']

        mat = _por_codigo(incompletas, 'MAT')
        self.assertIsNotNone(mat)
        self.assertIn('A', mat['motivos'])
        self.assertEqual(mat['max_observado'], 10.0)
        self.assertEqual(mat['cobertura'], 1.0)
        self.assertEqual(mat['nome'], get_simplified_name('MAT', disciplinas))

        self.assertIsNone(_por_codigo(incompletas, 'FIS'))


class TestSinalB(unittest.TestCase):
    """B — absoluto: máximo < 90% da pontuação. Cenário em que TODAS as
    disciplinas estão uniformemente baixas — o único caso que só B pega,
    porque nenhuma disciplina destoa das demais (a mediana também está
    baixa, deixando C cego) e a cobertura está completa (D fica de fora)."""

    def test_sinal_b_isolado_com_todas_as_disciplinas_uniformemente_baixas(self):
        disciplinas = {
            'MAT': 'MATEMÁTICA - 2ª SÉRIE',
            'FIS': 'FÍSICA - 2ª SÉRIE',
            'QUI': 'QUÍMICA - 2ª SÉRIE',
        }
        # As três disciplinas têm o mesmo máximo observado (17 de 20 = 85%):
        # abaixo de 90% (B), mas acima de 50% (A não dispara), e nenhuma
        # destoa das demais (C não dispara) — cobertura plena (D não dispara).
        df_notas = pd.DataFrame({
            'nome': [f'Aluno {i}' for i in range(1, 11)],
            'MAT': [17.0] * 10,
            'FIS': [17.0] * 10,
            'QUI': [17.0] * 10,
        })
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        incompletas = stats['disciplinas_incompletas']
        self.assertEqual({d['codigo'] for d in incompletas}, {'MAT', 'FIS', 'QUI'})
        for d in incompletas:
            self.assertEqual(d['motivos'], ['B'])
            self.assertEqual(d['max_observado'], 17.0)
            self.assertEqual(d['cobertura'], 1.0)


class TestSinalC(unittest.TestCase):
    """C — relativo: uma disciplina destoando das demais num bimestre em
    que as outras estão com desempenho alto (não uniformemente baixo, o que
    isola do cenário do sinal B)."""

    def test_sinal_c_dispara_para_a_disciplina_destoante(self):
        disciplinas = {
            'MAT': 'MATEMÁTICA - 2ª SÉRIE',
            'FIS': 'FÍSICA - 2ª SÉRIE',
            'QUI': 'QUÍMICA - 2ª SÉRIE',
            'POR': 'LÍNGUA PORTUGUESA - 2ª SÉRIE',
        }
        # FIS, QUI e POR ficam com máximo perto do teto (mediana 20).
        # MAT destoa: máximo 13 (65% de 20, acima do piso de 50% de A) mas
        # abaixo de 80% da mediana das demais (16) — dispara C. Como
        # 0,8×mediana(demais) nunca ultrapassa 0,9×pontuação do bimestre,
        # sempre que C dispara B também dispara (documentado no módulo) —
        # não é um bug deste teste, é a matemática dos dois sinais.
        df_notas = pd.DataFrame({
            'nome': [f'Aluno {i}' for i in range(1, 6)],
            'MAT': [13.0, 10.0, 9.0, 12.0, 13.0],
            'FIS': [20.0, 18.0, 19.0, 17.0, 20.0],
            'QUI': [19.0, 17.0, 18.0, 16.0, 19.0],
            'POR': [20.0, 19.0, 18.0, 17.0, 20.0],
        })
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        incompletas = stats['disciplinas_incompletas']

        mat = _por_codigo(incompletas, 'MAT')
        self.assertIsNotNone(mat)
        self.assertIn('C', mat['motivos'])
        self.assertNotIn('A', mat['motivos'])  # 13 > 50% de 20
        self.assertEqual(mat['max_observado'], 13.0)

        # As demais, todas perto do teto, não disparam nenhum sinal.
        self.assertIsNone(_por_codigo(incompletas, 'FIS'))
        self.assertIsNone(_por_codigo(incompletas, 'QUI'))
        self.assertIsNone(_por_codigo(incompletas, 'POR'))


class TestSinalD(unittest.TestCase):
    """D — cobertura: menos de 90% dos alunos com nota lançada, mesmo com
    quem lançou tirando o teto (nota máxima cheia)."""

    def test_sinal_d_isolado_com_cobertura_baixa_e_nota_maxima_cheia(self):
        disciplinas = {'MAT': 'MATEMÁTICA - 2ª SÉRIE', 'FIS': 'FÍSICA - 2ª SÉRIE'}
        # 10 alunos na turma; só 8 têm MAT lançada (80% < 90% → D), todos
        # tirando nota máxima (20 = pontuação do bimestre, sem A/B/C). FIS
        # tem cobertura plena, também no teto — sem nenhum sinal.
        notas_mat = [20.0] * 8 + [None, None]
        df_notas = pd.DataFrame({
            'nome': [f'Aluno {i}' for i in range(1, 11)],
            'MAT': notas_mat,
            'FIS': [20.0] * 10,
        })
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        incompletas = stats['disciplinas_incompletas']

        mat = _por_codigo(incompletas, 'MAT')
        self.assertIsNotNone(mat)
        self.assertEqual(mat['motivos'], ['D'])
        self.assertEqual(mat['max_observado'], 20.0)
        self.assertAlmostEqual(mat['cobertura'], 0.8)

        self.assertIsNone(_por_codigo(incompletas, 'FIS'))


class TestEstruturaDoDicionario(unittest.TestCase):
    """Cada entrada de `disciplinas_incompletas` é um dicionário com as
    cinco chaves esperadas (Passo 2-bis substitui a tupla de 3 posições)."""

    def test_chaves_do_dicionario(self):
        disciplinas = {'MAT': 'MATEMÁTICA - 2ª SÉRIE'}
        df_notas = pd.DataFrame({
            'nome': ['Ana', 'Beto'],
            'MAT': [8.0, 9.0],  # 45% de 20 → dispara A (e B e C não se aplicam/são vazios)
        })
        stats = calcular_estatisticas(df_notas, disciplinas, metadados=METADADOS)
        incompletas = stats['disciplinas_incompletas']
        self.assertEqual(len(incompletas), 1)
        entrada = incompletas[0]
        self.assertEqual(
            set(entrada.keys()), {'codigo', 'nome', 'max_observado', 'cobertura', 'motivos'})
        self.assertIsInstance(entrada['motivos'], list)


if __name__ == "__main__":
    unittest.main()
