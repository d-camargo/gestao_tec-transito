"""Testes unitários para o processamento do curso integrado Trânsito e Estradas."""

import unittest

from core.manipulacao import (
    processar_transito_estradas,
    processar_multiplos_bimestres,
    processar_multiplos_bimestres_transito_estradas,
    ArquivoInvalidoError,
)
from tests.conftest import mapa_sintetico


class TestTransitoEstradas(unittest.TestCase):
    """Testes para o fluxo conjunto de Trânsito + Estradas."""

    def test_processar_transito_estradas_sucesso(self):
        """Testa o processamento conjunto bem-sucedido de Trânsito e Estradas (mesmo bimestre)."""
        # Trânsito tem disciplinas técnicas e um aluno
        df_tt = mapa_sintetico(
            curso="TÉCNICO EM TRÂNSITO",
            bimestre=1,
            turma="TRA.1A",
            disciplinas={"1TT.009": "PLANEJAMENTO DE TRANSPORTES"},
            alunos=[("20260000001", "Aluno Trânsito", {"1TT.009": 18.0}, {"1TT.009": 0})],
        )

        # Estradas tem disciplinas de Ensino Médio + Técnica e dois alunos (um compartilhado com Trânsito)
        df_est = mapa_sintetico(
            curso="TÉCNICO EM ESTRADAS",
            bimestre=1,
            turma="EST.1A",
            disciplinas={
                "1MAT.006": "MATEMÁTICA - 1ª SÉRIE",
                "1TT.43": "LABORATÓRIO DE SOLOS",
            },
            alunos=[
                ("20260000001", "Aluno Trânsito", {"1MAT.006": 15.0, "1TT.43": 12.0}, {"1MAT.006": 2, "1TT.43": 1}),
                ("20260000002", "Aluno Estradas", {"1MAT.006": 12.0, "1TT.43": 15.0}, {"1MAT.006": 0, "1TT.43": 2}),
            ],
        )

        conjuntos_tt, conjuntos_est = processar_transito_estradas(df_tt, df_est)

        df_notas_tt, df_faltas_tt, disc_tt, meta_tt = conjuntos_tt
        df_notas_est, df_faltas_est, disc_est, meta_est = conjuntos_est

        # Valida metadados
        self.assertEqual(meta_tt["bimestre_num"], 1)
        self.assertEqual(meta_est["bimestre_num"], 1)
        self.assertEqual(meta_tt["curso_amigavel"], "Trânsito")
        self.assertEqual(meta_est["curso_amigavel"], "Estradas")

        # Trânsito deve ter herdado a disciplina de ensino médio (1MAT.006) de Estradas
        self.assertIn("1MAT.006", df_notas_tt.columns)
        self.assertEqual(df_notas_tt.loc[df_notas_tt["matricula"] == "20260000001", "1MAT.006"].values[0], 15.0)

        # Estradas deve ter removido o aluno de Trânsito para evitar dupla contagem
        self.assertNotIn("20260000001", df_notas_est["matricula"].values)
        self.assertIn("20260000002", df_notas_est["matricula"].values)

    def test_processar_transito_estradas_bimestres_diferentes(self):
        """Testa que bimestres divergentes entre Trânsito e Estradas geram erro."""
        df_tt = mapa_sintetico(
            curso="TÉCNICO EM TRÂNSITO",
            bimestre=1,
            turma="TRA.1A",
            disciplinas={"1TT.009": "PLANEJAMENTO DE TRANSPORTES"},
        )
        df_est = mapa_sintetico(
            curso="TÉCNICO EM ESTRADAS",
            bimestre=2,
            turma="EST.1A",
            disciplinas={"1MAT.006": "MATEMÁTICA - 1ª SÉRIE"},
        )

        with self.assertRaises(ArquivoInvalidoError) as ctx:
            processar_transito_estradas(df_tt, df_est)

        self.assertIn("cobrem bimestres diferentes", str(ctx.exception))

    def test_processar_multiplos_bimestres_transito_estradas_mensagem_acionavel(self):
        """Testa que ao enviar Trânsito + Estradas no modo de curso único a mensagem orienta usar o modo integrado."""
        df_tt = mapa_sintetico(
            curso="TÉCNICO EM TRÂNSITO - BH-1TRANS - A (2024)",
            bimestre=1,
            turma="TRA.1A",
        )
        df_est = mapa_sintetico(
            curso="TÉCNICO EM ESTRADAS - BH-1EST - A (2024)",
            bimestre=1,
            turma="EST.1A",
        )

        with self.assertRaises(ArquivoInvalidoError) as ctx:
            processar_multiplos_bimestres([df_tt, df_est])

        mensagem = str(ctx.exception)
        self.assertIn("Curso integrado Trânsito + Estradas (1ª série)", mensagem)
        self.assertIn("envie cada mapa no seu próprio campo", mensagem)

    def test_processar_multiplos_bimestres_outros_cursos_mensagem_padrao(self):
        """Testa que ao enviar turmas de outros cursos no modo curso único mantemos a mensagem padrão."""
        df_edif = mapa_sintetico(
            curso="TÉCNICO EM EDIFICAÇÕES",
            bimestre=1,
            turma="EDI.1A",
        )
        df_info = mapa_sintetico(
            curso="TÉCNICO EM INFORMÁTICA",
            bimestre=1,
            turma="INF.1A",
        )

        with self.assertRaises(ArquivoInvalidoError) as ctx:
            processar_multiplos_bimestres([df_edif, df_info])

        mensagem = str(ctx.exception)
        self.assertIn("Envie apenas arquivos da mesma turma.", mensagem)
        self.assertNotIn("Curso integrado", mensagem)

    def test_processar_multiplos_bimestres_transito_estradas_campos_invertidos(self):
        """Testa que enviar o mapa de Estradas no campo de Trânsito (ou vice-versa) levanta erro pedindo a troca."""
        df_tt = mapa_sintetico(
            curso="TÉCNICO EM TRÂNSITO",
            bimestre=1,
            turma="TRA.1A",
        )
        df_est = mapa_sintetico(
            curso="TÉCNICO EM ESTRADAS",
            bimestre=1,
            turma="EST.1A",
        )

        # Invertido: df_est no campo de Trânsito
        with self.assertRaises(ArquivoInvalidoError) as ctx:
            processar_multiplos_bimestres_transito_estradas([df_est], [df_tt])

        mensagem = str(ctx.exception)
        self.assertIn("O campo de Trânsito recebeu um arquivo do curso", mensagem)
        self.assertIn("TÉCNICO EM ESTRADAS", mensagem)
        self.assertIn("faça a troca dos campos", mensagem)

        # Invertido: df_tt no campo de Estradas
        with self.assertRaises(ArquivoInvalidoError) as ctx:
            processar_multiplos_bimestres_transito_estradas([df_tt], [df_tt])

        mensagem = str(ctx.exception)
        self.assertIn("O campo de Estradas recebeu um arquivo do curso", mensagem)
        self.assertIn("TÉCNICO EM TRÂNSITO", mensagem)
        self.assertIn("faça a troca dos campos", mensagem)


if __name__ == "__main__":
    unittest.main()
