"""Cruzamento e conciliação entre dados da DAE e do Mapa de Turma do app.

Este módulo realiza o cruzamento por matrícula entre a planilha de acompanhamento
discente da Diretoria de Assuntos Estudantis (DAE) e os mapas de turma processados
pelo app (core.manipulacao).

Decisões de arquitetura e premissas:
    - Minimização de dados / LGPD (D1, D4): scripts de cruzamento e saídas de terminal
      operam estritamente sobre contagens e estatísticas agregadas (mediana, p90), nunca
      imprimindo nomes, CPFs ou matrículas de estudantes.
    - Comparação por bimestre (D5, D6): compara as faltas do mapa de turma (soma das colunas
      de disciplina de df_faltas) com as faltas apuradas na DAE:
      Σ(ha_ofertadas - ha_presenciadas) dos meses correspondentes (via MESES_POR_BIMESTRE),
      gerando diff_faltas_bim_<n>.
    - Junção completa (outer join + indicator): preserva todos os estudantes de ambas as
      fontes e quantifica a cobertura:
        * só no app (right_only)
        * só na DAE (left_only)
        * nos dois (both)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
import pandas as pd

# Assegura que o diretório sandbox/dae e a raiz do projeto estejam no sys.path
_DIR_DAE = Path(__file__).resolve().parent
_DIR_RAIZ = _DIR_DAE.parent.parent
for _p in (_DIR_DAE, _DIR_RAIZ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from .carregar import _normalizar_matricula, carregar_dae
    from .frequencia import MESES_POR_BIMESTRE, mes_referencia, meses_lancados
except ImportError:
    from carregar import _normalizar_matricula, carregar_dae
    from frequencia import MESES_POR_BIMESTRE, mes_referencia, meses_lancados

from core.manipulacao import processar_multiplos_bimestres


def _extrair_colunas_disciplinas(df: pd.DataFrame) -> list[str]:
    """Identifica colunas que representam disciplinas (não identificação/metadados)."""
    colunas_identificacao = {
        "matricula",
        "nome",
        "turma",
        "etapa",
        "curso",
        "situacao",
        "total faltas",
        "_merge",
    }
    return [
        c
        for c in df.columns
        if str(c).strip().lower() not in colunas_identificacao
        and not str(c).startswith("faltas_bim_")
        and not str(c).startswith("diff_faltas_bim_")
    ]


def _consolidar_faltas_app(
    df_faltas_app: pd.DataFrame | dict[int, pd.DataFrame] | Sequence[object],
    bimestres: Sequence[int] | int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Consolida os dados de faltas do app para cruzamento.

    Aceita:
        - pd.DataFrame com colunas de disciplina ou colunas 'faltas_bim_<n>'
        - dict {bimestre_num: df_faltas}
        - lista de conjuntos [(df_notas, df_faltas, disc_dict, meta), ...]
          conforme retornado por core.manipulacao.processar_multiplos_bimestres.

    Retorna:
        - pd.DataFrame com 'matricula' (normalizada) e 'faltas_bim_<n>'
        - lista de bimestres detectados
    """
    # Caso 1: Lista de conjuntos do processar_multiplos_bimestres ou lista de tuplas
    if isinstance(df_faltas_app, (list, tuple)) and df_faltas_app:
        primeiro = df_faltas_app[0]
        if isinstance(primeiro, (tuple, list)) and len(primeiro) >= 4:
            # Formato de processar_multiplos_bimestres: (df_notas, df_faltas, disc, meta)
            dict_bimestres: dict[int, pd.DataFrame] = {}
            for i, conjunto in enumerate(df_faltas_app):
                df_f = conjunto[1]
                meta = conjunto[3] if len(conjunto) > 3 else {}
                b_num = meta.get("bimestre_num", i + 1)
                dict_bimestres[int(b_num)] = df_f
            return _consolidar_faltas_app(dict_bimestres, bimestres=bimestres)

    # Caso 2: Dicionário {bimestre_num: df_faltas}
    if isinstance(df_faltas_app, dict):
        if not df_faltas_app:
            return pd.DataFrame(columns=["matricula"]), []

        dfs_por_bimestre: list[pd.DataFrame] = []
        bimestres_detectados: list[int] = []

        for b_num in sorted(df_faltas_app.keys()):
            df_b = df_faltas_app[b_num].copy()
            if "matricula" not in df_b.columns:
                continue

            df_b["matricula"] = df_b["matricula"].apply(_normalizar_matricula)
            b_int = int(b_num)
            bimestres_detectados.append(b_int)

            cols_disc = _extrair_colunas_disciplinas(df_b)
            col_alvo = f"faltas_bim_{b_int}"

            if cols_disc:
                soma = (
                    df_b[cols_disc]
                    .apply(pd.to_numeric, errors="coerce")
                    .sum(axis=1)
                )
            elif col_alvo in df_b.columns:
                soma = pd.to_numeric(df_b[col_alvo], errors="coerce")
            else:
                soma = pd.Series(0.0, index=df_b.index)

            cols_saida = {"matricula": df_b["matricula"], col_alvo: soma}
            if "nome" in df_b.columns:
                cols_saida["nome"] = df_b["nome"]

            dfs_por_bimestre.append(pd.DataFrame(cols_saida))

        if not dfs_por_bimestre:
            return pd.DataFrame(columns=["matricula"]), []

        df_consolidado = dfs_por_bimestre[0]
        for df_outro in dfs_por_bimestre[1:]:
            df_consolidado = pd.merge(
                df_consolidado,
                df_outro,
                on="matricula",
                how="outer",
                suffixes=("", "_dup"),
            )
            if "nome_dup" in df_consolidado.columns:
                if "nome" in df_consolidado.columns:
                    df_consolidado["nome"] = df_consolidado["nome"].combine_first(
                        df_consolidado["nome_dup"]
                    )
                else:
                    df_consolidado["nome"] = df_consolidado["nome_dup"]
                df_consolidado = df_consolidado.drop(columns=["nome_dup"])

        return df_consolidado, bimestres_detectados

    # Caso 3: pd.DataFrame
    if isinstance(df_faltas_app, pd.DataFrame):
        df_app = df_faltas_app.copy()
        if "matricula" not in df_app.columns:
            return df_app, []

        df_app["matricula"] = df_app["matricula"].apply(_normalizar_matricula)

        # Verifica se já possui colunas no formato faltas_bim_<n>
        bimestres_existentes: list[int] = []
        for col in df_app.columns:
            match = re.match(r"^faltas_bim_(\d+)$", col)
            if match:
                bimestres_existentes.append(int(match.group(1)))

        if bimestres_existentes:
            return df_app, sorted(bimestres_existentes)

        # Se não possui faltas_bim_<n>, soma colunas de disciplina
        cols_disc = _extrair_colunas_disciplinas(df_app)
        if cols_disc:
            if bimestres is not None:
                if isinstance(bimestres, int):
                    b_alvo = bimestres
                elif isinstance(bimestres, (list, tuple, set)) and len(bimestres) > 0:
                    b_alvo = int(list(bimestres)[0])
                else:
                    b_alvo = 1
            else:
                b_alvo = 1

            soma = df_app[cols_disc].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            df_app[f"faltas_bim_{b_alvo}"] = soma
            return df_app, [b_alvo]

        return df_app, [1]

    raise TypeError(
        f"Tipo não suportado para df_faltas_app: {type(df_faltas_app)}. "
        "Esperado pd.DataFrame, dict ou lista de conjuntos."
    )


def cruzar(
    df_dae: pd.DataFrame,
    df_faltas_app: pd.DataFrame | dict[int, pd.DataFrame] | Sequence[object],
    bimestres: Sequence[int] | int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Cruza dados da DAE com dados de faltas do app por matrícula (how='outer').

    Compara, por bimestre, as faltas do mapa de turma (soma das disciplinas)
    com Σ(ha_ofertadas − ha_presenciadas) dos meses correspondentes na DAE
    (conforme MESES_POR_BIMESTRE), gerando diff_faltas_bim_<n>.

    Para estudantes que não constam em ambas as bases (left_only ou right_only),
    diff_faltas_bim_<n> é preenchido com NaN.

    Parâmetros:
        df_dae: DataFrame carregado via carregar_dae.
        df_faltas_app: DataFrame, dict ou lista de conjuntos de faltas do app.
        bimestres: Bimestres específicos a analisar (opcional; se None, deduzido).

    Retorna:
        - DataFrame unido com indicador de junção ('_merge') e diff_faltas_bim_<n>.
        - dict com o resumo de cobertura: 'so_app', 'so_dae', 'nos_dois' e 'total'.
    """
    df_dae_work = df_dae.copy()
    if "_merge" in df_dae_work.columns:
        df_dae_work = df_dae_work.drop(columns=["_merge"])

    if "matricula" in df_dae_work.columns:
        df_dae_work["matricula"] = df_dae_work["matricula"].apply(_normalizar_matricula)
    else:
        df_dae_work["matricula"] = pd.Series(dtype=str)

    df_app_work, bimestres_detectados = _consolidar_faltas_app(
        df_faltas_app, bimestres=bimestres
    )
    if "_merge" in df_app_work.columns:
        df_app_work = df_app_work.drop(columns=["_merge"])

    # Determina bimestres para cálculo da diferença
    if bimestres is not None:
        if isinstance(bimestres, int):
            lista_bimestres = [bimestres]
        else:
            lista_bimestres = sorted(int(b) for b in bimestres)
    elif bimestres_detectados:
        lista_bimestres = sorted(bimestres_detectados)
    else:
        lista_bimestres = [1]

    # Junção outer com indicator
    df_unido = pd.merge(
        df_dae_work,
        df_app_work,
        on="matricula",
        how="outer",
        indicator=True,
        suffixes=("_dae", "_app"),
    )

    # Consolida coluna 'nome' se houver colunas separadas
    if "nome_dae" in df_unido.columns and "nome_app" in df_unido.columns:
        df_unido["nome"] = df_unido["nome_dae"].combine_first(df_unido["nome_app"])

    # Resumo de cobertura
    contagem_right = int((df_unido["_merge"] == "right_only").sum())
    contagem_left = int((df_unido["_merge"] == "left_only").sum())
    contagem_both = int((df_unido["_merge"] == "both").sum())

    resumo = {
        "so_app": contagem_right,
        "so_dae": contagem_left,
        "nos_dois": contagem_both,
        "total": contagem_right + contagem_left + contagem_both,
    }

    # Comparação por bimestre apenas para quem casa (_merge == 'both')
    mask_casa = df_unido["_merge"] == "both"

    for b in lista_bimestres:
        meses_bim = MESES_POR_BIMESTRE.get(b, [])

        # Identifica meses que estão presentes e possuem horas lançadas (> 0)
        meses_validos_bim: list[str] = []
        for m in meses_bim:
            col_o = f"ha_ofertadas_{m}"
            if col_o in df_unido.columns:
                vals = pd.to_numeric(df_unido[col_o], errors="coerce")
                if (vals > 0).any():
                    meses_validos_bim.append(m)

        col_diff = f"diff_faltas_bim_{b}"

        if not meses_validos_bim:
            # Sem meses lançados na DAE para este bimestre
            df_unido[col_diff] = np.nan
            continue

        # Calcula Σ(ha_ofertadas − ha_presenciadas) dos meses válidos
        cols_ofer = [f"ha_ofertadas_{m}" for m in meses_validos_bim]
        df_ofer = df_unido[cols_ofer].apply(pd.to_numeric, errors="coerce")
        soma_ofer = df_ofer.sum(axis=1, min_count=1)

        cols_pres = [
            f"ha_presenciadas_{m}"
            for m in meses_validos_bim
            if f"ha_presenciadas_{m}" in df_unido.columns
        ]
        if cols_pres:
            df_pres = df_unido[cols_pres].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            soma_pres = df_pres.sum(axis=1, min_count=1)
        else:
            soma_pres = pd.Series(0.0, index=df_unido.index)

        faltas_dae = soma_ofer - soma_pres
        faltas_dae = faltas_dae.mask(soma_ofer.isna() | (soma_ofer <= 0), np.nan)

        col_app = f"faltas_bim_{b}"
        if col_app in df_unido.columns:
            faltas_app = pd.to_numeric(df_unido[col_app], errors="coerce")
        else:
            faltas_app = pd.Series(np.nan, index=df_unido.index)

        diff = faltas_app - faltas_dae
        # Restringe estritamente para quem casa
        diff = diff.mask(~mask_casa, np.nan)
        df_unido[col_diff] = diff

    return df_unido, resumo


def _executar_cli(argv: list[str] | None = None) -> int:
    """Executa a interface de linha de comando para cruzamento DAE x Mapas de Turma.

    Imprime o mês de referência, os meses usados por bimestre e APENAS agregados
    (contagens, mediana e percentil 90 de |diff_faltas| por bimestre).
    Nunca imprime nomes ou matrículas (LGPD).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cruzamento de dados da DAE com Mapas de Turma (.xls) do App. "
            "Exibe apenas estatísticas agregadas (LGPD)."
        )
    )
    parser.add_argument(
        "--dae",
        dest="dae",
        type=str,
        default=None,
        help="Caminho para o arquivo da DAE (.xlsx ou .csv)",
    )
    parser.add_argument(
        "--mapas",
        dest="mapas",
        nargs="*",
        default=None,
        help="Caminhos para arquivos de Mapa de Turma (.xls)",
    )
    parser.add_argument(
        "arquivos",
        nargs="*",
        help="Arquivos de entrada (detecta automaticamente DAE [.xlsx/.csv] e Mapas [.xls])",
    )

    args = parser.parse_args(argv)

    caminho_dae: str | None = args.dae
    caminhos_mapas: list[str] = list(args.mapas or [])

    # Processa argumentos posicionais se fornecidos
    for arq in args.arquivos:
        p = Path(arq)
        ext = p.suffix.lower()
        if ext in (".xlsx", ".csv"):
            if caminho_dae is None:
                caminho_dae = str(p)
        elif ext == ".xls":
            if str(p) not in caminhos_mapas:
                caminhos_mapas.append(str(p))

    if not caminho_dae:
        print("Erro: Arquivo da DAE (.xlsx ou .csv) não fornecido.", file=sys.stderr)
        return 1

    if not caminhos_mapas:
        print("Erro: Nenhum mapa de turma (.xls) fornecido.", file=sys.stderr)
        return 1

    # 1. Carrega dados da DAE
    df_dae = carregar_dae(caminho_dae)

    # 2. Processa múltiplos bimestres via core.manipulacao (só import)
    conjuntos = processar_multiplos_bimestres(caminhos_mapas)

    # 3. Executa o cruzamento
    df_unido, resumo = cruzar(df_dae, conjuntos)

    # 4. Imprime mês de referência
    ref = mes_referencia(df_dae)
    print(f"Mês de referência DAE: {ref or 'Não identificado'}")

    # 5. Imprime meses usados por bimestre
    bimestres_presentes = [
        meta.get("bimestre_num") for _, _, _, meta in conjuntos if "bimestre_num" in meta
    ]
    if not bimestres_presentes:
        bimestres_presentes = [1]

    todos_lancados = set(meses_lancados(df_dae))
    for b in sorted(bimestres_presentes):
        meses_previstos = MESES_POR_BIMESTRE.get(b, [])
        meses_usados = [m for m in meses_previstos if m in todos_lancados]
        print(f"Bimestre {b}: meses previstos = {meses_previstos} | meses usados = {meses_usados}")

    # 6. Imprime APENAS agregados (LGPD)
    print("\nResumo de Cobertura:")
    print(f"  - Só no App: {resumo['so_app']}")
    print(f"  - Só na DAE: {resumo['so_dae']}")
    print(f"  - Nos dois: {resumo['nos_dois']}")
    print(f"  - Total: {resumo['total']}")

    print("\nDiscrepância de Faltas (|diff_faltas|) por Bimestre:")
    for b in sorted(bimestres_presentes):
        col_diff = f"diff_faltas_bim_{b}"
        if col_diff in df_unido.columns:
            diff_abs = df_unido.loc[df_unido["_merge"] == "both", col_diff].dropna().abs()
            n_casados = len(diff_abs)
            if n_casados > 0:
                mediana = float(diff_abs.median())
                p90 = float(diff_abs.quantile(0.90))
                print(f"  Bimestre {b} ({n_casados} estudantes comparados):")
                print(f"    - Mediana (|diff_faltas|): {mediana:.2f}")
                print(f"    - Percentil 90 (|diff_faltas|): {p90:.2f}")
            else:
                print(f"  Bimestre {b}: sem dados de estudantes casados para comparação.")

    return 0


if __name__ == "__main__":
    sys.exit(_executar_cli())
