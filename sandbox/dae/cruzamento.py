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
    from .carregar import (
        _normalizar_matricula,
        _normalizar_pe_de_meia,
        _normalizar_rotulo,
        carregar_dae,
    )
    from .ch_efetiva import PADRAO_CH_EFETIVA
    from .det import (
        CURSOS_DET,
        ROTULO_DET,
        carregar_det,
        classificar_mapas,
        conjuntos_det,
        eh_det,
        resumo_det,
    )
    from .frequencia import MESES_POR_BIMESTRE, mes_referencia, meses_lancados
except ImportError:
    from carregar import (
        _normalizar_matricula,
        _normalizar_pe_de_meia,
        _normalizar_rotulo,
        carregar_dae,
    )
    from ch_efetiva import PADRAO_CH_EFETIVA
    from det import (
        CURSOS_DET,
        ROTULO_DET,
        carregar_det,
        classificar_mapas,
        conjuntos_det,
        eh_det,
        resumo_det,
    )
    from frequencia import MESES_POR_BIMESTRE, mes_referencia, meses_lancados

from core.manipulacao import processar_multiplos_bimestres

PASTA_DADOS: Path = _DIR_DAE / "dados"


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
    # Caso 0: conjuntos_det do DET (ou tupla (conjuntos_tt, conjuntos_est))
    if isinstance(df_faltas_app, conjuntos_det) or (
        isinstance(df_faltas_app, (tuple, list))
        and len(df_faltas_app) == 2
        and isinstance(df_faltas_app[0], (tuple, list))
        and isinstance(df_faltas_app[1], (tuple, list))
        and (
            not df_faltas_app[0]
            or (isinstance(df_faltas_app[0][0], (tuple, list)) and len(df_faltas_app[0][0]) >= 4)
        )
    ):
        conj_tt, conj_est = df_faltas_app[0], df_faltas_app[1]
        df_tt, bim_tt = _consolidar_faltas_app(conj_tt, bimestres=bimestres)
        df_est, bim_est = _consolidar_faltas_app(conj_est, bimestres=bimestres)

        df_consolidado = pd.concat([df_tt, df_est], ignore_index=True)
        if "matricula" in df_consolidado.columns:
            df_consolidado = df_consolidado.drop_duplicates(subset=["matricula"])
        bimestres_detectados = sorted(list(set(bim_tt) | set(bim_est)))
        return df_consolidado, bimestres_detectados

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


def _eh_matricula_valida(val: object) -> bool:
    r"""Verifica se a matrícula possui exatamente 11 dígitos após _normalizar_matricula (C10)."""
    norm = _normalizar_matricula(val)
    return bool(re.fullmatch(r"\d{11}", norm))


def resumo_mapa(conjuntos: list) -> list[dict]:
    """Extrai métricas e validações de cada conjunto do mapa de turma (C10).

    Um dict por conjunto (na ordem de ``conjuntos``), com as chaves:
    ``curso_amigavel``, ``turma``, ``bimestre_num``, ``periodo_letivo``,
    ``n_alunos``, ``n_matriculas_validas`` (11 dígitos após
    ``_normalizar_matricula``), ``n_matriculas_duplicadas``, ``n_disciplinas``,
    ``faltas_total`` e ``faltas_mediana_aluno``.

    Args:
        conjuntos: Lista retornada por core.manipulacao.processar_multiplos_bimestres.

    Returns:
        Lista de dicts, um por conjunto. Só agregados (LGPD).
    """
    resumo: list[dict] = []
    for i, (df_notas, df_faltas, legenda, meta) in enumerate(conjuntos):
        legenda = legenda if isinstance(legenda, dict) else {}
        meta = meta if isinstance(meta, dict) else {}

        matriculas = (
            df_faltas["matricula"]
            if "matricula" in df_faltas.columns
            else pd.Series(dtype=object)
        )
        norm = matriculas.apply(_normalizar_matricula)

        cols_disc = _extrair_colunas_disciplinas(df_faltas)
        if legenda:
            filtradas = [c for c in cols_disc if c in legenda]
            cols_disc = filtradas or cols_disc

        faltas_num = df_faltas[cols_disc].apply(pd.to_numeric, errors="coerce") if cols_disc else pd.DataFrame(index=df_faltas.index)
        faltas_por_aluno = faltas_num.fillna(0).sum(axis=1)

        resumo.append(
            {
                "curso_amigavel": str(meta.get("curso_amigavel") or meta.get("curso") or ""),
                "turma": str(meta.get("turma") or ""),
                "bimestre_num": int(meta.get("bimestre_num", i + 1)),
                "periodo_letivo": str(meta.get("periodo_letivo", "")),
                "n_alunos": len(df_faltas),
                "n_matriculas_validas": int(norm.apply(_eh_matricula_valida).sum()),
                "n_matriculas_duplicadas": int(norm.duplicated().sum()),
                "n_disciplinas": len(cols_disc),
                "faltas_total": int(round(faltas_por_aluno.sum())),
                "faltas_mediana_aluno": float(faltas_por_aluno.median()) if len(faltas_por_aluno) else 0.0,
            }
        )
    return resumo


def contagem_pe_de_meia(
    df_dae: pd.DataFrame,
    curso_contem: str | Sequence[str] | tuple[str, ...] | None = None,
    matriculas: Sequence[str] | set[str] | pd.Series | None = None,
) -> dict[str, int]:
    """Contabiliza os discentes por situação do programa Pé-de-Meia (C11).

    Mantém as 4 chaves de ``carregar_dae`` (``elegivel``, ``nao_elegivel``,
    ``nada_consta``, ``indefinida``) mais ``total``, zeros incluídos.
    ``curso_contem`` casa sem acento e sem caixa (aceita str ou tupla para união);
    ``matriculas`` restringe o universo de discentes.

    Args:
        df_dae: DataFrame com dados da DAE contendo a coluna 'pe_de_meia'.
        curso_contem: Substring ou tupla/sequência de substrings para filtrar a coluna 'curso'
                      (ex.: 'estradas' ou ('estradas', 'transito')).
        matriculas: Matrículas que delimitam o universo (ex.: alunos do mapa).

    Returns:
        Dicionário com as cinco contagens.
    """
    df_filtrado = df_dae
    if curso_contem is not None and "curso" in df_filtrado.columns:
        if isinstance(curso_contem, str):
            alvos = [_normalizar_rotulo(curso_contem)]
        else:
            alvos = [_normalizar_rotulo(x) for x in curso_contem]
        mask_curso = df_filtrado["curso"].apply(
            lambda c: any(alvo in _normalizar_rotulo(c) for alvo in alvos)
        )
        df_filtrado = df_filtrado[mask_curso]

    if matriculas is not None and "matricula" in df_filtrado.columns:
        mats_set = {_normalizar_matricula(m) for m in matriculas if _normalizar_matricula(m)}
        mask_mats = df_filtrado["matricula"].apply(_normalizar_matricula).isin(mats_set)
        df_filtrado = df_filtrado[mask_mats]

    if "pe_de_meia" in df_filtrado.columns:
        pdm_norm = df_filtrado["pe_de_meia"].apply(_normalizar_pe_de_meia)
        contagens = {
            "elegivel": int((pdm_norm == "elegivel").sum()),
            "nao_elegivel": int((pdm_norm == "nao_elegivel").sum()),
            "nada_consta": int((pdm_norm == "nada_consta").sum()),
            "indefinida": int((pdm_norm == "indefinida").sum()),
        }
    else:
        contagens = {"elegivel": 0, "nao_elegivel": 0, "nada_consta": 0, "indefinida": 0}

    contagens["total"] = sum(contagens.values())
    return contagens


def _executar_cli(argv: list[str] | None = None) -> int:
    """Executa a interface de linha de comando para cruzamento DAE x Mapas de Turma.

    Imprime resumo do mapa e, caso a planilha da DAE esteja disponível, exibe
    estatísticas agregadas e recortes de Pé-de-Meia (C10, C11).
    Caso apenas mapas estejam presentes, opera em modo só-mapa (PENDENTE: ..., retorno 0).
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
            if caminho_dae is None and not PADRAO_CH_EFETIVA.match(p.name):
                caminho_dae = str(p)
        elif ext == ".xls":
            if str(p) not in caminhos_mapas:
                caminhos_mapas.append(str(p))

    # Descoberta automática em PASTA_DADOS quando nenhum arquivo for fornecido
    if not caminho_dae and not caminhos_mapas and not args.arquivos:
        if PASTA_DADOS.exists() and PASTA_DADOS.is_dir():
            candidatos_dae: list[Path] = []
            for p in sorted(PASTA_DADOS.iterdir()):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext == ".xls":
                    caminhos_mapas.append(str(p))
                elif ext in (".xlsx", ".csv") and not PADRAO_CH_EFETIVA.match(p.name):
                    candidatos_dae.append(p)
            if len(candidatos_dae) > 1:
                # Mais de um candidato a DAE: o de mtime mais recente, dizendo só o nome
                escolhido = max(candidatos_dae, key=lambda p: p.stat().st_mtime)
                print(f"DAE: vários arquivos candidatos em dados/ — usando {escolhido.name} (mais recente).")
                caminho_dae = str(escolhido)
            elif candidatos_dae:
                caminho_dae = str(candidatos_dae[0])

    if not caminhos_mapas:
        print("Erro: Nenhum mapa de turma (.xls) fornecido ou encontrado em dados/.", file=sys.stderr)
        return 1

    try:
        mapas_classificados = classificar_mapas(caminhos_mapas)
        eh_det_mapas = eh_det(mapas_classificados)
    except Exception:
        mapas_classificados = {}
        eh_det_mapas = False

    # Modo só-mapa: mapas presentes, mas planilha da DAE ausente
    if not caminho_dae:
        if eh_det_mapas:
            det = carregar_det(mapas_classificados)
            r_det = resumo_det(det)
            est_n = r_det.alunos_por_curso.get("Estradas", 0)
            tt_n = r_det.alunos_por_curso.get("Trânsito", 0)
            print(
                f"{ROTULO_DET}: {r_det.alunos_total} alunos / "
                f"Estradas {est_n} / Trânsito {tt_n} / interseção {r_det.intersecao}"
            )
            print("PENDENTE: arquivo da DAE (.xlsx/.csv) ausente em sandbox/dae/dados/ — cruzamento não executado.")
            return 0

        conjuntos = processar_multiplos_bimestres(caminhos_mapas)
        resumo = resumo_mapa(conjuntos)
        r0 = resumo[0]
        print(f"Mapa de turma: {r0['n_alunos']} alunos / bimestre {r0['bimestre_num']} / {r0['n_disciplinas']} disciplinas")
        print(f"Total de faltas registradas: {r0['faltas_total']}")
        if r0["n_matriculas_duplicadas"] > 0:
            print(f"Matrículas duplicadas detectadas: {r0['n_matriculas_duplicadas']}")
        validas = r0["n_matriculas_validas"]
        invalidas = r0["n_alunos"] - validas
        if invalidas > 0:
            print(f"Matrículas fora do padrão (inválidas): {invalidas}")
        print("PENDENTE: arquivo da DAE (.xlsx/.csv) ausente em sandbox/dae/dados/ — cruzamento não executado.")
        return 0

    # Cruzamento completo com DAE
    df_dae = carregar_dae(caminho_dae)
    if eh_det_mapas:
        det = carregar_det(mapas_classificados)
        r_det = resumo_det(det)
        est_n = r_det.alunos_por_curso.get("Estradas", 0)
        tt_n = r_det.alunos_por_curso.get("Trânsito", 0)
        print(
            f"{ROTULO_DET}: {r_det.alunos_total} alunos / "
            f"Estradas {est_n} / Trânsito {tt_n} / interseção {r_det.intersecao}"
        )
        df_unido, resumo = cruzar(df_dae, conjuntos_det(det))
        bimestres_presentes = r_det.bimestres if r_det.bimestres else [1]
        curso_alvo = ("estradas", "transito")
        mats_mapa = set()
        for conj in det:
            for _, df_f, _, _ in conj:
                if "matricula" in df_f.columns:
                    mats_mapa.update(df_f["matricula"].apply(_normalizar_matricula).dropna().unique())
    else:
        conjuntos = processar_multiplos_bimestres(caminhos_mapas)
        resumo_mapa_lista = resumo_mapa(conjuntos)
        r0 = resumo_mapa_lista[0]
        df_unido, resumo = cruzar(df_dae, conjuntos)
        print(f"Mapa de turma: {r0['n_alunos']} alunos / bimestre {r0['bimestre_num']} / {r0['n_disciplinas']} disciplinas")
        bimestres_presentes = [
            meta.get("bimestre_num") for _, _, _, meta in conjuntos if "bimestre_num" in meta
        ]
        if not bimestres_presentes:
            bimestres_presentes = [1]
        curso_alvo = None
        for _, _, _, meta in conjuntos:
            if isinstance(meta, dict):
                curso_alvo = meta.get("curso_amigavel") or meta.get("curso")
                if curso_alvo:
                    break
        mats_mapa = set()
        for _, df_f, _, _ in conjuntos:
            if "matricula" in df_f.columns:
                mats_mapa.update(df_f["matricula"].apply(_normalizar_matricula).dropna().unique())

    ref = mes_referencia(df_dae)
    print(f"Mês de referência DAE: {ref or 'Não identificado'}")

    todos_lancados = set(meses_lancados(df_dae))
    for b in sorted(bimestres_presentes):
        meses_previstos = MESES_POR_BIMESTRE.get(b, [])
        meses_usados = [m for m in meses_previstos if m in todos_lancados]
        print(f"Bimestre {b}: meses previstos = {meses_previstos} | meses usados = {meses_usados}")

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

    # Recortes de Pé-de-Meia (C11)
    contagem_curso = contagem_pe_de_meia(df_dae, curso_contem=curso_alvo)
    contagem_mapa = contagem_pe_de_meia(df_dae, matriculas=mats_mapa)

    if isinstance(curso_alvo, (tuple, list)):
        rotulo_curso = f" no Curso ({' + '.join(CURSOS_DET)})"
    elif curso_alvo:
        rotulo_curso = f" no Curso ({curso_alvo})"
    else:
        rotulo_curso = ""

    print(f"\nRecorte Pé-de-Meia (DAE{rotulo_curso}):")
    print(f"  - Elegível: {contagem_curso['elegivel']}")
    print(f"  - Não elegível: {contagem_curso['nao_elegivel']}")
    print(f"  - N/C (Nada consta): {contagem_curso['nada_consta']}")
    if contagem_curso["indefinida"] > 0:
        print(f"  - Elegibilidade indefinida: {contagem_curso['indefinida']}")
    print(f"  - Total: {contagem_curso['total']}")

    print("\nRecorte Pé-de-Meia (Estudantes do Mapa):")
    print(f"  - Elegível: {contagem_mapa['elegivel']}")
    print(f"  - Não elegível: {contagem_mapa['nao_elegivel']}")
    print(f"  - N/C (Nada consta): {contagem_mapa['nada_consta']}")
    if contagem_mapa["indefinida"] > 0:
        print(f"  - Elegibilidade indefinida: {contagem_mapa['indefinida']}")
    print(f"  - Total: {contagem_mapa['total']}")

    return 0


if __name__ == "__main__":
    sys.exit(_executar_cli())
