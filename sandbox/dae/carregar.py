"""Carregamento e normalização das planilhas de acompanhamento discente da DAE.

Este módulo implementa a leitura isolada da planilha da Diretoria de Assuntos
Estudantis (DAE), aceitando arquivos nos formatos .xlsx (apenas a aba '2026 - geral')
ou .csv (assumindo o mesmo layout da aba '2026 - geral').

Pressuposto N/C (Decisão D7):
    Assume-se que 'N/C' na coluna 'Pé de meia' significa "Nada consta" (não se aplica
    ou sem registro). Esse pressuposto foi adotado na análise dos dados, mas não foi
    confirmado formalmente pela DAE. No processamento, 'N/C' é normalizado para
    'nada_consta' e NÃO marca o aluno como participante do programa Pé-de-Meia (PdM).

Minimização de dados (Decisão D4):
    - O CPF do estudante é descartado imediatamente na entrada e não compõe o DataFrame.
    - Em arquivos .xlsx, apenas a aba '2026 - geral' é aberta; abas com dados pessoais
      extras (como 'Emails' ou 'Acompanhamento PdM') nunca são lidas.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


def remover_acentos(txt: str) -> str:
    """Remove caracteres diacríticos (acentos, cedilha, etc.) de uma string."""
    if not isinstance(txt, str):
        txt = str(txt)
    return "".join(
        c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn"
    )


def _normalizar_rotulo(txt: object) -> str:
    """Normaliza texto para comparação (sem acento, minúsculas, sem espaços extras)."""
    if pd.isna(txt):
        return ""
    s = remover_acentos(str(txt)).lower().strip()
    return re.sub(r"\s+", " ", s)


def _normalizar_matricula(val: object) -> str:
    """Normaliza a matrícula do estudante mantendo apenas dígitos.

    Lida corretamente com números inteiros, números de ponto flutuante
    (ex.: 20261234567.0 vindo de Excel) e strings formatadas.
    """
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return ""
        return str(int(val))
    s = str(val).strip()
    # Remove terminação .0 resultante de coerção numérica prévia
    s = re.sub(r"\.0$", "", s)
    return re.sub(r"\D", "", s)


def _normalizar_pe_de_meia(val: object) -> str:
    """Normaliza a situação do Pé-de-Meia conforme Decisão D7.

    Domínio esperado:
        - 'elegivel': Elegível
        - 'nao_elegivel': Não elegível
        - 'nada_consta': N/C (pressuposto "Nada consta", sem vínculo PdM)
        - 'indefinida': Elegibilidade indefinida ou qualquer outro valor desconhecido/nulo.
    """
    if pd.isna(val):
        return "indefinida"
    s_norm = _normalizar_rotulo(val)
    if s_norm == "elegivel":
        return "elegivel"
    if s_norm in ("nao elegivel", "nao-elegivel"):
        return "nao_elegivel"
    if s_norm in ("n/c", "nc", "n / c", "nada consta", "nada_consta"):
        return "nada_consta"
    if s_norm in ("elegibilidade indefinida", "indefinida"):
        return "indefinida"
    return "indefinida"


def _normalizar_bolsa(val: object) -> bool:
    """Normaliza campos de bolsa ('Sim'/'Não') para booleano."""
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    s = _normalizar_rotulo(val)
    return s in ("sim", "s", "true", "1")


def _derivar_programas(pe_de_meia: str, bolsa_ba_bp: bool, bolsa_bce: bool) -> list[str]:
    """Gera lista de siglas dos programas do estudante em ordem fixa: PdM, BA/BP, BCE.

    'PdM' só é incluído se pe_de_meia == 'elegivel'. Valores 'nada_consta',
    'nao_elegivel' ou 'indefinida' NÃO marcam PdM (Decisão D7).
    """
    progs: list[str] = []
    if pe_de_meia == "elegivel":
        progs.append("PdM")
    if bolsa_ba_bp:
        progs.append("BA/BP")
    if bolsa_bce:
        progs.append("BCE")
    return progs


def _converter_numerico(val: object) -> float:
    """Converte valor para float numérico, convertendo vazios para np.nan."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if not s or s == "-":
        return np.nan
    if s.endswith("%"):
        s_num = s[:-1].strip().replace(",", ".")
        try:
            return float(s_num) / 100.0
        except ValueError:
            return np.nan
    s_clean = s.replace(",", ".")
    try:
        return float(s_clean)
    except ValueError:
        return np.nan


def _normalizar_nome_mes(txt: object) -> str:
    """Normaliza o nome do mês para minúsculas sem acento (ex.: 'Março' -> 'marco')."""
    s = _normalizar_rotulo(txt)
    # Remove eventual indicação de ano (ex.: 'fevereiro/2026' -> 'fevereiro')
    s = re.sub(r"[/_ -]?202\d", "", s).strip()
    return s


def carregar_dae(caminho: str | Path) -> pd.DataFrame:
    """Carrega e normaliza os dados da planilha de frequência da DAE.

    Aceita arquivos .xlsx (abrindo exclusivamente a aba '2026 - geral') ou .csv
    (assumindo o mesmo layout da aba '2026 - geral'). Ambos os formatos devem
    possuir os meses na 1ª linha (com forward-fill para rotular cada trio de
    colunas) e o cabeçalho das colunas na 2ª linha.

    Pressuposto N/C (Decisão D7):
        Assume-se que 'N/C' na coluna 'Pé de meia' significa "Nada consta"
        (não se aplica / sem registro). Esse pressuposto foi adotado na análise
        e normalizado para 'nada_consta', mas não foi confirmado formalmente
        pela DAE. Alunos com 'nada_consta' não são marcados no programa Pé-de-Meia.

    Colunas de saída:
        - matricula: somente dígitos (str)
        - nome: nome do estudante (str)
        - unidade: unidade / campus (str)
        - curso: curso do estudante (str)
        - pe_de_meia: normalizado para 'elegivel', 'nao_elegivel', 'nada_consta' ou 'indefinida'
        - bolsa_ba_bp: bool indicando recebimento de bolsa BA/BP
        - bolsa_bce: bool indicando recebimento de bolsa BCE
        - programas: lista de siglas dos programas do aluno (['PdM', 'BA/BP', 'BCE'])
        - ha_ofertadas_<mes>: total de horas-aula ofertadas no mês (float ou NaN se vazio)
        - ha_presenciadas_<mes>: total de horas-aula presenciadas no mês (float ou NaN se vazio)
        - acumulado_dae: percentual acumulado registrado pela DAE (float ou NaN se vazio)

    Minimização de dados (Decisão D4):
        - A coluna CPF é descartada na entrada.
        - Em arquivos .xlsx, apenas a aba '2026 - geral' é lida.

    Parâmetros:
        caminho: Caminho para o arquivo .xlsx ou .csv.

    Retorna:
        pd.DataFrame com os dados normalizados.

    Levanta:
        ValueError: Se a extensão não for .xlsx ou .csv, se a aba '2026 - geral'
                    não for encontrada no Excel, ou se colunas obrigatórias estiverem ausentes.
    """
    caminho = Path(caminho)
    ext = caminho.suffix.lower()

    if ext not in (".xlsx", ".csv"):
        raise ValueError(
            f"Extensão de arquivo não suportada: '{caminho.suffix}'. "
            "O arquivo tem de ser .xlsx ou .csv."
        )

    # 1. Leitura bruta sem cabeçalho (D4: abre apenas '2026 - geral')
    if ext == ".xlsx":
        try:
            df_raw = pd.read_excel(
                caminho,
                sheet_name="2026 - geral",
                header=None,
                engine="openpyxl",
            )
        except Exception as e:
            raise ValueError(
                f"Não foi possível abrir a aba '2026 - geral' no arquivo Excel. "
                f"O arquivo tem de ser a aba '2026 - geral'. Detalhe: {e}"
            ) from e
    else:
        try:
            with open(caminho, mode="r", encoding="utf-8-sig", errors="replace") as f:
                sample = f.read(4096)
                f.seek(0)
                delimiter = (
                    ";"
                    if (";" in sample and sample.count(";") > sample.count(","))
                    else ","
                )
                reader = csv.reader(f, delimiter=delimiter)
                raw_rows = list(reader)
            if not raw_rows:
                raise ValueError(
                    "O arquivo CSV está vazio. O arquivo tem de ser a aba '2026 - geral'."
                )
            max_cols = max((len(r) for r in raw_rows), default=0)
            padded = [r + [""] * (max_cols - len(r)) for r in raw_rows]
            df_raw = pd.DataFrame(padded)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(
                f"Erro ao ler arquivo CSV: {e}. O arquivo tem de ser a aba '2026 - geral'."
            ) from e

    if len(df_raw) < 2:
        raise ValueError(
            "O arquivo não possui linhas de cabeçalho suficientes. "
            "O arquivo tem de ser a aba '2026 - geral'."
        )

    row_meses = df_raw.iloc[0].copy()
    row_cols = df_raw.iloc[1].copy()

    # Forward-fill nos meses da linha 1 para rotular cada trio
    meses_series = row_meses.replace(r"^\s*$", np.nan, regex=True).ffill()

    idx_map: dict[str, int] = {}
    colunas_meses: list[tuple[int, str]] = []

    for i in range(df_raw.shape[1]):
        c_text = row_cols.iloc[i]
        c_norm = _normalizar_rotulo(c_text)
        m_val = meses_series.iloc[i]
        m_norm = _normalizar_nome_mes(m_val)

        if "matricula" in c_norm:
            idx_map["matricula"] = i
        elif any(k in c_norm for k in ("nome do estudante", "nome do aluno", "nome")):
            if "nome" not in idx_map:
                idx_map["nome"] = i
        elif "unidade" in c_norm or "campus" in c_norm:
            idx_map["unidade"] = i
        elif "curso" in c_norm:
            idx_map["curso"] = i
        elif "pe de meia" in c_norm or "pe-de-meia" in c_norm:
            idx_map["pe_de_meia"] = i
        elif "ba/bp" in c_norm or "ba_bp" in c_norm or "ba-bp" in c_norm:
            idx_map["bolsa_ba_bp"] = i
        elif "bce" in c_norm:
            idx_map["bolsa_bce"] = i
        elif "acumulado" in c_norm:
            idx_map["acumulado_dae"] = i
        elif "cpf" in c_norm:
            # CPF descartado na entrada (Decisão D4)
            pass
        elif "ha ofertada" in c_norm and m_norm:
            colunas_meses.append((i, f"ha_ofertadas_{m_norm}"))
        elif "ha presenciada" in c_norm and m_norm:
            colunas_meses.append((i, f"ha_presenciadas_{m_norm}"))

    colunas_obrigatorias = [
        "matricula",
        "nome",
        "unidade",
        "curso",
        "pe_de_meia",
        "bolsa_ba_bp",
        "bolsa_bce",
        "acumulado_dae",
    ]
    faltando = [c for c in colunas_obrigatorias if c not in idx_map]
    if faltando:
        raise ValueError(
            f"Coluna obrigatória ausente ({', '.join(faltando)}). "
            "O arquivo tem de ser a aba '2026 - geral'."
        )

    df_data = df_raw.iloc[2:].copy()
    matricula_limpa = df_data[idx_map["matricula"]].apply(_normalizar_matricula)

    # Filtrar linhas vazias ao final da planilha
    mask_valido = matricula_limpa != ""
    if not mask_valido.any():
        # Retorna DataFrame vazio estruturado
        cols_vazias = [
            "matricula",
            "nome",
            "unidade",
            "curso",
            "pe_de_meia",
            "bolsa_ba_bp",
            "bolsa_bce",
            "programas",
        ] + [c_name for _, c_name in colunas_meses] + ["acumulado_dae"]
        return pd.DataFrame(columns=cols_vazias)

    matricula_s = matricula_limpa[mask_valido].reset_index(drop=True)
    nome_s = (
        df_data.loc[mask_valido, idx_map["nome"]]
        .astype(str)
        .str.strip()
        .reset_index(drop=True)
    )
    unidade_s = (
        df_data.loc[mask_valido, idx_map["unidade"]]
        .astype(str)
        .str.strip()
        .reset_index(drop=True)
    )
    curso_s = (
        df_data.loc[mask_valido, idx_map["curso"]]
        .astype(str)
        .str.strip()
        .reset_index(drop=True)
    )
    pdm_s = (
        df_data.loc[mask_valido, idx_map["pe_de_meia"]]
        .apply(_normalizar_pe_de_meia)
        .reset_index(drop=True)
    )
    ba_bp_s = (
        df_data.loc[mask_valido, idx_map["bolsa_ba_bp"]]
        .apply(_normalizar_bolsa)
        .astype(bool)
        .reset_index(drop=True)
    )
    bce_s = (
        df_data.loc[mask_valido, idx_map["bolsa_bce"]]
        .apply(_normalizar_bolsa)
        .astype(bool)
        .reset_index(drop=True)
    )

    programas_list = [
        _derivar_programas(p, b1, b2)
        for p, b1, b2 in zip(pdm_s, ba_bp_s, bce_s)
    ]
    programas_s = pd.Series(programas_list, dtype=object)

    resultado: dict[str, object] = {
        "matricula": matricula_s,
        "nome": nome_s,
        "unidade": unidade_s,
        "curso": curso_s,
        "pe_de_meia": pdm_s,
        "bolsa_ba_bp": ba_bp_s,
        "bolsa_bce": bce_s,
        "programas": programas_s,
    }

    # Adicionar trios mensais (apenas ha_ofertadas_<mes> e ha_presenciadas_<mes>)
    for col_idx, col_name in colunas_meses:
        resultado[col_name] = (
            df_data.loc[mask_valido, col_idx]
            .apply(_converter_numerico)
            .astype("float64")
            .reset_index(drop=True)
        )

    # Adicionar acumulado_dae
    resultado["acumulado_dae"] = (
        df_data.loc[mask_valido, idx_map["acumulado_dae"]]
        .apply(_converter_numerico)
        .astype("float64")
        .reset_index(drop=True)
    )

    return pd.DataFrame(resultado)
