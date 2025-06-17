import pandas as pd
import unicodedata
import re

# --------------------------------
# Funções para remoção de acentos e identificação das colunas de identificação
# --------------------------------
def remover_acentos(txt):
    """Remove acentos de uma string."""
    if not isinstance(txt, str):
        txt = str(txt)
    return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')

# --------------------------------
# Extração e Processamento dos Dados
# --------------------------------
def extrair_dataframes(caminho_xls):
    """
    Lê o arquivo XLS, identifica o cabeçalho dinamicamente e monta os DataFrames de Notas e Faltas.
    Esta abordagem é robusta contra variações na formatação do cabeçalho e células mescladas.
    """
    try:
        # Lê a planilha inteira como texto para evitar erros de tipo
        df_full = pd.read_excel(caminho_xls, header=None, dtype=str, engine='xlrd')
    except Exception as e:
        raise ValueError(f"Não foi possível ler o arquivo Excel: {caminho_xls}. Erro: {e}")

    # Encontra a linha onde o cabeçalho começa procurando por "Matrícula"
    header_start_row = -1
    mat_col_idx = -1
    nome_col_idx = -1

    for i, row in df_full.iterrows():
        # Converte a linha para uma lista de strings minúsculas e sem acentos
        search_row = [remover_acentos(str(cell)).lower() for cell in row.tolist()]
        if 'matricula' in search_row:
            header_start_row = i
            mat_col_idx = search_row.index('matricula')
            # Procura por 'nome' na mesma linha
            if 'nome do aluno' in search_row:
                nome_col_idx = search_row.index('nome do aluno')
            elif 'nome' in search_row:
                 nome_col_idx = search_row.index('nome')
            break
    
    if header_start_row == -1 or mat_col_idx == -1 or nome_col_idx == -1:
        raise ValueError("Não foi possível localizar as colunas 'Matrícula' e 'Nome' no arquivo.")

    # Extrai as linhas do cabeçalho e os dados
    header_disciplinas = df_full.iloc[header_start_row].ffill() # Preenche células mescladas
    header_tipo_dado = df_full.iloc[header_start_row + 1]
    df_data = df_full.iloc[header_start_row + 2:].copy()
    
    # Extrai a base de identificação dos alunos
    df_identificacao = df_data[[mat_col_idx, nome_col_idx]].copy()
    df_identificacao.columns = ['matricula', 'nome']
    
    # Valida as matrículas (devem ter 11 dígitos e começar com '20')
    matriculas_limpas = df_identificacao['matricula'].str.replace(r'\D', '', regex=True)
    mask_validos = matriculas_limpas.str.match(r'^20\d{9}$', na=False)
    
    df_alunos_validos = df_identificacao[mask_validos].reset_index(drop=True)
    df_data_validos = df_data[mask_validos].reset_index(drop=True)

    # Prepara os DataFrames finais
    df_notas = df_alunos_validos.copy()
    df_faltas = df_alunos_validos.copy()

    # Itera pelas colunas para montar os dataframes de notas e faltas
    for i in range(len(header_disciplinas)):
        tipo_dado = str(header_tipo_dado.iloc[i]).strip().upper()
        disciplina = str(header_disciplinas.iloc[i]).strip()

        # Pula as colunas de identificação e colunas sem nome de disciplina
        if i in [mat_col_idx, nome_col_idx] or not disciplina:
            continue
            
        # Verifica se o nome da disciplina não é um número (evita usar matrículas como nome de coluna)
        if disciplina.isdigit():
            continue

        if 'N' in tipo_dado:
            df_notas[disciplina] = pd.to_numeric(df_data_validos.iloc[:, i], errors='coerce')
        elif 'F' in tipo_dado:
            df_faltas[disciplina] = pd.to_numeric(df_data_validos.iloc[:, i], errors='coerce')
            
    return df_notas, df_faltas

# --------------------------------
# Ajuste e Manipulação de Disciplinas
# --------------------------------
def ajustar_dataframe(df, disciplinas_dict):
    """
    Garante que o DataFrame contenha todas as colunas de disciplina esperadas e na ordem correta,
    descartando quaisquer colunas indesejadas que não estejam no dicionário de disciplinas.
    """
    colunas_finais = ['matricula', 'nome'] + list(disciplinas_dict.keys())
    
    for col in colunas_finais:
        if col not in df.columns:
            df[col] = pd.NA
            
    # Retorna um novo DataFrame contendo APENAS as colunas finais e na ordem correta.
    return df[colunas_finais]

# Dicionários de disciplinas definidos diretamente no código
disciplinas_ensino_medio = {
    '1DEFISD.006': 'EDUCAÇÃO FÍSICA - 2ª SÉRIE', '1LIN.003': 'LÍNGUA ESTRANGEIRA: INGLÊS - 2ª SÉRIE',
    '1MAT.006': 'MATEMÁTICA - 2ª SÉRIE', '1QUI.003': 'QUÍMICA - 2ª SÉRIE',
    '1TFIL2.1': 'FILOSOFIA - 2ª SÉRIE', '1TLP2.1': 'LÍNGUA PORTUGUESA - 2ª SÉRIE',
    '1TRED2.01': 'REDAÇÃO - 2ª SÉRIE', 'GEO.2': 'GEOGRAFIA - 2ª SÉRIE',
    'HIST.2': 'HISTÓRIA - 2ª SÉRIE', '1CIE.010': 'BIOLOGIA - 2ª SÉRIE',
    '1CIE.011': 'FÍSICA - 2ª SÉRIE', 'SOC.2': 'SOCIOLOGIA - 2ª SÉRIE'
}

disciplinas_tecnicas_transito = {
    '1TT.009': 'PLANEJAMENTO DE TRANSPORTES',
    '1TT.35': 'LABORATÓRIO DE DE PESQUISA DE TRANSPORTES E TRÂNSITO',
    '1TT.37': 'LABORATÓRIO DE TOPOGRAFIA URBANA',
    '1TT.62': 'LABORATÓRIO DE SEGURANÇA VIÁRIA',
    '4929': 'INTRODUÇÃO À ENGENHARIA DE TRÁFEGO'
}

disciplinas_tecnicas_estradas = {
    '1TT.43': 'LABORATÓRIO DE SOLOS',
    '1TT.44': 'LABORATÓRIO DE DESENHO TOPOGRÁFICO',
    '1TT.45': 'LABORATÓRIO DE TOPOGRAFIA',
    '5801': 'SOLOS',
    '5802': 'TOPOGRAFIA',
    '5811': 'MÁQUINAS E EQUIPAMENTOS'
}

def preencher_dados_faltantes(df_principal, df_fonte_para_preencher):
    """
    Preenche dados (notas ou faltas) de ensino médio no DF principal usando um DF fonte já processado.
    Esta versão é mais simples e direta para evitar erros de lógica anteriores.
    """
    # Seleciona apenas as colunas necessárias do DF fonte: a chave de merge e as colunas de dados a serem adicionadas.
    cols_para_adicionar = ['matricula'] + list(disciplinas_ensino_medio.keys())
    
    # Garante que estamos pegando apenas colunas que realmente existem no dataframe fonte para evitar erros.
    cols_existentes_no_fonte = [col for col in cols_para_adicionar if col in df_fonte_para_preencher.columns]
    df_fonte_subset = df_fonte_para_preencher[cols_existentes_no_fonte]

    # Realiza a fusão. Como as colunas de ensino médio não existem em df_principal, elas serão adicionadas diretamente.
    df_final = pd.merge(df_principal, df_fonte_subset, on='matricula', how='left')
    
    return df_final

# --------------------------------
# Geração dos DataFrames finais
# --------------------------------
def gerar_dataframes_cursos(arquivo_transito_xls, arquivo_est_xls):
    """
    Orquestra a geração dos DataFrames finais de notas e faltas para cada curso.
    """
    # Combina os dicionários de disciplinas
    disciplinas_transito_dict = {**disciplinas_tecnicas_transito, **disciplinas_ensino_medio}
    disciplinas_estradas_dict = {**disciplinas_tecnicas_estradas, **disciplinas_ensino_medio}

    # Extração primária dos dados
    df_notas_transito, df_faltas_transito = extrair_dataframes(arquivo_transito_xls)
    df_notas_est_completo, df_faltas_est_completo = extrair_dataframes(arquivo_est_xls)

    # Isola os alunos de Estradas (remove quem está em Trânsito)
    transito_matriculas = set(df_notas_transito['matricula'])
    df_notas_estradas = df_notas_est_completo[~df_notas_est_completo['matricula'].isin(transito_matriculas)].copy()
    df_faltas_estradas = df_faltas_est_completo[~df_faltas_est_completo['matricula'].isin(transito_matriculas)].copy()
    
    # Preenche dados faltantes do ensino médio para o curso de Trânsito
    df_notas_transito = preencher_dados_faltantes(df_notas_transito, df_notas_est_completo)
    df_faltas_transito = preencher_dados_faltantes(df_faltas_transito, df_faltas_est_completo)

    # Ajusta os DFs para terem todas as colunas esperadas e na ordem correta, removendo colunas indesejadas.
    df_notas_transito = ajustar_dataframe(df_notas_transito, disciplinas_transito_dict)
    df_faltas_transito = ajustar_dataframe(df_faltas_transito, disciplinas_transito_dict)
    df_notas_estradas = ajustar_dataframe(df_notas_estradas, disciplinas_estradas_dict)
    df_faltas_estradas = ajustar_dataframe(df_faltas_estradas, disciplinas_estradas_dict)

    return (df_notas_transito, df_faltas_transito), (df_notas_estradas, df_faltas_estradas)

# --------------------------------
# Execução do script principal
# --------------------------------
if __name__ == "__main__":
    arquivo_transito_xls = "/content/MapaTurma_2024.xls"
    arquivo_est_xls = "/content/MapaTurma_2024_EST.xls"
    output_path = "/content/"

    try:
        # Gera os dataframes
        (df_notas_transito, df_faltas_transito), (df_notas_estradas, df_faltas_estradas) = gerar_dataframes_cursos(arquivo_transito_xls, arquivo_est_xls)
        
        print("DataFrames gerados com sucesso. Salvando em arquivos CSV...")

        # Nomes dos arquivos de saída
        nome_arquivo_notas_transito = "notas_transito.csv"
        nome_arquivo_faltas_transito = "faltas_transito.csv"
        nome_arquivo_notas_estradas = "notas_estradas.csv"
        nome_arquivo_faltas_estradas = "faltas_estradas.csv"
        
        # Salva cada DataFrame em um arquivo CSV no diretório de saída
        df_notas_transito.to_csv(output_path + nome_arquivo_notas_transito, index=False)
        df_faltas_transito.to_csv(output_path + nome_arquivo_faltas_transito, index=False)
        df_notas_estradas.to_csv(output_path + nome_arquivo_notas_estradas, index=False)
        df_faltas_estradas.to_csv(output_path + nome_arquivo_faltas_estradas, index=False)
        
        print("\nArquivos salvos com sucesso em '{}':".format(output_path))
        print(f"- {nome_arquivo_notas_transito}")
        print(f"- {nome_arquivo_faltas_transito}")
        print(f"- {nome_arquivo_notas_estradas}")
        print(f"- {nome_arquivo_faltas_estradas}")
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado.")
        print(f"Verifique se o arquivo '{e.filename}' existe no diretório do Colab ('/content/').")
        print("Certifique-se também de que a biblioteca 'xlrd' está instalada (`pip install xlrd`).")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
