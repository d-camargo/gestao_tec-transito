import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import requests # Adicionado para chamadas de API
import json
import io # Adicionado para lidar com a imagem em memória
from google.colab import userdata # Adicionado para ler os segredos do Colab
from datetime import datetime
from babel.dates import format_date # Adicionado para formatar a data em português
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY # Adicionado para alinhamento
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Image, Spacer, Table, TableStyle, Frame, PageTemplate, BaseDocTemplate, PageBreak
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader # Adicionado para ler a imagem da URL

# --------------------------------
# 1. Leitura dos Dados
# --------------------------------
def ler_dados(caminho_notas_csv, caminho_faltas_csv):
    """Lê os arquivos CSV de notas e faltas gerados pelo primeiro script."""
    try:
        df_notas = pd.read_csv(caminho_notas_csv)
        df_faltas = pd.read_csv(caminho_faltas_csv)
        return df_notas, df_faltas
    except FileNotFoundError:
        print(f"AVISO: Arquivo não encontrado em '{caminho_notas_csv}' ou '{caminho_faltas_csv}'. Pulando este curso.")
        return None, None

# --------------------------------
# 2. Cálculos Estatísticos
# --------------------------------
def get_simplified_name(code, disciplinas_dict):
    """Extrai o nome simplificado de uma disciplina a partir do seu código."""
    full_name = disciplinas_dict.get(str(code), str(code))
    simplified = re.split(r' - |:', full_name)[0].strip().title()
    return simplified

def calcular_estatisticas(df_notas, disciplinas_dict, nome_curso):
    """Calcula as estatísticas básicas para um curso."""
    estatisticas = {}

    disciplinas_presentes = [col for col in disciplinas_dict.keys() if col in df_notas.columns]
    disciplinas_com_notas = [col for col in disciplinas_presentes if df_notas[col].notna().any()]

    df_apenas_notas = df_notas[disciplinas_com_notas]

    media_por_aluno = df_apenas_notas.mean(axis=1)

    estatisticas['total_alunos'] = len(df_notas)
    estatisticas['media_geral_turma'] = media_por_aluno.mean()
    estatisticas['desvio_padrao_medias'] = media_por_aluno.std()

    taxa_aprovacao = (df_apenas_notas >= 12.0).all(axis=1).mean() * 100
    estatisticas['taxa_aprovacao_geral'] = f"{taxa_aprovacao:.2f}%"

    media_por_disciplina = df_apenas_notas.mean().sort_values()
    if not media_por_disciplina.empty:
        disciplina_menor_code = media_por_disciplina.index[0]
        estatisticas['disciplina_menor_media_code'] = disciplina_menor_code
        estatisticas['disciplina_menor_media_nome'] = get_simplified_name(disciplina_menor_code, disciplinas_dict)
        estatisticas['menor_media'] = media_por_disciplina.iloc[0]
        estatisticas['desvio_padrao_disciplina_critica'] = df_apenas_notas[disciplina_menor_code].std()

        alunos_abaixo_12 = (df_apenas_notas[disciplina_menor_code] < 12.0).sum()
        estatisticas['alunos_abaixo_12_disciplina_critica'] = alunos_abaixo_12

        disciplina_maior_code = media_por_disciplina.index[-1]
        estatisticas['disciplina_maior_media_nome'] = get_simplified_name(disciplina_maior_code, disciplinas_dict)
        estatisticas['maior_media'] = media_por_disciplina.iloc[-1]
    else:
        for key in ['disciplina_menor_media_nome', 'disciplina_maior_media_nome', 'disciplina_menor_media_code']:
            estatisticas[key] = "N/A"
        for key in ['menor_media', 'maior_media', 'desvio_padrao_disciplina_critica', 'alunos_abaixo_12_disciplina_critica']:
            estatisticas[key] = 0

    df_notas['disciplinas_abaixo_12'] = (df_apenas_notas < 12.0).sum(axis=1)
    top_10_criticos = df_notas.sort_values(by='disciplinas_abaixo_12', ascending=False).head(10)
    estatisticas['top_10_alunos_criticos'] = top_10_criticos[['nome', 'disciplinas_abaixo_12']]

    # ATUALIZAÇÃO: Calcula o resumo estatístico para a tabela do boxplot
    summary_list = []
    for col in disciplinas_com_notas:
        stats = df_apenas_notas[col].describe()
        s_name = get_simplified_name(col, disciplinas_dict)
        summary_list.append({
            'Disciplina': s_name,
            'Média': f"{stats.get('mean', 0):.2f}",
            'Mediana': f"{stats.get('50%', 0):.2f}",
            'Desv. Padrão': f"{stats.get('std', 0):.2f}",
            'Mínimo': f"{stats.get('min', 0):.2f}",
            'Máximo': f"{stats.get('max', 0):.2f}"
        })
    estatisticas['boxplot_summary_df'] = pd.DataFrame(summary_list)


    return estatisticas

# --------------------------------
# 3. Geração de Gráficos e Análise de IA
# --------------------------------
def gerar_comentario_ia(estatisticas, nome_curso):
    """Gera um comentário analítico usando a API da OpenAI."""
    # Formata o resumo do boxplot para incluir no prompt
    summary_markdown = estatisticas['boxplot_summary_df'].to_markdown(index=False)

    prompt = f"""
    Você é um especialista em análise de dados educacionais. Com base nos dados a seguir, gere uma análise em português. Não faça em formato MarkDown, ou seja, não use * ou #.

    **Contexto:**
    - **Curso:** {nome_curso}
    - **Total de Alunos:** {estatisticas['total_alunos']}

    **Análise Geral da Turma:**
    - **Média Geral (0-20):** {estatisticas['media_geral_turma']:.2f}
    - **Dispersão das Médias (Desvio Padrão):** {estatisticas['desvio_padrao_medias']:.2f} (Valores mais altos indicam maior variação de desempenho entre os alunos).
    - **Taxa de Aprovação Geral (Nota >= 12 em tudo):** {estatisticas['taxa_aprovacao_geral']}

    **Resumo Estatístico por Disciplina:**
    {summary_markdown}

    **Instruções:**
    1.  Primeiro Parágrafo: Comente sobre o desempenho geral da turma. A média é satisfatória? A turma é homogênea (baixo desvio padrão geral) ou heterogênea? A taxa de aprovação geral é preocupante?
    2.  Segundo Parágrafo: Analise as disciplinas. Destaque a disciplina com menor média ({estatisticas['disciplina_menor_media_nome']}) e comente sobre o número de alunos com nota baixa nela ({estatisticas['alunos_abaixo_12_disciplina_critica']}).
    3.  Terceiro Parágrafo: Com base na tabela de resumo estatístico, identifique e comente sobre as disciplinas que apresentam um desempenho ruim dos alunos. Faça uma comparação entre os indicadores e aponte possibilidades para melhora-los.

    O tom deve ser profissional e objetivo.
    """

    try:
        # ATUALIZAÇÃO: Lê a chave de API de forma segura a partir dos segredos do Colab
        api_key = userdata.get('OPEN_IA')
    except Exception as e:
        print("AVISO: Não foi possível ler o segredo 'OPENAI_API_KEY'. Certifique-se de que ele foi configurado no Colab.")
        return "A análise por IA não pôde ser gerada pois a chave de API não foi encontrada."

    api_url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()

        if result.get('choices') and result['choices'][0].get('message', {}).get('content'):
            commentary = result['choices'][0]['message']['content']
            return commentary.replace('\n', '<br/>')
        else:
            return "Análise da IA indisponível (resposta inesperada da API)."

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
             print("Erro 401: Não autorizado. Verifique se sua chave de API da OpenAI é válida.")
             return "A análise por IA não pôde ser gerada. A chave de API da OpenAI é inválida ou expirou."
        elif e.response.status_code == 403:
            print("Erro 403: Acesso proibido. Verifique as permissões da sua chave de API.")
            return "A análise por IA não pôde ser gerada devido a um problema de permissão."
        else:
            print(f"Erro HTTP ao chamar a API da IA: {e}")
            return "Erro na comunicação com a IA para gerar o comentário."

    except requests.exceptions.RequestException as e:
        print(f"Erro de rede ao chamar a API da IA: {e}")
        return "Erro de comunicação com a IA para gerar o comentário."
    except Exception as e:
        print(f"Erro inesperado ao processar a resposta da IA: {e}")
        return "Erro ao processar o comentário da IA."

def gerar_grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict, output_dir):
    """Gera um histograma da distribuição das médias dos alunos."""
    disciplinas_presentes = [col for col in disciplinas_dict.keys() if col in df_notas.columns and df_notas[col].notna().any()]
    if not disciplinas_presentes:
        return None

    df_notas['media_aluno'] = df_notas[disciplinas_presentes].mean(axis=1)

    plt.figure(figsize=(10, 6))
    sns.histplot(df_notas['media_aluno'], kde=True, bins=15)
    plt.title(f'Distribuição das Médias Finais - {nome_curso}', fontsize=16)
    plt.xlabel('Média Final do Aluno (0 a 20)', fontsize=12)
    plt.ylabel('Número de Alunos', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    caminho_grafico = os.path.join(output_dir, f"distribuicao_notas_{nome_curso.lower()}.png")
    plt.savefig(caminho_grafico)
    plt.close()

    return caminho_grafico

def gerar_grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict, output_dir):
    """Gera um gráfico de barras com a média por disciplina, usando nomes simplificados."""
    disciplinas_presentes = [col for col in disciplinas_dict.keys() if col in df_notas.columns and df_notas[col].notna().any()]
    if not disciplinas_presentes:
        return None

    media_por_disciplina = df_notas[disciplinas_presentes].mean().sort_values(ascending=False)
    simplified_labels = [get_simplified_name(code, disciplinas_dict) for code in media_por_disciplina.index]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=media_por_disciplina.values, y=simplified_labels)
    plt.title(f'Média por Disciplina - {nome_curso}', fontsize=16)
    plt.xlabel('Média da Turma (0 a 20)', fontsize=12)
    plt.ylabel('Disciplina', fontsize=12)
    plt.xlim(0, 20)

    caminho_grafico = os.path.join(output_dir, f"media_disciplina_{nome_curso.lower()}.png")
    plt.tight_layout()
    plt.savefig(caminho_grafico)
    plt.close()

    return caminho_grafico

def gerar_grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict, output_dir):
    """Gera um boxplot para visualizar a dispersão das notas em cada disciplina."""
    disciplinas_presentes = [col for col in disciplinas_dict.keys() if col in df_notas.columns and df_notas[col].notna().any()]
    if not disciplinas_presentes:
        return None

    df_melted = df_notas.melt(value_vars=disciplinas_presentes, var_name='disciplina_code', value_name='nota')
    df_melted['disciplina_nome'] = df_melted['disciplina_code'].apply(lambda code: get_simplified_name(code, disciplinas_dict))

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='nota', y='disciplina_nome', data=df_melted, orient='h')
    plt.title(f'Dispersão de Notas por Disciplina - {nome_curso}', fontsize=16)
    plt.xlabel('Nota (0 a 20)', fontsize=12)
    plt.ylabel('Disciplina', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')

    caminho_grafico = os.path.join(output_dir, f"boxplot_disciplinas_{nome_curso.lower()}.png")
    plt.tight_layout()
    plt.savefig(caminho_grafico)
    plt.close()

    return caminho_grafico


def gerar_grafico_disciplina_critica(df_notas, disciplina_code, disciplina_nome, nome_curso, output_dir):
    """Gera um histograma de dispersão para a disciplina com menor média."""
    if disciplina_code == "N/A" or disciplina_code not in df_notas.columns:
        return None

    plt.figure(figsize=(10, 6))
    sns.histplot(df_notas[disciplina_code].dropna(), kde=True, bins=10, color='indianred')
    plt.title(f'Dispersão de Notas: {disciplina_nome} ({nome_curso})', fontsize=16)
    plt.xlabel('Nota na Disciplina (0 a 20)', fontsize=12)
    plt.ylabel('Número de Alunos', fontsize=12)
    plt.xlim(0, 20)
    plt.grid(True, linestyle='--', alpha=0.6)

    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', disciplina_nome)
    caminho_grafico = os.path.join(output_dir, f"dispersao_{safe_filename}_{nome_curso.lower()}.png")
    plt.savefig(caminho_grafico)
    plt.close()

    return caminho_grafico


# --------------------------------
# 4. Geração do Relatório em PDF
# --------------------------------
def adicionar_cabecalho(canvas, doc):
    """Adiciona os logos e o texto do cabeçalho no topo de cada página."""
    canvas.saveState()
    largura, altura = doc.pagesize

    # --- Logo Principal (Brasão) ---
    caminho_logo_principal = '/content/drive/MyDrive/Relatorios_CEFETMG/selo_115_anos_CEFET_RGB.png' # Usando o novo selo como principal
    if os.path.exists(caminho_logo_principal):
        try:
            logo = ImageReader(caminho_logo_principal)
            logo_width, logo_height = logo.getSize()
            aspect = logo_height / float(logo_width)
            display_height = 2.0 * cm
            display_width = display_height / aspect
            # Posiciona o logo à esquerda
            logo_x = 1.5 * cm
            logo_y = altura - 3 * cm
            canvas.drawImage(logo, logo_x, logo_y, width=display_width, height=display_height, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print(f"Não foi possível carregar o logo principal: {e}")
    else:
        print(f"AVISO: Imagem do selo não encontrada em '{caminho_logo_principal}'.")

    # --- Texto do Cabeçalho Centralizado ---
    canvas.setFont('Times-Bold', 10)
    canvas.setFillColor(colors.HexColor('#002060'))

    y_start = altura - 2 * cm
    canvas.drawCentredString(largura / 2.0, y_start, "Serviço Público Federal")
    canvas.drawCentredString(largura / 2.0, y_start - 0.5*cm, "Ministério da Educação")
    canvas.drawCentredString(largura / 2.0, y_start - 1.0*cm, "Centro Federal de Educação Tecnológica de Minas Gerais")

    # --- Rodapé ---
    canvas.setFont('Times-Italic', 8)
    canvas.setFillColor(colors.grey)
    texto_rodape = "Desenvolvido pelo Professor Diego Camargo (diegocamargo@cefetmg.br)."
    canvas.drawCentredString(largura / 2.0, 1.5 * cm, texto_rodape)


    canvas.restoreState()

def criar_relatorio_pdf(nome_curso, estatisticas, caminhos_graficos, output_pdf):
    """Cria um relatório em PDF com as estatísticas e gráficos, suportando múltiplas páginas."""
    doc = BaseDocTemplate(output_pdf, pagesize=A4)

    largura, altura = A4
    frame = Frame(2.5*cm, 2.5*cm, largura - 5*cm, altura - 6*cm, id='normal')
    template = PageTemplate(id='principal', frames=[frame], onPage=adicionar_cabecalho)
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', parent=styles['BodyText'], alignment=TA_JUSTIFY, fontName='Times-Roman'))
    style_titulo_capa = ParagraphStyle(name='Title', fontSize=22, alignment=TA_CENTER, leading=26, spaceAfter=1.5*cm, textColor=colors.HexColor('#002060'), fontName='Times-Bold')
    style_subtitulo_capa = ParagraphStyle(name='SubTitle', fontSize=18, alignment=TA_CENTER, spaceAfter=2*cm, textColor=colors.HexColor('#002060'), fontName='Times-Roman')
    style_data_capa = ParagraphStyle(name='Date', fontSize=12, alignment=TA_CENTER, spaceBefore=13*cm, fontName='Times-Roman')
    style_h2 = ParagraphStyle(name='h2', parent=styles['h2'], fontName='Times-Bold')
    style_corpo = styles['Justify']

    story = []

    # --- CAPA ---
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph("RELATÓRIO DE ACOMPANHAMENTO ACADÊMICO", style_titulo_capa))
    story.append(Paragraph(f"Curso Técnico em {nome_curso.title()}", style_subtitulo_capa))
    data_pt = format_date(datetime.now(), format="d 'de' MMMM 'de' y", locale='pt_BR')
    story.append(Paragraph(data_pt, style_data_capa))
    story.append(PageBreak())

    # --- CONTEÚDO ---
    story.append(Paragraph("Estatísticas Gerais da Turma", style_h2))
    story.append(Spacer(1, 0.5*cm))

    dados_tabela_geral = [
        ['Total de Alunos:', estatisticas['total_alunos']],
        ['Média Geral da Turma:', f"{estatisticas['media_geral_turma']:.2f}"],
        ['Desvio Padrão das Médias:', f"{estatisticas['desvio_padrao_medias']:.2f}"],
        ['Taxa de Aprovação (>= 12.0 em tudo):', estatisticas['taxa_aprovacao_geral']],
        ['Disciplina com Maior Média:', f"{estatisticas['disciplina_maior_media_nome']} ({estatisticas['maior_media']:.2f})"],
        ['Disciplina com Menor Média:', f"{estatisticas['disciplina_menor_media_nome']} ({estatisticas['menor_media']:.2f})"]
    ]
    tabela_geral = Table(dados_tabela_geral, colWidths=[7*cm, 9*cm])
    tabela_geral.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman')
    ]))
    story.append(tabela_geral)
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Alunos com Maior Número de Disciplinas Abaixo da Média", style_h2))
    story.append(Spacer(1, 0.5*cm))
    dados_alunos_criticos = [['Aluno', 'Disciplinas < 12']] + estatisticas['top_10_alunos_criticos'].values.tolist()
    tabela_alunos_criticos = Table(dados_alunos_criticos, colWidths=[12*cm, 4*cm])
    tabela_alunos_criticos.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkred),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Times-Roman')
    ]))
    story.append(tabela_alunos_criticos)
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Visualizações Gráficas Gerais", style_h2))
    story.append(Spacer(1, 0.5*cm))

    graficos_gerais = [
        caminhos_graficos.get('distribuicao_geral'),
        caminhos_graficos.get('media_disciplina'),
        caminhos_graficos.get('boxplot_disciplinas')
    ]

    for caminho in graficos_gerais:
        if caminho and os.path.exists(caminho):
            img = Image(caminho, width=16*cm, height=11*cm, kind='proportional')
            story.append(img)
            story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Resumo Estatístico por Disciplina", style_h2))
    story.append(Spacer(1, 0.5*cm))
    df_summary = estatisticas['boxplot_summary_df']
    table_data = [df_summary.columns.tolist()] + df_summary.values.tolist()
    tabela_boxplot_summary = Table(table_data, colWidths=[4.5*cm, 2*cm, 2*cm, 2.5*cm, 2*cm, 2*cm])
    tabela_boxplot_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.cadetblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(tabela_boxplot_summary)
    story.append(Spacer(1, 1*cm))

    if caminhos_graficos['disciplina_critica'] and os.path.exists(caminhos_graficos['disciplina_critica']):
        story.append(PageBreak())
        story.append(Paragraph("Análise da Disciplina com Menor Desempenho", style_h2))
        story.append(Spacer(1, 0.5*cm))

        dados_tabela_critica = [
            ["Disciplina:", estatisticas['disciplina_menor_media_nome']],
            ["Média da Turma:", f"{estatisticas['menor_media']:.2f}"],
            ["Desvio Padrão:", f"{estatisticas['desvio_padrao_disciplina_critica']:.2f}"],
            ["Alunos com Nota < 12.0:", f"{estatisticas['alunos_abaixo_12_disciplina_critica']}"]
        ]
        tabela_critica = Table(dados_tabela_critica, colWidths=[7*cm, 9*cm])
        tabela_critica.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.lightpink),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,-1), 'Times-Roman')
        ]))
        story.append(tabela_critica)
        story.append(Spacer(1, 0.5*cm))

        img_critica = Image(caminhos_graficos['disciplina_critica'], width=16*cm, height=11*cm, kind='proportional')
        story.append(img_critica)

    if estatisticas.get('comentario_ia'):
        story.append(PageBreak())
        story.append(Paragraph("Análise e Comentários (Gerado por Inteligência Artificial)", style_h2))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(estatisticas['comentario_ia'], style_corpo))

    doc.build(story)
    print(f"Relatório em PDF gerado com sucesso: '{output_pdf}'")

# --------------------------------
# Execução do Script Principal
# --------------------------------
if __name__ == "__main__":
    input_path = "/content/"
    output_path = "/content/"

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
        '1TT.35': 'LABORATÓRIO DE PESQUISA DE TRANSPORTES E TRÂNSITO',
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

    cursos = {
        "Transito": {
            "notas_csv": os.path.join(input_path, "notas_transito.csv"),
            "faltas_csv": os.path.join(input_path, "faltas_transito.csv"),
            "disciplinas_dict": {**disciplinas_tecnicas_transito, **disciplinas_ensino_medio}
        },
        "Estradas": {
            "notas_csv": os.path.join(input_path, "notas_estradas.csv"),
            "faltas_csv": os.path.join(input_path, "faltas_estradas.csv"),
            "disciplinas_dict": {**disciplinas_tecnicas_estradas, **disciplinas_ensino_medio}
        }
    }

    for nome_curso, info in cursos.items():
        print(f"\n--- Processando curso: {nome_curso} ---")
        df_notas, df_faltas = ler_dados(info["notas_csv"], info["faltas_csv"])

        if df_notas is not None and not df_notas.empty:
            disciplinas_dict = info["disciplinas_dict"]

            estatisticas = calcular_estatisticas(df_notas, disciplinas_dict, nome_curso)

            estatisticas['comentario_ia'] = gerar_comentario_ia(estatisticas, nome_curso)

            caminhos_graficos = {}
            caminhos_graficos['distribuicao_geral'] = gerar_grafico_distribuicao_notas(df_notas, nome_curso, disciplinas_dict, output_path)
            caminhos_graficos['media_disciplina'] = gerar_grafico_media_por_disciplina(df_notas, nome_curso, disciplinas_dict, output_path)
            caminhos_graficos['boxplot_disciplinas'] = gerar_grafico_boxplot_disciplinas(df_notas, nome_curso, disciplinas_dict, output_path)
            caminhos_graficos['disciplina_critica'] = gerar_grafico_disciplina_critica(
                df_notas,
                estatisticas['disciplina_menor_media_code'],
                estatisticas['disciplina_menor_media_nome'],
                nome_curso,
                output_path
            )

            output_pdf_path = os.path.join(output_path, f"relatorio_{nome_curso.lower()}.pdf")
            criar_relatorio_pdf(nome_curso, estatisticas, caminhos_graficos, output_pdf_path)
