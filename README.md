# 🎓 Gestão Acadêmica EPTNM — CEFET-MG

Aplicação web (Streamlit) para os coordenadores dos cursos técnicos de nível
médio (**EPTNM**) do CEFET-MG. O coordenador informa seu **e-mail institucional**,
envia o *Mapa de Turma* (`.xls`) e recebe o **relatório de acompanhamento
acadêmico (PDF) por e-mail**.

Evolução dos scripts originais que rodavam no Google Colab (no histórico do
repositório), agora como uma página acessível pelo navegador e aberta a qualquer
coordenador do EPTNM — não só Trânsito/Estradas.

## ✨ Como funciona

1. O coordenador informa o **e-mail `@cefetmg.br`** (o serviço só envia para esse
   domínio — é assim que se garante o uso restrito a professores do CEFET).
2. Informa o **nome do curso** e envia o **Mapa de Turma** (`.xls`).
3. O app processa notas e faltas, calcula estatísticas, gera gráficos e monta o
   relatório em PDF.
4. O PDF é **enviado por e-mail** ao coordenador. A tela mostra apenas a
   confirmação — nada de download nem dados expostos na página.

### Caso especial: Curso Técnico em Trânsito

Há um **checkbox na barra lateral**, "Curso Técnico em Trânsito". Quando marcado,
o app pede **2 arquivos**: o mapa de Trânsito **e** o mapa completo da escola
(necessário porque o mapa de Trânsito não traz as notas de ensino médio). Para os
demais cursos, basta **1 arquivo**, e as disciplinas são extraídas dinamicamente
do próprio mapa.

## 🗂️ Estrutura

```
.
├── app.py                  # Interface Streamlit (formulário, validação, envio)
├── core/
│   ├── disciplinas.py      # Catálogo de nomes amigáveis de disciplinas
│   ├── manipulacao.py      # Leitura/processamento dos .xls -> DataFrames
│   ├── relatorios.py       # Estatísticas, gráficos, IA e geração do PDF
│   └── email_sender.py     # Validação de e-mail e envio SMTP (Gmail)
├── assets/                 # Logo institucional opcional (logo_cefet.png)
├── .streamlit/
│   ├── config.toml         # Tema
│   └── secrets.toml.example
└── requirements.txt
```

## ⚙️ Configuração (segredos)

Copie `.streamlit/secrets.toml.example` para `.streamlit/secrets.toml` (local) ou
cole o conteúdo em **Settings → Secrets** no Streamlit Cloud:

| Segredo | Obrigatório | Para quê |
|---|---|---|
| `GMAIL_USER` | sim | Conta Gmail que **envia** os relatórios |
| `GMAIL_APP_PASSWORD` | sim | **Senha de app** do Gmail (não a senha normal) |
| `OPENAI_API_KEY` | não | Comentário analítico por IA (opcional) |

### Gerando a "Senha de app" do Gmail

1. Ative a **verificação em 2 etapas** na conta Google.
2. Acesse <https://myaccount.google.com/apppasswords> e gere uma senha de app.
3. Use essa senha (16 caracteres) em `GMAIL_APP_PASSWORD`.

> Recomenda-se uma conta Google Workspace institucional como remetente, para que
> os e-mails saiam de um endereço oficial.

## ▶️ Rodar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501`.

> No Ubuntu, se faltar o módulo de ambiente virtual: `sudo apt install python3-venv`.

## ☁️ Publicar no Streamlit Community Cloud (gratuito)

1. Suba este repositório para o GitHub.
2. Em <https://share.streamlit.io>, conecte o repo e aponte o *main file* para `app.py`.
3. Em **Settings → Secrets**, cole `GMAIL_USER`, `GMAIL_APP_PASSWORD` (e
   opcionalmente `OPENAI_API_KEY`).
4. *Deploy* — o Streamlit instala o `requirements.txt` automaticamente.

## 🔒 Privacidade e limitações

- Os mapas de turma e o PDF são processados **em memória** e não ficam salvos no
  servidor. O `.gitignore` impede o commit de `*.xls`, `*.csv` e `*.pdf`.
- A restrição por domínio `@cefetmg.br` é uma barreira simples: o relatório
  **só é entregue na caixa institucional** informada. Ela não verifica a posse da
  conta (qualquer um poderia digitar um endereço `@cefetmg.br` de terceiros, mas o
  resultado iria para a caixa daquela pessoa, não para quem enviou). Se for
  necessário garantir a posse, dá para adicionar um **código de verificação** por
  e-mail antes de processar.

---
Desenvolvido pelo Professor Diego Camargo — `diegocamargo@cefetmg.br`.
