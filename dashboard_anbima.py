import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard TI - ANBIMA", layout="wide", page_icon="📊")

# --- CSS PARA ESTILO E IMPRESSÃO ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .logo-img {
        height: 80px;
        width: 80px;
        object-fit: contain;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    div[data-testid="stColumn"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* --- ESTILO DE IMPRESSÃO (ESCONDER BARRA LATERAL E BOTÕES) --- */
    @media print {
        /* Esconde a barra lateral */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Esconde o cabeçalho padrão do Streamlit (faixa colorida) */
        header {
            display: none;
        }
        /* Esconde botões de upload e outros elementos interativos que não fazem sentido no papel */
        .stFileUploader, .stButton, .stDeployButton, [data-testid="stToolbar"] {
            display: none;
        }
        /* Ajusta a margem para usar a folha toda */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# --- FUNÇÃO AUXILIAR: CONVERTER IMAGEM PARA BASE64 ---
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# --- CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .logo-img {
        height: 80px;
        width: 80px;
        object-fit: contain;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    div[data-testid="stColumn"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABEÇALHO ---
col_head1, col_head2, col_head3 = st.columns([1, 6, 1])

with col_head1:
    img_anbima = get_img_as_base64("logo_anbima.png")
    if img_anbima:
        st.markdown(f'<img src="data:image/png;base64,{img_anbima}" class="logo-img">', unsafe_allow_html=True)
    else:
        st.warning("Falta 'logo_anbima.png'")

with col_head2:
    st.markdown("<h1 style='text-align: center; color: #005c58; margin: 0; padding: 0;'>Report Semanal - Service Desk</h1>", unsafe_allow_html=True)

with col_head3:
    img_gotobiz = get_img_as_base64("logo_gotobiz.png")
    if img_gotobiz:
        st.markdown(f'<img src="data:image/png;base64,{img_gotobiz}" class="logo-img">', unsafe_allow_html=True)
    else:
        st.warning("Falta 'logo_gotobiz.png'")

st.divider()

# --- 1. FUNÇÕES DE NEGÓCIO ---

def verificar_prazo(valor_tempo):
    if pd.isna(valor_tempo) or str(valor_tempo).strip() == '':
        return None
    valor_str = str(valor_tempo).strip()
    if valor_str.startswith('-'):
        return False # FORA
    return True

def classificar_criador(nome):
    if pd.isna(nome): return 'Desconhecido'
    nome = str(nome).lower().strip()
    tecnicos = ['daniel costa', 'paulo reis', 'felipe souza', 'leandro jesus', 'marco costa', 'alessandro pereira', 'enos oliveira', 'hernan silva', 'paulo saraiva', 'guilherme silva', 'daniel souza']
    sistemas = ['pipefy', 'automation for jira', 'administrador do jira', 'jira infraestrutura', 'usuário de serviço', 'system']
    if nome in tecnicos: return 'Técnico'
    elif any(s in nome for s in sistemas): return 'Sistema/Serviço'
    else: return 'Usuário Final'

def get_location(analyst_name):
    ANALYST_MAP = {'Daniel Costa': 'SP', 'Felipe Souza': 'SP', 'Leandro Jesus': 'SP', 'Paulo Reis': 'RJ', 'Marco Costa': 'SP', 'Alessandro Pereira': 'SP', 'Enos Oliveira': 'SP', 'Daniel souza': 'SP', 'Guilherme Silva': 'SP', 'Hernan Silva': 'RJ', 'Paulo Saraiva': 'SP'}
    return ANALYST_MAP.get(analyst_name, 'Outros')

# --- 2. BARRA LATERAL ---
st.sidebar.header("📂 Importação")
uploaded_jira = st.sidebar.file_uploader("1. Arquivo do Jira (CSV/XLSX)", type=["csv", "xlsx"], key="jira")
uploaded_lan = st.sidebar.file_uploader("2. Arquivo Lansweeper (CSV/XLSX)", type=["csv", "xlsx"], key="lan")

st.sidebar.markdown("---")
st.sidebar.header("📝 Dados Manuais")
st.sidebar.subheader("Rotinas e Metas")
qtd_rotinas = st.sidebar.number_input("Qtd. Rotinas (Volume)", min_value=0, value=0)
pct_rotinas_slm = st.sidebar.slider("% Rotinas Executadas (Meta)", 0, 100, 100)
qtd_pesquisas_resp = st.sidebar.number_input("Nº Total de Respostas", min_value=0, value=0)
qtd_avaliacoes_positivas = st.sidebar.number_input("Nº Avaliações 5 Estrelas", min_value=0, value=0)

st.sidebar.subheader("Estoque RJ (Manual)")
teclado_fio = st.sidebar.number_input("Teclado c/ fio", min_value=0, value=0)
teclado_sem_fio = st.sidebar.number_input("Teclado s/ fio", min_value=0, value=0)
mouse_fio = st.sidebar.number_input("Mouse c/ fio", min_value=0, value=0)
mouse_sem_fio = st.sidebar.number_input("Mouse s/ fio", min_value=0, value=0)
headset = st.sidebar.number_input("Headset", min_value=0, value=0)
estoque_rj_manual = teclado_fio + teclado_sem_fio + mouse_fio + mouse_sem_fio + headset

# --- 3. PROCESSAMENTO ---
df_jira = None
if uploaded_jira:
    try:
        if uploaded_jira.name.endswith('.csv'): df_jira = pd.read_csv(uploaded_jira)
        else: df_jira = pd.read_excel(uploaded_jira)
        df_jira.columns = df_jira.columns.str.strip()
        if 'Status' in df_jira.columns: df_jira = df_jira[df_jira['Status'] != 'Cancelado']
        if 'Criado' in df_jira.columns: df_jira['Criado'] = pd.to_datetime(df_jira['Criado'], dayfirst=True, errors='coerce')
        
        if 'Criado' in df_jira.columns and not df_jira['Criado'].isnull().all():
            min_date, max_date = df_jira['Criado'].min().date(), df_jira['Criado'].max().date()
            st.sidebar.subheader("📅 Período Jira")
            c1, c2 = st.sidebar.columns(2)
            start_date = c1.date_input("Início", min_date)
            end_date = c2.date_input("Fim", max_date)
            mask = (df_jira['Criado'].dt.date >= start_date) & (df_jira['Criado'].dt.date <= end_date)
            df_jira = df_jira.loc[mask].copy()

        df_jira['Localidade'] = df_jira['Responsável'].apply(get_location)
        col_criador = 'Criador' if 'Criador' in df_jira.columns else 'Creator'
        if col_criador in df_jira.columns: df_jira['Categoria_Criador'] = df_jira[col_criador].apply(classificar_criador)
        else: df_jira['Categoria_Criador'] = 'Desconhecido'
    except Exception as e: st.error(f"Erro Jira: {e}")

df_lan = None
if uploaded_lan:
    try:
        if uploaded_lan.name.endswith('.csv'): df_lan = pd.read_csv(uploaded_lan)
        else: df_lan = pd.read_excel(uploaded_lan)
        df_lan.columns = df_lan.columns.str.strip()
        if 'Model' not in df_lan.columns: df_lan['Model'] = "Desconhecido"
        if 'Statename' not in df_lan.columns: df_lan['Statename'] = "Desconhecido"
        modelos_excluir = ['nginx', 'Virtual Machine', 'VMware7,1', 't3a.large', 'Custom Pc']
        df_lan = df_lan[~df_lan['Model'].isin(modelos_excluir)]
        df_lan['Model'] = df_lan['Model'].replace('Pro 14 PC14250', 'Dell Pro 14 PC14250')
    except Exception as e: st.error(f"Erro Lansweeper: {e}")

# --- 4. DASHBOARD ---
if df_jira is None and df_lan is None:
    st.info("👋 Faça o upload dos arquivos na barra lateral.")
    st.stop()

tabs = st.tabs([t for t in ["Service Desk (Jira)", "🖥️ Inventário (Lansweeper)"] if (t=="Service Desk (Jira)" and df_jira is not None) or (t=="🖥️ Inventário (Lansweeper)" and df_lan is not None)])

# ABA JIRA
if df_jira is not None:
    with tabs[0]:
        st.caption(f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        st.divider()

        incidentes_df = df_jira[df_jira['Tipo de item'] == 'Incidente']
        requisicoes_df = df_jira[df_jira['Tipo de item'] != 'Incidente']
        total_chamados = len(df_jira)
        total_kpi = total_chamados + qtd_rotinas

        # SLA Resposta
        ok_resp, total_resp, pct_resp = 0, 0, 0
        if 'Tempo até a primeira resposta' in df_jira.columns:
            df_jira['SLA_Resp_Ok'] = df_jira['Tempo até a primeira resposta'].apply(verificar_prazo)
            total_resp = df_jira['SLA_Resp_Ok'].notnull().sum()
            ok_resp = df_jira[df_jira['SLA_Resp_Ok'] == True].shape[0]
            pct_resp = (ok_resp / total_resp * 100) if total_resp > 0 else 0

        # SLA Resolução
        ok_resol, total_resol, pct_resol = 0, 0, 0
        if 'Tempo de resolução' in df_jira.columns:
            df_jira['SLA_Res_Ok'] = df_jira['Tempo de resolução'].apply(verificar_prazo)
            total_resol = df_jira['SLA_Res_Ok'].notnull().sum()
            ok_resol = df_jira[df_jira['SLA_Res_Ok'] == True].shape[0]
            pct_resol = (ok_resol / total_resol * 100) if total_resol > 0 else 0

        # Backlog
        status_fechados = ['Resolvido', 'Fechado', 'Entrega Concluída', 'Devolução Concluída', 'Troca Concluída', 'Concluído']
        backlog_df = df_jira[~df_jira['Status'].isin(status_fechados)]
        qtd_backlog = len(backlog_df)
        pct_backlog = (qtd_backlog / total_chamados * 100) if total_chamados > 0 else 0

        # Satisfação
        pct_satisfacao = (qtd_avaliacoes_positivas / qtd_pesquisas_resp * 100) if qtd_pesquisas_resp > 0 else 0
        
        # Aderência
        base_aderencia = total_resol if total_resol > 0 else total_chamados
        pct_aderencia = (qtd_pesquisas_resp / base_aderencia * 100) if base_aderencia > 0 else 0

        # VISUALIZAÇÃO
        st.subheader("🔹 Volumetria")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Geral", total_kpi)
        k2.metric("Incidentes", len(incidentes_df), border=True)
        k3.metric("Requisições", len(requisicoes_df), border=True)
        k4.metric("Rotinas", qtd_rotinas, border=True)
        k5.metric("Chamados Jira", total_chamados)

        st.divider()
        st.subheader("🔹 Gestão de Nível de Serviço (SLM)")
        s1, s2, s3, s4 = st.columns(4)
        
        s1.metric("% Resp. Prazo", f"{pct_resp:.1f}%", f"{ok_resp} no prazo", help="Chamados respondidos dentro do SLA")
        s2.metric("% Resol. Prazo", f"{pct_resol:.1f}%", f"{ok_resol} no prazo", help="Chamados resolvidos dentro do SLA")
        
        s3.metric("% Rotinas", f"{pct_rotinas_slm}%", "Meta")
        s4.metric("% Backlog", f"{pct_backlog:.1f}%", f"{qtd_backlog} abertos", delta_color="inverse")

        st.divider()
        st.subheader("🔹 Pesquisa de Satisfação")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Satisfação", f"{pct_satisfacao:.1f}%", delta_color="normal" if pct_satisfacao >= 90 else "inverse")
        q2.metric("Aderência", f"{pct_aderencia:.1f}%")
        q3.metric("Volume SP", len(df_jira[df_jira['Localidade']=='SP']))
        q4.metric("Volume RJ", len(df_jira[df_jira['Localidade']=='RJ']))

        st.divider()
        st.subheader("🚀 Planos de Ação & Estoque")
        if 'df_planos' not in st.session_state:
            st.session_state.df_planos = pd.DataFrame([{"Descrição": "Ex: Troca de Switch", "Status": "Em Andamento", "Local": "RJ", "% Conclusão": 50, "Responsável": "Paulo", "Início": None}])
        st.data_editor(st.session_state.df_planos, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"% Conclusão": st.column_config.NumberColumn(format="%d%%", min_value=0, max_value=100)})
        
        st.markdown("#### 📦 Detalhamento Estoque RJ")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Teclado c/ Fio", teclado_fio)
        e2.metric("Teclado s/ Fio", teclado_sem_fio)
        e3.metric("Mouse c/ Fio", mouse_fio)
        e4.metric("Mouse s/ Fio", mouse_sem_fio)
        e5.metric("Headset", headset)
        st.caption(f"Total: {estoque_rj_manual} itens")

        st.divider()
        st.markdown("#### Detalhes Operacionais")
        t1, t2 = st.tabs(["Top 5 & Criadores", "Base Completa"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.error("🔥 Incidentes")
                if not incidentes_df.empty:
                    if 'Request Type' in incidentes_df.columns: st.dataframe(incidentes_df['Request Type'].value_counts().head(5).reset_index(name='Qtd'), use_container_width=True, hide_index=True)
                    if col_criador in incidentes_df.columns: st.dataframe(incidentes_df[col_criador].value_counts().head(5).reset_index(name='Qtd'), use_container_width=True, hide_index=True)
            with c2:
                st.success("📋 Requisições")
                if not requisicoes_df.empty:
                    if 'Request Type' in requisicoes_df.columns: st.dataframe(requisicoes_df['Request Type'].value_counts().head(5).reset_index(name='Qtd'), use_container_width=True, hide_index=True)
                    if col_criador in requisicoes_df.columns: st.dataframe(requisicoes_df[col_criador].value_counts().head(5).reset_index(name='Qtd'), use_container_width=True, hide_index=True)
        with t2: st.dataframe(df_jira)

# ABA LANSWEEPER
if df_lan is not None:
    # AQUI ESTAVA O ERRO - CORRIGIDO
    idx = 1 if df_jira is not None else 0
    
    with tabs[idx]:
        st.header("🖥️ Gestão de Ativos")
        st.divider()
        c1, c2 = st.columns([1.2, 1])
        model_counts = df_lan['Model'].value_counts().reset_index()
        model_counts.columns = ['Modelo', 'Quantidade']
        
        with c1:
            st.subheader("📍 Selecione um Modelo")
            st.info("Clique nas barras para filtrar")
            fig = px.bar(model_counts.head(20), x='Quantidade', y='Modelo', orientation='h', text='Quantidade', color_discrete_sequence=['#007bff'])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, clickmode='event+select', plot_bgcolor='rgba(0,0,0,0)', dragmode=False)
            sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
        
        total = len(df_lan)
        if len(sel["selection"]["points"]) > 0:
            mod = sel["selection"]["points"][0]["y"]
            qtd = sel["selection"]["points"][0]["x"]
            df_rosca = pd.DataFrame({'Categoria': [mod, 'Outros'], 'Quantidade': [qtd, total-qtd], 'Cor': ['#005c58', '#e0e0e0']})
            title, center = f"Representatividade: {mod}", f"<b>{qtd}</b><br>({(qtd/total*100):.1f}%)"
            df_table = df_lan[df_lan['Model'] == mod]
        else:
            df_rosca = model_counts.head(5).copy()
            df_rosca.columns = ['Categoria', 'Quantidade']
            if total - df_rosca['Quantidade'].sum() > 0: df_rosca = pd.concat([df_rosca, pd.DataFrame({'Categoria':['Outros'], 'Quantidade':[total - df_rosca['Quantidade'].sum()]})])
            df_rosca['Cor'] = px.colors.qualitative.Safe[:len(df_rosca)]
            title, center, df_table = "Visão Geral", f"<b>{total}</b><br>Ativos", df_lan

        with c2:
            st.subheader("📊 Análise")
            fig_pie = go.Figure(data=[go.Pie(labels=df_rosca['Categoria'], values=df_rosca['Quantidade'], hole=0.6, textinfo='none', marker=dict(colors=df_rosca['Cor']))])
            fig_pie.update_layout(title=title, showlegend=True, legend=dict(orientation="h", y=-0.2), annotations=[dict(text=center, x=0.5, y=0.5, font_size=20, showarrow=False)], hovermode=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        st.dataframe(df_table, use_container_width=True)