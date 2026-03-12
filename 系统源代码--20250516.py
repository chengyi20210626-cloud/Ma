import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from zhipuai import ZhipuAI
import os
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

# 配置页面
st.set_page_config(
    page_title="科研人员学术诚信风险预警平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* 统一所有按钮样式（高级感天青色系） */
.stButton>button[kind="primary"], .stDownloadButton>button {
    background-color: #4a90e2 !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.2) !important;
}

/* 统一按钮悬停状态 */
.stButton>button:hover, .stDownloadButton>button:hover, .sidebar .stButton>button:hover {  
    background-color: #357abd;
    box-shadow: 0 6px 18px rgba(74, 144, 226, 0.3);
    transform: translateY(-1px);
}

/* 统一按钮激活状态 */
.stButton>button:active, .stDownloadButton>button:active, .sidebar .stButton>button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(74, 144, 226, 0.4);
}

/* 侧边栏按钮之前的特殊样式清除，使用统一样式 */
.sidebar .stButton>button {
    padding: 10px 20px;  /* 取消之前的8px 16px，使用统一值 */
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.2);  /* 添加统一阴影 */
    background-color: #4a90e2;  /* 确保颜色一致 */
}

/* 其他原有样式保持不变... */
.main {
    max-width: 90%;
    margin: 0 auto; /* 水平居中 */
}

/* 调整侧边栏宽度（可选，保持与主内容比例协调） */
.sidebar .sidebar-content {
    max-width: 280px; /* 适当缩小侧边栏宽度 */
}

/* 确保宽屏设备下内容不溢出 */
.stApp {
    padding: 20px; /* 增加内边距提升舒适感 */
}

/* 表格滚动条优化 */
.scrollable-table {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px;
}
/* 页脚样式 */
.footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 1px solid #e0e0e0;
    color: #666;
    font-size: 14px;
    font-style: italic;
}

/* 研究人员信息卡片样式 */
.info-card {
    padding: 24px;
    margin: 20px 0;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.1);
    background-color: white;
}

.info-card h3 {
    font-size: 28px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 12px;
}

.info-item {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0;
}

.info-item label {
    font-weight: 500;
    color: #34495e;
    min-width: 80px;
    text-align: right;
}

.info-item value {
    color: #2c3e50;
    font-size: 16px;
}
.module {
    padding: 24px;
    margin: 20px 0;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.1);
    background-color: white;
}
</
</style>
""", unsafe_allow_html=True)


# 用户认证模块
USER_DATA_FILE = Path(__file__).parent / "users.json"


def create_user_file():
    """初始化用户数据文件"""
    if not USER_DATA_FILE.exists():
        with open(USER_DATA_FILE, "w") as f:
            json.dump([], f)


def load_users():
    """加载用户数据"""
    create_user_file()
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)


def save_user(username, password):
    """保存新用户"""
    users = load_users()
    if any(u["username"] == username for u in users):
        return False, "用户名已存在"
    users.append({"username": username, "password": password})
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)
    return True, "注册成功"


def authenticate_user(username, password):
    """用户认证"""
    users = load_users()
    for user in users:
        if user["username"] == username and user["password"] == password:
            return True, "登录成功"
    return False, "用户名或密码错误"


# 初始化会话状态
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'search_name' not in st.session_state:
    st.session_state.search_name = ''
if 'search_institution' not in st.session_state:
    st.session_state.search_institution = ''
if 'search_button_clicked' not in st.session_state:
    st.session_state.search_button_clicked = False
if 'selected' not in st.session_state:
    st.session_state.selected = None
if 'author_risk' not in st.session_state:
    st.session_state.author_risk = None
if 'paper_records' not in st.session_state:
    st.session_state.paper_records = pd.DataFrame()
if 'project_records' not in st.session_state:
    st.session_state.project_records = pd.DataFrame()
if 'related_people' not in st.session_state:
    st.session_state.related_people = []
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None


# 登录注册页面
# 在show_auth_page函数前添加以下样式代码
st.markdown(f"""
    <style>
        /* 全局容器样式 */
        .main {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        /* 认证卡片容器 */
        .auth-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin: 1rem;
        }}
        
        /* 输入框样式 */
        .stTextInput>div>div>input {{
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }}
        
        .stTextInput>div>div>input:focus {{
            border-color: #4a90e2;
            box-shadow: 0 0 8px rgba(74, 144, 226, 0.3);
        }}
        
        /* 按钮样式 */
        .stButton>button {{
            border-radius: 10px;
            background: linear-gradient(45deg, #4a90e2, #5d9cec);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            width: 100%;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3);
        }}
        
        /* 选项卡样式 */
        .stRadio [role=radiogroup] {{
            gap: 1rem;
            justify-content: center;
        }}
        
        .stRadio [role=radio] {{
            padding: 0.5rem 1rem;
            border-radius: 10px;
            background: #f0f0f0;
            transition: all 0.3s ease;
        }}
        
        .stRadio [role=radio][aria-checked=true] {{
            background: linear-gradient(45deg, #4a90e2, #5d9cec);
            color: white !important;
            box-shadow: 0 3px 10px rgba(74, 144, 226, 0.3);
        }}
        
        /* 系统简介样式 */
        .intro-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
            margin: 1rem;
        }}
        
        .intro-card h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #4a90e2;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        
        /* 响应式布局 */
        @media (max-width: 768px) {{
            .main {{
                padding: 1rem;
            }}
            .auth-card, .intro-card {{
                margin: 0.5rem;
                padding: 1.5rem;
            }}
        }}
    </style>
""", unsafe_allow_html=True)

# 修改后的show_auth_page函数
def show_auth_page():
    # 页面标题
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 style="color: #2c3e50; font-size: 2.5rem; margin-bottom: 0.5rem;">
                <span style="color: #4a90e2;"></span>正道人员诚信监测预警系统
            </h1>
            <p style="color: #7f8c8d; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                诚信科研不仅是对科学精神的尊重，更是对社会的责任与担当。在追求知识的道路上，坚守真实，捍卫公正。
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 双栏布局
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        # 系统简介卡片
        st.markdown("""
            <div class="intro-card">
                <h3>📘 系统简介</h3>
                <p style="line-height: 1.6; color: #34495e;">
                    “正道”人员诚信监测预警系统致力于为科研领域提供一个可靠的诚信监测平台。通过先进的技术手段和大数据分析，对科研人员的行为进行全面、客观的评估，及时发现并预警潜在的诚信问题，保障科研环境的公正与透明。
                </p>
                <div style="margin-top: 2rem; background: #f8f9fa; padding: 1rem; border-radius: 10px;">
                    <h4 style="color: #4a90e2; margin-bottom: 0.5rem;">✨ 核心功能</h4>
                    <ul style="color: #7f8c8d; padding-left: 1.2rem;">
                        <li>不端信息查询</li>
                        <li>大模型评价</li>
                        <li>关系网络图</li>
                        <li>信用报告</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # 认证卡片
        with st.container():
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            
            # 选项卡样式优化
            auth_tab = st.radio(
                "请选择操作",
                ["登录", "注册"],
                key="auth_tab",
                horizontal=True,
                label_visibility="collapsed"
            )

            if auth_tab == "登录":
                st.markdown("### 🔑 用户登录")
                username = st.text_input("用户名", key="login_username", 
                                       placeholder="请输入注册用户名")
                password = st.text_input("密码", type="password", key="login_password",
                                       placeholder="请输入登录密码")

                if st.button("立即登录", use_container_width=True):
                    valid, msg = authenticate_user(username, password)
                    if valid:
                        st.session_state.is_logged_in = True
                        st.session_state.current_user = username
                        st.success("✅ 登录成功！正在跳转...")
                        st.rerun()
                    else:
                        st.error("❌ 认证失败：用户名或密码错误")

            elif auth_tab == "注册":
                st.markdown("### 📝 新用户注册")
                new_username = st.text_input("用户名", key="register_username",
                                           placeholder="设置您的唯一用户名")
                new_password = st.text_input("密码", type="password", key="register_password",
                                           placeholder="设置登录密码（至少6位）")
                confirm_password = st.text_input("确认密码", type="password", key="register_confirm_password",
                                               placeholder="请再次输入密码")

                if st.button("立即注册", use_container_width=True):
                    if len(new_password) < 6:
                        st.error("⚠️ 密码长度需至少6位")
                    elif new_password != confirm_password:
                        st.error("⚠️ 两次输入的密码不一致")
                    else:
                        valid, msg = save_user(new_username, new_password)
                        if valid:
                            st.success("🎉 注册成功！自动登录中...")
                            st.session_state.is_logged_in = True
                            st.session_state.current_user = new_username
                            st.rerun()
                        else:
                            st.error(f"⚠️ {msg}")

            st.markdown('</div>', unsafe_allow_html=True)


# 未登录时显示登录注册页面
if not st.session_state.is_logged_in:
    show_auth_page()
    st.stop()

# 初始化智谱API
client = ZhipuAI(api_key="89c41de3c3a34f62972bc75683c66c72.ZGwzmpwgMfjtmksz")

# 数据预处理和风险值计算模块
@st.cache_data(show_spinner=False)
def process_risk_data():
    # 不端原因严重性权重
    misconduct_weights = {
        '伪造、篡改图片': 6,
        '篡改图片': 3,
        '篡改数据': 3,
        '篡改数据、图片': 6,
        '编造研究过程': 4,
        '编造研究过程、不当署名': 7,
        '篡改数据、不当署名': 6,
        '伪造通讯作者邮箱': 2,
        '实验流程不规范': 2,
        '数据审核不严': 2,
        '署名不当、实验流程不规范': 5,
        '篡改数据、代写代投、伪造通讯作者邮箱、不当署名': 13,
        '篡改数据、伪造通讯作者邮箱、不当署名': 8,
        '第三方代写、伪造通讯作者邮箱': 7,
        '第三方代写代投、伪造数据': 8,
        '一稿多投': 2,
        '第三方代写代投、伪造数据、一稿多投': 10,
        '篡改数据、剽窃': 8,
        '伪造图片': 3,
        '伪造图片、不当署名': 6,
        '委托实验、不当署名': 6,
        '伪造数据': 3,
        '伪造数据、篡改图片': 6,
        '伪造数据、不当署名、伪造通讯作者邮箱等': 8,
        '伪造数据、一图多用、伪造图片、代投问题': 14,
        '伪造数据、署名不当': 6,
        '抄袭剽窃他人项目申请书内容': 6,
        '伪造通讯作者邮箱、篡改数据和图片': 8,
        '篡改数据、不当署名': 6,
        '抄袭他人基金项目申请书': 6,
        '结题报告中存在虚假信息': 5,
        '抄袭剽窃': 5,
        '造假、抄袭': 5,
        '第三方代写代投': 5,
        '署名不当': 3,
        '第三方代写代投、署名不当': 8,
        '抄袭剽窃、伪造数据': 8,
        '买卖图片数据': 3,
        '买卖数据': 3,
        '买卖论文': 5,
        '买卖论文、不当署名': 8,
        '买卖论文数据': 8,
        '买卖论文数据、不当署名': 11,
        '买卖图片数据、不当署名': 6,
        '图片不当使用、伪造数据': 6,
        '图片不当使用、数据造假、未经同意使用他人署名': 9,
        '图片不当使用、数据造假、未经同意使用他人署名、编造研究过程': 13,
        '图片造假、不当署名': 9,
        '图片造假、不当署名、伪造通讯作者邮箱等': 11,
        '买卖数据、不当署名': 6,
        '伪造论文、不当署名': 6,
        '图片不当使用、数据造假、未经同意使用他人署名': 9,
        '图片不当使用、数据造假、未经同意使用他人署名、编造研究过程': 13,
        '其他轻微不端行为': 1
    }
    # 责任权重映射
    responsibility_weights = {
        "通讯作者": 0.45,
        "第一作者": 0.35,
        "合作者": 0.20
    }

    # 读取原始数据
    papers_df = pd.read_excel('C:\\Users\\86130\\Desktop\\project\\马丹薇\\实验数据.xlsx', sheet_name='论文')
    projects_df = pd.read_excel('C:\\Users\\86130\\Desktop\\project\\马丹薇\\实验数据.xlsx', sheet_name='项目')
    # 网络构建函数
    @st.cache_resource(show_spinner=False)
    def build_networks(papers, projects):
        # 作者-论文网络（处理论文数据，包含责任权重）
        G_papers = nx.Graph()
        for _, row in papers.iterrows():
            misconduct_weight = misconduct_weights.get(row["不端原因"], 1)
            responsibility = row.get("责任", "")  # 处理可能缺失的责任列
            responsibility_weight = responsibility_weights.get(responsibility, 0.2)  # 默认合作者权重
            total_weight = misconduct_weight * responsibility_weight
            G_papers.add_edge(row["姓名"], row["不端内容"], weight=total_weight)

        # 作者-项目网络（处理项目数据，无责任列，直接使用不端原因权重）
        G_projects = nx.Graph()
        for _, row in projects.iterrows():
            misconduct_weight = misconduct_weights.get(row["不端原因"], 1)
            total_weight = misconduct_weight * 1  # 项目不考虑责任权重，直接使用基础权重
            G_projects.add_edge(row["姓名"], row["不端内容"], weight=total_weight)

        # 作者-作者网络（合并论文和项目的连接关系）
        G_authors = nx.Graph()

        # 处理论文中的共同作者关系
        for _, row in papers.iterrows():
            authors = [row['姓名']]  # 这里假设每行只有一个作者（实际可能需要解析多作者）
            weight = misconduct_weights.get(row['不端原因'], 1)
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G_authors.has_edge(authors[i], authors[j]):
                        G_authors[authors[i]][authors[j]]['weight'] += weight
                    else:
                        G_authors.add_edge(authors[i], authors[j], weight=weight)

        # 处理项目中的共同作者关系（假设项目每行一个负责人）
        for _, row in projects.iterrows():
            author = row['姓名']
            weight = misconduct_weights.get(row['不端原因'], 1)
            # 项目通常只有一个负责人，暂不建立作者间连接，如需连接可根据机构或方向

        # 研究方向相似性连接（保留原逻辑）
        research_areas = papers.groupby('姓名')['研究方向'].apply(lambda x: ' '.join(x)).reset_index()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(research_areas['研究方向'])
        similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

        for i in range(len(research_areas)):
            for j in range(i + 1, len(research_areas)):
                if similarity_matrix[i, j] > 0.7:
                    a1 = research_areas.iloc[i]['姓名']
                    a2 = research_areas.iloc[j]['姓名']
                    G_authors.add_edge(a1, a2, weight=similarity_matrix[i, j], reason='研究方向相似')

        # 共同机构连接（保留原逻辑）
        institution_map = papers.set_index('姓名')['研究机构'].to_dict()
        for a1 in institution_map:
            for a2 in institution_map:
                if a1 != a2 and institution_map[a1] == institution_map[a2]:
                    G_authors.add_edge(a1, a2, weight=1, reason='研究机构相同')

        return G_authors

    # Word2Vec（Skip-gram）模型定义（保留原代码）
    class SkipGramModel(nn.Module):
        def __init__(self, vocab_size, embedding_size):
            super(SkipGramModel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.out = nn.Linear(embedding_size, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            outputs = self.out(embeds)
            return outputs

    # 数据集定义（保留原代码）
    class SkipGramDataset(Dataset):
        def __init__(self, walks, node2id):
            self.walks = walks
            self.node2id = node2id

        def __len__(self):
            return len(self.walks)

        def __getitem__(self, idx):
            walk = self.walks[idx]
            input_ids = [self.node2id[node] for node in walk[:-1]]
            target_ids = [self.node2id[node] for node in walk[1:]]
            return torch.tensor(input_ids), torch.tensor(target_ids)

    # DeepWalk实现（保留原代码）
    @st.cache_resource(show_spinner=False)
    def deepwalk(_graph, walk_length=30, num_walks=100, embedding_size=64):
        graph = _graph
        walks = []
        nodes = list(graph.nodes())

        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [str(node)]
                current = node
                for _ in range(walk_length - 1):
                    neighbors = list(graph.neighbors(current))
                    if neighbors:
                        current = random.choice(neighbors)
                        walk.append(str(current))
                    else:
                        break
                walks.append(walk)

        # 构建节点到ID的映射
        node2id = {node: idx for idx, node in enumerate(set([node for walk in walks for node in walk]))}
        id2node = {idx: node for node, idx in node2id.items()}

        # 构建数据集
        dataset = SkipGramDataset(walks, node2id)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 模型初始化
        model = SkipGramModel(len(node2id), embedding_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        for epoch in range(3):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, len(node2id)), targets.view(-1))
                loss.backward()
                optimizer.step()

        # 获取嵌入
        embeddings = {}
        with torch.no_grad():
            for node, idx in node2id.items():
                embeddings[node] = model.embeddings(torch.tensor([idx])).squeeze().numpy()

        return embeddings

    # 执行计算流程（保留原代码逻辑，调整数据处理）
    with st.spinner('正在构建合作网络...'):
        G_authors = build_networks(papers_df, projects_df)

    with st.spinner('正在训练DeepWalk模型...'):
        embeddings = deepwalk(G_authors)

    with st.spinner('正在计算风险指标...'):
        # 构建分类数据集（保留原逻辑，假设边权重转换为风险特征）
        X, y = [], []
        for edge in G_authors.edges():
            X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
            y.append(1)

        non_edges = list(nx.non_edges(G_authors))
        non_edges = random.sample(non_edges, len(y))
        for edge in non_edges:
            X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
            y.append(0)

        # 训练分类器（保留原代码）
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        # 计算节点风险值（使用嵌入向量的范数作为风险指标）
        initial_risks = {node: np.linalg.norm(emb) for node, emb in embeddings.items()}

    # 风险传播模型参数
    MAX_ITERATIONS = 10  # 最大迭代次数
    ALPHA = 0.2  # 传播衰减因子

    # 构建图的邻接表（包含权重）
    adj_list = {node: {} for node in G_authors.nodes()}
    for u, v, data in G_authors.edges(data=True):
        weight = data.get('weight', 1.0)
        adj_list[u][v] = weight
        adj_list[v][u] = weight  # 无向图双向添加

    # 初始化风险值（使用原始计算的风险值）
    current_risks = initial_risks.copy()

    # 风险传播迭代
    for _ in range(MAX_ITERATIONS):
        new_risks = current_risks.copy()
        for node in current_risks:
            # 计算邻居风险加权和
            neighbor_risk = 0.0
            total_weight = 0.0
            for neighbor, weight in adj_list[node].items():
                neighbor_risk += current_risks.get(neighbor, 0.0) * weight
                total_weight += weight

            # 归一化权重并计算传播影响
            if total_weight > 0:
                avg_neighbor_risk = neighbor_risk / total_weight
                new_risks[node] = current_risks[node] + ALPHA * avg_neighbor_risk  # 原始风险+邻居影响
            else:
                new_risks[node] = current_risks[node]  # 没有邻居则保持不变

        current_risks = new_risks

    risk_df = pd.DataFrame({
        '作者': list(current_risks.keys()),
        '风险值': list(current_risks.values())
    })
    risk_df.to_parquet('risk_scores.parquet', engine='pyarrow')
    return risk_df, papers_df, projects_df

# 调用智谱大模型进行评价
def get_zhipu_evaluation(selected, paper_records, project_records, related_people):
    # 构建输入文本
    related_people_str = ", ".join(related_people) if related_people else "无"
    input_text = f"请对科研人员 {selected} 进行评价，其论文不端记录为：{paper_records.to_csv(sep=chr(9), na_rep='nan')}，项目不端记录为：{project_records.to_csv(sep=chr(9), na_rep='nan')}。同时，请提及国家的一些科研诚信政策，并列举出与 {selected} 有关的一些人（{related_people_str}）。"
    try:
        response = client.chat.completions.create(
            model="glm-4v-plus",
            messages=[{"role": "user", "content": input_text}]
        )
        # 检查响应是否成功
        if response:
            return response.choices[0].message.content
        else:
            return f"请求失败，可能是网络问题或API调用异常"
    except Exception as e:
        return f"发生异常：{str(e)}"

# 分页显示表格
def show_paginated_table(df, page_size=10, key="pagination"):
    total_pages = (len(df) // page_size) + 1
    page = st.number_input('选择页码', 1, total_pages, 1, key=key)
    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(df.iloc[start:end], use_container_width=True)

# 侧边栏导航
with st.sidebar:
    st.title("导航")
    st.markdown(f"欢迎，{st.session_state.current_user}")
    if st.button("🏠 首页", use_container_width=True):
        st.session_state.page = 'home'
    if st.button("🔍 查询", use_container_width=True):
        st.session_state.page = 'search'
        # 清空搜索相关的session_state变量
        st.session_state.search_name = ''
        st.session_state.search_institution = ''
        st.session_state.selected = None
        st.session_state.author_risk = None
        st.session_state.paper_records = pd.DataFrame()
        st.session_state.project_records = pd.DataFrame()
        st.session_state.related_people = []
        st.session_state.evaluation = None
        st.session_state.search_button_clicked = False

# 主内容区域
st.markdown("<div class='navbar'><h1>科研人员学术诚信风险预警平台</h1></div>", unsafe_allow_html=True)

# 确保 risk_df 被正确加载
try:
    risk_df = pd.read_parquet('risk_scores.parquet')
    papers = pd.read_excel('C:\\Users\\86130\\Desktop\\project\\马丹薇\\实验数据.xlsx', sheet_name='论文')
    projects = pd.read_excel('‪C:\\Users\\86130\\Desktop\\project\\马丹薇\\实验数据.xlsx', sheet_name='项目')
except:
    with st.spinner("首次运行需要初始化数据..."):
        risk_df, papers, projects = process_risk_data()

# 添加风险等级划分
risk_df['风险等级'] = pd.cut(risk_df['风险值'], 
                           bins=[-float('inf'), 52, 56, float('inf')], 
                           labels=['低风险（<52）', '中风险（52-56）', '高风险（≥56）'],
                           include_lowest=True)
risk_df.to_parquet('risk_scores.parquet', engine='pyarrow')

if st.session_state.page == 'home':
    color_map = {'低风险（<52）': '#2ecc71', '中风险（52-56）': '#f39c12', '高风险（≥56）': '#e74c3c'}
    
    fig = go.Figure()
    
    for level, group in risk_df.groupby('风险等级'):
        fig.add_trace(go.Scatter(
            x=group['作者'],
            y=group['风险值'],
            mode='markers',
            text=group['风险值'].round(2),
            hovertext=group['风险等级'],
            name=level,
            marker=dict(
                color=color_map[level],
                size=10,
                line=dict(width=2, color='white')
            )
        ))
    
    fig.update_layout(
        title='科研人员风险预警总体态势',
        xaxis_title='作者',
        yaxis_title='风险值',
        legend_title='风险等级',
        hovermode='closest',
        xaxis=dict(
            tickangle=45,  # 设置x轴标签旋转45度（可调整为30/60等）
            tickfont=dict(size=10),  # 适当减小字体大小（避免标签过大）
            tickmode='auto'  # 自动优化标签密度（可选）
        )
    )
    st.plotly_chart(fig, use_container_width=True)
elif st.session_state.page == 'search':
    # 初始化查询页面时，确保状态为空（避免页面缓存导致的旧数据显示）
    if not st.session_state.search_button_clicked:
        st.session_state.search_name = ''
        st.session_state.search_institution = ''
        st.session_state.selected = None
        st.session_state.author_risk = None
        st.session_state.paper_records = pd.DataFrame()
        st.session_state.project_records = pd.DataFrame()
        st.session_state.related_people = []
        st.session_state.evaluation = None

    # 搜索模块
    with st.container():
        st.subheader("🔍 研究人员查询")
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            st.session_state.search_name = st.text_input("姓名", placeholder="输入研究人员姓名",
                                                         value=st.session_state.search_name)
        with col2:
            st.session_state.search_institution = st.text_input("机构", placeholder="输入研究机构",
                                                                value=st.session_state.search_institution)
        with col3:
            search_button = st.button("查询", type="primary", use_container_width=True)

    if search_button:
        st.session_state.selected = None
        st.session_state.author_risk = None
        st.session_state.paper_records = pd.DataFrame()
        st.session_state.project_records = pd.DataFrame()
        st.session_state.related_people = []
        st.session_state.evaluation = None
        st.session_state.search_button_clicked = True  # 新增，标记按钮已点击

        if not st.session_state.search_name:
            st.warning("请输入研究人员姓名进行查询")
            st.session_state.search_button_clicked = False
            st.stop()

        # 模糊匹配姓名
        name_candidates = risk_df[risk_df['作者'].str.contains(st.session_state.search_name)]

        if name_candidates.empty:
            st.warning("未找到匹配的研究人员（风险数据中无此作者）")
            st.session_state.search_button_clicked = False
            st.stop()

        # 仅根据姓名或姓名+机构搜索
        if st.session_state.search_institution:
            # 同时匹配机构（优化后的条件）
            paper_matches = papers[
                (papers['姓名'].str.contains(st.session_state.search_name)) &
                (papers['研究机构'].str.contains(st.session_state.search_institution))
            ]
            project_matches = projects[
                (projects['姓名'].str.contains(st.session_state.search_name)) &
                (projects['研究机构'].str.contains(st.session_state.search_institution))
            ]
        else:
            paper_matches = papers[papers['姓名'].str.contains(st.session_state.search_name)]
            project_matches = projects[projects['姓名'].str.contains(st.session_state.search_name)]

        if paper_matches.empty and project_matches.empty:
            st.warning("该作者在论文和项目中无相关记录")
            st.session_state.search_button_clicked = False
            st.stop()

        # 选择第一个匹配的作者（从 risk_df 中获取，确保存在）
        st.session_state.selected = name_candidates['作者'].iloc[0]

        # 获取详细信息（修复缩进，确保属于 if search_button 块）
        st.session_state.author_risk = risk_df[risk_df['作者'] == st.session_state.selected].iloc[0]['风险值']
        st.session_state.paper_records = papers[papers['姓名'] == st.session_state.selected]
        st.session_state.project_records = projects[projects['姓名'] == st.session_state.selected]

        # 查找与查询作者有关的人（修复缩进）
        related_people = papers[
            (papers['研究机构'] == papers[papers['姓名'] == st.session_state.selected]['研究机构'].iloc[0]) |
            (papers['研究方向'] == papers[papers['姓名'] == st.session_state.selected]['研究方向'].iloc[0]) |
            (papers['不端内容'] == papers[papers['姓名'] == st.session_state.selected]['不端内容'].iloc[0])
        ]['姓名'].unique()
        st.session_state.related_people = [person for person in related_people if person != st.session_state.selected]

        if st.session_state.search_name: 
            st.session_state.search_button_clicked = True
            # 模糊匹配
            name_candidates = risk_df[risk_df['作者'].str.contains(st.session_state.search_name)]
            paper_matches = papers[papers['姓名'].str.contains(st.session_state.search_name) & papers[
                '研究机构'].str.contains(st.session_state.search_institution)]
            project_matches = projects[projects['姓名'].str.contains(st.session_state.search_name) & projects[
                '研究机构'].str.contains(st.session_state.search_institution)]

            if len(paper_matches) == 0 and len(project_matches) == 0:
                st.warning("未找到匹配的研究人员")
                st.session_state.search_button_clicked = False
                st.stop()

            # 直接选择第一个匹配人员
            st.session_state.selected = name_candidates['作者'].iloc[0]

            # 获取详细信息
            st.session_state.author_risk = risk_df[risk_df['作者'] == st.session_state.selected].iloc[0]['风险值']
            st.session_state.paper_records = papers[papers['姓名'] == st.session_state.selected]
            st.session_state.project_records = projects[projects['姓名'] == st.session_state.selected]

            # 查找与查询作者有关的人
            st.session_state.related_people = papers[
                (papers['研究机构'] == papers[papers['姓名'] == st.session_state.selected]['研究机构'].iloc[0]) |
                (papers['研究方向'] == papers[papers['姓名'] == st.session_state.selected]['研究方向'].iloc[0]) |
                (papers['不端内容'] == papers[papers['姓名'] == st.session_state.selected]['不端内容'].iloc[0])
            ]['姓名'].unique()
            st.session_state.related_people = [person for person in st.session_state.related_people if
                                               person != st.session_state.selected]

        elif st.session_state.search_name and st.session_state.search_institution:
            st.session_state.search_button_clicked = True
            # 模糊匹配
            name_candidates = risk_df[risk_df['作者'].str.contains(st.session_state.search_name)]
            paper_matches = papers[papers['姓名'].str.contains(st.session_state.search_name) & papers[
                '研究机构'].str.contains(st.session_state.search_institution)]
            project_matches = projects[projects['姓名'].str.contains(st.session_state.search_name) & projects[
                '研究机构'].str.contains(st.session_state.search_institution)]

            if len(paper_matches) == 0 and len(project_matches) == 0:
                st.warning("未找到匹配的研究人员")
                st.session_state.search_button_clicked = False
                st.stop()

            # 直接选择第一个匹配人员
            st.session_state.selected = name_candidates['作者'].iloc[0]

            # 获取详细信息
            st.session_state.author_risk = risk_df[risk_df['作者'] == st.session_state.selected].iloc[0]['风险值']
            st.session_state.paper_records = papers[papers['姓名'] == st.session_state.selected]
            st.session_state.project_records = projects[projects['姓名'] == st.session_state.selected]

            # 查找与查询作者有关的人
            st.session_state.related_people = papers[
                (papers['研究机构'] == papers[papers['姓名'] == st.session_state.selected]['研究机构'].iloc[0]) |
                (papers['研究方向'] == papers[papers['姓名'] == st.session_state.selected]['研究方向'].iloc[0]) |
                (papers['不端内容'] == papers[papers['姓名'] == st.session_state.selected]['不端内容'].iloc[0])
            ]['姓名'].unique()
            st.session_state.related_people = [person for person in st.session_state.related_people if
                                               person != st.session_state.selected]

        elif not st.session_state.search_name and st.session_state.search_institution:
            st.warning("不支持这种检索，请输入研究人员姓名进行查询。")
            st.session_state.search_button_clicked = False
            st.stop()

    if st.session_state.search_button_clicked:
        # 研究人员基本信息卡片（位于搜索结果区域）
        with st.container(border=True):
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            # 基础信息布局（优化为4列，更紧凑）
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"<h3>{st.session_state.selected}</h3>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="info-item"><label>研究方向：</label><value>{}</value></div>'.format(
                    st.session_state.paper_records['研究方向'].iloc[0] if not st.session_state.paper_records.empty else "—"
                ), unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="info-item"><label>研究机构：</label><value>{}</value></div>'.format(
                    st.session_state.paper_records['研究机构'].iloc[0] if not st.session_state.paper_records.empty else "—"
                ), unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="info-item"><label>职称：</label><value>{}</value></div>'.format(
                    st.session_state.paper_records['职称'].iloc[0] if not st.session_state.paper_records.empty else "—"
                ), unsafe_allow_html=True)

            # 第二行信息（新增性别、居住地、ORCID）
            col5, col6, col7 = st.columns(3)
            with col5:
                st.markdown('<div class="info-item"><label>性别：</label><value>{}</value></div>'.format(
                    st.session_state.paper_records['性别'].iloc[0] if not st.session_state.paper_records.empty else "—"
                ), unsafe_allow_html=True)

            with col6:
                st.markdown('<div class="info-item"><label>居住地：</label><value>{}</value></div>'.format(
                    st.session_state.paper_records['居住地'].iloc[0] if not st.session_state.paper_records.empty else "—"
                ), unsafe_allow_html=True)

            with col7:
                st.markdown('<div class="info-item"><label>ORCID：</label><value>{}</value></div>'.format(
                    st.session_state.paper_records['ORCID'].iloc[0] if not st.session_state.paper_records.empty else "—"
                ), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        # 数据表格区域（论文记录）
        with st.container():
            st.subheader("📄 论文记录")
        if not st.session_state.paper_records.empty:
            # 添加竖向滚动条
            st.session_state.paper_records = st.session_state.paper_records.fillna("—")  # 将NaN替换为“—”
            st.markdown(
                """
                <style>
                .scrollable-table {
                    max-height: 300px;  /* 设置最大高度 */
                    overflow-y: auto;   /* 添加竖向滚动条 */
                    display: block;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # 将 DataFrame 转换为 HTML，并添加滚动条样式
            st.markdown(
                f'<div class="scrollable-table">{st.session_state.paper_records.to_html(escape=False, index=False)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("暂无论文不端记录",icon="ℹ️")

        # 项目记录表格
        with st.container():
            st.subheader("📋 项目记录")
        if not st.session_state.project_records.empty:
            st.markdown(
                f'<div class="scrollable-table">{st.session_state.project_records.to_html(escape=False, index=False, col_space=50)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("暂无项目不端记录", icon="ℹ️")


        with st.container(border=True):
            st.markdown("<h3>核心指标</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("风险值", f"{st.session_state.author_risk:.2f}", delta_color="inverse")
            with col2:
                # 修正 if-elif-else 语法
                if st.session_state.author_risk > 9:
                    risk_level = "high"
                elif 8 < st.session_state.author_risk < 9:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                # 根据风险等级设置显示文本
                display_text = '⚠️ 高风险' if risk_level == 'high' else '❗ 中风险' if risk_level == 'medium' else '✅ 低风险'
                st.metric(
                    label=risk_level,
                    value=display_text,
                    help="通讯作者和第一作者为主要责任人",
                    label_visibility="collapsed"
                )
  

        # 大模型评价区域
        with st.container(border=True):
            st.subheader("📝 智谱大模型评价")
            executor = ThreadPoolExecutor(max_workers=2)
            def async_evaluation(selected, paper_records, project_records, related_people):
                return get_zhipu_evaluation(
                    selected,
                    paper_records,
                    project_records,
                    related_people
                )
            # 创建两列布局放置按钮
            btn_col1, btn_col2 = st.columns(2, gap="small")

            with btn_col1:  # 大模型评价按钮列
                if st.session_state.search_button_clicked and st.session_state.selected:
                    if st.button(f"📝 获取 {st.session_state.selected} 的大模型评价", 
                                type="secondary", 
                                use_container_width=True):  # 使用容器宽度（列宽50%）
                        future = executor.submit(
                            async_evaluation,
                            st.session_state.selected,
                            st.session_state.paper_records,
                            st.session_state.project_records,
                            st.session_state.related_people
                        )
                        with st.spinner("正在调用智谱大模型进行评价..."):
                            st.session_state.evaluation = future.result()
            if st.session_state.evaluation is not None:
                st.info(st.session_state.evaluation, icon="💡")       
        with st.container(border=True):
            executor = ThreadPoolExecutor(max_workers=2)
            # 创建两列布局放置按钮
            btn_col1, btn_col2 = st.columns(2, gap="small")
            with btn_col1:
                if st.session_state.search_button_clicked and st.session_state.selected:
                    if st.button("🕸️ 查看合作关系网络", 
                                type="secondary", 
                                use_container_width=True):
                        def build_network_graph(author):
                            G = nx.Graph()
                            G.add_node(author)

                            # 查找与查询作者有共同研究机构、研究方向或不端内容的作者
                            related = papers[
                                (papers['研究机构'] == papers[papers['姓名'] == author]['研究机构'].iloc[0]) |
                                (papers['研究方向'] == papers[papers['姓名'] == author]['研究方向'].iloc[0]) |
                                (papers['不端内容'] == papers[papers['姓名'] == author]['不端内容'].iloc[0])
                            ]['姓名'].unique()

                            for person in related:
                                if person != author:
                                    reason = ''
                                    if papers[(papers['姓名'] == author) & (papers['研究机构'] == papers[
                                        papers['姓名'] == person]['研究机构'].iloc[0])].shape[0] > 0:
                                        reason = '研究机构相同'
                                    elif papers[(papers['姓名'] == author) & (papers['研究方向'] == papers[
                                        papers['姓名'] == person]['研究方向'].iloc[0])].shape[0] > 0:
                                        reason = '研究方向相似'
                                    else:
                                        reason = '不端内容相关'
                                    G.add_node(person)
                                    G.add_edge(author, person, label=reason)

                            # 使用 plotly 绘制网络图
                            pos = nx.spring_layout(G, k=0.5)  # 布局
                            edge_trace = []
                            edge_annotations = []  # 用于存储边的标注信息
                            for edge in G.edges(data=True):
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_trace.append(go.Scatter(
                                    x=[x0, x1, None], y=[y0, y1, None],
                                    line=dict(width=0.5, color='#888'),
                                    hoverinfo='text',
                                    mode='lines'
                                ))

                                # 计算边的中点位置，用于放置标注文字
                                mid_x = (x0 + x1) / 2
                                mid_y = (y0 + y1) / 2
                                edge_annotations.append(
                                    dict(
                                        x=mid_x,
                                        y=mid_y,
                                        xref='x',
                                        yref='y',
                                        text=edge[2]['label'],  # 相连的原因作为标注文字
                                        showarrow=False,
                                        font=dict(size=10, color='black')
                                    )
                                )

                            node_trace = go.Scatter(
                                x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                                marker=dict(
                                    showscale=True,
                                    colorscale='YlGnBu',
                                    size=10,
                                )
                            )
                            for node in G.nodes():
                                x, y = pos[node]
                                node_trace['x'] += tuple([x])
                                node_trace['y'] += tuple([y])
                                node_trace['text'] += tuple([node])

                            fig = go.Figure(
                                data=edge_trace + [node_trace],
                                layout=go.Layout(
                                    title='<br>合作关系网络图',
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    annotations=edge_annotations  # 添加边的标注信息
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        build_network_graph(st.session_state.selected)

        with st.container(border=True):
            import io
            from openpyxl import Workbook
            from openpyxl.utils.dataframe import dataframe_to_rows

            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')

            # 写入基本信息
            if not st.session_state.paper_records.empty:
                basic_cols = ['姓名', '研究机构', '研究方向', '职称', '性别', '居住地', 'ORCID']
                basic_info = st.session_state.paper_records[basic_cols].iloc[0:1]  # 取首行基本信息
                basic_info.to_excel(writer, sheet_name='基本信息', index=False)

            # 写入论文记录
            if not st.session_state.paper_records.empty:
                st.session_state.paper_records.to_excel(writer, sheet_name='论文记录', index=False)

            # 写入项目记录
            if not st.session_state.project_records.empty:
                st.session_state.project_records.to_excel(writer, sheet_name='项目记录', index=False)

            # 写入大模型评价（新增部分）
            if st.session_state.evaluation is not None:
                eval_df = pd.DataFrame({'大模型评价内容': [st.session_state.evaluation]})  # 转换为DataFrame
                eval_df.to_excel(writer, sheet_name='大模型评价', index=False)  # 写入新Sheet

            writer.save()
            output.seek(0)

            st.download_button(
                label="📥 下载查询结果",
                data=output,
                file_name=f"{st.session_state.selected}_查询结果.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

# 页脚
st.markdown("""
<div class="footer">
    <p>© 2025 科研人员学术诚信风险预警平台</p>
    <p>开发人员：马丹薇　指导老师：宋培彦 </p>
    <p>课题组网址：https://trustkos.github.io/  导师邮箱：songpy@tjnu.edu.cn</p>
    <p>内容仅供参考</p>
</div>
""", unsafe_allow_html=True)