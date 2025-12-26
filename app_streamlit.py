import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================
st.set_page_config(

    page_icon="📊",
    layout="wide"
)

st.title("🌐 МОДЕЛЬ РАСПРОСТРАНЕНИЯ КОНКУРИРУЮЩИХ ИНФОРМАЦИОННЫХ ПОТОКОВ")
st.markdown("---")

# =====================================================
# SESSION STATE (чтобы кнопки РАБОТАЛИ)
# =====================================================
def init_state():
    defaults = dict(
        model_type="basic",
        N=1000,
        time=100,
        beta1=0.3,
        gamma1=0.1,
        beta2=0.4,
        gamma2=0.08,
        I1_0=3,
        I2_0=1,
        model_param=0.2
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =====================================================
# МОДЕЛИ
# =====================================================
def basic_model(state, t, beta1, gamma1, beta2, gamma2, N, c):
    S, I1, R1, I2, R2 = state
    b1 = beta1 * (1 - c * I2 / (I1 + I2 + 1e-9))
    b2 = beta2 * (1 - c * I1 / (I1 + I2 + 1e-9))
    return [
        -b1*S*I1/N - b2*S*I2/N,
        b1*S*I1/N - gamma1*I1,
        gamma1*I1,
        b2*S*I2/N - gamma2*I2,
        gamma2*I2
    ]

def reinforced_model(state, t, beta1, gamma1, beta2, gamma2, N, r):
    S, I1, R1, I2, R2 = state
    b1 = beta1 * (1 + r*(I1+R1)/N)
    b2 = beta2 * (1 + r*(I2+R2)/N)
    return [
        -b1*S*I1/N - b2*S*I2/N,
        b1*S*I1/N - gamma1*I1,
        gamma1*I1,
        b2*S*I2/N - gamma2*I2,
        gamma2*I2
    ]

def forget_model(state, t, beta1, gamma1, beta2, gamma2, N, m):
    S, I1, R1, I2, R2 = state
    return [
        -beta1*S*I1/N - beta2*S*I2/N + m*(R1+R2),
        beta1*S*I1/N - gamma1*I1 + 0.3*m*R2,
        gamma1*I1 - m*R1,
        beta2*S*I2/N - gamma2*I2 + 0.3*m*R1,
        gamma2*I2 - m*R2
    ]

# =====================================================
# РЕШЕНИЕ СИСТЕМЫ
# =====================================================
def solve():
    t = np.linspace(0, st.session_state.time, st.session_state.time*2+1)
    S0 = st.session_state.N - st.session_state.I1_0 - st.session_state.I2_0
    y0 = [S0, st.session_state.I1_0, 0, st.session_state.I2_0, 0]

    model = {
        "basic": basic_model,
        "reinforced": reinforced_model,
        "forget": forget_model
    }[st.session_state.model_type]

    sol = odeint(
        model, y0, t,
        args=(
            st.session_state.beta1,
            st.session_state.gamma1,
            st.session_state.beta2,
            st.session_state.gamma2,
            st.session_state.N,
            st.session_state.model_param
        )
    )
    return t, sol.T

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("⚙️ ПАРАМЕТРЫ")

    st.session_state.model_type = st.selectbox(
        "Модель",
        ["basic", "reinforced", "forget"],
        format_func=lambda x:{
            "basic":"🎯 Конкуренция",
            "reinforced":"📈 Подкрепление",
            "forget":"🔄 Забывание"
        }[x]
    )

    st.session_state.N = st.slider("Население",100,5000,st.session_state.N,100)
    st.session_state.time = st.slider("Время",50,300,st.session_state.time,10)

    st.subheader("🔵 ПРАВДА")
    st.session_state.beta1 = st.slider("β₁",0.01,1.0,st.session_state.beta1,0.01)
    st.session_state.gamma1 = st.slider("γ₁",0.01,0.5,st.session_state.gamma1,0.01)
    st.session_state.I1_0 = st.slider("I₁₀",1,50,st.session_state.I1_0)

    st.subheader("🔴 СЛУХ")
    st.session_state.beta2 = st.slider("β₂",0.01,1.0,st.session_state.beta2,0.01)
    st.session_state.gamma2 = st.slider("γ₂",0.01,0.5,st.session_state.gamma2,0.01)
    st.session_state.I2_0 = st.slider("I₂₀",1,50,st.session_state.I2_0)

    st.session_state.model_param = st.slider("Параметр модели",0.0,0.5,st.session_state.model_param,0.01)

    st.subheader("🎮 СЦЕНАРИИ")
    if st.button("✅ Правда побеждает"):
        st.session_state.update(
            beta1=0.45,beta2=0.25,
            gamma1=0.08,gamma2=0.15,
            I1_0=6,I2_0=1,
            model_type="basic",model_param=0.3
        )

    if st.button("❌ Слух побеждает"):
        st.session_state.update(
            beta1=0.25,beta2=0.5,
            gamma1=0.15,gamma2=0.05,
            I1_0=2,I2_0=4,
            model_type="reinforced",model_param=0.25
        )

    if st.button("⚖️ Баланс сил"):
        st.session_state.update(
            beta1=0.35,beta2=0.35,
            gamma1=0.1,gamma2=0.1,
            I1_0=3,I2_0=3,
            model_type="forget",model_param=0.04
        )

    if st.button("🔄 Сброс"):
        st.session_state.clear()
        init_state()

# =====================================================
# РАСЧЁТ
# =====================================================
t, (S,I1,R1,I2,R2) = solve()

total_truth = np.max(I1+R1)
total_rumor = np.max(I2+R2)
truth_share = total_truth/(total_truth+total_rumor+1e-9)*100

c1,c2,c3,c4 = st.columns(4)
c1.metric("📈 Пик правды",int(np.max(I1)))
c2.metric("📉 Пик слуха",int(np.max(I2)))
c3.metric("🎯 Доля правды",f"{truth_share:.1f}%")
c4.metric("🏆 Победитель","ПРАВДА" if truth_share>50 else "СЛУХ")

st.markdown("---")

# =====================================================
# ВСЕ ГРАФИКИ + 3D
# =====================================================
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type":"xy"},{"type":"xy"}],
           [{"type":"scene"},{"type":"scene"}]],
    subplot_titles=[
        "📊 Динамика распространения",
        "📈 Сравнение потоков",
        "🌀 3D фазовая траектория",
        "🏔️ 3D поверхность влияния"
    ]
)

# 1 — динамика
fig.add_trace(go.Scatter(x=t,y=S,name="Не знают",line=dict(color="gray")),1,1)
fig.add_trace(go.Scatter(x=t,y=I1,name="Активная правда",line=dict(color="green")),1,1)
fig.add_trace(go.Scatter(x=t,y=R1,name="Знают правду",line=dict(color="lightgreen")),1,1)
fig.add_trace(go.Scatter(x=t,y=I2,name="Активный слух",line=dict(color="red")),1,1)
fig.add_trace(go.Scatter(x=t,y=R2,name="Знают слух",line=dict(color="lightcoral")),1,1)

# 2 — сравнение
fig.add_trace(go.Bar(
    x=["Всего","Пик","Доля"],
    y=[total_truth,np.max(I1),truth_share],
    name="Правда",marker_color="green"
),1,2)
fig.add_trace(go.Bar(
    x=["Всего","Пик","Доля"],
    y=[total_rumor,np.max(I2),100-truth_share],
    name="Слух",marker_color="red"
),1,2)

# 3 — фазовая траектория
idx = np.linspace(0,len(t)-1,100,dtype=int)
fig.add_trace(go.Scatter3d(
    x=I1[idx],y=I2[idx],z=t[idx],
    mode="lines+markers",
    marker=dict(size=3,color=t[idx],colorscale="Viridis"),
    line=dict(color="purple",width=3),
    name="Траектория"
),2,1)

# 4 — поверхность
X,Y = np.meshgrid(range(len(idx)),range(5))
Z = np.array([S[idx],I1[idx],R1[idx],I2[idx],R2[idx]])/st.session_state.N
fig.add_trace(go.Surface(z=Z,x=X,y=Y,colorscale="Viridis",opacity=0.85),2,2)

fig.update_layout(
    height=1000,
    legend=dict(orientation="h",y=-0.18,x=0.5,xanchor="center"),
    margin=dict(l=40,r=40,t=80,b=140)
)

st.plotly_chart(fig,use_container_width=True)

# =====================================================
# ОПИСАНИЕ МОДЕЛЕЙ
# =====================================================
with st.expander("📖 ОПИСАНИЕ МОДЕЛЕЙ"):
    st.markdown("""
### Категории
- **S** — не знают
- **I₁** — активно распространяют ПРАВДУ
- **R₁** — знают правду
- **I₂** — активно распространяют СЛУХ
- **R₂** — знают слух

### Модели
**🎯 Конкуренция**  
Информационные потоки подавляют друг друга.

**📈 Подкрепление**  
Популярная информация распространяется быстрее.

**🔄 Забывание**  
Люди могут менять мнение со временем.
""")

st.markdown("---")
st.markdown("**Запуск:** `streamlit run app_streamlit.py`")
