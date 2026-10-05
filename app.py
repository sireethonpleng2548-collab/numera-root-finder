import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify, diff, latex

st.set_page_config(page_title="Numera Root Finder", layout="wide")
st.title("🔢 Numera Root Finder")
st.caption("Bisection | Fixed-Point | Newton-Raphson — Input • Calculation • Output")

x = symbols('x')

# ---------- Sidebar Input ----------
def append_text(key, t):
    st.session_state[key] = (st.session_state.get(key, "") + t)

with st.sidebar:
    st.header("Input")
    method = st.selectbox("Method", ["Bisection", "Fixed-Point", "Newton-Raphson"])
    if "f_input" not in st.session_state:
        st.session_state["f_input"] = "x**3 - x - 2"
    st.text_input("f(x) =", key="f_input")
    f_str = st.session_state["f_input"]

    # Calculator-like buttons to reduce typing errors (for f)
    st.write("⌨️ แป้นเครื่องคิดเลข f(x):")
    btns = ["+", "-", "*", "/", "**", "(", ")", "x", "sin(x)", "cos(x)", "exp(x)", "sqrt(x)"]
    cols = st.columns(4)
    for i, b in enumerate(btns):
        cols[i % 4].button(b, key=f"btn_f_{b}", on_click=append_text, args=("f_input", b))

    decimals = st.slider("Decimals shown", 2, 10, 6)
    tol = 10**(-decimals)
    max_iter = st.number_input("Max iterations", 1, 200, 50)
    if method == "Bisection":
        a = st.number_input("a", value=1.0)
        b = st.number_input("b", value=2.0)
    elif method == "Fixed-Point":
        if "g_input" not in st.session_state:
            st.session_state["g_input"] = "(x + 2)**(1/3)"
        st.text_input("g(x) =", key="g_input")
        g_str = st.session_state["g_input"]
        st.write("⌨️ แป้นเครื่องคิดเลข g(x):")
        gcols = st.columns(4)
        for i, b in enumerate(btns):
            gcols[i % 4].button(b, key=f"btn_g_{b}", on_click=append_text, args=("g_input", b))
        x0 = st.number_input("x0", value=1.0)
    else:
        x0 = st.number_input("x0", value=1.5)
    run = st.button("Calculate ▶️")

# ---------- Parse functions ----------
try:
    f_expr = sympify(f_str)
    f = lambdify(x, f_expr, "numpy")
    fp = lambdify(x, diff(f_expr, x), "numpy")
except Exception as e:
    # Friendly check like scientific calculator
    msg = str(e)
    if "x" in f_str and "*" not in f_str:
        st.warning("💡 ลืม `*` รึเปล่า? เช่น พิมพ์ `2x` ต้องเป็น `2*x`")
    st.error(f"f(x) ผิด: {e}")
    st.stop()

# LaTeX preview like calculator display
st.latex(f"f(x) = {latex(f_expr)}")

if run:
    rows = []
    try:
        if method == "Bisection":
            if f(a)*f(b) > 0:
                st.error("f(a) และ f(b) ต้องคนละเครื่องหมาย")
                st.stop()
            converged = False
            for i in range(1, max_iter+1):
                c = (a+b)/2
                err = abs(b-a)/2
                rows.append([i, round(a,decimals), round(b,decimals), round(c,decimals), round(float(f(c)),decimals), round(err,decimals)])
                if err < tol or f(c) == 0:
                    converged = True
                    break
                if f(a)*f(c) < 0: b = c
                else: a = c
            cols = ["iter","a","b","c","f(c)","error"]
        elif method == "Fixed-Point":
            g = lambdify(x, sympify(g_str), "numpy")
            xi = x0
            converged = False
            for i in range(1, max_iter+1):
                xnew = float(g(xi))
                err = abs(xnew-xi)
                rows.append([i, round(xi,decimals), round(xnew,decimals), round(float(f(xnew)),decimals), round(err,decimals)])
                if err < tol:
                    converged = True
                    break
                xi = xnew
            cols = ["iter","x_old","x_new","f(x_new)","error"]
        else:
            xi = x0
            converged = False
            for i in range(1, max_iter+1):
                d = float(fp(xi))
                if d == 0:
                    st.error("f'(x) = 0 หารไม่ได้")
                    break
                xnew = xi - float(f(xi))/d
                err = abs(xnew-xi)
                rows.append([i, round(xi,decimals), round(xnew,decimals), round(float(f(xnew)),decimals), round(err,decimals)])
                if err < tol:
                    converged = True
                    break
                xi = xnew
            cols = ["iter","x_old","x_new","f(x_new)","error"]

        df = pd.DataFrame(rows, columns=cols)
        st.subheader("Output — ตารางผลลัพธ์")
        st.dataframe(df, use_container_width=True)
        if converged:
            st.success(f"ราก ≈ {df.iloc[-1,2] if method=='Bisection' else df.iloc[-1,2]} (หยุดที่ tol=1e-{decimals})")
        else:
            st.warning(f"⚠️ ไม่ลู่เข้าภายใน {max_iter} รอบ (error สุดท้าย = {df.iloc[-1,-1]})")
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "result.csv", "text/csv")

        # Error vs iteration graph
        st.subheader("Convergence — error vs iteration")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["iter"], df["error"], marker="o")
        ax2.set_yscale("log")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("error")
        ax2.grid(True)
        st.pyplot(fig2)

        # Graph auto-scale around root
        root = float(df.iloc[-1,2])
        span = max(2.0, abs(root)*0.8 + 2)
        xs = np.linspace(root-span, root+span, 400)
        ys = f(xs)
        lo, hi = np.percentile(ys[np.isfinite(ys)], [2, 98])
        pad = max(1.0, (hi-lo)*0.2)
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=f"f(x)={f_str}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(root, color="red", linestyle="--", label=f"root≈{root:.{decimals}f}")
        ax.set_ylim(lo-pad, hi+pad)
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"คำนวณไม่ได้: {e}")
else:
    st.info("ตั้งค่า Input ด้านซ้ายแล้วกด Calculate")
