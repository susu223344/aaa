import shap
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

if 'model1' not in st.session_state:
    model1 = joblib.load('clf1.pkl')
    model2 = joblib.load('clf2.pkl')
    st.session_state["model1"] = model1
    st.session_state["model2"] = model2
else:
    model1 = st.session_state["model1"]
    model2 = st.session_state["model2"]
scaler1 = joblib.load("Scaler1.pkl")
scaler2 = joblib.load("Scaler2.pkl")
continuous_vars1 = ['年龄（岁）', '清除血肿含量（mL）', '术中出血量（mL）', '引流时间（d）', '术后入住ICU时间（d）']
continuous_vars2 = ['术前白蛋白（g/L）', '术前血糖（mmol/L）', '手术时间（min）', '清除血肿含量（mL）']
st.set_page_config(layout='wide')

st.write("<h1 style='text-align: center'>颅内血肿清除术后患者器官/腔隙感染和院内死亡的综合预测工具</h1>",
         unsafe_allow_html=True)
st.markdown('-----')
st.warning('本预测工具由滕州市中心人民医院感染管理科开发，旨在早期预测颅内血肿清除术后患者的器官/腔隙感染和院内死亡情况，改善患者预后')

dic1 = {
    '男': 1,
    '女': 0
}
dic2 = {
    '是': 1,
    '否': 0
}
with st.sidebar:
    st.markdown("# 输入变量值")
    st.markdown('-----')
    性别 = st.radio("性别", ["男", "女"])
    年龄 = st.text_input("年龄（岁）")
    入院诊断为高血压脑出血 = st.radio("入院诊断为高血压脑出血", ["是", "否"])
    术前白蛋白 = st.text_input("术前白蛋白（g/L）")
    术前血糖 = st.text_input("术前血糖（mmol/L）")
    手术入路 = st.radio("手术入路为额部、颞顶、翼点和枕后以外的部位", ["是", "否"])
    手术入路为颞顶 = st.radio("手术入路为颞顶", ["是", "否"])
    清除血肿含量 = st.text_input("清除血肿含量（mL）")
    手术时间 = st.text_input("手术时间（min）")
    术中出血量 = st.text_input("术中出血量（mL）")
    硬膜外引流 = st.radio("硬膜外引流", ["是", "否"])
    引流时间 = st.text_input("脑内血肿引流时间（d）")
    中心静脉插管 = st.radio("中心静脉插管", ["是", "否"])
    气管切开 = st.radio("气管切开", ["是", "否"])
    术后入住ICU时间 = st.text_input("术后入住ICU时间（d）")

    
    
    
    
    
    
    
if st.sidebar.button("预测"):
    with st.spinner("运算中, 请等待..."):
        st.header('1. 颅内血肿清除术后患者器官/腔隙感染预测')
        test_df = pd.DataFrame([dic1[性别],float(年龄), float(清除血肿含量), float(术中出血量),
                                dic2[硬膜外引流], float(引流时间), dic2[中心静脉插管],
                                dic2[手术入路], float(术后入住ICU时间)],
                               index=['性别', '年龄（岁）', '清除血肿含量（mL）', '术中出血量（mL）', '硬膜外引流', '引流时间（d）',
                                      '中心静脉插管', '手术入路（其他）', '术后入住ICU时间（d）']).T
        test_df[continuous_vars1] = scaler1.transform(test_df[continuous_vars1])
        explainer = shap.Explainer(model1)  # 创建解释器
        shap_ = explainer.shap_values(test_df)
        shap_values = explainer.shap_values(test_df)
        plt.rcParams['font.sans-serif'] = ['kaiti']
        plt.rcParams['axes.unicode_minus'] = False
        shap.decision_plot(explainer.expected_value[0], shap_values[1][0], test_df.columns)
        plt.tight_layout()
        plt.savefig('shap1.png', dpi=300)
        col1, col2, col3 = st.columns([2, 5, 3])        
        with col2:
            st.image('shap1.png')
        st.success("颅内血肿清除术后患者器官/腔隙感染概率: {:.3f}%".format(model1.predict_proba(test_df)[:, 1][0] * 100))  
        plt.close()           


  

        
        st.header('2. 颅内血肿清除术后患者院内死亡概率预测')
        test_df2 = pd.DataFrame([float(术前白蛋白),float(术前血糖), float(手术时间), float(清除血肿含量),
                                dic2[中心静脉插管], dic2[气管切开],
                                dic2[入院诊断为高血压脑出血], dic2[手术入路为颞顶]],
                               index=['术前白蛋白（g/L）', '术前血糖（mmol/L）', '手术时间（min）', '清除血肿含量（mL）', '中心静脉插管', '气管切开',
                                      '入院诊断（高血压脑出血）', '手术入路（颞顶）']).T
        test_df2[continuous_vars2] = scaler2.transform(test_df2[continuous_vars2])
        explainer = shap.Explainer(model2)  # 创建解释器
        shap_ = explainer.shap_values(test_df2)
        shap_values2 = explainer.shap_values(test_df2)
        plt.rcParams['font.sans-serif'] = ['kaiti']
        plt.rcParams['axes.unicode_minus'] = False
        shap.decision_plot(explainer.expected_value[0], shap_values2[1][0], test_df2.columns)

        plt.tight_layout()
        plt.savefig('shap2.png', dpi=300)
        col4, col5, col6 = st.columns([2, 5, 3])
        with col5:
            st.image('shap2.png')
        st.success("颅内血肿清除术后患者院内死亡概率: {:.3f}%".format(model2.predict_proba(test_df2)[:, 1][0] * 100)) 
        
        
        
        
        
        
        
