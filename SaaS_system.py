import streamlit as st
import pandas as pd 
import shap 
import matplotlib.pyplot as plt 
import os 
import numpy as np
import platform 
import pyreadstat as pt
import matplotlib as mpl
#Module 불러오기
from Meta_Assist_Backend import build_ma, private_patient

#최소한의 UI,UX
st.set_page_config(layout='wide', page_title='Meta_Assist Dashboard')

#폰트 깨짐 방지, 화면에 그릴 때 적용
current_os=platform.system()
if current_os == 'Windows': # 윈도우
    plt.rcParams['font.family']='Malgun Gothic'
elif current_os == 'Darwin': # 맥
    plt.rcParams['font.family']='Arial Unicode MS'
else:
    plt.rcParams['font.family']='NanumGothic' # 이외에 os

plt.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.unicode_minus'] = False #마이너스 깨짐 방지

#메인
st.title('Meta_Assist: 대사증후군 AI 스크리닝 대시보드')
st.markdown('의료 업계 종사자를 위한 설명 가능한 AI (XAI) 기반 예측 시스템')
st.divider() #가로선 구분

# 왼쪾 사이드바 (컨트롤 패널)
with st.sidebar:
    st.header('환자 데이터 업로드')
    st.caption('파일 확장자가 sav 또는 csv 포맷의 EMR 데이터만 업로드가 가능')

    #사용자와 상호작용(sav,csv버튼)
    upload_file=st.file_uploader("",type=['sav','csv'])

    #파일 업로드가 완료될 시, 사용자가 직접 피처를 커스텀할 수 있는 기능 구현
    selected_features=[]
    use_hba=False
    edit_cate_list=[]
    patient_idx=0
    analyze_butn=False

    #환자 찾기
    if upload_file is not None:
        st.divider() #가로선 구분
        
        #분석 모델 설정(사용자 커스텀 UI)
        st.header('분석 모델 설정(사용자는 수정이 가능함)')

        #임시로 파일 분석, 컬럼명만 따로 뽑아오기 (메타데이터같은 무거운 데이터 제외)
        file_extension = upload_file.name.split('.')[-1] 
        if file_extension == 'sav': #.sav
            file_df,_=pt.read_sav(upload_file)
        else: #.csv
            file_df=pd.read_csv(upload_file)

        # 모델에 포함될 기본 컬럼 값과 전체 컬럼 리스트로 변환
        all_columns = file_df.columns.tolist() #다량의 컬럼을 리스트로 바꾸기 (for)
        default_cols = [col for col in ['Age', 'SBP', 'DBP', 'WC', 'BMI', 'GLU'] if col in all_columns ]
        
        #사용자 피처 다량 선택(구현)
        selected_features = st.multiselect('진단을 위한 수치형 항목 선택',options=all_columns,default=default_cols)
        
        #범주형 데이터 따로 구현
        st.markdown('고급 설정 (범주형 변수 예외 처리)')
        st.caption('모델이 인식하지 못하는 특수코드,분류형 변수를 추가')
        edit_cate_list=st.multiselect('범주형 항목 선택',options=all_columns)

        #결과 (당화혈색소 추가)
        if 'HbA' in all_columns:
            #수동 절차이기에 value=False (자동 구현 방지)
            use_hba=st.checkbox('고위험군(2) 환자가 나왔을 경우 당화혈색소 기준 추가 가능: HbA >= 6.5 반영',value=False)

            if use_hba and 'HbA' not in selected_features:
                selected_features.append('HbA')
        
        st.divider()
        st.header('개별 환자 분석')
        #사용자와 상호작용(직접 마우스를 올리고 내리거나, 입력 가능)
        patient_idx = st.number_input('분석할 환자의 인덱스 번호 입력: ', min_value=0, step=1, value=0)
        #type: 테마 강조색, use~:화면 너비에 맞게 버튼을 꽉 차게 늘려줌
        analyze_butn = st.button('AI 분석 시작', type='primary', use_container_width=True)

#메인 화면 (결과 출력)
if upload_file is not None:
    #1. 파일이 올라오면, 서버에서 처리할 수 있도록 임시 파일 저장
    #의사가 csv,sav를 올리든 확장자를 그대로 따서 저장[-1] 
    file_extension = upload_file.name.split('.')[-1] #EMR.csv->csv,
    file_name = f'uploaded.{file_extension}'#uploaded기본 파일명 설정, 확장자명 가져오기

    with open(file_name,'wb') as fn:
        #getbuffer() 메모리에 있는 데이터 덩어리의 주소를 직접 가르킴, 다른 곳을 복사,저장할 수 있도록 꺼내주는 역할
        fn.write(upload_file.getbuffer()) 

    #2. 업로드된 파일 확장자에 따라 Meta_Assist 엔진에 변수를 맞게 설정 
    with st.spinner('XGB 엔진이 데이터 학습 및 모델 최적화 중...'): #애니메이션
        if file_extension == 'sav':
            #sav->csv (백엔드로 selected,hba,cate 전부 보내기)
            model, explainer, f_name, X_test, y_test = build_ma(file_name, 'converted.csv', selected_features, use_hba, edit_cate_list)
        else:
            #csv파일을 올렸을 시 dummy.sav 가짜 이름 부여
            model, explainer, f_name, X_test, y_test = build_ma('dummy.sav', file_name,selected_features,use_hba,edit_cate_list)
    
    st.success('EMR 데이터 연동 및 XGB 엔진 빌드가 완료됨!')

    # AI 분석 시작 버튼 블록 생성
    if analyze_butn:
        st.divider()

        #사용자가 피처를 선택을 안 하고 분석을 하는 경우 방지
        if not selected_features:
            st.error('최소 1개 이상의 수치형 항목을 선택')

        # 에러 방지: 사용자가 입력한 번호가 전체 EMR 환자 수보다 많으면 경고
        elif patient_idx >= len(X_test): #파일이 아닌 이미 [0,1,2]가 부여된 test
            st.error(f'해당 번호의 환자는 존재하지 않음. (EMR 환자 수: {len(X_test)}명)')
        else:
            with st.spinner(f'{patient_idx}번 환자 데이터 정밀 분석 중...'):
                # 3.선택한 환자 데이터 한 줄 선택, 출력
                selected_p_df = X_test.iloc[[patient_idx]] #iloc을 사용해 idx 선택
                # 4.백엔드 함수 불러오기 (중요)
                result = private_patient(model, explainer, f_name, selected_p_df)

                #환자가 판정받은 진단 결과 인덱스 번호 저장
                pred_class_idx= result['prediction']

                #5. 진단 결과(브리핑)
                class_result=['정상(normal)', '주의(Warning)', '고위험(High Risk)']
                condition_result = class_result[pred_class_idx] #result(백엔드)에서 가져오기
                st.subheader(f'[{patient_idx}번 환자] 진단 결과')

                # # SaaS_system.py 내 그래프 코드 바로 위에 추가
                # st.write("--- 디버깅 데이터 확인 ---")
                # st.write(f"SHAP 값 길이: {len(result['shap_values'])}")
                # st.write(f"Base Value: {result['base_value']:.2f}")
                # st.write(np.round(result['shap_values'][:5],2)) # 앞의 5개만이라도 값이 있는지 확인

                st.divider()
                st.subheader(f'(AI 모델 기반) 현재 EMR 내 환자 주요 위험 요인 가이드 (Summary Plot)')
                
                #추가 기능: Clustering 기법을 이용해 EMR 전체 환자 수 분포 현황 나타내기
                st.caption('XGB 엔진에 연동된 EMR 내 환자들의 타겟(0,1,2) 분포 현황')

                val_counts = y_test.value_counts()
                #정상
                count_0=val_counts.get(0,0)
                #주의
                count_1=val_counts.get(1,0)
                #고위험
                count_2=val_counts.get(2,0)

                #UI에 지표 위젯으로 환자 수 띄우기
                col_a, col_b,col_c=st.columns(3)
                col_a.metric(label='정상 환자 수 ', value=f'{count_0}명')
                col_b.metric(label= '주의 환자 수', value=f'{count_1}명')
                col_c.metric(label='고위험 환자 수', value= f'{count_2}명')
                #html 태그 br를 직접 주입하여 원하는 만큼 강제로 빈 줄을 추가하는 명령어
                st.markdown("<br>", unsafe_allow_html=True)

                #Summary plot 출력
                fig=plt.figure(figsize=(10,6))

                #0,1,2 KMeans 스타일 군집화 출력을 위해 UI 추가
                st.markdown('각 환자 상태(정상/주의/고위험) 상세 위험 분포도')
                st.caption('아래 항목을 클릭하여 전체 EMR 내 환자들의 상태 확인이 가능함.')

                con1,con2,con3=st.tabs(['정상(0)','주의(1)','고위험(2)'])

                #preprocessor를 거친 X_test 설정
                X_test_transformed=model.named_steps['preprocessor'].transform(X_test)
                shap_vals_all=explainer.shap_values(X_test_transformed) # s

                #구버전(List), 신버전(3D) 호환용 추출함수
                def shap_target(class_idx):
                    if isinstance(shap_vals_all, list):
                        return shap_vals_all[class_idx]
                    else:
                        return shap_vals_all[:,:,class_idx]
                    
                # 각 항목 설계    
                with con1:
                    fig0=plt.figure(figsize=(10,6))
                    shap.summary_plot(shap_target(0),X_test_transformed,feature_names=f_name,show=False)
                    st.pyplot(fig0)
                    plt.clf()
                with con2:
                    fig1=plt.figure(figsize=(10,6))
                    shap.summary_plot(shap_target(1),X_test_transformed,feature_names=f_name,show=False)
                    st.pyplot(fig1)
                    plt.clf()
                with con3:
                    fig2=plt.figure(figsize=(10,6))
                    shap.summary_plot(shap_target(2),X_test_transformed,feature_names=f_name,show=False)
                    st.pyplot(fig2)
                    plt.clf()
                # with con4:
                #     fig_a=plt.figure(figsize=(10,6))
                #     shap.summary_plot(shap_vals_all,X_test_transformed,feature_names=f_name,show=False)
                #     st.pyplot(fig_a)
                #     plt.clf()

                #6. 개별 환자에 따른 bar plot (sum_plot이랑 분리)
                st.subheader(f'{patient_idx}번 환자 맞춤형 위험 기여도 (Bar Plot)')
                fig_bar = plt.figure(figsize=(10,4))
                shap.bar_plot(result['shap_values'],feature_names=f_name, show=False)
                st.pyplot(plt.gcf()) #Get Current Figure
                plt.clf() #Clear Current FIgure (데이터 꼬임 방지)
                # st.markdown('---') # 깔끔하게 나누기

                                #박스 UI
                col1, col2 = st.columns(2) # 1,2
                with col1:
                    st.info(f'AI 예측 결과: {condition_result}')
                with col2:
                    st.info(f"({patient_idx}번 환자 기준) 결정적인 위험 요인(Top2): {result['top_1']}, {result['top_2']}")
                st.markdown('---') #표시

                #7.SHAP Waterfall plot 시각화 띄우기
                st.subheader(f'{patient_idx}번 환자 위험 요인 차트 (Waterfall plot)')
                st.caption(
                    "- **[해당 환자의 최종 위험도 점수가 어떤 요인들로 인해 계산되었는지 보여주는 차트]** \n\n"
                    "- **[붉은색: 위험도를 높이는 요인 | 푸른색: 위험도를 낮추는 요인]** \n\n"
                    "- **[f(x) 상단 최종점: 모든 요인이 합산된 이 환자의 최종 모델 예측 점수]** \n\n"
                    "- **[E[f(x)] 하단 밑바닥: 현재 환자와 동일한 판정(정상,주의,고위험군)을 받은 상태 내에서 환자들의 평균 점수 (출발선)** "
                    )

                #Waterfall plot -shap.Explanation 객체 렌더링-> Streamlit에 설정 
                #(values,base_values,data,feature_names)
                exp = shap.Explanation(
                    values=result['shap_values'],
                    base_values=result['base_value'],
                    data=np.round(result['patient_transformed'],2),
                    feature_names=f_name
                )

                fig_chart = plt.figure(figsize=(10,6))
                shap.plots.waterfall(exp, show=False)
                st.pyplot(fig_chart) #그려진 그래프를 st에 주기 
                plt.clf() #그래프 초기화
    else:
        #파일 안 올렸을 시
        st.info('EMR 데이터 파일을 올려 분석을 할 수 있습니다.')