#%% [Cell 1] 라이브러리 로드 및 SAV 파일 변환 데이터 확인
import os
import platform
import pandas as pd
import numpy as np
import pyreadstat as pt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report 
from xgboost import XGBClassifier
import shap

# ---------------------------------------------------------
# 1. 웹 환경 구축 SaaS 연동을 위해 def로 묶기
# ---------------------------------------------------------

#X: selected_features | hba조건 추가: use_hba | edit_cate_list: 범주형 데이터 리스트(동적 구현)
def build_ma(sav_file,csv_file,selected_features,use_hba=False,edit_cate_list=[]):
    # 데이터 무결성 검사 및 변환 
    try:
        if not os.path.exists(csv_file):
            print("Meta_Assist: SAV 원본 데이터를 기반으로 의료 보안 가명 처리를 진행합니다...")
            sav_load, _ = pt.read_sav(sav_file)
            # 보안 폴더 자동 생성 후 저장
            os.makedirs('csv', exist_ok=True)
            sav_load.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print("Meta_Assist: 안전한 실무용 임베디드 데이터셋 변환 완료!")
    except Exception as e:
        print(f"데이터 연동 오류 발생: {e}")

    # 변환된 데이터 호출 및 모니터링 확인 (Pandas 디스플레이 강제 설정)
    df = pd.read_csv(csv_file)
    print("--- [의료진 모니터용 EMR 전자의무기록 상위 n개 데이터 추출] ---") #head에서 수정 가능(기본 출력 값: 5)
    pd.set_option('display.max_columns',None) #중간에 '...' 생략 방지
    pd.set_option('display.width',2000) #가로 길이 넓게 설정
    pd.set_option('display.expand_frame_repr',False) #화면이 좁아도 절대 밑으로 줄바꿈(X)
    pd.set_option('display.precision',1) #소수점 1자리까지만 출력
    print(df.head(4).to_string()) 

    # %% [Cell 2] 피처 분할 및 고유의 3단계 대사증후군 타겟 빌드
    # 대한민국 의학 표준 가이드라인 기반의 가상 3단계 분류 타겟(0:정상, 1:주의, 2:고위험) 설정
    # 실제 데이터의 컬럼 상황에 맞게 융통성 있게 작동하도록 안전 장치 가동

    # 대한비만학회 기준: 한국인 복부비만 기준 초과(남자 허리둘레 90cm 이상, 여자 85cm 이상)
    if 'WC' in df.columns and 'Gender' in df.columns:
        gender_wc = [
            (df['Gender'] == 1) & (df['WC'] >= 90), #남성
            (df['Gender'] == 2) & (df['WC'] >=85) #여성
        ]
        #대한비만학회는 주의 단계가 아닌 복부 비만 확진(고위험)을 절대적인 커트라인으로 지정
        info_wc = np.select(gender_wc,[1,1],default=0)
    else:
        info_wc = 0 # WC,Gender 컬럼이 없을 시 방어코드

    if 'target' not in df.columns:
        print("Meta_Assist: 대한민국 표준 진료 지침 기반 3단계 타겟 스코어링을 시작합니다...")
        ## 수동 설정 ##

        #사용자가 추가한 변수만 동적으로 타겟 스코어링에 포함되도록 방어 코드 설계
        danger_score = info_wc
        warning_score = 0

        # 대사증후군 진단 지표(공복혈당, 복부비만, 혈압 등) 조건문 시스템
        #1.고위험,주의(두 가지 범위를 같이 설계) 요소 카운트
        if 'GLU' in df.columns:
            danger_score += (df['GLU'] >= 126.0).astype(int) #당뇨 의심
            warning_score += (df['GLU'].between(100.0,125.9)).astype(int)
        if 'SBP' in df.columns:
            danger_score +=(df['SBP'] >= 140.0).astype(int) #고혈압 1기
            (df['SBP'].between(130.0, 139.9)).astype(int) #고혈압 전단계
        if 'DBP' in df.columns:
            danger_score +=(df['DBP'] >= 90.0).astype(int) #이완기 고혈압
            (df['DBP'].between(85.0, 89.9)).astype(int)  #이완기 주의혈압
        if 'BMI' in df.columns:
            danger_score +=(df['BMI'] >= 25.0).astype(int) #비만(WC 대체 가능)
            (df['BMI'].between(23.0, 24.9)).astype(int) #과체중
        
        # 데이터에 당화혈색소(HbA) 칼럼이 필요할 시 추가 기능 설계
        if use_hba and 'HbA' in df.columns:
            danger_score += (df['HbA'] >= 6.5).astype(int) #고위험군에만 조건을 추가

        #2.나머지 남은 조건 요소: 정상 카운트 

        #3.최종 타겟 부여
        conditions = [
            #class 2 (고위험): 심각한 수치 1개 OR 애매한 수치 3개 이상 겹칠 때(대사증후군 확진 기준)
            ((danger_score>=1) | ((danger_score+warning_score)>=3)),
            #class 1 (주의군): 애매한 수치가 1~2개 (생활습관 교정)
            (warning_score>=1)
        ]
        #순서에 맞춰서 2(고위험), 1(주의), 0(정상) 부여
        df['target'] = np.select(conditions,[2,1],default=0)

    ## 수동 설정 ##
    # 진료에 쓰일 핵심 독립변수(X)와 종속변수(y) 분리
    # 사용자가 직접 X에 넣을 변수를 선택할 수 있도록 selected_features 추가
    X = df[selected_features]
    y = df['target']

    # 근본 분할 비율 8:2 적용 및 무작위 고정 난수 42 매칭
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # %% [Cell 3] 데이터 누수 차단용 ColumnTransformer & 자동화 파이프라인 설계
    
    Master_cate_list = [
        'Gender', 'Smoke', 'Town', 'AgeGroup', 'Education', 
        'HP', 'DM', 'DSP', 'ALG', 'FamilyHistory', 'Drink'
    ]
    #총합 범주형
    final_cate_list = list(set(Master_cate_list+edit_cate_list))

    ## 수동 설정 ##
    # 연속형(수치) 변수와 범주형 변수를 분기하여 최적의 결측치 대체법 매칭
    categorical_features = [col for col in selected_features if col in final_cate_list] # 만약 성별이나 흡연 등의 컬럼이 포함될 시 여기에 수동 기입
    
    # 연속형(수치) 변수와 범주형 변수를 분기하여 최적의 결측치 대체법 매칭
    numeric_features = [col for col in selected_features if col not in final_cate_list]

    # 1. 수치형 데이터: 중앙값(median)으로 결측치를 메꾸고, 차원 조절용 표준화 스케일링 수행
    numeric_transformer = Pipeline(steps=[
        # 결측치 임의 채움->의료 사고
        # ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. 범주형 데이터: 최빈값(most_frequent)으로 메꾸고 원-핫 인코딩 적용
    categorical_transformer = Pipeline(steps=[
        # 결측치 임의 채움->의료 사고
        # ('imputer', SimpleImputer(strategy='most_frequent')),
        #모델이 학습을 할 때 새로운 카테고리가 기입됐을 경우 오류가 발생할 수 있기에 0으로 처리(오류 발생 방지)
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    #feature가 하나라도 비어있으면 Ctf가 에러 방지
    transformers = []
    if numeric_features:
        transformers.append(('num',numeric_transformer,numeric_features))
    if categorical_features:
        transformers.append(('cate',categorical_transformer,categorical_features))

    # 리스트에 있는 값들은 ColumnTransforme를 이용해 처리 
    preprocessor = ColumnTransformer(transformers=transformers)

    # %% [Cell 4] XGBoostClassifier 연동 및 GridSearchCV 최고의 조합 탐색 (시간 단축 핵심)
    # 전처리기와 메인 엔진인 XGBoost를 컨베이어 벨트로 묶기
    meta_assist_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #multi:softprob은 다중 결과를 출력할 때 사용하며 단순 결과를 출력하는게 아닌 각 카테고리 별 퍼센트 비율을 동시에 출력
        ('xgb', XGBClassifier(objective='multi:softprob', random_state=42, eval_metric='mlogloss'))
    ])

    # 최적의 속도와 정확도 밸런스를 찾는 하이퍼파라미터 그리드 격자 설정
    param_grid = {
        'xgb__n_estimators': [100, 200],  # 의사 오답 체크 릴레이 횟수
        'xgb__max_depth': [3, 5, 7],       # 나무의 종적 사고 깊이 수준
        'xgb__learning_rate': [0.05, 0.1]  # 기울기 미세조정 보폭
    }

    print('')
    print("Meta_Assist: 의료진의 분석 대기 시간 단축을 위한 오토 튜닝 엔진 가동...")
    #recall_macro 채점방식을 이용하여 누락된 값이 있을 시 누락되지 않은 값의 정확도가 높아도 누락된 값을 포함하여 결과를 도출(False Negative 방지)
    grid_search = GridSearchCV(meta_assist_pipe, param_grid, cv=5, scoring='recall_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 최고의 퍼포먼스를 보여주는 마스터 모델 추출
    print('')
    best_doctor_model = grid_search.best_estimator_
    print(f"최적의 하이퍼파라미터 조합: {grid_search.best_params_}")
    print('')
    print(f"내과/가정의학과 검증 데이터 최고 정확도: {grid_search.best_score_*100:.1f}%")

    # %% [Cell 5] 의료진 설득용 SHAP 데이터 설명 모델 엔진 구축
    print("\nMeta_Assist: 설명 가능한 AI (XAI) 시각화 엔진을 구축합니다...")

    # 학습, Meta-Assist 핵심 엔진
    X_train_transformed = best_doctor_model.named_steps['preprocessor'].transform(X_train)
    xgb_engine = best_doctor_model.named_steps['xgb'] 

    # 수치,범주 데이터 컬럼명 매칭
    feature_names = best_doctor_model.named_steps['preprocessor'].get_feature_names_out()

    # TreeExplainer를 통한 기여도 연산
    # 의사결정나무의 구조적 특성을 이용하여 연산 시간을 효과적으로 줄임
    explainer = shap.TreeExplainer(xgb_engine.get_booster()) #XGB내부를 못 읽기에 booster기법만 따로 뺴주기

    return best_doctor_model, explainer, feature_names, X_test, y_test

# ---------------------------------------------------------
# 2. 웹 연동용 SaaS 시각화 자료 '자동 해설' 함수 
# ---------------------------------------------------------

def private_patient(best_doctor_model,explainer,feature_names,patient_df):

    pred_class = best_doctor_model.predict(patient_df)[0] #0,1,2 중 하나 자동 추출

#1. 환자의 실전 데이터 전처리 (NaN값 냅두기, 스케일링/원핫만 진행)
    patient_preprocess = best_doctor_model.named_steps['preprocessor'].transform(patient_df)

#2. SHAP 값 계산 (환자 1명만!)
    shap_vals = explainer.shap_values(patient_preprocess)

#3. 위험군 설정 [0,1,2] (버전에 맞춰서 설정이 자동으로 변경됨) 인스턴스 수동 설정
    if isinstance(shap_vals,list):
        target_shap_val = shap_vals[pred_class][0,:] #구버전(리스트)일 경우, [n환자,:(모든 스탯)]
        base_value = explainer.expected_value[pred_class]
    else:
        target_shap_val = shap_vals[0,:,pred_class] #신버전(3D)일 경우 [전체환자,전체스탯,고위험군(2)]
        base_value = explainer.expected_value[pred_class]

    #4. NLG(자연어) 해설 argsort 로직 (n번째 환자가 왜 [0,1,2]인 지, 원인 컬럼 순위 매기기)
    ranked_col = np.argsort(target_shap_val)#argsort로 정렬, np.abs 절댓값 빼기
    top_col = ranked_col[-4:][::-1]#뒤에서 가장 큰 값(Defalut: ASC) 두개 출력 후 순서 뒤집기

    top1_feature = feature_names[top_col[0]] if (len(top_col)>0 and target_shap_val[top_col[0]]>0) else '위험 요인 없음(안전)'
    top2_feature = feature_names[top_col[1]] if (len(top_col)>1 and target_shap_val[top_col[1]]>0) else '1순위 외 유의미한 위험 인자 없음'

    print("SHAP 엔진 준비 완료.")

    return {
        'prediction': pred_class,
        'base_value': base_value,
        'shap_values': target_shap_val,
        'patient_transformed': patient_preprocess[0,:],
        'top_1':top1_feature,
        'top_2':top2_feature
    }
# ---------------------------------------------------------
# 3. 기존 터미널 스크립트 실행 구역
# ---------------------------------------------------------
if __name__ == '__main__':

# 파일 경로 설정 (사용자 경로에 맞춰서 설정sav)

    sav_file = 'sav'
    csv_file = 'csv'

#터미널에서 단독 실행시, 에러가 안 나도록 가짜 데이터 설정
    dummy_features = ['Gender', 'Age', 'SBP', 'DBP', 'WC', 'BMI', 'GLU']

#엔진 불러오기
    best_doctor_model, explainer, feature_names, X_test, y_test = build_ma(
        sav_file,csv_file,selected_features=dummy_features,use_hba=False,edit_cate_list=[])

# %% 의사용 최종 진단 레포트 및 오차 행렬 시각화 출력
    y_pred = best_doctor_model.predict(X_test)

    print("\n" + "="*60)
    print("   [META_ASSIST MEDICAL REPORT FOR CLINICIANS]   ")
    print("="*60)
    #딕셔너리 형태로 결과를 뽑아주기 위해 output_dict=True를 사용
    report_dict = classification_report(y_test, y_pred, target_names=['정상', '주의', '고위험'],output_dict=True)
    report_output=pd.DataFrame(report_dict).transpose()#변환
    report_output.rename(index={
        'accuracy': '정확도',
        'macro avg': '단순 평균', #각 그룹의 점수를 다 더해서 나타낸 평균
        'weighted avg': '가중 평균' #환자 수가 많은 그룹에 가중치를 더 줘서 계산한 현실적인 평균
    },
    columns={
        'precision': '정밀도',
        'recall': '재현율',
        'f1-score': '종합 균형 점수',
        'support': '평가 환자 수'
    },inplace=True)#기존 틀에서 수정
    pd.set_option('display.max_columns',None)
    pd.set_option('display.width',2000)
    pd.set_option('display.expand_frame_repr',False) 
    pd.set_option('display.unicode.east_asian_width',True) #서양언어보다 크기에 east_asian_width 추가
    print(report_output.round(2))
    print("="*60)

    # 파이프라인에서 전처리가 완료된 Train 데이터를 추출하여 SHAP에 주입
    # SHAP은 파이프라인 해석이 불가능하기에 named_steps(파이프라인에서 원하는 정보만 추출 가능)를 이용해 한 번 더 전처리 진행을 위한 코드 설계
    X_test_transformed = best_doctor_model.named_steps['preprocessor'].transform(X_test)
    shap_values = explainer.shap_values(X_test_transformed) #[0,1,2]

    #WINDOW,MAC OS 전부 사용 가능
    current_os=platform.system()

    if current_os == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif current_os == 'Darwin': # Apple
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False

    #요약,막대 플롯용 타겟 설정 (Defalut=2, 변수화 진행)
    visual_target_class=2
    if isinstance(shap_values, list):
        visual_target_shap = shap_values[visual_target_class]
    else:
        visual_target_shap = shap_values[:,:,visual_target_class] #환자,스탯,컨디션 범위 

    # 1. 요약 차트 (Summary Plot - 전체 환자 관점)
    plt.figure(figsize=(10, 6))
    plt.title("Meta_Assist: 전체 환자 대사증후군 위험 요인 (Summary)",fontsize=12)
    # 여기서 비로소 feature_names(이름표)를 씀, []은 0,1,2 클래스 설정 가능
    shap.summary_plot(visual_target_shap, X_test_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()

    # 2. 막대 그래프 (Bar Plot - 직관적인 영향력 순위)
    plt.figure(figsize=(10, 6))
    plt.title("Meta_Assist: 평균 위험도 기여도 순위 (Bar)",fontsize=12)
    shap.summary_plot(visual_target_shap, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()

    # 3. 개별 환자 맞춤형 포스 플롯 (Force Plot-주력)
    # 주피터/VS Code에서 인터랙티브(마우스 반응형)로 보려면 initjs() 필수
    shap.initjs()

    ##SaaS 환경 구축
    patient_idx = 10 # 웹에서는 이 값이 마우스 클릭값으로 설정
    test_patient_idx = X_test.iloc[[patient_idx]] 
    result = private_patient(best_doctor_model, explainer, feature_names, test_patient_idx)

    # AI 예측 클래스 명칭 변환용 리스트
    class_mapping = ['정상', '주의', '고위험']
    
    print(f"[개별환자 분석] {patient_idx} 환자 상세 진단 및 원인 분석")
    print(f"AI 자동 분석 결과: 이 환자는 [{class_mapping[result['prediction']]}]군으로 판정되었습니다.")
    print(f"판단 근거 리포트: 해당 컨디션을 유발한 가장 결정적인 컬럼은 '{result['top_1']}'와(과) '{result['top_2']}'입니다.")

    print(f"[개별 환자 분석] {patient_idx} 환자의 고위험군(High Risk) 판정 원인 분석")
    # 0번 환자의 고위험군(클래스 인덱스 2) 기여도 출력
    shap.force_plot(
        base_value=result['base_value'], 
        shap_values=result['shap_values'], #0번째 환자 정보만      
        features=np.round(result['patient_transformed'],2), #0번째 환자 정보만   
        feature_names=feature_names,
        out_names='AI 예측 위험도',
        matplotlib=True            
    )
    plt.tight_layout()
    plt.show()
    # 주의: force_plot은 plt.show()가 아니라 주피터 셀 결과창에 HTML 형태로 자동 출력됨!