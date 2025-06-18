
# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.graphics.mosaicplot import mosaic
import datetime
import streamlit as st
import numpy as np

@st.cache_data
# 데이터 로드 함수 정의
# def load_data(dataset_name):
#     df = sns.load_dataset(dataset_name)
#     return df
def load_data(dataset_name, uploaded_file, data_ready):
    # 직접 업로드하는 경우
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8', index_col=0)
            except UnicodeDecodeError:
                st.error("해당 파일을 불러올 수 없습니다. UTF-8로 변환해주세요.")
                return None
        else:
            st.warning("csv 파일만 업로드 가능합니다. ")
        return df
    # 시본 데이터 사용하는 경우
    elif dataset_name:
        try:
            df = sns.load_dataset(dataset_name)
            return df
        except ValueError:
            st.error("⚠ 데이터셋 이름을 다시 확인해주세요!")
    # 깃허브 주소에서 가져오는 경우
    elif data_ready:
        df = pd.read_csv(f"https://raw.githubusercontent.com/Surihub/stat_edu/main/data/{dataset_name}.csv", index_col=0)
        return df
        # try:
        # df = sns.load_dataset(dataset_name)
        # except ValueError:
        #     st.error("⚠ 데이터셋 이름을 다시 확인해주세요!")

# @st.cache_data
# def select_columns(df):
    



# def load_data(dataset_name, uploaded_file):
#     if dataset_name:
#         try:
#             df = sns.load_dataset(dataset_name)
#             return df
#         except ValueError:
#             st.error("⚠ 데이터셋 이름을 다시 확인해주세요!")
#     elif uploaded_file:
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         else:
#             st.warning("csv 파일만 업로드 가능합니다. ")
#         return df

@st.cache_data
def summarize(df):
    # 기초 통계량 요약 함수
    summ = df.describe()
    summ = np.round(summ, 2)

    summ.loc['분산'] = np.round(df.var(), 2)
    modes = df.mode().dropna()  # 최빈값을 계산하고 결측값 제거
    mode_str = ', '.join(modes.astype(str))  # 모든 최빈값을 문자열로 변환하고 쉼표로 연결
    summ.loc['최빈값'] = mode_str  # 문자열로 변환된 최빈값을 할당
    summ.index = ['개수', '평균', '표준편차', '최솟값', '제1사분위수', '중앙값', '제3사분위수', '최댓값', '분산', '최빈값']
    return summ


@st.cache_data
def _base_summary(df_num: pd.DataFrame) -> pd.DataFrame:
    """수치형 DataFrame → 기초 통계량(+분산·최빈값)"""
    summ = df_num.describe().T.round(2)
    summ["분산"]   = df_num.var().round(2)
    modes = df_num.mode().iloc[0].astype(str)
    summ["최빈값"] = modes
    summ = summ[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "분산", "최빈값"]
    ]
    summ.columns = [
        "개수", "평균", "표준편차", "최솟값", "제1사분위수",
        "중앙값", "제3사분위수", "최댓값", "분산", "최빈값"
    ]
    return summ
# 수정된 summarize: 그룹별(또는 전체) 통계량을 반환, 행은 그룹(전체 혹은 그룹명)·변수, 열은 통계량
# 그룹별 통계량 및 전체 통계량 반환 함수
from typing import Optional, Union
import pandas as pd
import numpy as np
@st.cache_data

# def summarize(df: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
#     num_cols = df.select_dtypes(include="number").columns.tolist()
#     if not num_cols:
#         return pd.DataFrame()
#     stats = ["count","mean","std","min","25%","50%","75%","max"]
#     labels = ["개수","평균","표준편차","최솟값","제1사분위수","중앙값","제3사분위수","최댓값","분산","최빈값"]
#     # 전체 통계량
#     if by is None:
#         desc = df[num_cols].describe().loc[stats].round(2)
#         desc.loc["var"]  = df[num_cols].var(ddof=1).round(2)
#         desc.loc["mode"] = df[num_cols].mode().iloc[0].astype(str)
#         desc.index = labels
#         return desc.T
#     # 그룹별 통계량
#     rows = []
#     for grp, sub in df.groupby(by):
#         desc = sub[num_cols].describe().loc[stats].round(2)
#         desc.loc["var"] = sub[num_cols].var(ddof=1).round(2)
#         modes = sub[num_cols].mode()
#         desc.loc["mode"] = modes.iloc[0].astype(str) if not modes.empty else np.nan
#         desc.index = labels
#         df_t = desc.T
#         df_t.insert(0, by, grp)
#         df_t.insert(1, "변수", df_t.index)
#         df_t = df_t.set_index([by, "변수"])
#         rows.append(df_t)
#     result = pd.concat(rows)
#     return result[labels]

def _freq_table(s: pd.Series) -> pd.DataFrame:
    """단일 범주형 시리즈 → 빈도·비율(%) 테이블"""
    vc  = s.value_counts(dropna=False)
    pct = (vc / len(s) * 100).round(2)
    tbl = pd.DataFrame({"빈도": vc, "비율(%)": pct})
    tbl.index.name = s.name
    return tbl

def summarize(df: pd.DataFrame, by: Optional[str] = None) -> Union[pd.DataFrame, dict]:
    """
    ▶ 수치형만 있을 때     : 기존 기초 통계량 반환
    ▶ 범주형만 있을 때     : 빈도·비율 테이블 반환 (단일 변수는 DataFrame, 다중은 dict)
    ▶ 수치+범주 혼합 선택 : {"numeric": DataFrame, "categorical": dict} 형태

    by : 그룹 변수 지정 시 그룹별 결과
         - 수치형 → (그룹, 변수)  MultiIndex 행 · 통계량 열
         - 범주형 → {변수: DataFrame(행=그룹, 열=범주값+%)}
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # ────────────────────────────────────────────
    # 1) 수치형 통계량 (기존 로직 유지)
    # ────────────────────────────────────────────
    def numeric_summary(data: pd.DataFrame, group: Optional[str]):
        stats  = ["count","mean","std","min","25%","50%","75%","max"]
        labels = ["개수","평균","표준편차","최솟값","제1사분위수","중앙값","제3사분위수","최댓값","분산","최빈값"]
        if group is None:
            desc = data.describe().loc[stats].round(2)
            desc.loc["var"]  = data.var(ddof=1).round(2)
            desc.loc["mode"] = data.mode().iloc[0].astype(str)
            desc.index = labels
            return desc.T
        rows = []
        for g, sub in data.groupby(group):
            d = sub.describe().loc[stats].round(2)
            # 수정 후 (숫자형 컬럼만 선택)
            numeric_cols = sub.select_dtypes(include='number')
            d.loc["var"] = numeric_cols.var(ddof=1).round(2)
            modes = sub.mode()
            d.loc["mode"] = modes.iloc[0].astype(str) if not modes.empty else np.nan
            d.index = labels
            t = d.T
            t.insert(0, group, g)
            t.insert(1, "변수", t.index)
            t = t.set_index([group, "변수"])
            rows.append(t)
        out = pd.concat(rows)
        return out[labels]

    # ────────────────────────────────────────────
    # 2) 범주형 빈도표
    # ────────────────────────────────────────────
    def categorical_summary(data: pd.DataFrame, group: Optional[str]):
        if group is None:
            tbls = {col: _freq_table(data[col]) for col in data.columns}
            return tbls[col] if len(tbls)==1 else tbls
        # 그룹별: 각 그룹마다 value_counts → DataFrame(행=그룹, 열=범주/비율)
        result = {}
        for col in data.columns:
            ct = (
                data.groupby(group)[col]
                .value_counts(dropna=False)
                .unstack(fill_value=0)
            )
            pct = (ct.div(ct.sum(axis=1), axis=0)*100).round(2).add_suffix("%")
            result[col] = pd.concat([ct, pct], axis=1)
        return result[col] if len(result)==1 else result

    out_num, out_cat = None, None
    if num_cols:
        out_num = numeric_summary(df[num_cols + ([] if by is None else [by])], by)
    if cat_cols:
        out_cat = categorical_summary(df[cat_cols + ([] if by is None else [by])], by)

    # 반환 형태 조정
    if out_num is not None and out_cat is None:
        return out_num
    if out_num is None and out_cat is not None:
        return out_cat
    return {"numeric": out_num, "categorical": out_cat}

@st.cache_data
def table_num(df, bin_width):
    """
    수치형 데이터의 도수분포표를 생성합니다.

    Parameters:
    - df (pd.Series): 도수분포를 계산할 수치형 데이터
    - bin_width (int or float): 각 구간의 너비

    Returns:
    - pd.DataFrame: 구간과 해당 구간의 도수를 포함하는 데이터 프레임
    """
    # 데이터의 최소값과 최대값을 기준으로 구간 경계를 설정
    min_val = df.min()
    max_val = df.max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # numpy의 histogram 함수를 사용하여 도수와 구간 경계 계산
    hist, bin_edges = np.histogram(df, bins=bins)

    # 도수분포표를 DataFrame으로 변환
    dosu_table = pd.DataFrame({
        '구간(이상-이하)': [f"{bin_edges[i]} - {bin_edges[i+1]}" for i in range(len(bin_edges)-1)],
        '도수': hist
    })

    return dosu_table

@st.cache_data
def table_cat(df):

    # 빈도 계산
    frequency = df.value_counts()
    modes = df.mode()  # 모든 최빈값

    # 빈도표 생성
    summary = pd.DataFrame({
        '빈도': frequency,
        '비율': np.round(frequency / len(df), 2)
    })

    # 최빈값 출력
    mode_text = ""
    for mode in modes:
        mode_text = mode_text+mode
        mode_text = mode_text+", "
    st.write("**최빈값**:", len(modes), "개", mode_text[:-2])
    st.error("평균을 구할 수 없습니다. ")
    st.error("중앙값을 구할 수 없습니다. ")
    return summary



@st.cache_data
def convert_column_types(df, user_column_types):
    # 사용자 입력에 따른 데이터 유형 변환
    for column, col_type in user_column_types.items():
        if col_type == 'Numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif col_type == 'Categorical':
            df[column] = df[column].astype('category')
    return df

@st.cache_data
def infer_column_types(df):
    column_types = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = 'Numeric'
        else:
            column_types[column] = 'Categorical'
    return column_types

    # st.session_state['column_types'] = column_types

    # # 사용자가 각 열의 데이터 유형을 설정할 수 있도록 입력 받기
    # user_column_types = {}
    # options_en = ['Numeric', 'Categorical']
    # options_kr = ["수치형", "범주형"]
    # options_dic = {'수치형': 'Numeric', '범주형': 'Categorical'}
    
    # # 반반 나눠서 나열
    # col1, col2 = st.columns(2)
    # keys = list(column_types.keys())
    # half = len(keys) // 2 

    # dict1 = {key: column_types[key] for key in keys[:half]}
    # dict2 = {key: column_types[key] for key in keys[half:]}

    # with col1:
    #     for column, col_type in dict1.items():
    #         default_index = options_en.index(col_type)
    #         user_col_type = st.radio(
    #             f"'{column}'의 유형:",
    #             options_kr,
    #             index=default_index,
    #             key=column
    #         )
    #         user_column_types[column] = options_dic[user_col_type]

    # with col2:
    #     for column, col_type in dict2.items():
    #         default_index = options_en.index(col_type)
    #         user_col_type = st.radio(
    #             f"'{column}'의 유형:",
    #             options_kr,
    #             index=default_index,
    #             key=column
    #         )
    #         user_column_types[column] = options_dic[user_col_type]

    # return user_column_types


@st.cache_data
# 수치형 데이터 변환
def transform_numeric_data(df, column, transformation):
    if transformation == '로그변환':
        df[column + '_log'] = np.log(df[column])
        transformed_column = column + '_log'
    elif transformation == '제곱근':
        df[column + '_sqrt'] = np.sqrt(df[column])
        transformed_column = column + '_sqrt'
    elif transformation == '제곱':
        df[column + '_squared'] = np.square(df[column])
        transformed_column = column + '_squared'
    else:
        transformed_column = column  # 변환 없을 경우 원본 열 이름을 그대로 사용

    # 원본 데이터 열 삭제
    df = df.drop(column, axis = 1)

    return df

pal = sns.color_palette(['#FB8500', '#FFB703', '#8E8444', '#1B536F', '#219EBC', '#A7D0E2'])

def palet(num_categories):
    if num_categories <= 6:
        p = sns.color_palette(['#FB8500', '#FFB703', '#8E8444', '#1B536F', '#219EBC', '#A7D0E2'])
        
    else:
        p = sns.color_palette("Set2", n_colors=num_categories)
    return p

import time
@st.cache_data
def 모든_그래프_그리기(df):
    user_column_types = infer_column_types(df)
    n = len(df.columns)
    # 범주의 수에 따라 팔레트 선택
    # 전체 그래프 개수 계산
    if n > 1:
        st.warning("각 변수마다 일변량, 이변량 데이터를 시각화하고 있어요. 오래 걸릴 수 있으니 기다려주세요!")
        progress_text = "📈 그래프를 그리는 중입니다...."
        count = 0
        # bar = st.progress(count , text=progress_text)
        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
        for i, col1 in enumerate(df.columns):
            # toast = st.toast(f"{col1}의 그래프를 그리는 중!", icon = '🍞')
            for j, col2 in enumerate(df.columns):
                # toast.toast(f"{col1}과 {col2}의 그래프", icon = '🥞')
                ax = axes[i, j]
                if i != j:
                    if user_column_types[col1] == 'Numeric' and user_column_types[col2] == 'Numeric':
                        sns.scatterplot(data=df, x=col1, y=col2, ax=ax, color = pal[0])
                    elif user_column_types[col1] == 'Categorical' and user_column_types[col2] == 'Numeric':
                        sns.boxplot(data=df, x=col1, y=col2, ax=ax, palette=pal)
                    elif user_column_types[col1] == 'Numeric' and user_column_types[col2] == 'Categorical':
                        # sns.histplot(data=df, x=col1, hue=col2, ax=ax, palette=pal)  # 여기를 수정
                        sns.kdeplot(data=df, x=col1, hue=col2, ax=ax, palette=pal)  # 여기를 수정
                    elif user_column_types[col1] == 'Categorical' and user_column_types[col2] == 'Categorical':
                        unique_values = df[col2].unique().astype(str)
                        # st.write(unique_values)
                        # 색상 매핑 생성
                        color_mapping = {val: color for val, color in zip(unique_values, palet(len(unique_values)))}
                        mosaic(df, [col1, col2], ax=ax, properties=lambda key: {'color': color_mapping[key[1]]}, gap=0.05)

                    ax.set_title(f'{col1} vs {col2}')
                else:
                    if user_column_types[col1] == 'Numeric':
                        sns.histplot(df[col1], ax=ax, color=pal[0])
                    else:
                        sns.countplot(x=df[col1], ax=ax, palette=pal)
                    ax.set_title(f'Distribution of {col1}')
                count = count + 1
                # bar.progress(count /(n*n), text=progress_text)
                # st.text(f'그려진 그래프: {completed_plots} / 총 그래프: {total_plots}')  # 진행 상황 업데이트
                time.sleep(0.1)
                # placeholder.empty()
        # st.toast("거의 다 그렸어요!", icon = "🍽")

        plt.tight_layout()
        # bar.empty()
        st.pyplot(fig)
    if n==1:
        st.warning("열을 하나만 선택하셨군요! 아래의 데이터 하나씩 시각화 영역에서 시각화하세요!")



from stemgraphic import stem_graphic

@st.cache_data
def 하나씩_그래프_그리기(df, width, height):
    user_column_types = infer_column_types(df)
    # 범주의 수에 따라 팔레트 선택
    # 전체 그래프 개수 계산s
    progress_text = "📈 그래프를 그리는 중입니다...."
    col = df.columns[0]
    # 범주형일 때, 막대, 원, 띠
    if user_column_types[col] == "Categorical":
        fig, axes = plt.subplots(1, 3, figsize=(width, height))

        # 막대 그래프
        sns.countplot(x=df[col], ax=axes[0], palette=pal)
        axes[0].set_title(f'{col} bar chart')

        # 원 그래프
        axes[1].pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%', startangle=90,  colors=pal)
        axes[1].set_title(f'{col} pie chart')

        # 띠 그래프
        # 데이터프레임에서 특정 열에 대한 값의 비율을 계산합니다.
        ddi = df.copy()
        ddi = ddi.dropna()
        ddi = pd.DataFrame(ddi[col])
        ddi['temp'] = '_'
        
        ddi_2 = pd.pivot_table(ddi, columns = col, aggfunc= 'count')
        # ddi_2.plot.bar(stacked = True, ax = axes[2])

        # 각 값이 전체 합계에 대한 비율이 되도록 변환합니다.
        ddi_percent = ddi_2.divide(ddi_2.sum(axis=1), axis=0)

        # 막대 그래프를 가로로 그리고, 누적해서 표시합니다.
        ddi_percent.plot(kind='barh', stacked=True, ax=axes[2], legend=False, color = pal)

        # 범례 설정
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, [label.split(', ')[-1][:-1] for label in labels], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(labels), frameon=False)

        # x축 레이블을 퍼센트로 표시
        axes[2].set_xlabel('(%)')

        # y축의 눈금과 레이블 제거
        axes[2].yaxis.set_ticks([])
        axes[2].yaxis.set_ticklabels([])

        # 그래프 제목 설정
        axes[2].set_title(f'{col} ribbon graph')

        plt.tight_layout()
        st.pyplot(fig)

    # 수치형일 때, 줄기잎, 히스토, 도다, 상자그림
    else:

        fig, axes = plt.subplots(2, 2, figsize=(width, height*2))            
        
        # 줄기잎그림

        # 히스토그램
        # 도다
        # 상자그림
        stem_graphic(df[col], ax = axes[0,0])
        sns.histplot(data = df, x = col, ax = axes[0,1], color=pal[0])
        # sns.boxplot(data = df, x = col, ax = axes[1,0], palette=pal)
        sns.boxplot(data = df, x = col, ax = axes[1,1], palette = pal)


        # 데이터를 히스토그램으로 나누어 계급 구하기
        df_copy = df.dropna()
        counts, bin_edges = np.histogram(df_copy[col], bins=10)

        # 도수분포다각형을 그리기 위한 x값(계급의 중앙값) 계산
        # 양 끝의 계급에 대한 도수를 0으로 추가
        counts = np.insert(counts, 0, 0)
        counts = np.append(counts, 0)
        bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[1] - bin_edges[0]))
        bin_edges = np.append(bin_edges, bin_edges[-1] + (bin_edges[-1] - bin_edges[-2]))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 도수분포다각형 그리기
        axes[1,0].plot(bin_centers, counts, marker='o', linestyle='-')

        plt.tight_layout()
        st.pyplot(fig)

# @st.cache_data
# def 선택해서_그래프_그리기(df, col, graph_type, option = None, rot_angle = 0):
#     fig, ax = plt.subplots()
    
#     if graph_type == '막대그래프':
#         horizontal = option[0]
#         # order = option[1]
#         # sns.countplot(y=df.columns[0], data=df, order = option[1], ax=ax, palette=pal)
#         if horizontal : 
#             sns.countplot(y = df.columns[0], data = df, ax = ax, palette=pal) # order = order, 
#         else:
#             sns.countplot(x = df.columns[0], data = df, ax = ax, palette=pal) # order = order, 


#     elif graph_type == '원그래프':
#         ax.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%', startangle=90,  colors=pal)
#     elif graph_type == '띠그래프':
#         # 띠 그래프
#         # 데이터프레임에서 특정 열에 대한 값의 비율을 계산합니다.
#         ddi = df.copy()
#         ddi = ddi.dropna()
#         ddi = pd.DataFrame(ddi[col])
#         ddi['temp'] = '_'

#         ddi_2 = pd.pivot_table(ddi, columns=col, aggfunc='count')

#         # 각 값이 전체 합계에 대한 비율이 되도록 변환합니다.
#         ddi_percent = ddi_2.divide(ddi_2.sum(axis=1), axis=0)*100

#         # 막대 그래프를 가로로 그리고, 누적해서 표시합니다.
#         ddi_percent.plot(kind='barh', stacked=True, ax=ax, legend=False, color=pal)

#         # 범례 설정 - 세로로 배치
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles, [label.split(', ')[-1][:] for label in labels], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=True)

#         # x축 레이블을 퍼센트로 표시
#         ax.set_xlabel('(%)')

#         # y축의 눈금과 레이블 제거
#         ax.yaxis.set_ticks([])
#         ax.yaxis.set_ticklabels([])

#         # 그래프 제목 설정
#         ax.set_title(f'{col} 띠그래프')

#         plt.tight_layout()
#     elif graph_type == '꺾은선그래프':
#         # 이변량에서....
#         temp = df[col].value_counts()
#         plt.plot(temp.sort_index().index, temp.sort_index().values, marker='o', linestyle='-', color='black')
#         if option == None:
#             plt.ylim(0, temp.sort_index().max() * 1.2)
#         else:
#             plt.ylim(temp.sort_index().min * 0.8, temp.sort_index().max() * 1.2)  
#     elif graph_type == '히스토그램':
#         if pd.api.types.is_numeric_dtype(df[col]):
#             if df[col].max() - df[col].min() < option:
#                 st.error(f"오류: 계급의 크기를 확인해주세요.")
#             else:
#                 sns.histplot(data = df, x = col, ax = ax, color=pal[0], binwidth = option)        
#         else:
#             st.error(f"오류: '{col}' 에 대해 히스토그림을 그릴 수 없어요. ")
#     elif graph_type == '도수분포다각형':
#         if pd.api.types.is_numeric_dtype(df[col]):
#             if df[col].max() - df[col].min() < option:
#                 st.error(f"오류: 계급의 크기를 확인해주세요.")
#             else:
#                 sns.histplot(data = df, x = col, ax = ax, element = "poly", color=pal[0], binwidth = option)    
#         else:
#             st.write(np.array(df[col]).dtype)
#             st.error(f"오류: '{col}' 에 대해 도수분포다각형을 그릴 수 없어요. ")



#     elif graph_type == '줄기와잎그림':
#         try:
#             # 숫자형 데이터가 아닐 경우 오류가 발생할 수 있음
#             stem_graphic(df[col], ax=ax)
#         except TypeError:
#             st.error(f"오류: '{col}'에 대해 줄기와 잎 그림을 그릴 수 없어요.")
#         except Exception as e:
#             st.write(f"알 수 없는 오류가 발생했습니다: {e}")
#     elif graph_type == '상자그림':
#         # st.write(df.dtypes) 임시로 처리함
#         if pd.api.types.is_categorical_dtype(df[col]):
#             pass
#         else:
#             sns.boxplot(data = df, x = col, color=pal[0], showmeans=True,
#                     meanprops={'marker':'o',
#                        'markerfacecolor':'white', 
#                        'markeredgecolor':'black',
#                        'markersize':'8'})    
#         # else:
#         #     st.error(f"오류: '{col}' 에 대해 상자그림을 그릴 수 없어요. ")
            
#     else:
#         st.error("지원되지 않는 그래프입니다. ")
#         return None
#     return fig
# --- 단변량 그래프 그리기 ----------------------------------------------
@st.cache_data
def 선택해서_그래프_그리기(
    df: pd.DataFrame,
    col: str,
    graph_type: str,
    *,
    order: tuple|list|None = None,   # ← 범주 순서(막대·원·띠 그래프용)
    horizontal: bool = False,        # ← 막대그래프 가로/세로
    binwidth: int|float|None = None,
    rot_angle: int = 0,
    palette: list|str = "Set2"
):
    """단변량 시각화 함수 (order 지정 시 순서형 처리)"""
    fig, ax = plt.subplots()

    # 0) 범주 순서 처리
    if order is not None:
        df[col] = pd.Categorical(df[col], categories=list(order), ordered=True)

    # 1) 막대그래프
    if graph_type == "막대그래프":
        if horizontal:
            sns.countplot(y=col, data=df, order=order, palette=palette, ax=ax)
        else:
            sns.countplot(x=col, data=df, order=order, palette=palette, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rot_angle, ha="right")

    # 2) 원그래프
    elif graph_type == "원그래프":
        val_counts = df[col].value_counts().reindex(order) if order else df[col].value_counts()
        ax.pie(val_counts, labels=val_counts.index,
               autopct="%1.1f%%", startangle=90, colors=palette)

    # 3) 띠그래프
    elif graph_type == "띠그래프":
        vc = df[col].value_counts(normalize=True).mul(100).reindex(order).fillna(0)
        ax.barh([""], vc, color=sns.color_palette(palette, len(vc)))
        ax.set_xlabel("(%)"); ax.set_yticks([]); ax.set_title(f"{col} 띠그래프")
        for i, (v, lbl) in enumerate(zip(vc.cumsum(), vc.index)):
            ax.text(v - vc.iloc[i]/2, 0, lbl, ha="center", va="center", fontsize=8)

    # 4) 꺾은선그래프 (빈도 시계열 느낌)
    elif graph_type == "꺾은선그래프":
        freq = df[col].value_counts().sort_index()
        ax.plot(freq.index, freq.values, marker="o", color="black")
        ax.set_xticklabels(freq.index, rotation=rot_angle, ha="right")

    # 5) 히스토그램 / 도수분포다각형
    elif graph_type in ("히스토그램", "도수분포다각형"):
        element = "poly" if graph_type == "도수분포다각형" else "step"
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], binwidth=binwidth, element=element,
                         color=sns.color_palette(palette)[0], ax=ax)
        else:
            st.error(f"'{col}' 은(는) 수치형이 아니어서 {graph_type} 불가")
            return None

    # 6) 줄기와잎그림
    elif graph_type == "줄기와잎그림":
        try:
            stem_graphic(df[col], ax=ax)
        except Exception:
            st.error(f"'{col}' 에 대해 줄기와잎그림을 그릴 수 없어요.")
            return None

    # 7) 상자그림
    elif graph_type == "상자그림":
        sns.boxplot(x=df[col], color=sns.color_palette(palette)[0], showmeans=True,
                    meanprops=dict(marker="o", markerfacecolor="white",
                                   markeredgecolor="black"), ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rot_angle, ha="right")

    else:
        st.error("지원되지 않는 그래프 유형입니다.")
        return None

    fig.tight_layout()
    return fig



# @st.cache_data
# def 선택해서_그래프_그리기_이변량(df, x_var, y_var, graph_type, option = None, rot_angle = 0):
#     if option is None:
#         option = () 
#     col = df.columns[0]
#     fig, ax = plt.subplots()
    
#     if graph_type == '막대그래프':
#         # # 순서 지정은 나중에
#         # if option != None: 
#         #     st.write(option)
#         #     st.write('hhhh')
#         #     sns.histplot(data=df, x = x_var, hue = y_var, order=option,  ax=ax, palette=pal)
#         if option == True: 
#             # sns.histplot(data=df, x=x_var, hue=y_var, multiple="stack", shrink=0.8, palette=pal, stat="count")
#             sns.histplot(data=df, x=x_var, hue=y_var, multiple="stack", shrink=0.8, palette=pal, edgecolor = None,  stat="percent")
#         else:
#             sns.countplot(data=df, x = x_var, hue = y_var, ax=ax, palette=pal)

#     elif graph_type == '꺾은선그래프':
#         # 이변량에서....
#         if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):

#             plt.plot(df[x_var], df[y_var], marker='o', linestyle='-', color='black')
#             if option == None:
#                 plt.ylim(0, df[y_var].max() * 1.2)
#             else:
#                 plt.ylim(df[y_var].min() * 0.8, df[y_var].max()*1.2)
#         else:
#             st.error("꺾은선그래프를 그릴 수 없어요. ")
#     elif graph_type == '히스토그램':
#         if option:
#             sns.histplot(data = df, x = x_var, hue = y_var, element = "step", binwidth = option)
#         else:
#             sns.histplot(data = df, x = x_var, hue = y_var)

#     elif graph_type == '도수분포다각형':
#         if option:
#             sns.histplot(data = df, x = x_var, hue = y_var, element = "poly", binwidth = option)
#         else:
#             sns.histplot(data = df, x = x_var, hue = y_var)

#     elif graph_type == '상자그림':
#         # 완료
#         sns.boxplot(data = df, x = x_var, y = y_var, showmeans=True,
#                     color = pal[0],
#                     meanprops={'marker':'o',
#                        'markerfacecolor':'white', 
#                        'markeredgecolor':'black',
#                        'markersize':'8'})
#     elif graph_type == "산점도":
#         # option: (hue, style, size, regline) 형태일 수도 있고 비어 있을 수도 있음
#         hue   = option[0] if len(option) > 0 else None
#         style = option[1] if len(option) > 1 else None
#         size  = option[2] if len(option) > 2 else None
#         reg   = option[3] if len(option) > 3 else False

#         sns.scatterplot(data=df, x=x_var, y=y_var,
#                         hue=hue, style=style, size=size)

#         if reg:  # 회귀선 추가
#             if hue is not None:  # 범주별 회귀선
#                 unique_cats = df[hue].unique()
#                 palette = sns.color_palette("hsv", len(unique_cats))
#                 col_dict = dict(zip(unique_cats, palette))
#                 for cat in unique_cats:
#                     cat_data = df[df[hue] == cat]
#                     sns.regplot(data=cat_data, x=x_var, y=y_var,
#                                 scatter=False, color=col_dict[cat],
#                                 label=f"Reg {cat}", ci=None)
#             else:  # 전체 회귀선
#                 sns.regplot(data=df, x=x_var, y=y_var,
#                             scatter=False, label="Total Reg", ci=None)
#     else:
#         st.error("지원되지 않는 그래프입니다. ")
#         return None
#     return fig

# --- 선택해서_그래프_그리기_이변량 ----------------------------------------
@st.cache_data
def 선택해서_그래프_그리기_이변량(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    graph_type: str,
    *,
    order: tuple|list|None = None,     # ← ① x축 순서(범주형) 전달
    hue_order: tuple|list|None = None, # ← ② hue 순서 전달
    rot_angle: int = 0,
    binwidth: int|float|None = None,
    palette: list|str = "Set2"
):
    """
    * order / hue_order 에 순서형(ordinal) 리스트를 넘기면
      자동으로 Categorical(ordered=True) 로 변환 후 그립니다.
    * 기타 로직·그래프 타입은 기존과 동일
    """
    fig, ax = plt.subplots()

    # --- 0) 순서형 처리 -------------------------------------------------
    if order is not None:
        df[x_var] = pd.Categorical(df[x_var], categories=list(order), ordered=True)
    if hue_order is not None and y_var in df.columns:
        df[y_var] = pd.Categorical(df[y_var], categories=list(hue_order), ordered=True)

    # --- 1) 막대그래프 ---------------------------------------------------
    if graph_type == "막대그래프":
        sns.countplot(
            data=df,
            x=x_var,
            hue=y_var,
            order=order,
            hue_order=hue_order,
            palette=palette,
            ax=ax
        )

    # --- 2) 꺾은선그래프 --------------------------------------------------
    elif graph_type == "꺾은선그래프":
        if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):
            ax.plot(df[x_var], df[y_var], marker="o", color="black")
        else:
            st.error("꺾은선그래프는 두 변수 모두 수치형일 때만 지원합니다.")
            return None

    # --- 3) 히스토그램 / 도수분포다각형 ----------------------------------
    elif graph_type in ("히스토그램", "도수분포다각형"):
        element = "poly" if graph_type == "도수분포다각형" else "step"
        sns.histplot(
            data=df,
            x=x_var,
            hue=y_var,
            binwidth=binwidth,
            element=element,
            palette=palette,
            ax=ax
        )

    # --- 4) 상자그림 ------------------------------------------------------
    elif graph_type == "상자그림":
        sns.boxplot(
            data=df,
            x=x_var,
            y=y_var,
            showmeans=True,
            color=sns.color_palette(palette)[0],
            meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black")
        )

    # --- 5) 산점도 --------------------------------------------------------
    elif graph_type == "산점도":
        sns.scatterplot(
            data=df,
            x=x_var,
            y=y_var,
            hue=hue_order[0] if hue_order else None,
            palette=palette,
            ax=ax
        )

    else:
        st.error("지원되지 않는 그래프 유형입니다.")
        return None

    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot_angle, ha="right")
    fig.tight_layout()
    return fig
