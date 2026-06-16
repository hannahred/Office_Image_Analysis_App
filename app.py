import streamlit as st
from openai import OpenAI
import pandas as pd
from io import BytesIO
import base64
import json

st.set_page_config(
    page_title="Office Image Analysis App",
    layout="wide"
)

st.title("Office Image Analysis App")
st.write("실내공간 이미지를 AI로 분석하고, 결과를 Excel 파일로 정리합니다.")

# API Key 확인
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. Streamlit Cloud의 Secrets에 API Key를 추가해주세요.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 분석 키워드
default_keywords = """bookshelf / built-in shelf
table
chair
lounge seating
library / reading area
wood material
white wall
plant / greenery
open office
meeting room
workstation / desk
lighting fixture
natural light
glass wall / partition
acoustic panel
carpet / rug
ceiling feature
storage / cabinet"""

st.subheader("1. 분석할 키워드 입력")

keyword_text = st.text_area(
    "이미지에서 확인할 키워드를 한 줄에 하나씩 입력하세요.",
    value=default_keywords,
    height=260
)

keywords = [k.strip() for k in keyword_text.splitlines() if k.strip()]

st.subheader("2. 이미지 업로드")

uploaded_files = st.file_uploader(
    "분석할 이미지 파일을 업로드하세요. 여러 장 선택 가능합니다.",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

def image_to_base64(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    encoded = base64.b64encode(bytes_data).decode("utf-8")
    mime_type = uploaded_file.type
    return f"data:{mime_type};base64,{encoded}"

def analyze_image(uploaded_file, keywords):
    image_data_url = image_to_base64(uploaded_file)

    prompt = f"""
You are analyzing interior design images for academic research.

For each keyword below, judge whether it is visible or reasonably inferable in the image.

Keywords:
{chr(10).join("- " + k for k in keywords)}

Return ONLY valid JSON in the following structure:

{{
  "space_type": "short description of the space type",
  "items": [
    {{
      "keyword": "keyword text",
      "judgment": "있음 / 없음 / 불명확",
      "confidence": 0.0,
      "reason": "brief reason in Korean"
    }}
  ]
}}

Rules:
- Use only one of these judgments: 있음, 없음, 불명확.
- Confidence must be a number between 0 and 1.
- Reason must be short and written in Korean.
- Do not include markdown.
- Do not include any extra explanation outside JSON.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url}
                ]
            }
        ]
    )

    result_text = response.output_text.strip()

    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError:
        result_json = {
            "space_type": "분석 실패",
            "items": [
                {
                    "keyword": "JSON parsing error",
                    "judgment": "불명확",
                    "confidence": 0,
                    "reason": result_text[:200]
                }
            ]
        }

    return result_json

def make_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="image_analysis")
    output.seek(0)
    return output

st.subheader("3. 분석 실행")

if uploaded_files:
    st.write(f"업로드된 이미지 수: {len(uploaded_files)}장")

    if st.button("이미지 분석 시작"):
        rows = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"분석 중: {uploaded_file.name}")

            try:
                result = analyze_image(uploaded_file, keywords)
                space_type = result.get("space_type", "")

                for item in result.get("items", []):
                    rows.append({
                        "파일명": uploaded_file.name,
                        "공간유형": space_type,
                        "키워드 / 코드": item.get("keyword", ""),
                        "판단": item.get("judgment", ""),
                        "추정 신뢰도": item.get("confidence", ""),
                        "이유": item.get("reason", "")
                    })

            except Exception as e:
                rows.append({
                    "파일명": uploaded_file.name,
                    "공간유형": "오류",
                    "키워드 / 코드": "오류",
                    "판단": "분석 실패",
                    "추정 신뢰도": 0,
                    "이유": str(e)
                })

            progress.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(rows)

        st.success("분석이 완료되었습니다.")
        st.dataframe(df, use_container_width=True)

        excel_file = make_excel(df)

        st.download_button(
            label="Excel 파일 다운로드",
            data=excel_file,
            file_name="image_analysis_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("먼저 이미지 파일을 업로드하세요.")
