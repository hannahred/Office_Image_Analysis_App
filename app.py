import streamlit as st
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import pandas as pd
from io import BytesIO
import base64
import json
import zipfile
import os
import mimetypes
import time
import re
from PIL import Image


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="E-magazine Image & Text Analysis App",
    layout="wide"
)

st.title("E-magazine Image & Text Analysis App")
st.write(
    "e-magazine 캡처본에서 이미지 영역과 텍스트 영역을 구분하고, "
    "이미지는 별도 분석하며 텍스트는 따로 추출해 Excel로 정리합니다."
)

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. Streamlit Cloud의 Secrets에 API Key를 추가해주세요.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =========================================================
# 세션 상태
# =========================================================

DEFAULT_KEYWORDS = """office interior
open office
workstation / desk
meeting room
conference room
lounge seating
break area
collaboration area
focus area
phone booth
library / reading area
bookshelf / built-in shelf
table
chair
sofa
plant / greenery
wood material
glass wall / partition
acoustic panel
carpet / rug
natural light
lighting fixture
ceiling feature
storage / cabinet
colorful interior
warm atmosphere
wellbeing element
sustainable material"""

if "screenshots" not in st.session_state:
    st.session_state.screenshots = []

if "visual_regions" not in st.session_state:
    st.session_state.visual_regions = []

if "text_blocks" not in st.session_state:
    st.session_state.text_blocks = []

if "cropped_images" not in st.session_state:
    st.session_state.cropped_images = []

if "keyword_text" not in st.session_state:
    st.session_state.keyword_text = DEFAULT_KEYWORDS

if "image_analysis_df" not in st.session_state:
    st.session_state.image_analysis_df = None

if "wide_analysis_df" not in st.session_state:
    st.session_state.wide_analysis_df = None


# =========================================================
# 유틸 함수
# =========================================================

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def is_image_file(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in IMAGE_EXTENSIONS


def sanitize_filename(text):
    text = re.sub(r"[^a-zA-Z0-9가-힣_\-\.]+", "_", text)
    text = text.strip("_")
    return text[:120] if len(text) > 120 else text


def prepare_image_bytes(file_bytes, max_size=1400):
    """
    이미지 크기를 줄여 API 오류와 비용을 줄입니다.
    긴 변 기준 max_size 이하로 축소합니다.
    """
    img = Image.open(BytesIO(file_bytes))
    img.thumbnail((max_size, max_size))

    if img.mode != "RGB":
        img = img.convert("RGB")

    output = BytesIO()
    img.save(output, format="JPEG", quality=85)

    width, height = img.size
    return output.getvalue(), "image/jpeg", width, height


def bytes_to_data_url(file_bytes, mime_type="image/jpeg"):
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def safe_json_loads(text):
    """
    AI 응답에서 JSON 부분만 최대한 안전하게 추출합니다.
    """
    if text is None:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


def parse_keywords(text):
    return [k.strip() for k in text.splitlines() if k.strip()]


def extract_images_from_uploads(uploaded_files):
    """
    직접 업로드한 이미지와 ZIP 안 이미지 모두 읽기.
    """
    screenshots = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(file_bytes), "r") as z:
                    for info in z.infolist():
                        if info.is_dir():
                            continue

                        inner_path = info.filename

                        if not is_image_file(inner_path):
                            continue

                        try:
                            raw_bytes = z.read(info)
                            prepared_bytes, mime_type, width, height = prepare_image_bytes(raw_bytes)

                            screenshots.append({
                                "source_file": filename,
                                "folder_path": os.path.dirname(inner_path),
                                "filename": os.path.basename(inner_path),
                                "full_path": inner_path,
                                "bytes": prepared_bytes,
                                "mime_type": mime_type,
                                "width": width,
                                "height": height
                            })

                        except Exception as e:
                            st.warning(f"ZIP 안 이미지 읽기 실패: {inner_path} / {e}")

            except Exception as e:
                st.error(f"ZIP 파일을 열 수 없습니다: {filename} / {e}")

        else:
            if is_image_file(filename):
                try:
                    prepared_bytes, mime_type, width, height = prepare_image_bytes(file_bytes)

                    screenshots.append({
                        "source_file": "direct_upload",
                        "folder_path": "",
                        "filename": filename,
                        "full_path": filename,
                        "bytes": prepared_bytes,
                        "mime_type": mime_type,
                        "width": width,
                        "height": height
                    })

                except Exception as e:
                    st.warning(f"이미지 읽기 실패: {filename} / {e}")

    return screenshots


def call_openai_with_retry(prompt, image_bytes, mime_type, model, max_retries=3, base_wait=8, max_output_tokens=2500):
    """
    429 에러 대비:
    실패하면 일정 시간 기다렸다가 재시도합니다.
    """
    image_data_url = bytes_to_data_url(image_bytes, mime_type)

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data_url}
                        ]
                    }
                ],
                max_output_tokens=max_output_tokens
            )
            return response.output_text

        except RateLimitError as e:
            last_error = e
            if attempt >= max_retries:
                raise e

            wait_time = base_wait * (2 ** attempt)
            st.warning(f"429 Rate Limit 발생: {wait_time}초 기다린 뒤 재시도합니다. ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)

        except (APITimeoutError, APIError) as e:
            last_error = e
            if attempt >= max_retries:
                raise e

            wait_time = base_wait * (2 ** attempt)
            st.warning(f"API 일시 오류 발생: {wait_time}초 기다린 뒤 재시도합니다. ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    raise last_error


def bbox_norm_to_pixels(bbox_norm, width, height):
    """
    bbox_norm: [x1, y1, x2, y2], 0~1000 기준
    """
    if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    except Exception:
        return None

    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    px1 = int(x1 / 1000 * width)
    py1 = int(y1 / 1000 * height)
    px2 = int(x2 / 1000 * width)
    py2 = int(y2 / 1000 * height)

    return px1, py1, px2, py2


def crop_image_region(screenshot, bbox_norm, padding=5):
    bbox = bbox_norm_to_pixels(bbox_norm, screenshot["width"], screenshot["height"])

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(screenshot["width"], x2 + padding)
    y2 = min(screenshot["height"], y2 + padding)

    if (x2 - x1) < 80 or (y2 - y1) < 80:
        return None

    img = Image.open(BytesIO(screenshot["bytes"])).convert("RGB")
    cropped = img.crop((x1, y1, x2, y2))

    output = BytesIO()
    cropped.save(output, format="JPEG", quality=90)

    return output.getvalue(), cropped.size[0], cropped.size[1], [x1, y1, x2, y2]


# =========================================================
# AI 함수 1: 캡처본에서 이미지/텍스트 영역 구분
# =========================================================

def segment_screenshot(screenshot, model, max_retries, base_wait):
    prompt = f"""
You are analyzing a screenshot of an e-magazine or web magazine page.

The image size after resizing is:
width = {screenshot["width"]} pixels
height = {screenshot["height"]} pixels

Task:
1. Detect visual regions that are meaningful for interior design research.
2. Detect readable text blocks from the article.
3. Separate article content from browser UI, menus, buttons, ads, logos, and unrelated interface elements.

Return ONLY valid JSON.

Use this exact structure:

{{
  "page_summary": "brief Korean summary",
  "visual_regions": [
    {{
      "region_id": "v1",
      "region_type": "interior_photo / rendering / floor_plan / diagram / advertisement / logo / ui / other",
      "bbox_norm": [x1, y1, x2, y2],
      "confidence": 0.0,
      "reason": "brief Korean reason"
    }}
  ],
  "text_blocks": [
    {{
      "text_id": "t1",
      "text_type": "title / subtitle / heading / body / caption / source / date / ui / other",
      "bbox_norm": [x1, y1, x2, y2],
      "text": "extracted text",
      "confidence": 0.0
    }}
  ]
}}

Important rules:
- bbox_norm must use normalized coordinates from 0 to 1000.
- x1,y1 is top-left. x2,y2 is bottom-right.
- Extract Korean text as accurately as possible.
- If text is unreadable, skip it.
- For visual_regions, include only meaningful rectangular image areas.
- If an interior image contains a small caption overlay, still classify it as interior_photo or rendering if the main content is interior space.
- Do not include markdown.
- Do not include explanation outside JSON.
"""

    result_text = call_openai_with_retry(
        prompt=prompt,
        image_bytes=screenshot["bytes"],
        mime_type=screenshot["mime_type"],
        model=model,
        max_retries=max_retries,
        base_wait=base_wait,
        max_output_tokens=3000
    )

    parsed = safe_json_loads(result_text)

    if not parsed:
        return {
            "page_summary": "JSON 파싱 실패",
            "visual_regions": [],
            "text_blocks": [
                {
                    "text_id": "error",
                    "text_type": "other",
                    "bbox_norm": [0, 0, 0, 0],
                    "text": result_text[:500],
                    "confidence": 0
                }
            ]
        }

    return parsed


# =========================================================
# AI 함수 2: 추출 이미지 분석
# =========================================================

def analyze_cropped_image(cropped_image, keywords, model, max_retries, base_wait):
    keyword_list = "\n".join("- " + k for k in keywords)

    prompt = f"""
You are analyzing interior design images for academic research.

For each keyword below, judge whether it is visible or reasonably inferable in the image.

Keywords:
{keyword_list}

Return ONLY valid JSON in this exact structure:

{{
  "space_type": "short English or Korean description of the space type",
  "items": [
    {{
      "keyword": "keyword text",
      "judgment": "있음 / 없음 / 불명확",
      "confidence": 0.0,
      "reason": "brief Korean reason"
    }}
  ]
}}

Rules:
- Use only one of these judgments: 있음, 없음, 불명확.
- Confidence must be a number between 0 and 1.
- Reason must be short and written in Korean.
- Judge based on visible evidence.
- If the item may exist but is not clearly visible, use 불명확.
- Do not include markdown.
- Do not include explanation outside JSON.
"""

    result_text = call_openai_with_retry(
        prompt=prompt,
        image_bytes=cropped_image["bytes"],
        mime_type="image/jpeg",
        model=model,
        max_retries=max_retries,
        base_wait=base_wait,
        max_output_tokens=3000
    )

    parsed = safe_json_loads(result_text)

    if not parsed:
        return {
            "space_type": "분석 실패",
            "items": [
                {
                    "keyword": "JSON parsing error",
                    "judgment": "불명확",
                    "confidence": 0,
                    "reason": result_text[:300]
                }
            ]
        }

    return parsed


# =========================================================
# Excel 생성 함수
# =========================================================

def make_wide_format(long_df):
    if long_df is None or long_df.empty:
        return pd.DataFrame()

    base_cols = [
        "source_file",
        "원본캡처",
        "추출이미지명",
        "region_type",
        "공간유형"
    ]

    try:
        wide_df = long_df.pivot_table(
            index=base_cols,
            columns="키워드 / 코드",
            values="판단",
            aggfunc="first"
        ).reset_index()

        wide_df.columns.name = None
        return wide_df

    except Exception:
        return pd.DataFrame()


def make_excel_file():
    output = BytesIO()

    screenshot_df = pd.DataFrame([
        {
            "source_file": s["source_file"],
            "폴더경로": s["folder_path"],
            "파일명": s["filename"],
            "전체경로": s["full_path"],
            "width": s["width"],
            "height": s["height"]
        }
        for s in st.session_state.screenshots
    ])

    visual_df = pd.DataFrame(st.session_state.visual_regions)
    text_df = pd.DataFrame(st.session_state.text_blocks)
    cropped_df = pd.DataFrame([
        {
            "source_file": c["source_file"],
            "원본캡처": c["screenshot_full_path"],
            "추출이미지명": c["cropped_filename"],
            "region_type": c["region_type"],
            "bbox_norm": c["bbox_norm"],
            "bbox_pixels": c["bbox_pixels"],
            "width": c["width"],
            "height": c["height"],
            "confidence": c["confidence"],
            "reason": c["reason"]
        }
        for c in st.session_state.cropped_images
    ])

    image_analysis_df = st.session_state.image_analysis_df
    wide_df = st.session_state.wide_analysis_df

    keyword_df = pd.DataFrame({"keyword": parse_keywords(st.session_state.keyword_text)})

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        screenshot_df.to_excel(writer, index=False, sheet_name="screenshot_list")
        visual_df.to_excel(writer, index=False, sheet_name="visual_regions")
        cropped_df.to_excel(writer, index=False, sheet_name="cropped_images")
        text_df.to_excel(writer, index=False, sheet_name="text_extraction")
        keyword_df.to_excel(writer, index=False, sheet_name="keywords")

        if image_analysis_df is not None and not image_analysis_df.empty:
            image_analysis_df.to_excel(writer, index=False, sheet_name="image_analysis_long")

        if wide_df is not None and not wide_df.empty:
            wide_df.to_excel(writer, index=False, sheet_name="image_analysis_wide")

    output.seek(0)
    return output


def make_cropped_images_zip():
    output = BytesIO()

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for img in st.session_state.cropped_images:
            z.writestr(img["cropped_filename"], img["bytes"])

    output.seek(0)
    return output


# =========================================================
# 사이드바 설정
# =========================================================

st.sidebar.header("설정")

model = st.sidebar.selectbox(
    "OpenAI model",
    ["gpt-4o-mini", "gpt-4.1-mini"],
    index=0
)

max_retries = st.sidebar.number_input(
    "429 발생 시 재시도 횟수",
    min_value=0,
    max_value=5,
    value=2,
    step=1
)

base_wait = st.sidebar.number_input(
    "재시도 전 기본 대기시간초",
    min_value=3,
    max_value=60,
    value=10,
    step=1
)

delay_between_requests = st.sidebar.number_input(
    "각 이미지 처리 사이 대기시간초",
    min_value=0.0,
    max_value=30.0,
    value=3.0,
    step=0.5
)

st.sidebar.caption(
    "429 에러가 계속 나면 대기시간을 10~20초 이상으로 늘리고, "
    "한 번에 처리하는 이미지 수를 줄여보세요."
)


# =========================================================
# 1. 캡처본 업로드
# =========================================================

st.header("1. e-magazine 캡처본 업로드")

uploaded_files = st.file_uploader(
    "캡처 이미지 여러 장 또는 캡처 이미지 폴더를 압축한 ZIP 파일을 업로드하세요.",
    type=["jpg", "jpeg", "png", "webp", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("업로드 파일 읽기"):
        st.session_state.screenshots = extract_images_from_uploads(uploaded_files)
        st.session_state.visual_regions = []
        st.session_state.text_blocks = []
        st.session_state.cropped_images = []
        st.session_state.image_analysis_df = None
        st.session_state.wide_analysis_df = None

        if len(st.session_state.screenshots) == 0:
            st.warning("읽을 수 있는 이미지가 없습니다. jpg, jpeg, png, webp 또는 zip 파일을 확인해주세요.")
        else:
            st.success(f"총 {len(st.session_state.screenshots)}장의 캡처본을 읽었습니다.")

if st.session_state.screenshots:
    screenshot_df = pd.DataFrame([
        {
            "source_file": s["source_file"],
            "폴더경로": s["folder_path"],
            "파일명": s["filename"],
            "전체경로": s["full_path"],
            "width": s["width"],
            "height": s["height"]
        }
        for s in st.session_state.screenshots
    ])

    st.dataframe(screenshot_df, use_container_width=True)

    with st.expander("캡처본 미리보기"):
        cols = st.columns(3)
        for idx, s in enumerate(st.session_state.screenshots[:9]):
            with cols[idx % 3]:
                st.image(s["bytes"], caption=s["full_path"], use_container_width=True)

        if len(st.session_state.screenshots) > 9:
            st.caption(f"미리보기는 9장까지만 표시합니다. 전체 캡처 수: {len(st.session_state.screenshots)}장")


# =========================================================
# 2. 이미지/텍스트 영역 분리
# =========================================================

st.header("2. 캡처본에서 이미지 영역과 텍스트 추출")

if st.session_state.screenshots:
    max_pages = st.number_input(
        "영역 분리할 캡처본 수",
        min_value=1,
        max_value=len(st.session_state.screenshots),
        value=min(1, len(st.session_state.screenshots)),
        step=1
    )

    st.info("처음에는 반드시 1장만 테스트하세요. 정상 작동하면 3장, 5장 순서로 늘리는 것을 추천합니다.")

    if st.button("이미지/텍스트 영역 분리 시작"):
        target_screenshots = st.session_state.screenshots[:max_pages]

        visual_rows = []
        text_rows = []
        cropped_images = []

        progress = st.progress(0)

        for i, screenshot in enumerate(target_screenshots):
            st.write(f"처리 중: {screenshot['full_path']}")

            try:
                result = segment_screenshot(
                    screenshot=screenshot,
                    model=model,
                    max_retries=max_retries,
                    base_wait=base_wait
                )

                page_summary = result.get("page_summary", "")

                # visual regions
                visual_regions = result.get("visual_regions", [])
                if not isinstance(visual_regions, list):
                    visual_regions = []

                for idx, region in enumerate(visual_regions):
                    region_id = region.get("region_id", f"v{idx + 1}")
                    region_type = region.get("region_type", "other")
                    bbox_norm = region.get("bbox_norm", [])
                    confidence = region.get("confidence", "")
                    reason = region.get("reason", "")

                    visual_rows.append({
                        "source_file": screenshot["source_file"],
                        "원본캡처": screenshot["full_path"],
                        "page_summary": page_summary,
                        "region_id": region_id,
                        "region_type": region_type,
                        "bbox_norm": bbox_norm,
                        "confidence": confidence,
                        "reason": reason
                    })

                    # 의미 있는 이미지 영역 crop
                    if region_type in ["interior_photo", "rendering", "floor_plan", "diagram"]:
                        crop_result = crop_image_region(screenshot, bbox_norm)

                        if crop_result is not None:
                            crop_bytes, crop_w, crop_h, bbox_pixels = crop_result

                            base_name = os.path.splitext(os.path.basename(screenshot["filename"]))[0]
                            clean_type = sanitize_filename(region_type)
                            cropped_filename = f"{sanitize_filename(base_name)}_{idx + 1}_{clean_type}.jpg"

                            cropped_images.append({
                                "source_file": screenshot["source_file"],
                                "screenshot_full_path": screenshot["full_path"],
                                "cropped_filename": cropped_filename,
                                "region_id": region_id,
                                "region_type": region_type,
                                "bbox_norm": bbox_norm,
                                "bbox_pixels": bbox_pixels,
                                "confidence": confidence,
                                "reason": reason,
                                "width": crop_w,
                                "height": crop_h,
                                "bytes": crop_bytes
                            })

                # text blocks
                text_blocks = result.get("text_blocks", [])
                if not isinstance(text_blocks, list):
                    text_blocks = []

                for idx, block in enumerate(text_blocks):
                    text_rows.append({
                        "source_file": screenshot["source_file"],
                        "원본캡처": screenshot["full_path"],
                        "page_summary": page_summary,
                        "text_id": block.get("text_id", f"t{idx + 1}"),
                        "text_type": block.get("text_type", "other"),
                        "bbox_norm": block.get("bbox_norm", []),
                        "text": block.get("text", ""),
                        "confidence": block.get("confidence", "")
                    })

            except Exception as e:
                st.error(f"처리 실패: {screenshot['full_path']} / {e}")

            progress.progress((i + 1) / len(target_screenshots))

            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

        st.session_state.visual_regions = visual_rows
        st.session_state.text_blocks = text_rows
        st.session_state.cropped_images = cropped_images

        st.success(
            f"영역 분리 완료: visual region {len(visual_rows)}개, "
            f"text block {len(text_rows)}개, 추출 이미지 {len(cropped_images)}개"
        )

else:
    st.info("먼저 캡처본 또는 ZIP 파일을 업로드하고 '업로드 파일 읽기'를 눌러주세요.")


# =========================================================
# 3. 분리 결과 확인
# =========================================================

st.header("3. 분리 결과 확인")

tab_visual, tab_text, tab_crop = st.tabs(["Visual regions", "Text extraction", "Cropped images"])

with tab_visual:
    if st.session_state.visual_regions:
        st.dataframe(pd.DataFrame(st.session_state.visual_regions), use_container_width=True)
    else:
        st.info("아직 visual region 결과가 없습니다.")

with tab_text:
    if st.session_state.text_blocks:
        text_df = pd.DataFrame(st.session_state.text_blocks)
        st.dataframe(text_df, use_container_width=True)

        st.subheader("추출 텍스트 모아보기")
        combined_text = "\n\n".join([
            f"[{row.get('원본캡처', '')} / {row.get('text_type', '')}]\n{row.get('text', '')}"
            for row in st.session_state.text_blocks
            if row.get("text", "")
        ])
        st.text_area("전체 추출 텍스트", value=combined_text, height=300)
    else:
        st.info("아직 텍스트 추출 결과가 없습니다.")

with tab_crop:
    if st.session_state.cropped_images:
        cropped_df = pd.DataFrame([
            {
                "source_file": c["source_file"],
                "원본캡처": c["screenshot_full_path"],
                "추출이미지명": c["cropped_filename"],
                "region_type": c["region_type"],
                "bbox_norm": c["bbox_norm"],
                "bbox_pixels": c["bbox_pixels"],
                "width": c["width"],
                "height": c["height"],
                "confidence": c["confidence"],
                "reason": c["reason"]
            }
            for c in st.session_state.cropped_images
        ])
        st.dataframe(cropped_df, use_container_width=True)

        with st.expander("추출 이미지 미리보기"):
            cols = st.columns(4)
            for idx, img in enumerate(st.session_state.cropped_images[:16]):
                with cols[idx % 4]:
                    st.image(img["bytes"], caption=img["cropped_filename"], use_container_width=True)

        zip_file = make_cropped_images_zip()
        st.download_button(
            label="추출 이미지 ZIP 다운로드",
            data=zip_file,
            file_name="cropped_images.zip",
            mime="application/zip"
        )

    else:
        st.info("아직 추출된 이미지가 없습니다.")


# =========================================================
# 4. 이미지 분석 키워드
# =========================================================

st.header("4. 이미지 분석 키워드 설정")

st.write("추출된 이미지에 대해 확인할 키워드를 한 줄에 하나씩 입력하세요.")

st.text_area(
    "분석 키워드",
    key="keyword_text",
    height=320
)

keywords = parse_keywords(st.session_state.keyword_text)
st.caption(f"현재 키워드 수: {len(keywords)}개")


# =========================================================
# 5. 추출 이미지 분석
# =========================================================

st.header("5. 추출 이미지 AI 분석")

if st.session_state.cropped_images and keywords:
    region_types = sorted(list(set([c["region_type"] for c in st.session_state.cropped_images])))

    selected_region_types = st.multiselect(
        "분석할 이미지 유형 선택",
        options=region_types,
        default=[t for t in region_types if t in ["interior_photo", "rendering"]]
    )

    filtered_images = [
        c for c in st.session_state.cropped_images
        if c["region_type"] in selected_region_types
    ]

    st.write(f"분석 대상 이미지 수: {len(filtered_images)}장")

    if filtered_images:
        max_images_to_analyze = st.number_input(
            "분석할 추출 이미지 수",
            min_value=1,
            max_value=len(filtered_images),
            value=min(1, len(filtered_images)),
            step=1
        )

        st.warning("처음에는 1장만 테스트하세요. 429 에러가 나면 대기시간을 늘리거나 OpenAI Usage/Billing을 확인하세요.")

        if st.button("추출 이미지 분석 시작"):
            target_images = filtered_images[:max_images_to_analyze]
            rows = []

            progress = st.progress(0)

            for i, img in enumerate(target_images):
                st.write(f"분석 중: {img['cropped_filename']}")

                try:
                    result = analyze_cropped_image(
                        cropped_image=img,
                        keywords=keywords,
                        model=model,
                        max_retries=max_retries,
                        base_wait=base_wait
                    )

                    space_type = result.get("space_type", "")
                    items = result.get("items", [])

                    if not isinstance(items, list):
                        items = []

                    for item in items:
                        rows.append({
                            "source_file": img["source_file"],
                            "원본캡처": img["screenshot_full_path"],
                            "추출이미지명": img["cropped_filename"],
                            "region_type": img["region_type"],
                            "공간유형": space_type,
                            "키워드 / 코드": item.get("keyword", ""),
                            "판단": item.get("judgment", ""),
                            "추정 신뢰도": item.get("confidence", ""),
                            "이유": item.get("reason", "")
                        })

                except Exception as e:
                    rows.append({
                        "source_file": img["source_file"],
                        "원본캡처": img["screenshot_full_path"],
                        "추출이미지명": img["cropped_filename"],
                        "region_type": img["region_type"],
                        "공간유형": "오류",
                        "키워드 / 코드": "오류",
                        "판단": "분석 실패",
                        "추정 신뢰도": 0,
                        "이유": str(e)
                    })

                progress.progress((i + 1) / len(target_images))

                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)

            long_df = pd.DataFrame(rows)
            wide_df = make_wide_format(long_df)

            st.session_state.image_analysis_df = long_df
            st.session_state.wide_analysis_df = wide_df

            st.success("이미지 분석이 완료되었습니다.")

else:
    st.info("추출된 이미지와 키워드가 있어야 분석할 수 있습니다.")


# =========================================================
# 6. 결과 확인 및 다운로드
# =========================================================

st.header("6. 결과 확인 및 Excel 다운로드")

if st.session_state.image_analysis_df is not None and not st.session_state.image_analysis_df.empty:
    tab_long, tab_wide = st.tabs(["Image analysis long", "Image analysis wide"])

    with tab_long:
        st.dataframe(st.session_state.image_analysis_df, use_container_width=True)

    with tab_wide:
        if st.session_state.wide_analysis_df is not None and not st.session_state.wide_analysis_df.empty:
            st.dataframe(st.session_state.wide_analysis_df, use_container_width=True)
        else:
            st.info("Wide format 결과가 없습니다.")
else:
    st.info("아직 이미지 분석 결과가 없습니다.")

if (
    st.session_state.screenshots
    or st.session_state.visual_regions
    or st.session_state.text_blocks
    or st.session_state.cropped_images
    or st.session_state.image_analysis_df is not None
):
    excel_file = make_excel_file()

    st.download_button(
        label="전체 결과 Excel 다운로드",
        data=excel_file,
        file_name="e_magazine_analysis_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
