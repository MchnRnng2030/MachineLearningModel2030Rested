# ======================================================
# Persona Policy API
# - 페르소나 항목 입력 → 정책 결과 반환
# - 직업교육 여부에 따른 취업확률 변화 포함
# - 모델 재추론 ❌ / 관측 기반 시뮬레이션 ⭕
# ======================================================

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# ------------------------------------------------------
# 1️⃣ 앱 및 데이터 로드
# ------------------------------------------------------
app = Flask(__name__)

DATA_PATH = "../results/final/merged_policy_analysis.csv"
df = pd.read_csv(DATA_PATH)

# 안전장치: 필요한 컬럼 체크
REQUIRED_COLS = [
    "성별코드", "만연령", "교육정도_학력구분코드", "혼인상태코드",
    "직업훈련경험",
    "p_employ_lr", "p_employ_rf",
    "지원필요도점수", "진성쉬었음", "정책유형"
]

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"❌ 데이터에 필요한 컬럼이 없습니다: {missing}")

def shrinkage(mean_group, n_group, mean_global, k=200):
    """
    표본 수가 적을수록 전체 평균 쪽으로 당김
    k: 보정 강도 (200~500 권장)
    """
    w = n_group / (n_group + k)
    return w * mean_group + (1 - w) * mean_global


def query_persona(df, persona, tol_age=1, min_n=30):
    """
    - 표본 충분: 그대로
    - 표본 부족: shrinkage
    - 극단적 부족: 학력 완화
    """

    # --------------------------------------------------
    # 1️⃣ 1차 조건 (정밀 페르소나)
    # --------------------------------------------------
    base_cond = (
        (df["성별코드"] == persona["성별코드"]) &
        (df["교육정도_학력구분코드"] == persona["교육정도_학력구분코드"]) &
        (df["혼인상태코드"] == persona["혼인상태코드"]) &
        (df["만연령"].between(
            persona["만연령"] - tol_age,
            persona["만연령"] + tol_age
        ))
    )

    no_train  = df.loc[base_cond & (df["직업훈련경험"] == 0)]
    yes_train = df.loc[base_cond & (df["직업훈련경험"] == 1)]

    # --------------------------------------------------
    # 2️⃣ 전체 평균 (보정용 기준)
    # --------------------------------------------------
    global_no  = df[df["직업훈련경험"] == 0]
    global_yes = df[df["직업훈련경험"] == 1]

    p_no_global  = ((global_no["p_employ_lr"] + global_no["p_employ_rf"]) / 2).mean()
    p_yes_global = ((global_yes["p_employ_lr"] + global_yes["p_employ_rf"]) / 2).mean()

    # --------------------------------------------------
    # 3️⃣ 표본 충분한 경우
    # --------------------------------------------------
    if len(no_train) >= min_n and len(yes_train) >= min_n:
        p_no  = ((no_train["p_employ_lr"] + no_train["p_employ_rf"]) / 2).mean()
        p_yes = ((yes_train["p_employ_lr"] + yes_train["p_employ_rf"]) / 2).mean()

        method = "direct"

    # --------------------------------------------------
    # 4️⃣ 표본 부족 → Shrinkage
    # --------------------------------------------------
    elif len(no_train) >= 10 and len(yes_train) >= 10:
        p_no_raw  = ((no_train["p_employ_lr"] + no_train["p_employ_rf"]) / 2).mean()
        p_yes_raw = ((yes_train["p_employ_lr"] + yes_train["p_employ_rf"]) / 2).mean()

        p_no  = shrinkage(p_no_raw,  len(no_train),  p_no_global)
        p_yes = shrinkage(p_yes_raw, len(yes_train), p_yes_global)

        method = "shrinkage"

    # --------------------------------------------------
    # 5️⃣ 극단적 부족 → 학력 완화
    # --------------------------------------------------
    else:
        # 저학력 / 고학력으로 완화
        if persona["교육정도_학력구분코드"] <= 3:
            edu_cond = df["교육정도_학력구분코드"] <= 3
            edu_group = "저학력(≤고졸)"
        else:
            edu_cond = df["교육정도_학력구분코드"] >= 4
            edu_group = "고학력(대졸+)"

        relaxed_cond = (
            (df["성별코드"] == persona["성별코드"]) &
            edu_cond &
            (df["혼인상태코드"] == persona["혼인상태코드"]) &
            (df["만연령"].between(
                persona["만연령"] - 2,
                persona["만연령"] + 2
            ))
        )

        no_train  = df.loc[relaxed_cond & (df["직업훈련경험"] == 0)]
        yes_train = df.loc[relaxed_cond & (df["직업훈련경험"] == 1)]

        if len(no_train) < 10 or len(yes_train) < 10:
            return {
                "message": "유사 페르소나 표본이 매우 부족합니다.",
                "level": "insufficient",
                "count_no_training": int(len(no_train)),
                "count_training": int(len(yes_train))
            }

        p_no  = shrinkage(
            ((no_train["p_employ_lr"] + no_train["p_employ_rf"]) / 2).mean(),
            len(no_train), p_no_global
        )
        p_yes = shrinkage(
            ((yes_train["p_employ_lr"] + yes_train["p_employ_rf"]) / 2).mean(),
            len(yes_train), p_yes_global
        )

        method = f"relaxed({edu_group})"

    # --------------------------------------------------
    # 6️⃣ 결과 반환
    # --------------------------------------------------
    return {
        # -------------------------------
        # 1️⃣ 입력 페르소나
        # -------------------------------
        "페르소나": persona,

        # -------------------------------
        # 2️⃣ 표본 및 보정 정보
        # -------------------------------
        "보정방식": method,
        "표본수": {
            "직업교육_미이수": int(len(no_train)),
            "직업교육_이수": int(len(yes_train))
        },

        # -------------------------------
        # 3️⃣ 취업확률 (현기씨 모델 축)
        # -------------------------------
        "취업확률": {
            "직업교육_미이수": round(p_no, 3),
            "직업교육_이수": round(p_yes, 3),
            "증가폭(p)": round(p_yes - p_no, 3)
        },

        # -------------------------------
        # 4️⃣ 정책 매트릭스 결과 (옥주 축)
        # -------------------------------
        "정책판단": {
            "지원필요도점수": round(
                yes_train["지원필요도점수"].mean(), 3
            ),
            "진성쉬었음비율": round(
                yes_train["진성쉬었음"].mean(), 3
            ),
            "정책유형": yes_train["정책유형"].mode().iloc[0]
        }
    }




# ------------------------------------------------------
# 3️⃣ API 엔드포인트
# ------------------------------------------------------
@app.route("/persona", methods=["POST"])
def persona_api():
    """
    POST /persona
    Body(JSON):
    {
        "성별코드": 2,
        "만연령": 26,
        "교육정도_학력구분코드": 5,
        "혼인상태코드": 1
    }
    """
    persona = request.get_json()

    required_inputs = [
        "성별코드",
        "만연령",
        "교육정도_학력구분코드",
        "혼인상태코드"
    ]

    # 입력값 검증
    for key in required_inputs:
        if key not in persona:
            return jsonify({"error": f"입력값 누락: {key}"}), 400

    result = query_persona(df, persona)
    return jsonify(result)


# ------------------------------------------------------
# 4️⃣ 실행
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
