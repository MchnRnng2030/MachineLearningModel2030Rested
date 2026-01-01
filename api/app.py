# ======================================================
# Persona API - POLICY / CBM / ALL (FINAL - COMPLETE)
# ======================================================

from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
import os

# ======================================================
# 1️⃣ APP & DATA LOAD
# ======================================================
app = Flask(__name__)

# ---------------- POLICY DATA ----------------
POLICY_DATA_PATH = "../results/final/merged_policy_analysis.csv"
df_policy = pd.read_csv(POLICY_DATA_PATH)

# ---------------- CBM MODEL ------------------
CBM_MODEL_PATH = "../models/RYUI/cat_model.cbm"
cbm_model = CatBoostClassifier()
cbm_model.load_model(CBM_MODEL_PATH)

# CatBoost feature names fallback
CBM_FEATURES = getattr(cbm_model, "feature_names_", None)
if not CBM_FEATURES:
    feat_path = os.path.join(os.path.dirname(CBM_MODEL_PATH), "cat_features.txt")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            CBM_FEATURES = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("❌ CBM_FEATURES를 찾을 수 없습니다.")

# ======================================================
# 2️⃣ COMMON UTILITIES
# ======================================================
def validate_json(body):
    if body is None:
        return False, {"error": "JSON body 필요 (Content-Type: application/json)"}
    return True, None


def shrinkage(mean_group, n_group, mean_global, k=200):
    w = n_group / (n_group + k)
    return w * mean_group + (1 - w) * mean_global

# ======================================================
# 3️⃣ POLICY MODEL (집단 구조 판단)
# ======================================================
def progressive_filter(df, persona, min_n=30):
    age = persona.get("만연령")
    gender = persona.get("성별코드")
    marry = persona.get("혼인상태코드", None)

    if age is None or gender is None:
        return None, "INVALID_INPUT"

    def with_marry(cond):
        if marry is None:
            return cond
        return cond & (df["혼인상태코드"] == marry)

    cond1 = with_marry(
        (df["성별코드"] == gender) &
        (df["만연령"].between(age - 1, age + 1))
    )
    if df[cond1].shape[0] >= min_n:
        return df[cond1], "L1(성별+연령)"

    if persona.get("교육정도_학력구분코드") is not None:
        cond2 = cond1 & (
            df["교육정도_학력구분코드"] == persona["교육정도_학력구분코드"]
        )
        if df[cond2].shape[0] >= min_n:
            return df[cond2], "L2(+학력)"

    cond3 = with_marry(
        (df["성별코드"] == gender) &
        (df["만연령"].between(age - 2, age + 2))
    )
    if df[cond3].shape[0] >= min_n:
        return df[cond3], "L3(연령완화)"

    cond4 = with_marry(df["성별코드"] == gender)
    if df[cond4].shape[0] >= min_n:
        return df[cond4], "L4(성별)"

    return None, "INSUFFICIENT"


def run_policy(persona):
    df_grp, level = progressive_filter(df_policy, persona)

    if level == "INVALID_INPUT":
        return {"level": level, "message": "필수 입력(성별코드, 만연령) 누락"}

    if df_grp is None:
        return {"level": level, "message": "정책 페르소나 표본 부족"}

    no_train = df_grp[df_grp["직업훈련경험"] == 0]
    yes_train = df_grp[df_grp["직업훈련경험"] == 1]

    global_no = df_policy[df_policy["직업훈련경험"] == 0]
    global_yes = df_policy[df_policy["직업훈련경험"] == 1]

    p_no_global = ((global_no["p_employ_lr"] + global_no["p_employ_rf"]) / 2).mean()
    p_yes_global = ((global_yes["p_employ_lr"] + global_yes["p_employ_rf"]) / 2).mean()

    def calc(sub, global_mean):
        if len(sub) >= 30:
            return ((sub["p_employ_lr"] + sub["p_employ_rf"]) / 2).mean()
        elif len(sub) >= 10:
            raw = ((sub["p_employ_lr"] + sub["p_employ_rf"]) / 2).mean()
            return shrinkage(raw, len(sub), global_mean)
        else:
            return global_mean

    p_no = calc(no_train, p_no_global)
    p_yes = calc(yes_train, p_yes_global)

    policy_type = None
    support_score = None
    rest_ratio = None
    if len(yes_train) > 0:
        policy_type = yes_train["정책유형"].mode().iloc[0]
        support_score = round(float(yes_train["지원필요도점수"].mean()), 3)
        rest_ratio = round(float(yes_train["진성쉬었음"].mean()), 3)

    return {
        "level": level,
        "표본수": {"미이수": int(len(no_train)), "이수": int(len(yes_train))},
        "취업확률": {
            "직업교육_미이수": round(float(p_no), 3),
            "직업교육_이수": round(float(p_yes), 3),
            "증가폭": round(float(p_yes - p_no), 3)
        },
        "정책판단": {
            "정책유형": policy_type,
            "지원필요도점수": support_score,
            "진성쉬었음비율": rest_ratio
        }
    }

# ======================================================
# 4️⃣ CBM MODEL (개인 이직 성공 확률)
# ======================================================
def run_cbm(persona):
    row = {f: persona.get(f, 0) for f in CBM_FEATURES}
    X = pd.DataFrame([row])
    prob = float(cbm_model.predict_proba(X)[0, 1])
    return {
        "이직성공확률": round(prob, 3),
        "판정": "성공 가능" if prob >= 0.5 else "실패 가능"
    }

# ======================================================
# 5️⃣ PERSONAL STABILITY (개인 안정성)
# ======================================================
def derive_personal_stability(persona, cbm_res):
    score = 0
    reasons = []

    if persona.get("만연령", 0) >= 30:
        score += 1; reasons.append("30세 이상")
    if persona.get("교육정도_학력구분코드", 0) >= 4:
        score += 1; reasons.append("대졸 이상")
    if persona.get("직업훈련경험") == 1:
        score += 1; reasons.append("직업훈련 이수")
    if persona.get("구직사항_구직활동기간", 0) >= 1:
        score += 1; reasons.append("구직활동 경험")
    if cbm_res.get("이직성공확률", 0) >= 0.4:
        score += 1; reasons.append("이직 잠재력 존재")

    if score >= 4:
        level = "안정"
    elif score >= 2:
        level = "중간"
    else:
        level = "불안정"

    return {"개인안정성": level, "안정성점수": score, "근거": reasons}

## 기타
def interpret_result(policy, cbm):
    if "message" in policy:
        return "유사한 정책 페르소나 표본이 부족하여 해석이 제한됩니다."

    p_type = policy.get("정책판단", {}).get("정책유형")
    judge = cbm.get("판정")

    if p_type is None:
        return "정책유형 산출에 필요한 표본이 부족해 일반 해석만 제공합니다."

    if p_type == "안정군" and judge == "실패 가능":
        return "현재 위치는 안정적이나 동일 조건으로 이직 시 실패 가능성이 높습니다."
    if p_type == "안정군" and judge == "성공 가능":
        return "현재 상태와 이직 모두 비교적 안정적인 선택입니다."
    if p_type == "취업취약 비단절군" and judge == "성공 가능":
        return "현재 일자리는 불안정하나 이직을 통한 개선 가능성이 있습니다."
    if p_type == "취업취약 비단절군" and judge == "실패 가능":
        return "조건 개선 없이 이직 시 실패 가능성이 높습니다."
    if p_type == "재연결 가능 단절군":
        return "노동시장 재진입 가능성이 있어 준비 후 재도전이 권장됩니다."
    if p_type == "고착형 단절 위험군":
        return "구조적 위험이 높아 개인 이직보다는 정책 개입이 우선됩니다."

    return "현재 상태에 대한 일반적인 해석입니다."


def recommend_action(policy, cbm):
    if "message" in policy:
        return "추가 정보 필요"

    p_type = policy.get("정책판단", {}).get("정책유형")
    judge = cbm.get("판정")

    if p_type is None:
        return "표본 부족으로 일반 권고만 제공"

    if p_type == "안정군" and judge == "실패 가능":
        return "현 직장 유지"
    if p_type == "안정군" and judge == "성공 가능":
        return "선택적 이직 가능"

    if p_type == "취업취약 비단절군" and judge == "성공 가능":
        return "준비 후 이직 권장"
    if p_type == "취업취약 비단절군" and judge == "실패 가능":
        return "조건 개선 후 재도전"

    if p_type == "재연결 가능 단절군":
        return "정책 지원 + 취업 준비 병행"

    if p_type == "고착형 단절 위험군":
        return "정책 개입 우선 대상"

    return "개별 상담 권장"

def build_extra_advice(policy, cbm):
    p_type = policy.get("정책판단", {}).get("정책유형")

    if p_type == "안정군":
        return {
            "개인행동": "현 직무 유지 중심",
            "우선순위": "내부 이동 또는 점진적 준비",
            "단기전략": "핵심 직무 스킬 심화",
            "중기전략": "조건 좋은 기회 발생 시 이직",
            "주의사항": "불필요한 이직 리스크 회피"
        }

    if p_type == "취업취약 비단절군":
        return {
            "개인행동": "조건 개선 후 이직",
            "우선순위": "직업교육 → 구직활동 → 이동",
            "단기전략": "구직활동 방식 개선",
            "중기전략": "직무 일치도 높은 이동",
            "주의사항": "단기 이직 반복 주의"
        }

    if p_type == "재연결 가능 단절군":
        return {
            "개인행동": "즉시 이직 시도 비권장",
            "우선순위": "정책 지원 우선",
            "단기전략": "공공 프로그램 참여",
            "중기전략": "실습 기반 노동시장 재진입",
            "주의사항": "개별 구직은 효율 낮음"
        }

    if p_type == "고착형 단절 위험군":
        return {
            "개인행동": "개인 이직 시도 비권장",
            "우선순위": "정책 개입 최우선",
            "단기전략": "소득·활동 안정화",
            "중기전략": "단계적 노동시장 복귀",
            "주의사항": "개인 실패로 해석 금지"
        }

    return {
        "개인행동": "추가 정보 필요",
        "우선순위": "상담 권장"
    }
# ======================================================
# 6️⃣ API ENDPOINTS
# ======================================================
@app.route("/persona/policy", methods=["POST"])
def persona_policy():
    persona = request.get_json()
    ok, err = validate_json(persona)
    if not ok:
        return jsonify(err), 400
    return jsonify(run_policy(persona))


@app.route("/persona/cbm", methods=["POST"])
def persona_cbm():
    persona = request.get_json()
    ok, err = validate_json(persona)
    if not ok:
        return jsonify(err), 400
    return jsonify(run_cbm(persona))


@app.route("/persona/all", methods=["POST"])
def persona_all():
    persona = request.get_json()
    ok, err = validate_json(persona)
    if not ok:
        return jsonify(err), 400

    policy_res = run_policy(persona)
    cbm_res = run_cbm(persona)
    personal_res = derive_personal_stability(persona, cbm_res)

    return jsonify({
        "policy": policy_res,
        "cbm": cbm_res,
        "개인판단": personal_res,
        "해석": interpret_result(policy_res, cbm_res),
        "권고행동": recommend_action(policy_res, cbm_res),
        "추가권고": build_extra_advice(policy_res, cbm_res)
    })

# ======================================================
# 7️⃣ RUN
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
