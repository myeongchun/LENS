"""Streamlit adapter for https://github.com/myeongchun/LENS.

Place alongside main.py and config.json in the original repository.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MAX_BYTES = 50 * 1024 * 1024
MAX_EXPANDED = 100 * 1024 * 1024
MAX_ROWS = 20000
MAX_FEATURES = 512


def read_csv(raw: bytes, separator: str = ",") -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), sep=separator, encoding="utf-8-sig",
                        nrows=MAX_ROWS + 1)
    if len(frame) > MAX_ROWS:
        raise ValueError(f"최대 {MAX_ROWS:,}행까지 지원합니다.")
    return frame


def read_npz(raw: bytes) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED:
            raise ValueError("압축 해제 크기는 100 MiB 이하여야 합니다.")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        # Never deserialize Python objects from an uploaded archive.
        return {key: archive[key] for key in archive.files}


def validate_features(values) -> np.ndarray:
    X = np.asarray(values)
    if X.ndim != 2 or not 3 <= X.shape[0] <= MAX_ROWS:
        raise ValueError(f"X는 3~{MAX_ROWS:,}행의 2차원 배열이어야 합니다.")
    if not 2 <= X.shape[1] <= MAX_FEATURES:
        raise ValueError(f"수치형 특성을 2~{MAX_FEATURES}개 선택하세요.")
    if X.dtype.kind not in "biuf":
        raise ValueError("특성에는 실수형 숫자만 사용할 수 있습니다.")
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("특성에 결측값 또는 무한대가 있습니다. 데이터를 정리해 주세요.")
    if not np.any(np.ptp(X, axis=0) > 0):
        raise ValueError("모든 특성이 상수입니다. 변동이 있는 데이터를 사용하세요.")
    return X


def validate_labels(values, count: int) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(y) != count:
        raise ValueError("레이블은 X와 행 수가 같은 1차원 배열이어야 합니다.")
    if pd.isna(y).any():
        raise ValueError("레이블에 결측값이 있습니다.")
    if y.dtype.kind in "fc" and not np.isfinite(y).all():
        raise ValueError("레이블에 무한대가 있습니다.")
    return y.astype(str)


@st.cache_resource
def execution_lock():
    # main.run_lens changes global random seeds; serialize concurrent sessions.
    return threading.Lock()


def execute(X, config, seed):
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from main import run_lens

    # Match utils/data_loader.py in the original repository.
    scaled = MinMaxScaler().fit_transform(X)
    with execution_lock():
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            scores = np.asarray(run_lens(tensor, config, seed)).reshape(-1)
    if len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError("LENS가 유효하지 않은 점수를 반환했습니다.")
    return scores


def main():
    st.set_page_config(page_title="LENS Demo", layout="wide")
    st.title("LENS · 이상 탐지 데모")
    st.write("CSV 또는 NPZ를 업로드하고 레이블을 지정해 LENS를 실행하세요.")
    st.caption("CPU 실행 · 레이블은 평가에만 사용 · 점수가 클수록 이상 가능성이 높음")
    root = Path(__file__).resolve().parent
    if not (root / "main.py").exists() or not (root / "config.json").exists():
        st.error("이 app.py를 LENS 저장소의 main.py, config.json과 같은 폴더에 넣으세요.")
        return
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        for key in ("N", "r", "filtering_ratio", "rho"):
            if key not in config:
                raise ValueError(f"config.json에 {key}가 없습니다.")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    with st.sidebar:
        st.header("실행 설정")
        config["N"] = st.number_input("앙상블 크기 N", 1, 50, int(config["N"]))
        config["r"] = st.slider("설명 분산 비율 r", 0.01, 0.99, float(config["r"]))
        config["filtering_ratio"] = st.slider("필터링 비율", 0.0, 0.9,
                                               float(config["filtering_ratio"]))
        config["rho"] = st.slider("신뢰도 계수 rho", 0.0, 1.0, float(config["rho"]))
        seed = st.number_input("난수 시드", 0, 2147483647, 0)
        st.caption("제한: 파일 50 MiB, 20,000행, 특성 512개. 큰 데이터는 오래 걸릴 수 있습니다.")
    uploaded = st.file_uploader("데이터셋", type=["csv", "npz"])
    if uploaded is None:
        return
    if uploaded.size > MAX_BYTES:
        st.error("파일은 50 MiB 이하여야 합니다.")
        return
    raw = uploaded.getvalue()
    file_key = hashlib.sha256(raw).hexdigest()[:16]
    y = None
    feature_names = []
    try:
        if uploaded.name.lower().endswith(".csv"):
            separator = st.selectbox("CSV 구분자", [",", ";", "\t"], key=f"sep-{file_key}")
            frame = read_csv(raw, separator)
            st.dataframe(frame.head(10), hide_index=True)
            columns = list(frame.columns)
            default_label = next((c for c in ("label", "y", "target") if c in columns), None)
            label = st.selectbox("레이블 열 (없으면 선택하지 않음)", [None] + columns,
                                 index=columns.index(default_label) + 1 if default_label else 0,
                                 format_func=lambda v: "레이블 없음" if v is None else str(v),
                                 key=f"label-{file_key}")
            options = [c for c in columns if c != label]
            defaults = [c for c in options if pd.api.types.is_numeric_dtype(frame[c])]
            feature_names = st.multiselect("사용할 수치형 특성 (ID 열은 제외하세요)", options,
                                           default=defaults, key=f"features-{file_key}-{label}")
            X = validate_features(frame[feature_names].to_numpy())
            if label is not None:
                y = validate_labels(frame[label].to_numpy(), len(X))
        else:
            arrays = read_npz(raw)
            keys = list(arrays)
            candidates = [k for k in keys if arrays[k].ndim == 2]
            if not candidates:
                raise ValueError("NPZ에 2차원 특성 배열이 없습니다.")
            xkey = st.selectbox("특성 배열", candidates,
                                index=candidates.index("X") if "X" in candidates else 0,
                                key=f"x-{file_key}")
            X = validate_features(arrays[xkey])
            ykeys = [None] + [k for k in keys if k != xkey]
            ykey = st.selectbox("레이블 배열", ykeys,
                                index=ykeys.index("y") if "y" in ykeys else 0,
                                format_func=lambda v: "레이블 없음" if v is None else v,
                                key=f"y-{file_key}")
            if ykey is not None:
                y = validate_labels(arrays[ykey], len(X))
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            st.dataframe(pd.DataFrame(X[:10], columns=feature_names), hide_index=True)
        anomaly_values = []
        if y is not None:
            unique = list(pd.unique(y))
            st.write("레이블별 개수", pd.Series(y).value_counts().rename("count"))
            anomaly_values = st.multiselect("이상을 나타내는 레이블 값", unique,
                                            default=["1"] if "1" in unique else [],
                                            key=f"anomaly-{file_key}-{hashlib.sha256(y.tobytes()).hexdigest()[:8]}")
            st.caption("선택한 값은 이상(1), 나머지 값은 모두 정상(0)으로 평가합니다.")
            if not anomaly_values or len(anomaly_values) == len(unique):
                raise ValueError("평가하려면 이상 레이블을 선택하고 정상 레이블을 하나 이상 남겨 주세요. 점수만 필요하면 '레이블 없음'을 선택하세요.")
        st.write(f"{len(X):,}개 샘플 · {X.shape[1]}개 특성")
    except Exception as exc:
        st.error(f"입력 확인: {exc}")
        return

    signature = hashlib.sha256(raw + X.tobytes() + json.dumps(
        [config, int(seed), feature_names, anomaly_values,
         y.tolist() if y is not None else None], ensure_ascii=False).encode()).hexdigest()
    if st.button("LENS 실행", type="primary"):
        st.session_state.pop("lens_result", None)
        try:
            started = time.perf_counter()
            with st.spinner("LENS 실행 중입니다…"):
                scores = execute(X, config, int(seed))
            metrics = {}
            result = pd.DataFrame({"sample_index": np.arange(len(X)), "anomaly_score": scores})
            if y is not None:
                from sklearn.metrics import average_precision_score, roc_auc_score
                binary = np.isin(y, anomaly_values).astype(int)
                metrics = {"AUROC": float(roc_auc_score(binary, scores)),
                           "AUPRC (Average Precision)": float(average_precision_score(binary, scores))}
                result["original_label"] = y
                result["is_anomaly"] = binary
            report = {"source_file": uploaded.name, "samples": len(X), "features": feature_names,
                      "config": config, "seed": int(seed), "device": "cpu",
                      "preprocessing": "MinMaxScaler [0, 1]", "anomaly_values": anomaly_values,
                      "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
            st.session_state["lens_result"] = (signature, result, report)
        except Exception as exc:
            st.error(f"실행 실패: {exc}")
            st.info("requirements-demo.txt의 패키지와 원본 LENS 코드가 모두 설치되어 있는지 확인하세요.")
    saved = st.session_state.get("lens_result")
    if saved and saved[0] == signature:
        _, result, report = saved
        st.success(f"완료 · {report['elapsed_seconds']:.2f}초")
        for name, value in report["metrics"].items():
            st.metric(name, f"{value:.4f}")
        if not report["metrics"]:
            st.info("레이블 없이 점수만 계산했습니다.")
        st.dataframe(result.head(100), hide_index=True)
        st.download_button("샘플별 결과 CSV 다운로드", result.to_csv(index=False).encode("utf-8-sig"),
                           "lens-results.csv", "text/csv")
        st.download_button("설정·평가 결과 JSON 다운로드",
                           json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                           "lens-report.json", "application/json")


if __name__ == "__main__":
    main()
