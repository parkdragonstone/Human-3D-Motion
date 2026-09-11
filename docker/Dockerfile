# H3DM 컨테이너 이미지. CPU / GPU 두 변형을 하나의 Dockerfile 로 만든다.
#
#   docker build --build-arg VARIANT=cpu ...   → onnxruntime + openvino
#   docker build --build-arg VARIANT=gpu ...   → onnxruntime-gpu[cuda,cudnn]
#
# GPU 변형도 베이스는 python:3.12-slim 그대로다. onnxruntime-gpu 의
# [cuda,cudnn] extras 가 CUDA 13 런타임과 cuDNN 9 를 pip 패키지로 가져오므로
# nvidia/cuda 베이스 이미지가 필요 없다. 호스트에는 NVIDIA 드라이버와
# NVIDIA Container Toolkit 만 있으면 된다.
#
# ⚠️ linux/amd64 전용이다. opensim 은 Linux aarch64 휠을 배포하지 않고
#    openvino 도 최신 버전은 x86_64 전용이라, arm64 로 빌드하면 pip 에서
#    죽는다. Apple Silicon 에서는 에뮬레이션으로 돌아가며 추론이 약 50배
#    느리다 — macOS 는 Docker 대신 네이티브 실행을 쓸 것 (README 참고).

# ── 1단계: 프론트엔드 TypeScript 빌드 ───────────────────────────────────
# BUILDPLATFORM 이면 빌드 머신의 네이티브 아키텍처로 돈다. tsc 산출물은
# 플랫폼 독립적이라 에뮬레이션을 거칠 이유가 없다.
FROM --platform=$BUILDPLATFORM node:22-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY webapp/frontend ./webapp/frontend
# tsconfig 의 outDir 이 ../presentation/static/dist 이므로 여기로 떨어진다.
RUN npx tsc -p webapp/frontend/tsconfig.json


# ── 2단계: 런타임 ──────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

ARG VARIANT=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/data/tmp \
    MPLBACKEND=Agg \
    H3DM_VARIANT=${VARIANT}

# libgl1 / libglib2.0-0 : opencv-python 런타임 의존
# ffmpeg               : pipelines/utils/video_utils.py 의 transcode_to_h264 가
#                        오버레이 영상을 브라우저 재생 가능한 H.264 로 재인코딩한다.
#                        없으면 mp4v 로 남아 Chrome/Safari 에서 재생되지 않는다.
# tini                 : 분석 스레드와 ffmpeg 자식 프로세스의 좀비 수거
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 사용자를 COPY 보다 먼저 만든다. 마지막에 `chown -R h3dm /app` 을 돌리면
# overlayfs 가 /app 전체를 새 레이어로 복사해서(copy-up) 261MB 를 낭비한다
# — 실측값이다. COPY --chown 은 복사 시점에 소유자를 정해 추가 레이어가 없다.
RUN useradd --create-home --uid 10001 h3dm \
    && mkdir -p /data/recordings /data/webapp_data /data/tmp \
    && chown -R h3dm:h3dm /data

# 두 변형의 requirements 를 모두 복사한 뒤 VARIANT 로 고른다. 이렇게 하면
# 변형을 바꿔도 앞 레이어(apt)는 캐시가 살아 있다.
COPY docker/requirements-common.txt docker/requirements-cpu.txt docker/requirements-gpu.txt ./
RUN pip install --no-cache-dir -r "requirements-${VARIANT}.txt"

# pipelines/ 에는 models/(215MB, .gitignore 대상)와 OpenSim_Setup/(27MB)이
# 함께 들어온다. 모델이 없으면 분석이 detector_model_not_found 로 죽는다.
COPY --chown=h3dm:h3dm pipelines/ ./pipelines/
COPY --chown=h3dm:h3dm webapp/ ./webapp/
COPY --chown=h3dm:h3dm images/ ./images/
COPY --chown=h3dm:h3dm scripts/ ./scripts/
COPY --chown=h3dm:h3dm --from=frontend /build/webapp/presentation/static/dist ./webapp/presentation/static/dist
COPY --chown=h3dm:h3dm docker/entrypoint.sh /usr/local/bin/h3dm-entrypoint
RUN chmod +x /usr/local/bin/h3dm-entrypoint

USER h3dm

EXPOSE 9090

# HTTPS 모드에서는 자체서명 인증서라 curl 이 -k 를 필요로 한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsSk -o /dev/null "http://127.0.0.1:9090/api/cameras" \
     || curl -fsSk -o /dev/null "https://127.0.0.1:9090/api/cameras" \
     || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/h3dm-entrypoint"]
