# H3DM 배포 계획 (GPU 클라우드 미사용 · 로컬 저장경로 유지 · 단일 비밀번호)

작성일: 2026-09-10
대상 커밋: `e3b2a6d`

---

## 0. 결론 요약

**앱은 계속 내 PC에서 돌리고, 클라우드는 "주소 + 문 앞 자물쇠" 역할만 시킨다.**

```
   사용자 브라우저                    Cloudflare (무료)              내 랩 PC (GPU/카메라/디스크)
  ┌──────────────┐   HTTPS   ┌────────────────────┐  터널  ┌───────────────────────────┐
  │ h3dm.example │──────────▶│  TLS 종료 + 도메인   │◀──────│ cloudflared               │
  │    .com      │           │  (선택) Access 게이트 │       │      ↕ 127.0.0.1:9090     │
  └──────────────┘           └────────────────────┘        │ h3dm (Flask + SocketIO)   │
                                                            │  ├ RTMPose / YOLO / OpenSim│
                                                            │  ├ storage_root = 로컬 폴더 │
                                                            │  └ 비밀번호 게이트           │
                                                            └───────────────────────────┘
```

| 요구사항 | 해결 방식 |
| --- | --- |
| GPU 클라우드 안 씀 | 추론·OpenSim 전부 랩 PC에서 실행. 클라우드는 TCP 중계만 함 (연산 0) |
| 웹사이트인데 로컬 저장경로 | 저장 경로 = **서버(랩 PC)의 경로**. 원격 접속 시 OS 폴더 다이얼로그가 안 뜨므로 **앱 내장 폴더 브라우저**로 교체 필요 (§3) |
| 비밀번호 하나 | `HUMAN_3D_MOTION_PASSWORD_HASH` 환경변수 + Flask 세션 게이트 (§4) |
| 비용 | 도메인 ~₩15,000/년. 그 외 $0 |

**단, 시작 전에 반드시 확정해야 할 두 가지가 있습니다 → §7 (저장 디스크 주체), §8 (AGPL 라이선스).**

---

## 1. 현재 앱 진단 — 왜 "그냥 클라우드 배포"가 안 되는가

코드를 읽어보면 이 앱은 웹앱의 외형을 한 **데스크톱 앱**입니다. 클라우드에 그대로 올리면 아래가 전부 깨집니다.

| 항목 | 코드 위치 | 클라우드에서 깨지는 이유 |
| --- | --- | --- |
| 로컬 파일 시스템에 직접 read/write | `file_settings.py:15-32`, `FileSessionCatalog` | 저장 경로가 클라우드 컨테이너 디스크가 됨. 영상은 GB 단위 → 스토리지 과금 폭발, 재시작 시 소실 |
| **네이티브 OS 폴더 다이얼로그** | `tkinter_directory_selector.py:12-17` | 서버에 GUI가 없음. macOS는 `osascript`, Windows는 tkinter를 **서버 화면에** 띄움 → 원격 사용자는 영원히 대기 |
| Sony 카메라 제어 | `UrlCameraController` → `http://169.254.200.200/` | 링크로컬 주소. 물리적으로 같은 LAN이 아니면 접근 불가 |
| 폰 카메라 캡처 | `phone_capture_service.py`, Socket.IO | QR로 폰이 서버에 붙음. `HUMAN_3D_MOTION_PUBLIC_URL`은 이미 지원됨 (`main.py:43`, `flask_app.py:707`) |
| GPU 추론 | `pipelines/pose_estimation/backend.py:26-42` | CUDA→ROCm→MPS→CPU 자동 감지. 클라우드 GPU는 비쌈 (T4 기준 월 $200~400) |
| OpenSim (conda 전용) | `pipelines/kinematics/` | pip 설치 불가, 이미지 크기 수 GB |

**따라서 "서버를 클라우드로 옮기는" 배포는 불가능하고, 불필요합니다. 옮겨야 할 것은 서버가 아니라 URL뿐입니다.**

이미 배포를 염두에 둔 흔적도 있습니다:
- `HUMAN_3D_MOTION_PUBLIC_URL` — 외부 도메인 주입 지원
- `HUMAN_3D_MOTION_HTTPS=0` — TLS를 앞단에 위임 가능
- `StorageRootService.select()`의 `manual_required` 플래그 (`storage_root_service.py:40-46`) — 다이얼로그 실패 시 수동 입력 폴백이 **이미 구현되어 있음**

---

## 2. 배포 아키텍처 3안 비교

### A안. 로컬 실행 + Cloudflare Tunnel  ⭐ 권장

랩 PC에서 `h3dm`를 띄우고 `cloudflared`가 아웃바운드 연결로 Cloudflare에 붙습니다. 공인 IP·포트포워딩·공유기 설정이 전부 불필요합니다.

- ✅ GPU/카메라/디스크 전부 로컬 유지, 연산 비용 0
- ✅ 진짜 도메인 + 정상 TLS 인증서 → **자체서명 경고 사라짐** (폰 브라우저 WebSocket 호환성 크게 개선)
- ✅ 방화벽에 인바운드 구멍 안 뚫음
- ⚠️ **무료/Pro 플랜 업로드 본문 100MB 제한** → 폰 영상 업로드가 막힐 수 있음 (§6 대응)
- ⚠️ 랩 PC가 꺼져 있으면 사이트도 죽음

### B안. 로컬 실행 + Tailscale

- ✅ 100MB 제한 없음, 공개 노출 자체가 없어 보안 최상
- ✅ 무료 (개인 3명 / 100디바이스)
- ❌ 접속자 전원이 Tailscale 설치 + 초대 수락 필요 → "웹사이트"라는 느낌이 안 남
- → **대용량 업로드가 문제되면 A안에서 갈아탈 대안**

### C안. 정적 사이트(다운로드 페이지) + 각자 로컬 설치

Cloudflare Pages/GitHub Pages에 소개·문서·설치파일 배포. 사용자는 `Human3DMotion.exe` / `.app`을 자기 PC에 설치.

- ✅ 이미 PyInstaller 빌드 존재 (`scripts/build.cmd`, `scripts/build.sh`)
- ✅ 서버 운영 부담 0, 저장경로 문제 자연 해결(각자 자기 디스크)
- ❌ 비밀번호 게이트의 의미가 약해짐 (배포물 자체는 공개)
- → **제품화 단계의 최종 형태. A안과 병행 가능**

### 선택 가이드

| 상황 | 선택 |
| --- | --- |
| 소수 인원에게 URL+비번 주고 내 랩 PC를 쓰게 함 | **A안** |
| 사내/연구실 내부 인원만, 대용량 업로드 잦음 | **B안** |
| 외부에 제품으로 나눠줌 | **C안** |

이 문서는 **A안**을 기준으로 작성하고, C안은 §9 Phase 4에 둡니다.

---

## 3. 저장 경로 — 가장 중요한 설계 포인트

### 3.1 먼저 짚어야 할 사실

> **저장 경로는 "브라우저를 켠 PC"가 아니라 "서버가 도는 PC"의 경로입니다.**

브라우저는 보안상 서버에게 클라이언트 디스크 경로를 넘겨줄 수 없습니다. 즉 A안에서는 **모든 영상과 분석 결과가 랩 PC 디스크에 쌓입니다.** 이게 의도한 바라면 그대로 진행하면 되고, "접속한 사람 각자의 PC에 저장"이 목표라면 A안이 아니라 **C안**을 골라야 합니다. (→ §7에서 확정)

### 3.2 지금 그대로 원격 배포하면 발생하는 버그

`capture.html`의 **Select Path** 버튼 → `POST /api/settings/storage-root/select` → `TkinterDirectorySelector.select_directory()`.

원격 사용자가 이 버튼을 누르면:
- macOS 서버: `osascript`가 **랩 PC 화면에** 폴더 선택창을 띄우고, 아무도 안 누르므로 HTTP 요청이 무한 대기
- Windows 서버: tkinter 창이 세션 0 또는 서버 데스크톱에 뜸 (동일 증상)

**이건 배포 시 반드시 고쳐야 하는 실질적 결함입니다.**

### 3.3 Phase 1 — 즉시 조치 (약 20줄, 30분)

원격 모드에서는 네이티브 다이얼로그를 아예 시도하지 않고 곧바로 수동 입력으로 넘깁니다. 프론트엔드 폴백(`capture.ts:266-272`, `analysis.ts:2565-2570`)이 이미 있어서 **서버만 고치면 바로 동작합니다.**

`webapp/infrastructure/system/manual_directory_selector.py` 신규:

```python
from __future__ import annotations


class ManualDirectorySelector:
    """원격 배포 모드용. 네이티브 다이얼로그를 서버 화면에 띄우지 않고
    프론트엔드의 manual_required 폴백으로 넘긴다."""

    def select_directory(self, initial_dir: str) -> str | None:
        raise RuntimeError("folder_dialog_disabled_in_remote_mode")
```

`webapp/bootstrap.py`의 `directory_selector = TkinterDirectorySelector()` (57행) 을 교체:

```python
def _create_directory_selector():
    mode = os.environ.get("HUMAN_3D_MOTION_DIRECTORY_PICKER", "auto").lower()
    if mode == "native":
        return TkinterDirectorySelector()
    if mode == "manual" or os.environ.get("HUMAN_3D_MOTION_PUBLIC_URL", "").strip():
        return ManualDirectorySelector()
    return TkinterDirectorySelector()
```

`HUMAN_3D_MOTION_PUBLIC_URL`이 설정돼 있으면 = 원격 배포 = 자동으로 manual. 로컬 실행 시 동작은 **완전히 그대로**입니다.

### 3.4 Phase 2 — 앱 내장 폴더 브라우저 (권장, 반나절)

`window.prompt`로 경로를 타이핑하게 하는 건 임시방편입니다. 서버 측 디렉터리를 목록으로 보여주고 클릭해서 고르는 UI를 넣습니다.

**보안 요구사항: 비밀번호를 아는 사람이 랩 PC 디스크 전체를 훑을 수 있으면 안 됩니다. 반드시 허용 루트 화이트리스트를 둡니다.**

`webapp/domain/ports.py` 에 포트 추가:

```python
@dataclass(frozen=True)
class DirectoryListing:
    path: str
    parent: str | None
    entries: tuple[str, ...]


class DirectoryBrowser(Protocol):
    def list_directories(self, path: str) -> DirectoryListing:
        ...
```

`webapp/infrastructure/system/local_directory_browser.py` 신규:

```python
from __future__ import annotations

import os
from pathlib import Path

from webapp.domain.ports import DirectoryListing


class LocalDirectoryBrowser:
    def __init__(self, allowed_roots: tuple[Path, ...]) -> None:
        self._allowed_roots = tuple(root.expanduser().resolve() for root in allowed_roots)

    def list_directories(self, path: str) -> DirectoryListing:
        target = self._resolve_within_allowed(path)
        entries = sorted(
            child.name
            for child in target.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
        return DirectoryListing(
            path=str(target),
            parent=self._parent_within_allowed(target),
            entries=tuple(entries),
        )

    def _resolve_within_allowed(self, path: str) -> Path:
        candidate = Path(path).expanduser().resolve() if str(path).strip() else self._allowed_roots[0]
        for root in self._allowed_roots:
            if candidate == root or root in candidate.parents:
                return candidate
        raise PermissionError("path_outside_allowed_roots")

    def _parent_within_allowed(self, target: Path) -> str | None:
        if target in self._allowed_roots:
            return None
        return str(target.parent)


def allowed_roots_from_env() -> tuple[Path, ...]:
    raw = os.environ.get("HUMAN_3D_MOTION_ALLOWED_ROOTS", "").strip()
    if raw:
        return tuple(Path(item) for item in raw.split(os.pathsep) if item.strip())
    return (Path.home() / "Documents" / "Human3DMotion",)
```

라우트 (`flask_app.py`, 비밀번호 게이트 뒤):

```python
@app.get("/api/settings/browse")
def api_browse_directories():
    try:
        listing = directory_browser.list_directories(request.args.get("path") or "")
    except PermissionError:
        return jsonify({"error": "path_outside_allowed_roots"}), 403
    except OSError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"path": listing.path, "parent": listing.parent, "entries": list(listing.entries)})
```

프론트엔드는 `capture.ts` / `calibration.ts` / `analysis.ts`의 `window.prompt` 자리에 공통 모달 컴포넌트(`webapp/frontend/src/directory_picker.ts`)를 넣고, 선택 결과를 기존 `{ manual: true, storage_root }` 요청으로 그대로 전달하면 됩니다. **백엔드 API 시그니처 변경 없음.**

> ⚠️ 같은 화이트리스트를 `/api/analysis/video` (`flask_app.py:180`), `/api/analysis/session-root/select`, 캘리브레이션 업로드 등 **임의 경로를 받는 기존 라우트에도 적용**해야 합니다. 지금은 경로 검증이 없어, 인증만 통과하면 서버의 아무 파일이나 읽힐 수 있습니다.

---

## 4. 비밀번호 게이트 (단일 공용 비밀번호)

### 4.1 설계

- 계정 개념 없음. **비밀번호 1개** → 통과하면 Flask 세션 쿠키 발급
- 평문 비밀번호를 코드/환경변수에 두지 않고 **해시**를 환경변수로 (`werkzeug.security`, scrypt)
- `SECRET_KEY`는 반드시 랜덤 + 영속화 (현재 기본값이 `"dev"` — `flask_app.py:32`. 이대로면 쿠키 위조 가능)
- **Socket.IO 연결도 반드시 함께 막아야 함.** HTTP만 막으면 소켓으로 카메라 상태·프리뷰가 그대로 새어나감
- 폰 캡처 경로는 QR 토큰(`secrets.token_urlsafe(18)` = 144bit)을 자격증명으로 인정 → 폰에서 비밀번호 타이핑 불필요

### 4.2 `webapp/presentation/auth.py` (신규)

```python
from __future__ import annotations

import os
import secrets
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash

_PUBLIC_ENDPOINTS = {"login", "static", "app_logo"}
_PHONE_ENDPOINTS = {"phone_capture_page", "api_upload_phone_video", "api_phone_session_qr"}
_MAX_ATTEMPTS = 10
_LOCKOUT_SECONDS = 300

_attempts: dict[str, tuple[int, float]] = {}


def password_gate_enabled() -> bool:
    return bool(os.environ.get("HUMAN_3D_MOTION_PASSWORD_HASH", "").strip())


def resolve_secret_key(data_dir: Path) -> bytes:
    configured = os.environ.get("HUMAN_3D_MOTION_SECRET", "").strip()
    if configured:
        return configured.encode("utf-8")
    key_path = data_dir / "secret_key"
    if key_path.is_file():
        return key_path.read_bytes()
    key = secrets.token_bytes(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    key_path.chmod(0o600)
    return key


def is_authenticated() -> bool:
    return not password_gate_enabled() or bool(session.get("h3dm_auth"))


def install_password_gate(app: Flask, phone_service) -> None:
    if not password_gate_enabled():
        app.logger.warning("HUMAN_3D_MOTION_PASSWORD_HASH is not set; the app is open to anyone who can reach it.")
        return

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("HUMAN_3D_MOTION_COOKIE_SECURE", "1") not in {"0", "false", "no"},
        PERMANENT_SESSION_LIFETIME=int(os.environ.get("HUMAN_3D_MOTION_SESSION_HOURS", "12")) * 3600,
    )

    @app.before_request
    def _require_password():
        endpoint = request.endpoint or ""
        if endpoint in _PUBLIC_ENDPOINTS or session.get("h3dm_auth"):
            return None
        if endpoint in _PHONE_ENDPOINTS and phone_service.is_valid_token(request.view_args.get("session_token", "")):
            return None
        if request.path.startswith("/api/"):
            return jsonify({"error": "authentication_required"}), 401
        return redirect(url_for("login", next=request.full_path))

    @app.get("/login")
    def login():
        return render_template("login.html", error=None)

    @app.post("/login")
    def submit_login():
        if _locked_out(request.remote_addr):
            return render_template("login.html", error="too_many_attempts"), 429
        password = request.form.get("password", "")
        if check_password_hash(os.environ["HUMAN_3D_MOTION_PASSWORD_HASH"], password):
            _attempts.pop(request.remote_addr or "", None)
            session.clear()
            session.permanent = True
            session["h3dm_auth"] = True
            return redirect(request.args.get("next") or url_for("capture_page"))
        _record_failure(request.remote_addr)
        return render_template("login.html", error="invalid_password"), 401

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))


def _locked_out(remote_addr: str | None) -> bool:
    count, until = _attempts.get(remote_addr or "", (0, 0.0))
    return count >= _MAX_ATTEMPTS and time.time() < until


def _record_failure(remote_addr: str | None) -> None:
    key = remote_addr or ""
    count, _ = _attempts.get(key, (0, 0.0))
    _attempts[key] = (count + 1, time.time() + _LOCKOUT_SECONDS)
```

### 4.3 Socket.IO 게이트

현재 `on_connect`는 무조건 허용합니다 (`flask_app.py:629-631`). 아래처럼 교체:

```python
@socketio.on("connect")
def on_connect(auth=None):
    if not _socket_allowed(auth):
        return False          # False 반환 시 핸드셰이크 거부
    _emit_camera_status(socketio, capture_service)


def _socket_allowed(auth) -> bool:
    if is_authenticated():
        return True
    token = str((auth or {}).get("token") or "")
    return phone_service.is_valid_token(token)
```

폰 클라이언트(`phone_capture.ts`)는 `io({ auth: { token: config.sessionToken } })`로 토큰을 실어 보내도록 수정합니다.

`PhoneCaptureService`에 검증 메서드 추가:

```python
def is_valid_token(self, token: str) -> bool:
    token = str(token or "")
    if not token:
        return False
    return (
        token in self._active_sessions
        or token in self._active_calibrations
        or (self._draft is not None and secrets.compare_digest(token, self._draft.token))
    )
```

### 4.4 CORS 조이기

`cors_allowed_origins="*"` (`flask_app.py:37`) → 공개 URL만 허용:

```python
allowed_origins = os.environ.get("HUMAN_3D_MOTION_PUBLIC_URL", "").strip().rstrip("/") or "*"
socketio = SocketIO(app, cors_allowed_origins=allowed_origins, async_mode="threading", ...)
```

### 4.5 비밀번호 해시 생성 방법

```bash
conda activate human-3d-motion
python -c "from werkzeug.security import generate_password_hash; import getpass; print(generate_password_hash(getpass.getpass('password: ')))"
```

출력된 `scrypt:32768:8:1$...` 문자열을 `HUMAN_3D_MOTION_PASSWORD_HASH`에 넣습니다. **평문은 어디에도 저장하지 않습니다.**

### 4.6 (선택) Cloudflare Access 이중 게이트

Cloudflare Zero Trust의 Access는 50유저까지 무료이고, 앱에 도달하기 **전에** 이메일 OTP로 막아줍니다. 앱 비밀번호와 병행하면 봇 스캔 트래픽이 앱까지 오지 않습니다. 다만 폰 캡처 경로(`/phone-capture/*`, `/api/phone-sessions/*`)는 **bypass 정책으로 예외 처리**해야 QR 흐름이 깨지지 않습니다.

---

## 5. 배포용 실행 설정

### 5.1 환경변수 (`.env.production` — **git 커밋 금지**)

```bash
# 네트워크 — 터널만 접근 가능하도록 루프백 바인딩
HUMAN_3D_MOTION_HOST=127.0.0.1
HUMAN_3D_MOTION_PORT=9090
HUMAN_3D_MOTION_HTTPS=0                              # TLS는 Cloudflare가 종료
HUMAN_3D_MOTION_PUBLIC_URL=https://h3dm.example.com  # QR·리다이렉트가 이 주소를 씀
HUMAN_3D_MOTION_OPEN_BROWSER=0

# 인증
HUMAN_3D_MOTION_PASSWORD_HASH='scrypt:32768:8:1$...'
HUMAN_3D_MOTION_SECRET=                              # 비우면 webapp_data/secret_key 자동 생성
HUMAN_3D_MOTION_COOKIE_SECURE=1
HUMAN_3D_MOTION_SESSION_HOURS=12

# 저장 경로
HUMAN_3D_MOTION_STORAGE=/Users/yongseok/Documents/H3DM-Data
HUMAN_3D_MOTION_ALLOWED_ROOTS=/Users/yongseok/Documents/H3DM-Data
HUMAN_3D_MOTION_DIRECTORY_PICKER=manual
```

`.gitignore`에 `.env*` 추가 필요.

### 5.2 `~/.cloudflared/config.yml`

```yaml
tunnel: h3dm
credentials-file: /Users/yongseok/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: h3dm.example.com
    service: http://127.0.0.1:9090
    originRequest:
      connectTimeout: 30s
      noHappyEyeballs: true
  - service: http_status:404
```

### 5.3 상시 구동 (macOS launchd 예시)

`~/Library/LaunchAgents/com.h3dm.app.plist` 로 앱을, `brew services start cloudflared` 로 터널을 상시 구동합니다. Windows는 `nssm` 또는 작업 스케줄러(시스템 시작 시 실행)를 씁니다.

---

## 6. 알려진 제약과 대응

| 제약 | 영향 | 대응 |
| --- | --- | --- |
| **Cloudflare 무료 플랜 요청 본문 100MB** | 폰 영상은 단일 multipart POST (`phone_capture.ts:278-290`). 720p/60fps 30초면 이미 위험권 | ① 폰만 LAN 직결(`https://<랩PC IP>:9090`)로 쓰고 분석 화면만 터널 사용 ② 청크 업로드로 개선 ③ **B안(Tailscale)으로 전환** |
| Werkzeug 개발 서버 (`allow_unsafe_werkzeug=True`, `main.py:56`) | 동시 접속·대용량 업로드에 취약 | 소수 인원이면 허용 가능. 인원 늘면 `eventlet`/`gevent` 또는 `waitress` 전환 검토 |
| 랩 PC 전원/네트워크 = 서비스 가용성 | PC 꺼지면 사이트 다운 | 절전 해제, UPS, 상시 구동 설정 |
| 전역 상태 단일 세션 (`capture_service.active_capture()`) | 두 사람이 동시에 캡처 시작하면 서로 덮어씀 | v1은 "동시 1명" 운영 규칙으로 회피. 멀티테넌시는 별도 과제 |
| 자체서명 인증서 + 폰 | WebSocket/getUserMedia 차단 사례 | 터널 사용 시 정상 인증서로 해결됨 (터널의 큰 이점) |
| `webapp_data/settings.json` 전역 공유 | 접속자가 서로의 저장경로/카메라 설정을 바꿈 | 동일. v1은 운영 규칙으로 회피 |

---

## 7. ⚠️ 먼저 확정해야 할 질문 — "파일이 누구 디스크에 쌓이는가"

이 답에 따라 전체 계획이 갈립니다.

| 답 | 해당 시나리오 | 선택할 안 |
| --- | --- | --- |
| **랩 PC 한 곳에 모인다** | 내 카메라·내 GPU를 남들이 원격으로 쓰는 형태 | **A안** (이 문서 본문) |
| **접속자 각자의 PC에 저장된다** | 각자 자기 장비로 촬영·분석 | **C안** (설치형 배포) — A안으로는 원리상 불가능 |
| **둘 다** | 랩 PC는 시연·공용, 진짜 사용자는 설치형 | A안 먼저 → Phase 4에서 C안 추가 |

---

## 8. ⚠️ 라이선스 — 배포 전 반드시 결정

README(§Licensing note)에 이미 적혀 있는 내용이지만, **"배포"를 하는 순간 성격이 달라지므로** 여기 다시 정리합니다.

### Ultralytics YOLO — AGPL-3.0 (핵심 이슈)

AGPL은 **네트워크를 통해 서비스를 제공하는 것만으로도** 소스 공개 의무가 발생합니다. 즉 A안으로 URL을 열어 남에게 쓰게 하는 순간 트리거됩니다. 선택지:

1. **H3DM 전체 소스를 AGPL-3.0으로 공개** — 가장 간단. 연구용/오픈소스면 문제 없음
2. **Ultralytics 상용 라이선스 구매** — 비공개 유지 가능. 유료
3. **디텍터를 Apache-2.0 모델로 교체** — 추천 대안

3번이 생각보다 쉽습니다. `WrappingDetector`(`models.py:25-34`)가 `(frame_bgr) -> Nx4 float32 배열` 한 가지 계약만 노출하므로, `setup_detector`만 갈아끼우면 됩니다. 이미 의존성에 들어 있는 `rtmlib`이 **RTMDet / YOLOX (둘 다 Apache-2.0)** 를 제공하므로 신규 의존성도 필요 없습니다.

```python
# models.py — 교체 스케치
from rtmlib.tools.object_detection import RTMDet

def setup_detector(device, det_score_threshold, det_iou, det_nms, mode: str = "normal"):
    detector = RTMDet(str(MODEL_DIR / mode / "rtmdet.onnx"), backend=backend, device=device)
    return detector, {}
```

정확도 검증(기존 YOLO 대비 keypoint RMSE 비교)이 필요하므로 **Phase 3에 별도 작업으로 잡아둡니다.**

### VideoPose3D — CC BY-NC 4.0

비상업 전용. 자동 캘리브레이션에서만 쓰이고 체크포인트를 번들하지 않으므로 **상업 배포 시 자동 캘리브레이션 기능을 끄면** 회피됩니다. Object/CheckerBoard 캘리브레이션은 영향 없습니다.

> 법률 자문이 아닙니다. 상업적 배포라면 라이선스 결정권자의 확인을 받으세요.

---

## 9. 실행 계획

### Phase 0 — 결정 (30분, 코드 변경 없음)
- [ ] §7 저장 디스크 주체 확정 → A안/C안 확정
- [ ] §8 AGPL 대응 방침 확정 (공개 / 구매 / 교체)
- [ ] 도메인 확보 + Cloudflare에 네임서버 위임

### Phase 1 — 최소 배포 가능 상태 (반나절) ★ 여기까지만 해도 배포 가능
- [ ] `ManualDirectorySelector` 추가 + `bootstrap.py` 선택 로직 (§3.3)
- [ ] `auth.py` 비밀번호 게이트 + `login.html` 템플릿 (§4.2)
- [ ] `SECRET_KEY` 랜덤 생성·영속화 (§4.2 `resolve_secret_key`)
- [ ] Socket.IO `connect` 게이트 + `PhoneCaptureService.is_valid_token` (§4.3)
- [ ] CORS를 공개 URL로 제한 (§4.4)
- [ ] `.gitignore`에 `.env*` 추가
- [ ] 로컬에서 `HUMAN_3D_MOTION_PUBLIC_URL` 세팅 후 회귀 테스트: 캡처 → 캘리브레이션 → 분석 → 리포트 전 구간

### Phase 2 — 원격 사용성 (1일)
- [ ] `LocalDirectoryBrowser` + `/api/settings/browse` + 화이트리스트 (§3.4)
- [ ] 프론트 폴더 선택 모달 (`directory_picker.ts`), 3개 페이지 `window.prompt` 교체
- [ ] **기존 경로 수용 라우트 전부에 화이트리스트 검증 적용** (`/api/analysis/video`, 캘리브레이션 업로드 등)
- [ ] `cloudflared` 터널 구성 + 상시 구동 등록 (§5.2, §5.3)
- [ ] 폰 QR 흐름을 공개 URL로 실기기 검증 (100MB 제한 실측 포함)

### Phase 3 — 안정화 (1~2일)
- [ ] AGPL 대응 실행 (Phase 0 결정에 따라 — 디텍터 교체 시 정확도 회귀 비교)
- [ ] 업로드 100MB 제한 대응 (청크 업로드 or LAN 직결 운영 문서화)
- [ ] 로그인 실패 로깅 / 접속 로그 확인 절차
- [ ] 백업 정책 (`storage_root` + `webapp_data/settings.json`)
- [ ] 운영 문서: 재시작 방법, 비밀번호 교체 절차, 장애 대응

### Phase 4 — 배포 확장 (선택)
- [ ] Cloudflare Pages에 소개·다운로드 페이지 (C안)
- [ ] `scripts/build.cmd` / `build.sh` 산출물 릴리스 자동화
- [ ] macOS 코드 서명·공증 (현재 미적용 — README 126행)

---

## 10. 비용

| 항목 | 비용 |
| --- | --- |
| 도메인 (.com 기준) | ~₩15,000 / 년 |
| Cloudflare Tunnel | 무료 |
| Cloudflare Access (50유저) | 무료 |
| Cloudflare Pages | 무료 |
| GPU/서버 | **₩0** (기존 랩 PC 사용) |
| 전기·인터넷 | 기존 비용에 포함 |
| **합계** | **연 ~₩15,000** |

비교: 클라우드 GPU(T4급) 상시 구동 시 월 $200~400 = 연 ₩300만~700만.

---

## 11. 참고 — 변경 대상 파일

| 파일 | 변경 내용 | Phase |
| --- | --- | --- |
| `webapp/infrastructure/system/manual_directory_selector.py` | 신규 | 1 |
| `webapp/infrastructure/system/local_directory_browser.py` | 신규 | 2 |
| `webapp/infrastructure/system/__init__.py` | export 추가 | 1, 2 |
| `webapp/presentation/auth.py` | 신규 | 1 |
| `webapp/presentation/templates/login.html` | 신규 | 1 |
| `webapp/presentation/flask_app.py` | 게이트 설치, SECRET_KEY, CORS, socket connect, browse 라우트, 경로 검증 | 1, 2 |
| `webapp/bootstrap.py` | selector 분기, browser 주입 | 1, 2 |
| `webapp/domain/ports.py` | `DirectoryBrowser`, `DirectoryListing` | 2 |
| `webapp/application/phone_capture_service.py` | `is_valid_token` | 1 |
| `webapp/frontend/src/directory_picker.ts` | 신규 | 2 |
| `webapp/frontend/src/{capture,calibration,analysis}.ts` | prompt → 모달 | 2 |
| `webapp/frontend/src/phone_capture.ts` | socket auth 토큰 | 1 |
| `pipelines/pose_estimation/models.py` | 디텍터 교체 (선택) | 3 |
| `.gitignore` | `.env*` | 1 |
| `README.md` | 배포 섹션 추가 | 3 |
