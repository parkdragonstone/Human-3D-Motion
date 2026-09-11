#!/bin/sh
# 서버 기동 방식을 HUMAN_3D_MOTION_HTTPS 로 고른다.
#
#   1 (기본) : 앱이 직접 자체서명 TLS 를 띄운다. 폰 캡처의 getUserMedia 는
#              보안 컨텍스트(HTTPS)를 요구하므로 로컬 사용의 기본값이다.
#              webapp/main.py 가 socketio.run(ssl_context="adhoc") 으로 뜬다.
#   0        : gunicorn 으로 평문 HTTP 를 서빙한다. TLS 를 리버스 프록시가
#              종료하는 서버 배포용.
#
# 워커는 어느 쪽이든 1개여야 한다. 분석 잡 상태·캡처 세션·폰 토큰이 전부
# 프로세스 메모리에 있어서(AnalysisJobService._jobs 등) 워커가 둘이면
# /api/analysis/jobs/<id> 가 404 를 뱉는다.
set -e

case "${HUMAN_3D_MOTION_HTTPS:-1}" in
	0 | false | no | FALSE | NO)
		echo "[h3dm] gunicorn (HTTP) — TLS 는 앞단 프록시가 종료한다고 가정한다"
		exec gunicorn \
			--worker-class gthread \
			--workers 1 \
			--threads 8 \
			--timeout 0 \
			--access-logfile - \
			--error-logfile - \
			--bind "0.0.0.0:${HUMAN_3D_MOTION_PORT:-9090}" \
			webapp.main:app
		;;
	*)
		echo "[h3dm] 내장 서버 (자체서명 HTTPS) — 폰 캡처를 쓰려면 이 모드여야 한다"
		echo "[h3dm] 폰 브라우저에서 인증서 경고를 한 번 수락해야 한다"
		exec python -m webapp.main
		;;
esac
