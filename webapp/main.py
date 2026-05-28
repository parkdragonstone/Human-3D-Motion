import os
import logging

from webapp.presentation.flask_app import create_app


app, socketio = create_app()


class _WerkzeugAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in ("/socket.io/", " /static/", " /favicon.ico "))


if __name__ == "__main__":
    if os.environ.get("BASEBALL_MOTION_VERBOSE_ACCESS_LOG", "0").lower() not in {"1", "true", "yes"}:
        logging.getLogger("werkzeug").addFilter(_WerkzeugAccessLogFilter())
    https_value = os.environ.get("BASEBALL_MOTION_HTTPS", "1").lower()
    ssl_context = None if https_value in {"0", "false", "no"} else "adhoc"
    public_url = os.environ.get("BASEBALL_MOTION_PUBLIC_URL", "").strip()
    if public_url:
        print(f"Baseball Motion public URL: {public_url}", flush=True)
    socketio.run(
        app,
        host=os.environ.get("BASEBALL_MOTION_HOST", "0.0.0.0"),
        port=int(os.environ.get("BASEBALL_MOTION_PORT", "9090")),
        debug=os.environ.get("BASEBALL_MOTION_DEBUG") == "1",
        use_reloader=False,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context,
    )
