import logging
import os
import socket
import threading
import time
import webbrowser

from webapp.presentation.flask_app import create_app


app, socketio = create_app()


class _WerkzeugAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in ("/socket.io/", " /static/", " /favicon.ico "))


def main() -> None:
    if os.environ.get("HUMAN_3D_MOTION_VERBOSE_ACCESS_LOG", "0").lower() not in {"1", "true", "yes"}:
        logging.getLogger("werkzeug").addFilter(_WerkzeugAccessLogFilter())
    https_value = os.environ.get("HUMAN_3D_MOTION_HTTPS", "1").lower()
    ssl_context = None if https_value in {"0", "false", "no"} else "adhoc"
    public_url = os.environ.get("HUMAN_3D_MOTION_PUBLIC_URL", "").strip()
    host = os.environ.get("HUMAN_3D_MOTION_HOST", "0.0.0.0")
    port = int(os.environ.get("HUMAN_3D_MOTION_PORT", "9090"))
    browser_url = _browser_url(public_url, host, ssl_context, port)
    print(f"Human 3D Motion URL: {browser_url}", flush=True)
    if os.environ.get("HUMAN_3D_MOTION_OPEN_BROWSER", "1").lower() not in {"0", "false", "no"}:
        threading.Thread(target=_open_browser, args=(browser_url,), daemon=True).start()
    socketio.run(
        app,
        host=host,
        port=port,
        debug=os.environ.get("HUMAN_3D_MOTION_DEBUG") == "1",
        use_reloader=False,
        allow_unsafe_werkzeug=True,
        ssl_context=ssl_context,
    )


def _browser_url(public_url: str, host: str, ssl_context, port: int) -> str:
    if public_url:
        return public_url.rstrip("/") + "/"
    scheme = "http" if ssl_context is None else "https"
    browser_host = os.environ.get("HUMAN_3D_MOTION_BROWSER_HOST", "").strip() or _browser_host(host)
    return f"{scheme}://{browser_host}:{port}"


def _browser_host(host: str) -> str:
    if host and host not in {"0.0.0.0", "::"}:
        return host
    return _local_ipv4_address()


def _local_ipv4_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if address:
                return address
    except OSError:
        pass
    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass
    return "127.0.0.1"


def _open_browser(url: str) -> None:
    time.sleep(1.0)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
