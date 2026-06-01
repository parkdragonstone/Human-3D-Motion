from __future__ import annotations

from pathlib import Path

from PIL import Image


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "images" / "human-3d-motion.png"
    icon_path = repo_root / "packaging" / "app_icon.icns"
    if not source_path.is_file():
        raise FileNotFoundError(f"App logo source is missing: {source_path}")

    with Image.open(source_path) as source:
        source.convert("RGBA").save(icon_path, format="ICNS")

    print(f"macOS app icon generated: {icon_path}")


if __name__ == "__main__":
    main()
