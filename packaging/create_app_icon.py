from __future__ import annotations

from pathlib import Path

from PIL import Image


ICON_SIZES = ((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "images" / "baseball-motion.png"
    icon_path = repo_root / "packaging" / "app_icon.ico"
    if not source_path.is_file():
        raise FileNotFoundError(f"App logo source is missing: {source_path}")

    with Image.open(source_path) as source:
        source.convert("RGBA").save(icon_path, sizes=ICON_SIZES)

    print(f"App icon generated: {icon_path}")


if __name__ == "__main__":
    main()
