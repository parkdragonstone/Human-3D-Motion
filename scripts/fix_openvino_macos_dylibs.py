"""Repair OpenVINO dylibs that macOS ``codesign`` refuses to re-sign.

The OpenVINO wheel ships at least one dylib (``libtbb.12.dylib``) whose
``__LINKEDIT`` segment is declared a few bytes longer than the end of its
embedded code signature. ``codesign --force`` fails on such a file with
"internal error in Code Signing subsystem", which breaks the PyInstaller
build because PyInstaller ad-hoc signs every binary it collects.

Trimming ``__LINKEDIT`` down to the end of the signature (and truncating the
trailing padding) makes the file signable again. Only the padding after the
signature is removed, so the loadable content of the library is untouched.

The script is idempotent: dylibs that are already consistent are left alone.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
LC_CODE_SIGNATURE = 0x1D


def repair(path: Path) -> bool:
    """Trim __LINKEDIT to the end of the code signature. Return True if changed."""
    data = bytearray(path.read_bytes())
    if len(data) < 32 or struct.unpack_from("<I", data, 0)[0] != MH_MAGIC_64:
        return False  # not a thin 64-bit Mach-O (fat/other archs are left alone)

    ncmds = struct.unpack_from("<I", data, 16)[0]
    offset = 32
    linkedit_offset = linkedit_fileoff = linkedit_filesize = None
    signature_end = None

    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<2I", data, offset)
        if cmd == LC_SEGMENT_64:
            name = bytes(data[offset + 8:offset + 24]).rstrip(b"\0").decode()
            if name == "__LINKEDIT":
                linkedit_offset = offset
                linkedit_fileoff, linkedit_filesize = struct.unpack_from("<2Q", data, offset + 40)
        elif cmd == LC_CODE_SIGNATURE:
            dataoff, datasize = struct.unpack_from("<2I", data, offset + 8)
            signature_end = dataoff + datasize
        offset += cmdsize

    if linkedit_offset is None or signature_end is None:
        return False
    if signature_end >= linkedit_fileoff + linkedit_filesize:
        return False  # already consistent

    struct.pack_into("<Q", data, linkedit_offset + 48, signature_end - linkedit_fileoff)
    del data[signature_end:]
    path.write_bytes(data)
    # Re-sign ad-hoc: the original signature no longer matches the load commands.
    subprocess.run(
        ["/usr/bin/codesign", "-s", "-", "--force", "--all-architectures", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main() -> int:
    if sys.platform != "darwin":
        return 0

    try:
        import openvino
    except ImportError:
        print("OpenVINO is not installed; nothing to repair.")
        return 0

    lib_dir = Path(openvino.__file__).resolve().parent / "libs"
    if not lib_dir.is_dir():
        print(f"OpenVINO lib directory not found: {lib_dir}")
        return 0

    repaired = [dylib.name for dylib in sorted(lib_dir.glob("*.dylib")) if repair(dylib)]
    if repaired:
        print("Repaired OpenVINO dylibs for codesign:", ", ".join(repaired))
    else:
        print("OpenVINO dylibs are already signable; no repair needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
