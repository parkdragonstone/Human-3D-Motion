#!/usr/bin/env bash
set -euo pipefail

HOST_IP=""
STORAGE_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-ip)
      HOST_IP="${2:-}"
      if [[ -z "$HOST_IP" ]]; then
        echo "--host-ip requires an IPv4 address." >&2
        exit 1
      fi
      shift 2
      ;;
    --storage-dir)
      STORAGE_DIR="${2:-}"
      if [[ -z "$STORAGE_DIR" ]]; then
        echo "--storage-dir requires a WSL path." >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Usage: bash docker/run_cpu.sh [--host-ip <windows-lan-ip>] [--storage-dir <wsl-path>]" >&2
      exit 1
      ;;
  esac
done

detect_windows_lan_ip() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "powershell.exe is unavailable in WSL. Pass --host-ip explicitly." >&2
    exit 1
  fi

  mapfile -t candidates < <(
    powershell.exe -NoProfile -Command '
      Get-NetIPConfiguration |
        Where-Object { $_.NetAdapter.Status -eq "Up" -and $_.IPv4DefaultGateway -and $_.IPv4Address } |
        ForEach-Object {
          $config = $_
          foreach ($address in $config.IPv4Address) {
            if ($address.IPAddress -notlike "127.*" -and $address.IPAddress -notlike "169.254.*") {
              Write-Output "$($config.InterfaceAlias)`t$($address.IPAddress)"
            }
          }
        }
    ' | tr -d '\r'
  )

  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No Windows LAN IPv4 address with a default gateway was found. Pass --host-ip explicitly." >&2
    exit 1
  fi

  echo "Detected Windows host LAN IPv4 candidates:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  printf '%s\n' "${candidates[0]##*$'\t'}"
}

select_windows_storage_dir() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "powershell.exe is unavailable in WSL. Pass --storage-dir explicitly." >&2
    exit 1
  fi

  windows_dir="$(
    powershell.exe -NoProfile -Command '
      Add-Type -AssemblyName System.Windows.Forms
      $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
      $dialog.Description = "Select Baseball Motion storage folder"
      $dialog.ShowNewFolderButton = $true
      if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        Write-Output $dialog.SelectedPath
      }
    ' | tr -d '\r'
  )"

  if [[ -z "$windows_dir" ]]; then
    echo "Storage folder selection was cancelled." >&2
    exit 1
  fi

  wslpath -u "$windows_dir"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SELECTED_HOST_IP="${HOST_IP:-$(detect_windows_lan_ip)}"
SELECTED_STORAGE_DIR="${STORAGE_DIR:-$(select_windows_storage_dir)}"

export BASEBALL_MOTION_HTTPS="1"
export BASEBALL_MOTION_PUBLIC_URL="https://${SELECTED_HOST_IP}:5000"
export BASEBALL_MOTION_RECORDINGS_HOST="$SELECTED_STORAGE_DIR"

echo "Selected Docker host LAN URL: ${BASEBALL_MOTION_PUBLIC_URL}"
echo "Selected Windows storage folder mount: ${BASEBALL_MOTION_RECORDINGS_HOST} -> /app/recordings"
echo "Local Docker URL: https://127.0.0.1:5000"
echo "WSL LAN access may require mirrored networking or a Windows portproxy/firewall rule for port 5000."

docker compose -f "$REPO_ROOT/docker/docker-compose.yml" --profile cpu up --build --force-recreate
