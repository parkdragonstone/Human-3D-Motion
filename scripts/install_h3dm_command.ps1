param(
    [string]$EnvironmentName = "human-3d-motion"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

function Assert-LastCommandSucceeded {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$prefixOutput = conda run -n $EnvironmentName python -c "import sys; print(sys.prefix)"
Assert-LastCommandSucceeded "Resolving Conda environment prefix"
$environmentPrefix = (
    $prefixOutput |
        Where-Object { $_ -and $_.Trim() } |
        Select-Object -Last 1
).Trim()
if (-not $environmentPrefix) {
    throw "Could not resolve Conda environment prefix for '$EnvironmentName'."
}

$scriptsDir = Join-Path $environmentPrefix "Scripts"
New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null

$commandPath = Join-Path $scriptsDir "h3dm.exe"
$tempCommandPath = Join-Path $scriptsDir "h3dm.tmp.exe"
$legacyCommandPath = Join-Path $scriptsDir "h3dm.cmd"
$repoRootLiteral = $repoRoot.Replace('"', '""')
$source = @"
using System;
using System.Diagnostics;
using System.IO;
using System.Text;

public static class H3dmLauncher
{
    public static int Main(string[] args)
    {
        Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
        };

        string repoRoot = Environment.GetEnvironmentVariable("H3DM_REPO_ROOT");
        if (string.IsNullOrWhiteSpace(repoRoot))
        {
            repoRoot = @"$repoRootLiteral";
        }
        Directory.SetCurrentDirectory(repoRoot);

        string launcherPath = System.Reflection.Assembly.GetExecutingAssembly().Location;
        string scriptDirectory = Path.GetDirectoryName(launcherPath);
        string environmentRoot = Path.GetDirectoryName(scriptDirectory);
        string pythonPath = Path.Combine(environmentRoot, "python.exe");

        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.FileName = File.Exists(pythonPath) ? pythonPath : "python";
        startInfo.Arguments = "-u -m webapp.main" + BuildArguments(args);
        startInfo.WorkingDirectory = repoRoot;
        startInfo.UseShellExecute = false;
        startInfo.RedirectStandardOutput = true;
        startInfo.RedirectStandardError = true;

        using (Process process = new Process())
        {
            process.StartInfo = startInfo;
            process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
            {
                if (e.Data != null)
                {
                    Console.Out.WriteLine(e.Data);
                }
            };
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
            {
                if (e.Data != null)
                {
                    Console.Error.WriteLine(e.Data);
                }
            };

            if (!process.Start())
            {
                return 1;
            }
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
            return process.ExitCode;
        }
    }

    private static string BuildArguments(string[] args)
    {
        if (args == null || args.Length == 0)
        {
            return "";
        }

        StringBuilder builder = new StringBuilder();
        foreach (string arg in args)
        {
            builder.Append(' ');
            builder.Append(QuoteArgument(arg ?? ""));
        }
        return builder.ToString();
    }

    private static string QuoteArgument(string value)
    {
        if (value.Length > 0 && value.IndexOfAny(new char[] { ' ', '\t', '\n', '\v', '"' }) < 0)
        {
            return value;
        }

        StringBuilder builder = new StringBuilder();
        builder.Append('"');
        int backslashCount = 0;
        foreach (char current in value)
        {
            if (current == '\\')
            {
                backslashCount++;
            }
            else if (current == '"')
            {
                builder.Append('\\', backslashCount * 2 + 1);
                builder.Append('"');
                backslashCount = 0;
            }
            else
            {
                builder.Append('\\', backslashCount);
                builder.Append(current);
                backslashCount = 0;
            }
        }
        builder.Append('\\', backslashCount * 2);
        builder.Append('"');
        return builder.ToString();
    }
}
"@

Remove-Item -Force -ErrorAction SilentlyContinue $tempCommandPath
Add-Type -TypeDefinition $source -OutputAssembly $tempCommandPath -OutputType ConsoleApplication
Move-Item -Force -Path $tempCommandPath -Destination $commandPath
Remove-Item -Force -ErrorAction SilentlyContinue $legacyCommandPath
Write-Host "Installed h3dm command: $commandPath"
