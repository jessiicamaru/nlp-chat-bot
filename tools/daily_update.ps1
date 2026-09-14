<#
.SYNOPSIS
  Cập nhật dữ liệu hằng ngày cho chatbot tin tức: sao lưu -> crawl -> dựng lại index -> kiểm thử.

.DESCRIPTION
  Xem hướng dẫn đầy đủ ở docs/08-runbook-cap-nhat-du-lieu.md.

  Mã thoát:
    0  cập nhật thành công, kiểm thử đạt
    1  lỗi — corpus đã được KHÔI PHỤC về bản sao lưu trước khi crawl
    2  dữ liệu đã cập nhật nhưng kiểm thử hồi quy có câu không đạt -> cần xem log

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1
  powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -PerCategory 30 -RestartApi
  powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -SkipCrawl      # chỉ dựng lại + kiểm thử
#>
param(
    [int]$PerCategory = 20,      # số bài MỚI tối đa mỗi chuyên mục trong một lần chạy
    [switch]$SkipCrawl,          # bỏ bước crawl (dùng khi chỉ muốn dựng lại index / kiểm thử)
    [switch]$SkipTests,          # bỏ kiểm thử hồi quy (nhanh hơn, không khuyến nghị)
    [switch]$RestartApi,         # khởi động lại web server (src/api.py) để nạp dữ liệu mới
    [int]$KeepBackups = 14       # số bản sao lưu corpus giữ lại
)

# 'Continue' chứ không phải 'Stop': trên Windows PowerShell 5.1, cảnh báo Python in
# ra stderr sẽ bị coi là lỗi dừng script. Mã thoát được kiểm tra thủ công qua $LASTEXITCODE.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # đọc đúng tiếng Việt từ Python
$env:PYTHONIOENCODING = 'utf-8'

$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project
$Py = Join-Path $Project '.venv\Scripts\python.exe'
$Corpus = Join-Path $Project 'data\raw\corpus_raw.csv'
$BackupDir = Join-Path $Project 'data\raw\backups'
$LogDir = Join-Path $Project 'logs'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Log = Join-Path $LogDir "daily_update_$Stamp.log"

New-Item -ItemType Directory -Force -Path $BackupDir, $LogDir | Out-Null

function Write-Log([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Invoke-Py([string]$Title, [string[]]$PyArgs) {
    Write-Log "=== $Title"
    $output = & $Py @PyArgs
    $code = $LASTEXITCODE
    foreach ($l in $output) { Write-Host $l }
    if ($output) { Add-Content -Path $Log -Value $output -Encoding UTF8 }
    Write-Log "    -> mã thoát $code"
    return $code
}

function Get-CorpusCount {
    return @(Import-Csv -Path $Corpus).Count
}

function Restore-Corpus([string]$BackupFile, [string]$Reason) {
    Write-Log "LỖI: $Reason"
    Copy-Item -Path $BackupFile -Destination $Corpus -Force
    Write-Log "Đã khôi phục corpus từ $BackupFile"
    Invoke-Py 'Dựng lại index trên corpus đã khôi phục' @('tools/rebuild_index.py') | Out-Null
}

# ---------------------------------------------------------------- 0. kiểm tra
Write-Log "Bắt đầu cập nhật (PerCategory=$PerCategory, SkipCrawl=$SkipCrawl, SkipTests=$SkipTests, RestartApi=$RestartApi)"
if (-not (Test-Path $Py))     { Write-Log "LỖI: không thấy $Py — hãy tạo .venv (docs/08, mục 2)"; exit 1 }
if (-not (Test-Path $Corpus)) { Write-Log "LỖI: không thấy $Corpus"; exit 1 }

# ---------------------------------------------------------------- 1. sao lưu
$before = Get-CorpusCount
$backup = Join-Path $BackupDir "corpus_raw_$Stamp.csv"
Copy-Item -Path $Corpus -Destination $backup
Write-Log "Sao lưu corpus ($before bài) -> $backup"

# ---------------------------------------------------------------- 2. crawl
if (-not $SkipCrawl) {
    $code = Invoke-Py "Crawl VnExpress (tối đa $PerCategory bài mới/chuyên mục)" @('src/crawler.py', '--per-category', "$PerCategory")
    if ($code -ne 0) { Restore-Corpus $backup "crawler thoát với mã $code"; exit 1 }
}

$after = Get-CorpusCount
if ($after -lt $before) { Restore-Corpus $backup "corpus bị giảm từ $before xuống $after bài"; exit 1 }
Write-Log "Corpus: $before -> $after bài (+$($after - $before))"

# ---------------------------------------------------------------- 3. dựng lại index
$code = Invoke-Py 'Dựng lại index + kiểm tra khói' @('tools/rebuild_index.py')
if ($code -ne 0) { Restore-Corpus $backup "rebuild_index thoát với mã $code"; exit 1 }

# ---------------------------------------------------------------- 4. kiểm thử
$exitCode = 0
if (-not $SkipTests) {
    $t1 = Invoke-Py 'Kiểm thử TF-IDF / BM25' @('tests/test_vectorizer.py')
    $t2 = Invoke-Py 'Kiểm thử hồi quy chatbot' @('tests/test_chatbot.py')
    if ($t1 -ne 0 -or $t2 -ne 0) {
        Write-Log 'CẢNH BÁO: có kiểm thử không đạt. Dữ liệu mới VẪN được giữ.'
        Write-Log '          Xem docs/08 mục 7: bài mới có thể được xếp trên bài mà kiểm thử mong đợi.'
        $exitCode = 2
    }
}

# ---------------------------------------------------------------- 5. dọn sao lưu cũ
Get-ChildItem -Path $BackupDir -Filter 'corpus_raw_*.csv' |
    Sort-Object Name -Descending |
    Select-Object -Skip $KeepBackups |
    ForEach-Object { Remove-Item $_.FullName; Write-Log "Xóa sao lưu cũ $($_.Name)" }

# ---------------------------------------------------------------- 6. web server
if ($RestartApi) {
    $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'src[\\/]api\.py' }
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force
        Write-Log "Dừng web server cũ (PID $($p.ProcessId))"
    }
    $server = Start-Process -FilePath $Py -ArgumentList 'src/api.py' -WorkingDirectory $Project `
        -WindowStyle Hidden -PassThru
    Write-Log "Khởi động web server mới (PID $($server.Id)) -> http://127.0.0.1:8000"
} else {
    Write-Log 'Nhắc: web server đang chạy (nếu có) vẫn dùng dữ liệu CŨ — khởi động lại src/api.py để nạp dữ liệu mới.'
}

Write-Log "Kết thúc với mã $exitCode. Log: $Log"
exit $exitCode
