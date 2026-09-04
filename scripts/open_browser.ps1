$url = "http://127.0.0.1:8000"
$health = "$url/api/health"

for ($i = 0; $i -lt 120; $i++) {
    try {
        Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 1 | Out-Null
        Start-Process $url
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
