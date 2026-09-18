$ErrorActionPreference = 'Stop'
$taskOut = Join-Path $PSScriptRoot 'evidence'
New-Item -ItemType Directory -Force -Path $taskOut | Out-Null
$taskEnv = @{}
Get-Content (Join-Path $PSScriptRoot '../../.env') | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)$') {
        $taskEnv[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'")
    }
}
$taskHeaders = @{'APCA-API-KEY-ID'=$taskEnv['API_KEY']; 'APCA-API-SECRET-KEY'=$taskEnv['API_SECRET']}
function Save-Read($name, $uri, $headers) {
    try {
        $result = Invoke-RestMethod -Uri $uri -Headers $headers -TimeoutSec 45
        ConvertTo-Json -InputObject $result -Depth 30 | Set-Content -LiteralPath (Join-Path $taskOut "$name.json") -Encoding utf8
        Write-Output "$name saved"
        return $result
    } catch { Write-Output "$name failed: $($_.Exception.Message)"; return $null }
}
$account = Invoke-RestMethod 'https://paper-api.alpaca.markets/v2/account' -Headers $taskHeaders
$account | Select-Object status,created_at,currency,cash,equity,last_equity,long_market_value,short_market_value | ConvertTo-Json | Set-Content (Join-Path $taskOut 'account.json')
Save-Read 'clock' 'https://paper-api.alpaca.markets/v2/clock' $taskHeaders | Out-Null
Save-Read 'history' 'https://paper-api.alpaca.markets/v2/account/portfolio/history?start=2026-07-17T00%3A00%3A00-04%3A00&timeframe=1D&cashflow_types=ALL' $taskHeaders | Out-Null
Save-Read 'positions' 'https://paper-api.alpaca.markets/v2/positions' $taskHeaders | Out-Null
Save-Read 'orders' 'https://paper-api.alpaca.markets/v2/orders?status=all&after=2026-07-17T00%3A00%3A00Z&limit=500&direction=asc&nested=true' $taskHeaders | Out-Null
$activities = @()
$page = ''
do {
    $uri = 'https://paper-api.alpaca.markets/v2/account/activities?after=2026-07-17T00%3A00%3A00Z&page_size=100&direction=asc' + $page
    $batch = @(Invoke-RestMethod $uri -Headers $taskHeaders -TimeoutSec 45 | ForEach-Object { $_ })
    $activities += $batch
    if ($batch.Count -eq 100) { $page = '&page_token=' + [uri]::EscapeDataString($batch[-1].id) }
} while ($batch.Count -eq 100)
ConvertTo-Json -InputObject $activities -Depth 20 | Set-Content (Join-Path $taskOut 'activities.json') -Encoding utf8
Save-Read 'workflow_runs' 'https://api.github.com/repos/sbalta01/alpaca-quant-trading/actions/workflows/deploying-weekly-momentum.yml/runs?per_page=100' @{} | Out-Null
Save-Read 'aug28_jobs' 'https://api.github.com/repos/sbalta01/alpaca-quant-trading/actions/runs/33230922926/jobs' @{} | Out-Null
Save-Read 'benchmark_bars' 'https://data.alpaca.markets/v2/stocks/bars?symbols=SPY%2CQQQ%2CBIL&timeframe=1Day&start=2026-07-17T00%3A00%3A00Z&end=2026-09-18T00%3A00%3A00Z&limit=10000&adjustment=all&feed=sip' $taskHeaders | Out-Null
[DateTime]::UtcNow.ToString('o') | Set-Content (Join-Path $taskOut 'retrieved_at.txt')
Write-Output "Collected $($activities.Count) account activities."
