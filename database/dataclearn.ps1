$baseDate = Get-Date "2016-01-01"

Get-ChildItem -Filter "temp/*.txt" | ForEach-Object {

    $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $datePart = $name.Substring($name.Length - 6)  # YYMMDD

    try {
        # 20YYMMDD ¨ yyyyMMdd Œ`®‚Æ‚µ‚Ä•ÏŠ·
        $fileDate = [datetime]::ParseExact("20$datePart", "yyyyMMdd", $null)

        if ($fileDate -le $baseDate) {
            Remove-Item $_.FullName
        }
    }
    catch {
        Write-Host "“ú•t•ÏŠ·¸”s: $($_.Name)"
    }
}