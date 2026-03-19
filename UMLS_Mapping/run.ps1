$ProjectRoot = $PSScriptRoot
$DataPath    = "$ProjectRoot\data"
$Image       = "sparkle-umls:latest"
$CacheVolume = "scispacy_cache"

# Create volume if it doesn't exist
docker volume inspect $CacheVolume > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    docker volume create $CacheVolume | Out-Null
}

# Run and mount the scispacy cache at /root/.scispacy (where the model files landed)
docker run --rm `
    --mount type=bind,source="$DataPath",target=/app/data,readonly=false `
    --mount source=$CacheVolume,target=/root/.scispacy `
    $Image `
    --in /app/data/Note_terms_input.csv `
    --out /app/data/Note_terms_output.csv `
    --verbose