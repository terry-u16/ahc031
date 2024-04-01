Write-Host "[Compile]"
cargo build --release
Move-Item ./target/release/ahc031 . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
$env:AHC030_SHOW_COMMENT = 0
dotnet marathon run-local
./rel -d ./data/results -o min