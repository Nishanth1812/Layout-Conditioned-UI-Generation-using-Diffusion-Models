param(
	[string]$KaggleDataset = $env:KAGGLE_DATASET,
	[string]$KaggleFile = "",
	[string]$SecretName = "Kaggle_Secret",
	[int]$Epochs = 20,
	[int]$BatchSize = 8,
	[int]$NumWorkers = 8,
	[string]$Precision = "bf16",
	[int]$TrainSampleLimit = 30000
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Users\Devab\AppData\Local\Programs\Python\Python312\python.exe'

if (-not $KaggleDataset) {
	throw 'Provide -KaggleDataset "owner/dataset-slug" or set KAGGLE_DATASET environment variable.'
}

$env:MODAL_KAGGLE_SECRET_NAME = $SecretName

$dataObject = [ordered]@{
	output_dir = '/data/ui-gen/checkpoints'
	epochs = $Epochs
	batch_size = $BatchSize
	num_workers = $NumWorkers
	precision = $Precision
	train_sample_limit = $TrainSampleLimit
	force_preprocess = $true
	download_from_kaggle = $true
	kaggle_dataset = $KaggleDataset
	clear_raw_before_kaggle = $true
	cleanup_downloads = $true
}

if ($KaggleFile) {
	$dataObject.kaggle_file = $KaggleFile
}

$data = $dataObject | ConvertTo-Json -Compress
$dataBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($data))

& $python -m modal run -d main.py::run_pipeline_kaggle --data ("base64:" + $dataBase64)
exit $LASTEXITCODE