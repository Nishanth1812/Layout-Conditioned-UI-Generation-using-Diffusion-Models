$ErrorActionPreference = 'Stop'
& 'H:\Personal\Projects\Layout-Conditioned-UI-Generation-using-Diffusion-Models\.venv\Scripts\cerebrium.exe' run main.py::run_pipeline --data '{"source":"/persistent-storage/ui-gen/downloads/dataset.zip","preprocess_workers":8,"tensor_workers":8,"log_every":10,"epochs":20,"batch_size":4,"num_workers":4,"precision":"bf16"}'
