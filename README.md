# NuPIC Simple Benchmark

This repository contains the instructions and code used to benchmark NuPIC Inference Server in [this blogpost](https://tryolabs.com/blog/from-brain-to-binary-cpus-future-ai-inference).

## Preparation
You need to have your Inference Server up and running in `localhost:8000`, and have the `nupic.client` package installed (see the blogpost for more details). Now you need to install the required dependencies with the following command:

```bash
pip install transformers[torch]
```

## Download dataset
Run the following commands to download and prepare the data:
```bash
pip install kaggle
kaggle datasets download -d sbhatti/financial-sentiment-analysis
unzip financial-sentiment-analysis.zip
```

## Run
Now you can run each of the experiments as:

```bash
python run_nupic.py
python run_cpu.py
python run_gpu.py
```

Each experiment will print inference time used to process the `4846` sentences.



> Contributions are welcome. Feel free to open any issues you might find when executing the code.
