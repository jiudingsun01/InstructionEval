# InstructionEval

This is the repo for our preprint paper [Evaluating the Zero-shot Robustness of Instruction-tuned Language Models](https://arxiv.org/abs/2306.11270).  

## Resources

We evaluate the zero-shot O.O.D. robustness of instructions of the instruction-following LLMs tuned with the [Flan](https://github.com/google-research/FLAN) collection and [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) collection. 

### Observed Instructions

For observed instructions, we collected all the instructions that are **applicable** as general question-answering, multi-classification, and binary-classification. The preprocessing code of all the user instructions is put under [preprocessor.py](configs/preprocessor).

### Unobserved Instructions

For the unobserved instructions, we collected 319 natural, valid prompts on the benchmarks [MMLU](https://github.com/hendrycks/test) and [BBH](https://github.com/google/BIG-bench). All the templates and preprocessors are stored under the folder [config](configs)

## Results

We save the results disaggregated to the performance of each instruction under the folder [result_csv](results_csv)

| Folder Name | Description | Section | 
| ----------- | ----------- |----------- |
| Main   | The results for the main experiment of comparing "observed" and "unobserved" | 4.2 |
| Adversarial | The "Closer Look" experiment where the performance of models using "observed" but "incorrect" instructions       | 4.3 |
| Scaling   | The results for Flan-T5 at various size | 4.4 |
| Shots   | The results for models with one-shot in-context example | 4.6 |

## Evaluation

Please refer to instructions.json for the list of all the available task_name and instructions.

### Reproduce
To reproduce the results we reported in the paper, you may use the [main.py](main.py) file directly to process and run the dataset with the instruction template you specify. 

```python
from experiment import Experiment

if __name__ == "__main__":
    experiment = Experiment("google/flan-t5-xxl", devices=[0], precision="bf16")
    experiment.add_tasks_by_name(
        task_name="Conceptual_Combinations_Adversarial",
        output_dir="./test/",
        batch_size="8",
        instruction="BBL/Unobserved/4",
        shot_count="0",
        eval_by_logit=True,
    )
    experiment.inference()
```

### Embedding Distance

The notebooks under the folder [embedding_evaluation](embedding_evaluation) show the process of how we obtain the plots and distance for datasets with different types of instructions.

### Inference

To use our code to generate the processed PyTorch dataset with the template you want, you may import the corresponding config file and run the following code:

```python
import importlib

spec = importlib.util.spec_from_file_location("config", config_dir="./configs/Adv/conceptual_combinations.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
dataset = config.load_data(
    input_dir="./data/benchmark_tasks/conceptual_combinations",
    instruction="BBL/Unobserved/4",
    shot_count=1,
    eval_by_logits=False,
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base")
)
```

To check the config_dir and input_dir for each dataset, please refers to dirs.json

## News

### July.8th 2023

We publish the instructions and results reported in our paper.

