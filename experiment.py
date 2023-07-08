from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from metrics import LogitsMetric, OutputMetric
from pytorch_lightning import seed_everything
from inference_modules import LitDataModule
from lightning.fabric import Fabric
from tqdm import tqdm
import importlib
import argparse
import torch
import json
import time
import os

precisions_dict = {
    "fp16": (16, torch.float16),
    "fp32": (32, torch.float32),
    "bf16": ("bf16-mixed", torch.bfloat16),
}


DATASET2CONFIGS = {
    # data_dir, config_dir
    "MMLU_General": ("./data/MMLU", "./configs/MMLU/general.py",),
    "MMLU_Specific": ("./data/MMLU", "./configs/MMLU/specific.py",),
    "BBQ_Lite": (
        "./data/benchmark_tasks/bbq_lite_json",
        "./configs/BBH/multiple_choice/bbq_lite.py",
    ),
    "Code_Line_Description": (
        "./data/benchmark_tasks/code_line_description",
        "./configs/BBH/multiple_choice/code_line_description.py",
    ),
    "Logical_Deduction": (
        "./data/benchmark_tasks/logical_deduction",
        "./configs/BBH/multiple_choice/logical_deduction.py",
    ),
    "Play_Dialog": (
        "./data/benchmark_tasks/play_dialog_same_or_different",
        "./configs/BBH/binary_classification/play_dialog_same_or_different.py",),
    "Vitaminc_Fact_Verification": (
        "./data/benchmark_tasks/vitaminc_fact_verification",
        "./configs/BBH/classification/vitaminc_fact_verification.py",
    ),
    "StrategyQA": (
        "./data/benchmark_tasks/strategyqa",
        "./configs/BBH/binary_classification/strategy_qa.py",
    ),
    "Strange_Stories": (
        "./data/benchmark_tasks/strange_stories",
        "./configs/BBH/binary_classification/strange_stories.py",
    ),
    "Language_Identification": (
        "./data/benchmark_tasks/language_identification",
        "./configs/BBH/classification/language_identification.py",
    ),
    "Language_Identification_Paraphrased": (
        "./data/benchmark_tasks/language_identification",
        "./configs/Paraphrase/language_identification.py",
    ),
    "Language_Identification_Adversial": (
        "./data/benchmark_tasks/language_identification",
        "./configs/Adv/language_identification.py",
    ),
    "Known_Unknowns": (
        "./data/benchmark_tasks/known_unknowns",
        "./configs/BBH/multiple_choice/known_unknowns.py",
    ),
    "Hindu_Knowledge": (
        "./data/benchmark_tasks/hindu_knowledge",
        "./configs/BBH/multiple_choice/hindu_knowledge.py",
    ),
    "Novel_Concepts": (
        "./data/benchmark_tasks/novel_concepts",
        "./configs/BBH/multiple_choice/novel_concepts.py",
    ),
    "Winowhy": (
        "./data/benchmark_tasks/winowhy",
        "./configs/BBH/binary_classification/winowhy.py",
    ),
    "Intent_Recognition": (
        "./data/benchmark_tasks/intent_recognition",
        "./configs/Adv/intent_recognition.py",
    ),
    "Logic_Grid_Puzzle": (
        "./data/benchmark_tasks/logic_grid_puzzle",
        "./configs/BBH/multiple_choice/logic_grid_puzzle.py",
    ),
    "Conceptual_Combinations": (
        "./data/benchmark_tasks/conceptual_combinations",
        "./configs/BBH/multiple_choice/conceptual_combinations.py",
    ),
    "Conceptual_Combinations_Adversarial": (
        "./data/benchmark_tasks/conceptual_combinations",
        "./configs/Adv/conceptual_combinations.py",
    ),
    "Empirical_Judgments": (
        "./data/benchmark_tasks/empirical_judgments",
        "./configs/Adv/empirical_judgments.py",
    ),
    "Crash_Blossom": (
        "./data/benchmark_tasks/crash_blossom",
        "./configs/Adv/crash_blossom.py",
    ),
    "Common_Morpheme": (
        "./data/benchmark_tasks/common_morpheme",
        "./configs/Adv/common_morpheme.py",
    ),
    "Logical_Sequence": (
        "./data/benchmark_tasks/logical_sequence",
        "./configs/Adv/logical_sequence.py",
    ),
    "Epistemic_Reasoning": (
        "./data/benchmark_tasks/epistemic_reasoning",
        "./configs/Adv/epistemic_reasoning.py",
    ),
}


class Experiment:

    def __init__(self, model_name_or_path, devices, seed=42, precision="fp16"):

        if "alpaca" in model_name_or_path:
            ModelClass, TokenizerClass = AutoModelForCausalLM, LlamaTokenizer
        else:
            ModelClass, TokenizerClass = AutoModelForSeq2SeqLM, AutoTokenizer

        seed_everything(seed)
        self.model_name_or_path = model_name_or_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gpus = torch.cuda.device_count()
        print("Loading model...")
        if precision == "bf16":
            torch.set_float32_matmul_precision("high")
        fabric_precision, precision = precisions_dict[precision]
        self.model = ModelClass.from_pretrained(model_name_or_path, torch_dtype=precision)
        self.tokenizer = TokenizerClass.from_pretrained(model_name_or_path)
        strategy = "ddp" if len(devices) > 1 else "auto"
        self.fabric = Fabric(accelerator="cuda", devices=devices, precision=fabric_precision, strategy=strategy)
        self.fabric.launch()
        self.model.eval()
        self.model = self.fabric.setup(self.model)
        self._tasks = []
        self.fabric.barrier()

    def add_tasks(
            self,
            input_dir: str,
            output_dir: str,
            config_dir: str,
            batch_size: str,
            instruction: str,
            shot_count: str,
            eval_by_logit: bool
    ) -> None:
        # if output_dir not in [task["output_dir"] for task in self._tasks]:
        self._tasks.append({
            "input_dir": input_dir,
            "output_dir": output_dir,
            "config_dir": config_dir,
            "batch_size": int(batch_size),
            "instruction": instruction,
            "shot_count": int(shot_count),
            "eval_by_logit": eval_by_logit
        })

    def add_tasks_by_name(self,
            task_name: str,
            output_dir: str,
            batch_size: str,
            instruction: str,
            shot_count: str,
            eval_by_logit: bool
    ) -> None:
        if task_name not in DATASET2CONFIGS.keys():
            raise ValueError("Task name not found")
        else:
            input_dir, config_dir = DATASET2CONFIGS[task_name]
            self.add_tasks(input_dir, output_dir, config_dir, batch_size, instruction, shot_count, eval_by_logit)

    def inference(self):

        for i, task in enumerate(self._tasks):

            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            print("Inference on Task {}/{}...".format(i, len(self._tasks)))
            input_dir, output_dir, config_dir, batch_size, instruction, shot_count, eval_by_logits = list(task.values())
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # try:
            spec = importlib.util.spec_from_file_location("config", config_dir)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            print("Loading datasets")
            test_set = config.load_data(input_dir, instruction, shot_count, eval_by_logits, self.tokenizer)
            example = test_set[0]["input_text"]
            """except Exception:
                print(instruction)
                print("Encountered Exception while loading config file from {}; continue...".format(config_dir))
                continue"""

            data_module = LitDataModule(batch_size, test_set, self.tokenizer)
            test_set = data_module.test_dataloader()
            test_set = self.fabric.setup_dataloaders(test_set)
            self.fabric.barrier()

            metric = LogitsMetric(self.fabric) if eval_by_logits else OutputMetric()

            all_classes, all_gold_classes = [], []
            all_pred, all_gold = [], []
            with torch.no_grad():
                for batch in tqdm(test_set):
                    input_ids, attention_mask, labels, label_cls, label_spaces_ids, sample_to = batch.values()
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": 32
                    }
                    outputs = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
                    scores = outputs.scores
                    logits = torch.stack(scores, dim=1)
                    if eval_by_logits:
                        classes = metric.classify(logits, label_spaces_ids, sample_to)
                        all_classes.extend(classes.cpu().numpy())
                        all_gold_classes.extend(label_cls.cpu().numpy())

                    pred_ids = outputs.sequences
                    all_pred.extend(pred_ids.cpu().numpy())
                    all_gold.extend(labels.cpu().numpy())

            assert len(all_pred) == len(all_gold) and len(all_classes) == len(all_gold_classes)
            preds = self.tokenizer.batch_decode(all_pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            golds = self.tokenizer.batch_decode(all_gold, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            output_file = "output.txt" if self.fabric.world_size == 1 else "output_{}.txt".format(self.fabric.local_rank)
            golden_file = "golden.txt" if self.fabric.world_size == 1 else "golden_{}.txt".format(self.fabric.local_rank)

            with open(os.path.join(output_dir, output_file), "w") as f:
                for n, pred in enumerate(preds):
                    f.write(str(n) + "\t" + pred + "\n")
                f.close()

            with open(os.path.join(output_dir, golden_file), "w") as f:
                for n, gold in enumerate(golds):
                    f.write(str(n) + "\t" + gold + "\n")
                f.close()

            if eval_by_logits:
                class_file = "classes.txt" if self.fabric.world_size == 1 else "classes_{}.txt".format(
                    self.fabric.local_rank)
                with open(os.path.join(output_dir, class_file), "w") as f:
                    for n, cls in enumerate(all_classes):
                        f.write(str(n) + "\t" + str(cls) + "\n")
                    f.close()

            correct, total = metric(all_classes, all_gold_classes) if eval_by_logits else metric(preds, golds)
            self.fabric.barrier()
            if self.fabric.world_size > 1:
                correct = self.fabric.all_reduce(correct, reduce_op="sum")
                total = self.fabric.all_reduce(total, reduce_op="sum")

            accuracy = correct.float() / total.float()
            accuracy = accuracy.cpu().item()

            if self.fabric.global_rank in [0, -1]:
                print("The Accuracy is {}".format(accuracy))
                time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

                info_dict = {
                    "time": time_,
                    "performance": accuracy,
                    "example": example,
                    "instruction": instruction,
                    "shots": shot_count,
                    "eval_by_logits": eval_by_logits
                }
                with open(os.path.join(output_dir, "info.json"), "w") as f:
                    json.dump(info_dict, f)
                    f.close()
            
        print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--tasks_dir', required=True)
    parser.add_argument('--precision', default="fp16", choices=["fp16", "fp32", "bf16"], type=str)
    parser.add_argument('--devices', default=[0], type=int, nargs="+")

    args = parser.parse_args()

    experiment = Experiment(args.model_name_or_path, devices=args.devices, precision=args.precision)
    print("tasks_dir is: {}".format(args.tasks_dir))
    tasks_args = json.load(open(args.tasks_dir, "r"))
    for args in tasks_args:
        experiment.add_tasks(**args)

    experiment.inference()


if __name__ == "__main__":
    main()














