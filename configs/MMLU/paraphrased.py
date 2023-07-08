import json
import os.path
import pandas as pd

from datasets import Dataset

from configs.utils import (
    OptionMatchingAccuracy,
    ClassificationAccuracy,
)

from configs.processors.NIV2_tasks import NIV2Tasks
from configs.processors.NIV2_templates import NIV2Templates

import multiprocessing as mp


SUBSET = False

special_tokens = []
fewshot_examples = {
    "abstract_algebra_test": [1],
    "astronomy_test": [8],
    "college_medicine_test": [2],
    "international_law_test": [9],
    "us_foreign_policy_test": [2]
}


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    def to_qa_dict(item: dict) -> dict:
        item["answer"] = item[item.pop("answer")]
        item["options"] = [item.pop(x) for x in ["A", "B", "C", "D"]]
        assert list(item.keys()) == ["question", "answer", "options"]
        return item

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    items, examples = [], []
    for file in files:
        df = pd.read_csv(os.path.join(input_dir, file), names=["question", "A", "B", "C", "D", "answer"])
        file_items = df.to_dict("records")
        key = file.replace(".csv", "")
        if key in fewshot_examples.keys() and shot_count != 0:
            for i in fewshot_examples[key]:
                examples.append(to_qa_dict(file_items.pop(i)))
                shot_count -= 1
                if shot_count != 0:
                    break

        for item in file_items:
            to_qa_dict(item)
            if SUBSET:
                text = "{} {} {} {} {}".format(item["answer"], *item["options"])
                if len(tokenizer(text, truncation=True)["input_ids"]) >= 150:
                    continue
            items.append(item)

    test_set = Dataset.from_list(items)
    preprocessor = MMLUParaphrasedPreprocessor(instruction)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "answer", "options"], num_proc=1)

    return test_set


class MMLUParaphrasedPreprocessor(NIV2Tasks, NIV2Templates):

    def __init__(self, instruction):

        self.idx_to_func = {
            "73": self.niv2_73_paraphrased,
            "229": self.niv2_229_paraphrased,
            "1286": self.niv2_1286_paraphrased,
            "1420": self.niv2_1420_paraphrased,
            "1565": self.niv2_1565_paraphrased,
        }

        task_id, prompt_id, paraphrase_id = instruction.split('/')
        definition_text = json.load(open(os.path.join(
            "./data/multiple_choice_qa_paraphrases", task_id,
            "{}.json".format(prompt_id)), "r"))["choices"][int(paraphrase_id)]["message"]["content"]
        self.proc_func = self.idx_to_func[task_id](definition_text)

    def processor(self, item):
        definition_text, input_text, _, answer, label_space = self.proc_func(**item)
        input_text, output_text = self.niv2_zs_template_1(definition_text, input_text, answer)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict





