import os.path
import json

from datasets import Dataset
from configs.utils import ClassificationMatchAccuracy
from configs.preprocessor import Preprocessor
import multiprocessing as mp

special_tokens = []
fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):
    items, examples = [], []
    file_items = json.load(open(os.path.join(input_dir, "task.json"), "r"))
    items = file_items["examples"]
    for example_id in fewshot_examples:
        if shot_count != 0:
            item = items.pop(example_id)
            item.pop("comment") if "comment" in item.keys() else None
            assert list(item.keys()) == ["input", "target"]
            examples.append(item)
            shot_count -= 1
        else:
            break

    for item in items:
        item.pop("comment") if "comment" in item.keys() else None

    test_set = Dataset.from_list(items)
    preprocessor = RepeatCopyLogicPreprocessor(instruction, examples, input_dir)

    preprocess = preprocessor.preprocess
    metric = preprocessor.metric

    test_set = test_set.map(preprocess, remove_columns=["input", "target"], num_proc=1)

    return [], [], test_set, metric


class RepeatCopyLogicPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, input_dir):
        super(RepeatCopyLogicPreprocessor, self).__init__()
        self.str2preprocessor = {
            "Flan/Default/1": self.default,
            "Flan/Seen/1": self.flan_seen1,
            "Flan/Seen/2": self.flan_seen2,
            "Flan/Seen/3": self.flan_seen3,
            "Flan/Seen/4": self.flan_seen4,
            "Flan/Unseen/1": self.flan_unseen1,

        }
        self.str2preprocessor_fs = {
            "Flan/Default/1": self.default_fs,
            "Flan/Seen/1": self.flan_seen1_fs,
            "Flan/Seen/2": self.flan_seen2_fs,
            "Flan/Seen/3": self.flan_seen3_fs,
            "Flan/Seen/4": self.flan_seen4_fs,
            "Flan/Unseen/1": self.flan_unseen1_fs,
        }

        self._preprocessor = self.str2preprocessor[instruction] if not len(examples) \
            else self.str2preprocessor_fs[instruction]

        self.examples = examples

        self.metric = ClassificationMatchAccuracy(match_multiple=True)

        schema = json.load(open(os.path.join(input_dir, "task.json"), "r"))
        self.description = schema["description"] if "task_prefix" not in schema.keys() else schema["task_prefix"]
        self.example_output_prefix = schema["example_output_prefix"] if "example_output_prefix" in schema.keys() else \
            ""
        self.example_input_prefix = schema["example_input_prefix"] if "example_input_prefix" in schema.keys() else \
            "\n"
        self.few_shot_example_separator = "\n\n" if "few_shot_example_separator" in schema.keys() else \
            "\n\n"

    def preprocess(self, item):
        return self._preprocessor(item)

    def default(self, item):
        input_text = self.description + " " + self.example_input_prefix + item["input"] + self.example_input_prefix
        output_text = item["target"]
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def default_fs(self, item):
        input_text = self.description + " "
        for example in self.examples:
            input_text += self.example_input_prefix + example["input"] + self.example_input_prefix + example["target"] + \
                self.few_shot_example_separator

        input_text += self.example_input_prefix + item["input"] + self.example_input_prefix
        output_text = item["target"]
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen1(self, item):
        input_text, output_text = self.trivia_qa_3(item["input"], item["target"])
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen2(self, item):
        input_text, output_text = self.trivia_qa_10(item["input"], item["target"])
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen3(self, item):
        input_text, output_text = self.natural_questions_7(item["input"], item["target"])
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen4(self, item):
        input_text, output_text = self.natural_questions_10(item["input"], item["target"])
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen1_fs(self, item):
        input_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text = self.trivia_qa_3(example["input"], example["target"])
            input_text += ex_input_text + ex_output_text + self.few_shot_example_separator

        new_input_text, output_text = self.trivia_qa_3(item["input"], item["target"])
        input_text += new_input_text
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen2_fs(self, item):
        input_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text = self.trivia_qa_10(example["input"], example["target"])
            input_text += ex_input_text + ex_output_text + self.few_shot_example_separator

        new_input_text, output_text = self.trivia_qa_10(item["input"], item["target"])
        input_text += new_input_text
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen3_fs(self, item):
        input_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text = self.natural_questions_7(example["input"], example["target"])
            input_text += ex_input_text + ex_output_text + self.few_shot_example_separator

        new_input_text, output_text = self.natural_questions_10(item["input"], item["target"])
        input_text += new_input_text
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_seen4_fs(self, item):
        input_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text = self.natural_questions_10(example["input"], example["target"])
            input_text += ex_input_text + ex_output_text + self.few_shot_example_separator

        new_input_text, output_text = self.natural_questions_7(item["input"], item["target"])
        input_text += new_input_text
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_unseen1(self, item):
        instruction = item["input"]
        input_text = "Follow the instructions using logic to repeat part of the input.\n\n" \
                     "Instruction: {instruction}".format(instruction=instruction)
        output_text = item["target"]
        return_dict = {"input_text": input_text, "output_text": output_text}
        return return_dict

    def flan_unseen1_fs(self, item):
        input_text = "Follow the instructions using logic to repeat part of the input.\n\n"
        for example in self.examples:
            ex_instruction = example["input"]
            ex_answer = example["target"]
            input_text += "Instruction: {instruction} Answer: {answer}\n".format(instruction=ex_instruction, answer=ex_answer)
        input_text += "Instruction: {instruction} Answer: ".format(instruction=item["input"])
        output_text = item["target"]
        return_dict = {"input_text": input_text, "output_text": output_text}

        return return_dict





