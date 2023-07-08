import os.path
import json
from configs.preprocessor import Preprocessor
from configs.processors.NIV2_tasks import NIV2Tasks
from configs.processors.NIV2_templates import NIV2Templates
import multiprocessing as mp
from datasets import Dataset
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file, ClassificationGivenLabel


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = ParaphrasedPreprocessor(instruction)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class ParaphrasedPreprocessor(NIV2Tasks, NIV2Templates):

    def __init__(self, instruction):

        self.idx_to_func = {
            "1370": self.niv2_1370_paraphrased,
        }

        task_id, prompt_id, paraphrase_id = instruction.split('/')
        self.definition = json.load(open(os.path.join(
            "./data/paraphrases", task_id,
            "{}.json".format(prompt_id)), "r"))["choices"][int(paraphrase_id)]["message"]["content"]
        self.proc_func = self.idx_to_func[task_id]

    def processor(self, item):
        definition_text, input_text, _, answer, label_space = self.proc_func(**item, definition=self.definition)
        input_text, output_text = self.niv2_zs_template_1(definition_text, input_text, answer)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict