import os.path
import json
import random
from datasets import Dataset
from configs.utils import ClassificationMatchAccuracy, OptionMatchingAccuracy, ClassificationAccuracy, ClassificationGivenLabel, load_BBL_file
from configs.preprocessor import Preprocessor
import multiprocessing as mp

special_tokens = []

fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)

    test_set = Dataset.from_list(items)
    preprocessor = PlayDialogPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

def load_data_testing(input_dir, instruction, args):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = PlayDialogPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set


class PlayDialogPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(PlayDialogPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["BBL/Default/1"] = self.default_Classification
        self.instr2preprocessor["BBL/Unobserved/1"] = self.unobserved1
        self.instr2preprocessor["BBL/Unobserved/2"] = self.unobserved2
        self.instr2preprocessor["BBL/Unobserved/3"] = self.unobserved3
        self.instr2preprocessor["BBL/Unobserved/4"] = self.unobserved4
        self.instr2preprocessor["BBL/Unobserved/5"] = self.unobserved5
        self.instr2preprocessor["BBL/Unobserved/6"] = self.unobserved6
        self.instr2preprocessor["BBL/Unobserved/7"] = self.unobserved7
        self.instr2preprocessor["BBL/Unobserved/8"] = self.unobserved8
        self.instr2preprocessor["BBL/Unobserved/9"] = self.unobserved9
        self.instr2preprocessor["BBL/Unobserved/10"] = self.unobserved10

        self.instr2preprocessor_fs["BBL/Default/1"] = self.default_Classification_fs
        self.instr2preprocessor_fs["BBL/Unobserved/1"] = self.unobserved1_fs
        self.instr2preprocessor_fs["BBL/Unobserved/2"] = self.unobserved2_fs
        self.instr2preprocessor_fs["BBL/Unobserved/3"] = self.unobserved3_fs
        self.instr2preprocessor_fs["BBL/Unobserved/4"] = self.unobserved4_fs
        self.instr2preprocessor_fs["BBL/Unobserved/5"] = self.unobserved5_fs
        self.instr2preprocessor_fs["BBL/Unobserved/6"] = self.unobserved6_fs
        self.instr2preprocessor_fs["BBL/Unobserved/7"] = self.unobserved7_fs
        self.instr2preprocessor_fs["BBL/Unobserved/8"] = self.unobserved8_fs
        self.instr2preprocessor_fs["BBL/Unobserved/9"] = self.unobserved9_fs
        self.instr2preprocessor_fs["BBL/Unobserved/10"] = self.unobserved10_fs

        self.instr2preprocessor["NIV2/BC/1"] = lambda item: self.niv2_binary_classification_1(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/2"] = lambda item: self.niv2_binary_classification_2(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/3"] = lambda item: self.niv2_binary_classification_3(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/4"] = lambda item: self.niv2_binary_classification_4(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/5"] = lambda item: self.niv2_binary_classification_5(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/6"] = lambda item: self.niv2_binary_classification_6(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/7"] = lambda item: self.niv2_binary_classification_7(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/8"] = lambda item: self.niv2_binary_classification_8(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/9"] = lambda item: self.niv2_binary_classification_9(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/10"] = lambda item: self.niv2_binary_classification_10(self.map_to_niv2_56(item))

        self.instr2preprocessor["FLAN/BC/1"] = lambda item: self.flan_binary_classification_1(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/2"] = lambda item: self.flan_binary_classification_2(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/3"] = lambda item: self.flan_binary_classification_3(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/4"] = lambda item: self.flan_binary_classification_4(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/5"] = lambda item: self.flan_binary_classification_5(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/6"] = lambda item: self.flan_binary_classification_6(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/7"] = lambda item: self.flan_binary_classification_7(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/8"] = lambda item: self.flan_binary_classification_8(self.map_to_flan_multirc(item))

        self.instr2preprocessor["Alpaca/BC/1"] = lambda item: self.alpaca_dialog_1(item)
        self.instr2preprocessor["Alpaca/BC/2"] = lambda item: self.alpaca_dialog_2(item)

        self.instr2preprocessor["Alpaca/Default/1"] = self.alpaca_default
        self.instr2preprocessor["Alpaca/Unobserved/1"] = self.alpaca_unobserved1
        self.instr2preprocessor["Alpaca/Unobserved/2"] = self.alpaca_unobserved2
        self.instr2preprocessor["Alpaca/Unobserved/3"] = self.alpaca_unobserved3
        self.instr2preprocessor["Alpaca/Unobserved/4"] = self.alpaca_unobserved4
        self.instr2preprocessor["Alpaca/Unobserved/5"] = self.alpaca_unobserved5
        self.instr2preprocessor["Alpaca/Unobserved/6"] = self.alpaca_unobserved6
        self.instr2preprocessor["Alpaca/Unobserved/7"] = self.alpaca_unobserved7
        self.instr2preprocessor["Alpaca/Unobserved/8"] = self.alpaca_unobserved8
        self.instr2preprocessor["Alpaca/Unobserved/9"] = self.alpaca_unobserved9
        self.instr2preprocessor["Alpaca/Unobserved/10"] = self.alpaca_unobserved10

        self.instr2preprocessor_fs["NIV2/BC/1"] = lambda item: self.niv2_binary_classification_1_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/2"] = lambda item: self.niv2_binary_classification_2_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/3"] = lambda item: self.niv2_binary_classification_3_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/4"] = lambda item: self.niv2_binary_classification_4_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/5"] = lambda item: self.niv2_binary_classification_5_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/6"] = lambda item: self.niv2_binary_classification_6_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/7"] = lambda item: self.niv2_binary_classification_7_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/8"] = lambda item: self.niv2_binary_classification_8_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/9"] = lambda item: self.niv2_binary_classification_9_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/10"] = lambda item: self.niv2_binary_classification_10_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/1"] = lambda item: self.flan_binary_classification_1_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/2"] = lambda item: self.flan_binary_classification_2_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/3"] = lambda item: self.flan_binary_classification_3_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/4"] = lambda item: self.flan_binary_classification_4_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/5"] = lambda item: self.flan_binary_classification_5_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/6"] = lambda item: self.flan_binary_classification_6_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/7"] = lambda item: self.flan_binary_classification_7_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/8"] = lambda item: self.flan_binary_classification_8_fs(self.map_to_flan_multirc_few_shot(item))

    def map_to_niv2_56(self, item):
        paragraph, question = item["question"].split("\n", maxsplit=1)
        output_text = "Yes" if item["answer"] == "different" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "different",
            "output_text": output_text
        }
        return new_item

    def map_to_niv2_56_few_shot(self, item):
        paragraph, question = item["question"].split("\n", maxsplit=1)
        output_text = "Yes" if item["answer"] == "different" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "different",
            "output_text": output_text
        }
        if "paragraph" not in self.examples[0].keys():
            for i in range(len(self.examples)):
                new_example = self.map_to_niv2_56(self.examples[i])
                self.examples[i] = new_example

        return new_item

    def map_to_flan_multirc(self, item):
        paragraph, question = item["question"].split("\n", maxsplit=1)
        answer = "Yes" if item["answer"] == "different" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "different",
            "answer": answer
        }
        return new_item

    def map_to_flan_multirc_few_shot(self, item):
        paragraph, question = item["question"].split("\n", maxsplit=1)
        answer = "Yes" if item["answer"] == "different" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "different",
            "answer": answer
        }
        if "paragraph" not in self.examples[0].keys():
            for i in range(len(self.examples)):
                new_example = self.map_to_flan_multirc(self.examples[i])
                self.examples[i] = new_example
        return new_item

    def map_to_alpaca_binary(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = "True" if item["answer"] == "same" else "False"
        new_item = {
            "question": play + "In the preceding dialogue, the lines \"" + line1 + "\" and \"" + line2 + "\" were spoken by the same person.",
            "options": ["True", "False"],
            "answer": answer
        }
        return new_item

    def map_to_alpaca_binary_yesno(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = "Yes" if item["answer"] == "same" else "No"
        new_item = {
            "question": play + "In the preceding dialogue, were the lines \"" + line1 + "\" and \"" + line2 + "\" spoken by the same person?",
            "options": ["Yes", "No"],
            "answer": answer
        }
        return new_item
        
    def unobserved_template(self, item, input_template):
        input_text = input_template.format(**item)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template1_few_shot(self, item, input_template_prefix, input_template, example_template):
        input_text = input_template_prefix
        for example in self.examples:
            input_text += example_template.format(**example)
        input_text += input_template.format(**item)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved_template2_few_shot(self, item, input_template_prefix, input_template, example_template):
        def get_splited_input(item) -> dict:
            play, lines = item["question"].split("In the preceding dialogue, were the lines")
            _, line1, _, line2, _ = lines.split("\"")
            answer = "yes" if item["answer"] == "same" else "no"
            options = ["yes", "no"]
            new_item = {
                "play": play,
                "line1": line1,
                "line2": line2,
                "answer": answer,
                "options": options,
            }
            return new_item
        
        input_text = input_template_prefix
        for example in self.examples:
            new_example = get_splited_input(example)
            input_text += example_template.format(**new_example)
        
        new_item = get_splited_input(item)
        input_text += input_template.format(**new_item)
        output_text = new_item["answer"]
        label_space = new_item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved_template3_few_shot(self, item, input_template_prefix, input_template, example_template):
        def get_splited_input(item) -> dict:
            play, lines = item["question"].split("In the preceding dialogue, were the lines")
            _, line1, _, line2, _ = lines.split("\"")
            answer = item["answer"]
            options = ["same", "different"]
            new_item = {
                "play": play,
                "line1": line1,
                "line2": line2,
                "answer": answer,
                "options": options,
            }
            return new_item
        
        input_text = input_template_prefix
        for example in self.examples:
            new_example = get_splited_input(example)
            input_text += example_template.format(**new_example)
        
        new_item = get_splited_input(item)
        input_text += input_template.format(**new_item)
        output_text = new_item["answer"]
        label_space = new_item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved_template4_few_shot(self, item, input_template_prefix, input_template, example_template):
        def get_splited_input(item) -> dict:
            play, lines = item["question"].split("In the preceding dialogue, were the lines")
            _, line1, _, line2, _ = lines.split("\"")
            answer = "A" if item["answer"] == "same" else "B"
            options = ["A", "B"]
            new_item = {
                "play": play,
                "line1": line1,
                "line2": line2,
                "answer": answer,
                "options": options,
            }
            return new_item
        
        input_text = input_template_prefix
        for example in self.examples:
            new_example = get_splited_input(example)
            input_text += example_template.format(**new_example)
        
        new_item = get_splited_input(item)
        input_text += input_template.format(**new_item)
        output_text = new_item["answer"]
        label_space = new_item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "The following transcripts of dialogues have been taken from Shakespeare's plays, but the " \
                     "transcripts do not say who said what. Based on these contents and styles, your task is to " \
                     "identify whether the sentences in question were spoken by the same or different people." \
                     "\n{question}\n\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved1_fs(self, item):
        input_template_prefix = "The following transcripts of dialogues have been taken from Shakespeare's plays, but the " \
                     "transcripts do not say who said what. Based on these contents and styles, your task is to " \
                     "identify whether the sentences in question were spoken by the same or different people.\n\n"
        example_template = "{question}\nAnswer: {answer}\n\n"
        input_template = "{question}\nAnswer: "
        return self.unobserved_template1_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved1(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "The following transcripts of dialogues have been taken from Shakespeare's plays, but the " \
                     "transcripts do not say who said what. Based on these contents and styles, your task is to " \
                     "identify whether the sentences in question were spoken by the same or different people." \
                     ""
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=options)

    def unobserved2(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = "yes" if item["answer"] == "same" else "no"
        options = ["yes", "no"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "Below are transcripts of dialogues from Shakespeare plays.\n{play}\nPlease identify whether " \
                         "the two scripts were spoken by the same people. Answer yes or no.\nScript1: {line1}\nScript2:" \
                         " {line2}\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = "Below are transcripts of dialogues from Shakespeare plays. Please identify whether " \
                                "the two scripts were spoken by the same people. Answer yes or no."
        example_template = "\n{play}\nScript1: {line1}\nScript2: {line2}\nAnswer: {answer}"
        input_template = "\n{play}\nScript1: {line1}\nScript2: {line2}\nAnswer: "
        return self.unobserved_template2_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved2(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = "yes" if item["answer"] == "same" else "no"
        options = ["yes", "no"]
        instruction = f"Below are transcripts of dialogues from Shakespeare plays.\n{play}\nPlease identify whether " \
                         "the two scripts were spoken by the same people. Answer yes or no."
        inputs = f"\nScript1: {line1}\nScript2: {line2}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved3(self, item):
        answer, options = item["answer"], item["options"]
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "You have read all the plays by Shakespeare. You surely recognized this dialogue:{play}\n" \
                         "Now, are these two lines spoken by the same character or different characters? Answer " \
                         "from \"same\" or \"different\".\nLine1: {line1}\nLine2: {line2}\nYour answer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "You have read all the plays by Shakespeare. Now, are these two lines spoken by the same character or different characters? Answer " \
                                "from \"same\" or \"different\".\n\n"
        
        example_template = "You surely recognized this dialogue:{play}\nLine1: {line1}\nLine2: {line2}\nYour answer: {answer}\n\n"
        input_template = "You surely recognized this dialogue:{play}\nLine1: {line1}\nLine2: {line2}\nYour answer: "
        return self.unobserved_template3_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved3(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        instruction = f"You have read all the plays by Shakespeare. You surely recognized this dialogue:{play}\n" \
                         "Now, are these two lines spoken by the same character or different characters? Answer " \
                         "from \"same\" or \"different\"."

        inputs = f"\nLine1: {line1}\nLine2: {line2}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved4(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "The following paragraph is a dialogue from one of Shakespeare's plays, but without the " \
                         "information of the corresponding speaking character. You need to decide whether the two " \
                         "lines I give you are lines of the same character.\n{play}\nFrom the above dialogue, are " \
                         "the lines {line1} and {line2} spoken by the same or different characters?\n Answer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "The following paragraph is a dialogue from one of Shakespeare's plays, but without the " \
                         "information of the corresponding speaking character. You need to decide whether the two " \
                         "lines I give you are lines of the same character."
        
        example_template = "\n{play}\nFrom the above dialogue, are the lines {line1} and {line2} spoken by the same or different characters?\n Answer: {answer}"
        input_template = "\n{play}\nFrom the above dialogue, are the lines {line1} and {line2} spoken by the same or different characters?\n Answer: "
        return self.unobserved_template3_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved4(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        instruction = "The following paragraph is a dialogue from one of Shakespeare's plays, but without the " \
                         "information of the corresponding speaking character. You need to decide whether the two " \
                         "lines I give you are lines of the same character."
        inputs = f"\n{play}\nFrom the above dialogue, are the lines {line1} and {line2} spoken by the same or different characters?\n Answer: "
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved5(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "Now you are a dramatist. The following transcripts of dialogues are taken from Shakespeare " \
                         "plays, but the transcripts do not mark who said what.  Your task is to identify whether " \
                         "the sentences in question were spoken by the same or different people. Here is the play:\n" \
                         "{play}\nQuestion: In the preceding dialogue, were the lines {line1} and {line2} spoken by " \
                         "the same person or different people? Please just give a short answer: same or different." \
                         "\n\nYour Answer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = "Now you are a dramatist. The following transcripts of dialogues are taken from Shakespeare " \
                         "plays, but the transcripts do not mark who said what. Your task is to identify whether " \
                         "the sentences in question were spoken by the same or different people. "
        
        example_template = "Here is the play:\n" \
                         "{play}\nQuestion: In the preceding dialogue, were the lines {line1} and {line2} spoken by " \
                         "the same person or different people? Please just give a short answer: same or different." \
                         "\n\nYour Answer: {answer}\n\n\n"
        input_template = "Here is the play:\n" \
                         "{play}\nQuestion: In the preceding dialogue, were the lines {line1} and {line2} spoken by " \
                         "the same person or different people? Please just give a short answer: same or different." \
                         "\n\nYour Answer: "
        return self.unobserved_template3_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved5(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        instruction = "Now you are a dramatist. The following transcripts of dialogues are taken from Shakespeare " \
                         "plays, but the transcripts do not mark who said what.  Your task is to identify whether " \
                         "the sentences in question were spoken by the same or different people."
        inputs = f"Here is the play:\n{play}\nQuestion: In the preceding dialogue, were the lines {line1} and {line2} spoken by the same person or different people? Please just give a short answer: same or different." \
                         "\n\n"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved6(self, item):
        input_template = "The following transcripts of dialogues have been taken from Shakespeare plays, but the " \
                         "transcripts do not say who said what. We have two sentences selected from the transcripts, " \
                         "please make a judgement whether the sentences are spoken by the same people.\n{question}"
        return self.unobserved_template(item, input_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "The following transcripts of dialogues have been taken from Shakespeare plays, but the " \
                         "transcripts do not say who said what. We have two sentences selected from the transcripts, " \
                         "please make a judgement whether the sentences are spoken by the same people."
        
        example_template = "\n{question}A: {answer}"
        input_template = "\n{question}A: "
        return self.unobserved_template1_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved6(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "The following transcripts of dialogues have been taken from Shakespeare plays, but the " \
                         "transcripts do not say who said what. We have two sentences selected from the transcripts, " \
                         "please make a judgement whether the sentences are spoken by the same people."
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=options)

    def unobserved7(self, item):
        input_template = "Dialogue: {question}\nAnswer \"same\" or \"different\".\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = ""
        
        example_template = "Dialogue: {question}\nAnswer \"same\" or \"different\".\nAnswer: {answer}\n\n"
        input_template = "Dialogue: {question}\nAnswer \"same\" or \"different\".\nAnswer: "
        return self.unobserved_template1_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved7(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Answer \"same\" or \"different\""
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=options)

    def unobserved8(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "In the context of the Shakespeare play, {play}, assess the given dialogue transcripts. " \
                         "Determine whether the sentences {line1} and {line2} were spoken by a single person or by " \
                         "different people.\nAnswer: "
        return self.unobserved_template(item, input_template)

    def unobserved8_fs(self, item):
        input_template_prefix = ""
        
        example_template = "In the context of the Shakespeare play, {play}, assess the given dialogue transcripts. " \
                         "Determine whether the sentences {line1} and {line2} were spoken by a single person or by " \
                         "different people.\nAnswer: {answer}\n\n"
        input_template = "In the context of the Shakespeare play, {play}, assess the given dialogue transcripts. " \
                         "Determine whether the sentences {line1} and {line2} were spoken by a single person or by " \
                         "different people.\nAnswer: "
        return self.unobserved_template3_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved8(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        instruction = f"In the context of the Shakespeare play, {play}, assess the given dialogue transcripts. " \
                         f"Determine whether the sentences {line1} and {line2} were spoken by a single person or by " \
                         "different people."
        return self.alpaca_template_without_input(instruction=instruction, answer=answer, label_space=options)

    def unobserved9(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = "Yes" if item["answer"] == "same" else "No"
        options = ["Yes", "No"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "Play: {play}\n In this play written by Shakespeare, classify whether\nCharacter A: {line1}" \
                         "\nCharacter B: {line2}are spoken by the same character or different ones? Answer 'Yes' or" \
                         " 'No' only. "
        return self.unobserved_template(item, input_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = ""
        
        example_template = "Play: {play}\n In this play written by Shakespeare, classify whether\nCharacter A: {line1}" \
                         "\nCharacter B: {line2}are spoken by the same character or different ones? Answer 'yes' or" \
                         " 'no' only. {answer}\n\n"
        input_template = "Play: {play}\n In this play written by Shakespeare, classify whether\nCharacter A: {line1}" \
                         "\nCharacter B: {line2}are spoken by the same character or different ones? Answer 'yes' or" \
                         " 'no' only. "
        return self.unobserved_template2_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved9(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = "Yes" if item["answer"] == "same" else "No"
        options = ["Yes", "No"]
        instruction = f"Play: {play}\n In this play written by Shakespeare, classify whether\nCharacter A: {line1}" \
                         f"\nCharacter B: {line2}are spoken by the same character or different ones? Answer 'Yes' or" \
                         " 'No' only. "
        return self.alpaca_template_without_input(instruction=instruction, answer=answer, label_space=options)

    def unobserved10(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        item = {
            "play": play,
            "line1": line1,
            "line2": line2,
            "answer": answer,
            "options": options,
        }
        input_template = "Context: {play}\nQuestion: read this dialogue selected from a play written by Shakespeare, " \
                         "are the lines {line1} and {line2} from the same character?\nOptions:\nA) Yes\nB) No. Answer: "
        item["answer"] = "A" if item["answer"] == "same" else "B"
        item["options"] = ["A", "B"]
        return self.unobserved_template(item, input_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = ""
        
        example_template = "Context: {play}\nQuestion: read this dialogue selected from a play written by Shakespeare, " \
                         "are the lines {line1} and {line2} from the same character?\nOptions:\nA) Yes\nB) No. Answer: {answer}\n\n"
        input_template = "Context: {play}\nQuestion: read this dialogue selected from a play written by Shakespeare, " \
                         "are the lines {line1} and {line2} from the same character?\nOptions:\nA) Yes\nB) No. Answer: "
        return self.unobserved_template4_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved10(self, item):
        play, lines = item.pop("question").split("In the preceding dialogue, were the lines")
        _, line1, _, line2, _ = lines.split("\"")
        answer = item["answer"]
        options = ["same", "different"]
        instruction = f"Context: {play}\nQuestion: read this dialogue selected from a play written by Shakespeare, " \
                         f"are the lines {line1} and {line2} from the same character?\nOptions:\nA) Yes\nB) No. Answer: "
        answer = "A" if item["answer"] == "same" else "B"
        options = ["A", "B"]
        return self.alpaca_template_without_input(instruction=instruction, answer=answer, label_space=options)
