import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
import random
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file


special_tokens = []
fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = NovelConceptsPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

def load_data_testing(input_dir, instruction, args):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = NovelConceptsPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

class NovelConceptsPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Let's do some find-the-common-concept problems. In these problems, your goal is to identify the underlying concept or theme that relates the things listed. Make sure to answer carefully.\n",
        "Please choose the best option from the listed choices that precisely express the given things in common.",
        "Identify and output the commonality among the given objects",
        "You are given three objects; choose the option from below where the objects share the greatest similarity",
        "Please select the best option to indicate the commonality between the objects; Give your answer as one of A, B, C, D, E.",
        "Pick the most correct description",
        "One correct common thing among all the choices below is?",
        "Answer the question below. Give your answer with letter, then explain your choice in the next line",
        "What is the common phenomenon in these objects, choose the best answer from the following.",
        "Pick one of the choices A, B, C, D, E.",
        "You will be given several objects or activities. From a series of choices (A, B, C, D, E), identify an aspect that they all share in common. If there are multiple aspects, identify the one best fitting.\n\nWhat do these objects have in common"
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(NovelConceptsPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["BBL/Default/1"] = self.default_QA
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

        self.instr2preprocessor_fs["BBL/Default/1"] = self.default_QA_fs
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

        self.instr2preprocessor["Alpaca/Default/1"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[0])
        self.instr2preprocessor["Alpaca/Unobserved/1"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[1])
        self.instr2preprocessor["Alpaca/Unobserved/2"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[2])
        self.instr2preprocessor["Alpaca/Unobserved/3"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[3])
        self.instr2preprocessor["Alpaca/Unobserved/4"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[4])
        self.instr2preprocessor["Alpaca/Unobserved/5"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[5])
        self.instr2preprocessor["Alpaca/Unobserved/6"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[6])
        self.instr2preprocessor["Alpaca/Unobserved/7"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[7])
        self.instr2preprocessor["Alpaca/Unobserved/8"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[8])
        self.instr2preprocessor["Alpaca/Unobserved/9"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[9])
        self.instr2preprocessor["Alpaca/Unobserved/10"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[10])
        
    def unobserved_template_QA(self, item, input_temptlate):
        question, options, answer = item["question"], item["options"], item["answer"]
        try:
            A, B, C, D, E = options
        except ValueError:
            print(options)
            raise
        choice = ["A", "B", "C", "D", "E"][options.index(answer)]
        input_text = input_temptlate.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E)
        output_text = choice
        label_space = ["A", "B", "C", "D", "E"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_question, example_options, example_answer = example["question"], example["options"], example["answer"]
            A, B, C, D, E = example_options
            example_choice = ["A", "B", "C", "D", "E"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E, answer=example_choice)

        A, B, C, D, E = options
        choice = ["A", "B", "C", "D", "E"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E)
        output_text = choice
        label_space = ["A", "B", "C", "D", "E"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved1(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "Please choose the best option from the listed choices that precisely express the given " \
                         "things in common. {question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} E. " \
                         "{choiceE}\nPlease answer with your choice only without any other words."
        return self.unobserved_template_QA(item, input_template)

    def unobserved2(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "Identify and output the commonality among the given objects:\n Objects:{question}\nA." \
                         "{choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer:"
        return self.unobserved_template_QA(item, input_template)

    def unobserved3(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "You are given three objects {question}, choose the option from below where the objects" \
                         "share the greatest similarity. A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved4(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "Please select the best option to indicate the commonality between the objects: {question}\nA." \
                         "{choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nGive your answer as one " \
                         "of A, B, C, D, E. Answer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved5(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "{question}\nPick the most correct description from:\nA." \
                         "{choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nMy answer is: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved6(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "{question}\n\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nOne correct common thing among " \
                         "all the choices above is "
        return self.unobserved_template_QA(item, input_template)

    def unobserved7(self, item):
        input_template = "Answer the question below: {question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. " \
                         "{choiceD}\nE. {choiceE}\nGive your answer with letter, then explain your choice in the " \
                         "next line"
        return self.unobserved_template_QA(item, input_template)

    def unobserved8(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "What is the common phenomenon in these objects, {question}\n" \
                         "choose the best answer from the following\n" \
                         "A. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved9(self, item):
        input_template = "{question}\nA.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD} E.  {choiceE}\n" \
                         "Pick one of the choices A, B, C, D, E.\n The answer is "
        return self.unobserved_template_QA(item, input_template)

    def unobserved10(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        input_template = "You will be given several objects or activities. From a series of choices (A, B, C, D, E)," \
                         " identify an aspect that they all share in common. If there are multiple aspects, identify" \
                         " the one best fitting.\n\nWhat do these objects have in common: {question}\nA. {choiceA}\nB." \
                         " {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\n\nAnswer: "

        return self.unobserved_template_QA(item, input_template)
    
    def unobserved1_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = "Please choose the best option from the listed choices that precisely express the given " \
                                "things in common. Please answer with your choice only without any other words."

        example_template = " {question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} Answer: {answer}"

        input_template = " {question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved2_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = "Identify and output the commonality among the given objects:"

        example_template = " Objects:{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer: {answer}"

        input_template = " Objects:{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved3_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = "You are given three objects , choose the option from below where the objects share the greatest similarity."

        example_template = "\n{question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} Answer: {answer}"

        input_template = "\n{question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = "Please select the best option to indicate the commonality between the objects: Give your answer as one of A, B, C, D, E."

        example_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer: {answer}"

        input_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = ""

        example_template = "{question}\nPick the most correct description from:\nA." \
                         "{choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nMy answer is: {answer}\n"

        input_template = "{question}\nPick the most correct description from:\nA." \
                         "{choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nMy answer is: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = ""

        example_template = "{question}\n\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nOne correct common thing among all the choices above is {answer}\n"

        input_template = "{question}\n\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nOne correct common thing among all the choices above is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = "Answer the question below. Give your answer with letter, then explain your choice in the next line"

        example_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer: {answer}"

        input_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = "Answer the question below. Give your answer with letter, then explain your choice in the next line"

        example_template = "What is the common phenomenon in these objects, {question}\n" \
                         "choose the best answer from the following\n" \
                         "A. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\n{answer}"

        input_template = "What is the common phenomenon in these objects, {question}\n" \
                         "choose the best answer from the following\n" \
                         "A. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\n"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\nA.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD} E.  {choiceE}\nPick one of the choices A, B, C, D, E.\n The answer is {answer}\n"

        input_template = "{question}\nA.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD} E.  {choiceE}\nPick one of the choices A, B, C, D, E.\n The answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        item["question"] = item["question"].replace("What do the following have in common? ", "")
        for example in self.examples:
            example["question"] = example["question"].replace("What do the following have in common? ", "")

        input_template_prefix = "You will be given several objects or activities. From a series of choices (A, B, C, D, E)," \
                         " identify an aspect that they all share in common. If there are multiple aspects, identify" \
                         " the one best fitting."

        example_template = "\n\nWhat do these objects have in common: {question}\nA. {choiceA}\nB." \
                         " {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\n\nAnswer: {answer}"

        input_template = "\n\nWhat do these objects have in common: {question}\nA. {choiceA}\nB." \
                         " {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)