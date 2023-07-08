import os.path
import json
from configs.preprocessor import Preprocessor
import random
import multiprocessing as mp
from datasets import Dataset
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = LogicalDeductionPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

def load_data_testing(input_dir, instruction, args):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = LogicalDeductionPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class LogicalDeductionPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Give an English language description of Python code.",
        "Suppose you are an experienced code reviewer, please give an English language description of Python code. Which is the most suitable?",
        "You are given a simple line of Python code. Try to find out its English equivalency from the following short sentences.",
        "Choose a right English language description of the given Python Code from the four candidates.",
        "To paint in words the function of the given code, which one of A, B, C, D is the most accurate description:",
        "Now you are a code explainer. Question: Here is a Python code. Please choose the right interpretation of the code from the following",
        "We have the following Python code, which one is the correct interpretation? Output the best choice from \"A\", \"B\", \"C\", and \"D\".",
        "One of the following options: A, B, C or D is the actual annotation of the python code. Which one is it?",
        "For the Python code snippet, select the appropriate English description from the options below (output both the choice and the description)",
        "Question: Give the most suitable annotation to this code",
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(LogicalDeductionPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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


    def unobserved_template_QA(self, item, input_temptlate):
        question, options, answer = item["question"], item["options"], item["answer"]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text = input_temptlate.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_QA_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_question, example_options, example_answer = example["question"], example["options"], example["answer"]
            A, B, C, D = example_options
            example_choice = ["A", "B", "C", "D"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, choiceA=A, choiceB=B, choiceC=C,
                                                  choiceD=D, answer=example_choice)

        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "If you are an experienced code reviewer, please give an English language description of " \
                         "Python code{question}. Which is the most suitable? A.  {choiceA} B.  {choiceB} C.  " \
                         "{choiceC} D.  {choiceD}\n\nAnswer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved2(self, item):
        input_template = "You are given a simple line of Python code {question}. Try to find out its English " \
                         "equivalency from the following short sentences: A) {choiceA}, B) {choiceB} C) {choiceC}, " \
                         "D) {choiceD}. The equivalent sentence is: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved3(self, item):
        input_template = "Choose a right English language description of the given Python Code from the four " \
                         "candidates.\nPython Code: {question}\nCandidates: A. {choiceA}, B. {choiceB} C. {choiceC}, " \
                         "D. {choiceD}\nAnswer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved4(self, item):
        input_template = "To paint in words the function of the given code {question}, which one of A. {choiceA} B. " \
                         "{choiceB} C. {choiceC} D.  {choiceD} is the most accurate description:"
        return self.unobserved_template_QA(item, input_template)

    def unobserved5(self, item):
        input_template = "Now you are a code explainer.  Question: Here is a Python code {question} Please choose the" \
                         " right interpretation of the code from the following: A. {choiceA} B. {choiceB} C. " \
                         "{choiceC} D.  {choiceD}\n\nAnswer:"
        return self.unobserved_template_QA(item, input_template)

    def unobserved6(self, item):
        input_template = "We have the following Python code {question}, which one is the correct interpretation, " \
                         "output the best choice from \"A\", \"B\", \"C\", and \"D\".\n- A. {choiceA}\n- B. {choiceB}" \
                         "\n- C. {choiceC}\n- D.  {choiceD}\nThe best choice is "
        return self.unobserved_template_QA(item, input_template)

    def unobserved7(self, item):
        input_template = "One of the following options: A.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD} is the" \
                         " actual annotation of the python code: \"{question}\". Which one is it? Answer:"
        return self.unobserved_template_QA(item, input_template)

    def unobserved8(self, item):
        input_template = "For the Python code snippet {question}, select the appropriate English description from " \
                         "the options below (output both the choice and the description):\nA. {choiceA}\nB. {choiceB}\nC. " \
                         "{choiceC}\nD. {choiceD}\nOutput: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved9(self, item):
        input_template = "Question: Give the most suitable annotation to this code:\n{question}\nA. {choiceA}\nB. " \
                         "{choiceB}\nC. {choiceC}\nD. {choiceD}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved10(self, item):
        input_template = "A. // {choiceA}\n{question}\n\nB. // {choiceB}\n{question}\n\nC. // {choiceC}\n{question}" \
                         "D. // {choiceD}\n{question}\n\n\nFrom the four different python code A, B, C, and D, choose" \
                         " the code with the most correct specification."
        return self.unobserved_template_QA(item, input_template)

    def unobserved1_fs(self, item):
        input_template_prefix = "If you are an experienced code reviewer, please give an English language description " \
                                "of Python code.\nFor example,"

        example_template = "\nFor {question}. A.  {choiceA} B.  {choiceB} C.  {choiceC} D.  " \
                           "{choiceD}\nAnswer: {answer}"

        input_template = "\nNow, For {question}. A.  {choiceA} B.  {choiceB} C.  {choiceC} D.  " \
                         "{choiceD}\nAnswer: "
        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved2_fs(self, item):
        input_template_prefix = "You are given a simple line of Python code. Try to find out its English " \
                                "equivalency from the following short sentences\n"

        example_template = "{question} A) {choiceA}, B) {choiceB} C) {choiceC}, D) {choiceD}. The equivalent sentence" \
                           " is {answer}\n"

        input_template = "{question} A) {choiceA}, B) {choiceB} C) {choiceC}, D) {choiceD}. The equivalent sentence" \
                         " is "
        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved3_fs(self, item):
        input_template_prefix = "Choose a right English language description of the given Python Code from the " \
                                "four candidates.\n"

        example_template = "Python Code: {question}\nCandidates: A. {choiceA}, B. {choiceB} C. {choiceC}, D. " \
                           "{choiceD}\nAnswer: {answer}\n"

        input_template = "Python Code: {question}\nCandidates: A. {choiceA}, B. {choiceB} C. {choiceC}, D. " \
                         "{choiceD}\nAnswer:"

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved4_fs(self, item):
        input_template_prefix = "I will show you a piece of python code, you need to pick the best description of " \
                                "its function from four choices. For example:\n"

        example_template = "\n{question} A. {choiceA}, B. {choiceB} C. {choiceC}, D. " \
                           "{choiceD} The answer to this one is {answer}"

        input_template = "\nPlease answer by yourself {question} A. {choiceA}, B. {choiceB} C. {choiceC}, D. {choiceD} " \
                         "The answer to this one is "
        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved5_fs(self, item):
        input_template_prefix = "Now you are a code explainer. You should give an English language description of " \
                                "any Python code. "

        example_template = "Here's an example: {question} A. {choiceA}, B. {choiceB} C. {choiceC}, D. {choiceD}\n\n" \
                           "Your English language description:\n{answer}\n\n"

        input_template = "Now you do this: {question} A. {choiceA}, B. {choiceB} C. {choiceC}, D. {choiceD}\n\n" \
                         "Your English language description:\n"

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved6_fs(self, item):
        input_template_prefix = "Please give an English description to interpret a piece of Python code. We have the" \
                                " following examples:\n\n"

        example_template = "Python Code: {question}\n- A. {choiceA}\n- B. {choiceB}\n- C. {choiceC},\n- D. {choiceD}\n" \
                           "English description: {answer}\n"

        input_template = "Python Code: {question}\n- A. {choiceA}\n- B. {choiceB}\n- C. {choiceC},\n- D. {choiceD}\n" \
                         "English description: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved7_fs(self, item):
        input_template_prefix = ""

        example_template = "One of the following options: A.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD} is the" \
                         " actual annotation of the python code: \"{question}\". Which one is it? Answer: {answer}\n"

        input_template = "One of the following options: A.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD} is the" \
                         " actual annotation of the python code: \"{question}\". Which one is it? Answer:"

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved8_fs(self, item):
        input_template_prefix = "To provide an English language description of the Python code, consider the " \
                                "following examples:\n"

        example_template = "\nExample {id}: {question}\nOptions:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. " \
                           "{choiceD}\nEnglish language description:\n{answer}"

        input_template = "\Question: {question}\nOptions:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. " \
                           "{choiceD}\nEnglish language description:\nAnswer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = "Question: Give the most suitable annotation to this code:\n"

        example_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: {answer}"

        input_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = ""

        example_template = "\nA. // {choiceA}\n{question}\n\nB. // {choiceB}\n{question}\n\nC. // {choiceC}\n{question}" \
                         "D. // {choiceD}\n{question}\n\n\nAnswer: {answer}"

        input_template = "\nA. // {choiceA}\n{question}\n\nB. // {choiceB}\n{question}\n\nC. // {choiceC}\n{question}" \
                         "D. // {choiceD}\n{question}\n\n\nFrom the four different python code A, B, C, and D, choose" \
                         " the code with the most correct specification.\nAnswer:"

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)
