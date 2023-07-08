import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
from configs.utils import load_BBL_file
import random

special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    for item in items:
        item["options"] = ["1", "2", "3", "4", "5"]
    test_set = Dataset.from_list(items)
    preprocessor = LogicGridPuzzlePreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

def load_data_testing(input_dir, instruction, args):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    for item in items:
        item["options"] = ["1", "2", "3", "4", "5"]
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = LogicGridPuzzlePreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class LogicGridPuzzlePreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Solve logic grid puzzles",
        "Here's a puzzle for you as well as some clus. Now, answer the following question:",
        "You are given a logic grid puzzle to test your sense of space and positions. You are given a context and some clues to pick the correct answer from the options to answer a question.",
        "Answer this question based on the following context and clues:",
        "You are a master at solving logic grid puzzles. Solve this:",
        "The task is to solve a logic grid puzzle. You will have the context to the problem and some clues to solve the puzzle. Output your answer as one of \"A\", \"B\", \"C\", \"D\", \"E\".",
        "Here's a complex puzzle. Utitlize your skills to solve the puzzle by logic grid tables.",
        "This question can only be answered if you fully understand the context:",
        "You are tested for your ability to answer logic grid problems correctly.",
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(LogicGridPuzzlePreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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

        self.instr2preprocessor["Alpaca/Default/1"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[0])
        self.instr2preprocessor["Alpaca/Unobserved/1"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[1])
        self.instr2preprocessor["Alpaca/Unobserved/2"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[2])
        self.instr2preprocessor["Alpaca/Unobserved/3"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[3])
        self.instr2preprocessor["Alpaca/Unobserved/4"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[4])
        self.instr2preprocessor["Alpaca/Unobserved/5"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[5])
        self.instr2preprocessor["Alpaca/Unobserved/6"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[6])
        self.instr2preprocessor["Alpaca/Unobserved/7"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[7])
        self.instr2preprocessor["Alpaca/Unobserved/8"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[8])

    def unobserved_template(self, item, input_temptlate):
        options, answer = item["options"], item["answer"]
        context, clues, question = item["question"].split("\n\n")
        A, B, C, D, E = options
        choice = ["A", "B", "C", "D", "E"][options.index(answer)]
        input_text = input_temptlate.format(context=context, clues=clues, question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E)
        output_text = choice
        label_space = ["A", "B", "C", "D", "E"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict


    def unobserved1(self, item):
        input_template = "Here's a puzzle for you {context}\n\nHere are some clus:\n{clues}\n\n\nNow, answer the following question:\n{question}\n\n\nWhat is the correct answer?\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\n\n\nThe correct answer is: "
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "You are given a logic grid puzzle to test your sense of space and positions. You are given a context and some clues to pick the correct answer from the options to answer a question." \
                         "Context: {context}\n{clues}\nQuestion: {question}\nOptions:\n(A) {choiceA}\n(B) {choiceB}\n(C) {choiceC}\n(D) {choiceD}\n(E) {choiceE}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved3(self, item):
        answer = item["answer"]
        context, clues, question = item["question"].split("\n\n")
        input_text = "Question: {question}. Answer this question based on the following context and clues:\n\nContext: {context}\n{clues}\n\nThe answer is one of \"1\", \"2\", \"3\", \"4\", \"5\". The correct answer is: ".format(context=context, clues=clues, question=question)
        output_text = answer
        label_space = ["1", "2", "3", "4", "5"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved4(self, item):
        input_template = "You are a master at solving logic grid puzzles. Solve this: {context}\n\n{clues}\n\n{question}"
        return self.unobserved_template(item, input_template)
    
    def unobserved5(self, item):
        input_template = "{context}\n{clues}Based on the puzzle provided above, {question}\n\nOptions are: A: {choiceA}, B: {choiceB}, C: {choiceC}, D: {choiceD}, E: {choiceE}\n\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved6(self, item):
        input_template = "The task is to solve a logic grid puzzle. You will have the context to the problem and some clues to " \
                         "solve the puzzle.\nContext: {context}{clues}\nThe question is: {question}\nOptions:\nA. {choiceA}\nB. " \
                         "{choiceB}\nC. {choiceC}\nD. {choiceD}\nE. {choiceE}\nOutput your answer as one of \"A\", \"B\", \"C\", \"D\", \"E\"."
        return self.unobserved_template(item, input_template)
    
    def unobserved7(self, item):
        input_template = "Here's a complex puzzle. Utitlize your skills to solve the puzzle by logic grid tables.{context}\n{clues}\n{question}"
        return self.unobserved_template(item, input_template) 
    
    def unobserved8(self, item):
        input_template = "{question}\nThis question can only be answered if you fully understand the context: " \
                         "{context}\n{clues}. Your choices are:\n(A) {choiceA}\n(B) {choiceB}\n(C) {choiceC}\n(D) {choiceD}\n(E) {choiceE}\n\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved9(self, item):
        answer = item["answer"]
        context, clues, question = item["question"].split("\n\n")
        input_text = "You are tested for your ability to answer logic grid problems correctly. {question}\n{clues}\n{question}\nThe answer is always one of \'1\', \'2\', \'3\', \'4\', or \'5\'. Output your answer and give explanation".format(context=context, clues=clues, question=question)
        output_text = answer
        label_space = ["1", "2", "3", "4", "5"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved10(self, item):
        input_template = "Context: {context}\n\nThe following clues are always true: {clues}\n\nNow, infer the answer to this question: \"{question}\" and pick the correct answer from A) {choiceA} B) {choiceB} C) {choiceC} D) {choiceD} E) {choiceE}\n\nAnswer:"
        return self.unobserved_template(item, input_template)

    