import os.path
import pandas as pd
import random
from datasets import Dataset

from configs.utils import (
    OptionMatchingAccuracy,
    ClassificationAccuracy,
)

from configs.preprocessor import Preprocessor
import multiprocessing as mp

SUBSET = False

special_tokens = []
fewshot_examples = {
    "abstract_algebra_test": [0],
    "astronomy_test": [0],
    "college_medicine_test": [0],
    "international_law_test": [0],
    "us_foreign_policy_test": [0]
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

    preprocessor = MMLUGeneralPreprocessor(instruction, examples, eval_by_logits)

    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "answer", "options"], num_proc=1)

    return test_set


def load_data_testing(input_dir, instruction, args):
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
        for item in file_items:
            to_qa_dict(item)
            items.append(item)

    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = MMLUGeneralPreprocessor(instruction, examples, True)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "answer", "options"], num_proc=1)

    return test_set


class MMLUGeneralPreprocessor(Preprocessor):
    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Please act as a domain expert to choose the most suitable answer from the given choices to the question below. Please answer the question with your choice only without any other words.",
        "Solve the question with professional knowledge and output the best option for the question from \"A\", \"B\", \"C\", \"D\" without other words:",
        "Solve the question which requires deep understanding to the field.",
        "You are an expert. Answer the following question.",
        "Solve the following question which needs expert knowledge. Choose the best option that you think is correct from the following options.",
        "Please solve this question (requires expert level knowledge). Output one of \"A\", \"B\", \"C\", or \"D\" to indicate your answer",
        "You are given multiple-choice questions from a variety of domains. For each question, please select an answer from A, B, C, and D, and explain your reasoning.",
        "You are given a multiple-choice question that requires expert domain knowledge to answer correctly. Please choose the best answer from A, B, C, and D.",
        "Imagine you are a college student. Pick the correct answer, given the question",
        "The following is a multiple-choice question that requires expert-level domain knowledge. Please select the correct answer to the question below from the options \"A\", \"B\", \"C\", or \"D\" after carefully examining each choice provided below"
        "Please provide the correct answer to the following question, which requires expert level knowledge by choosing one of the options below and outputting it as your answer",
        "Please solve this question (requires expert level knowledge). Output one of \"A\", \"B\", \"C\", or \"D\" to indicate your answer",
        "Answer the following multiple choice question by selecting one of the options in the list (A, B, C, D):",
        "Please answer the following question, you should choose the most appropriate answer only from the four options",
        "What is the correct answer to the following multiple choice question?",
        "Given the question, output one of A, B, C, and D to indicate the correct choice.",
        "What is the answer to the question?",
        "You are given a question that requires knowledge from a specific domain.",
        "You are a college student, and you are tested in an exam with this question",
        "I want to know the answer to this question. Please select from the following and indicate your answer with the letter.",
        "Task: Multiple-choice question answering.",
        "I am working with an exam question that has four different options.",
        "A multiple-choice question is given. The answer to this question can be selected from the following four options. Use your knowledge to find the correct choice",
        "A question is given following with 4 options. Select the most correct options, output one of \"A\", \"B\", \"C\", or \"D\", and explain your choice with chain of thought.",
        "Please answer the appropriate questions based on knowledge you have. Each question will give four options, please output the corresponding option (i.e. A, B, C or D) to represent the corresponding answer. Do not include text in your output",
        "This is a single-choice question coming from exams. Use your knowledge to solve the following question and select the correct answer among \"A\", \"B\", \"C\", and \"D\". Just output the answer with the corresponding letter!",
        "This is a multiple choice test on college-level knowledge. Please choose the correct answer among A, B, C and D by finding the most correct one.",
        "Please answer the question using your knowledge. Output one of \"A\", \"B\", \"C\", or \"D\" to indicate your answer",
        "Answer the following question with your knowledge. Note that there may be more than one correct option.",
        "Now you are an expert with vast knowledge from different domains. Output one of \"A\", \"B\", \"C\", or \"D\" to indicate your answer for the following question",
        "Based on the knowledge of your expertises, given the following question, output the best choice from \"A\", \"B\", \"C\", and \"D\".",
        "Please use your domain-specific knowledge to answer the following questions",
        "Employ your knowledge to tackle the given question. Choose the right answer as \"A\", \"B\", \"C\", or \"D\"",
        "You are a useful assistant at answering questions from various domain. Please solve this question with an output of \"A\", \"B\", \"C\", or \"D\""
    ]

    def __init__(self, instruction, examples, eval_by_logits):
        super(MMLUGeneralPreprocessor, self).__init__(instruction, examples, eval_by_logits)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["MMLU/Unobserved/1"] = self.unobserved1
        self.instr2preprocessor["MMLU/Unobserved/2"] = self.unobserved2
        self.instr2preprocessor["MMLU/Unobserved/3"] = self.unobserved3
        self.instr2preprocessor["MMLU/Unobserved/4"] = self.unobserved4
        self.instr2preprocessor["MMLU/Unobserved/5"] = self.unobserved5
        self.instr2preprocessor["MMLU/Unobserved/6"] = self.unobserved6
        self.instr2preprocessor["MMLU/Unobserved/7"] = self.unobserved7
        self.instr2preprocessor["MMLU/Unobserved/8"] = self.unobserved8
        self.instr2preprocessor["MMLU/Unobserved/9"] = self.unobserved9
        self.instr2preprocessor["MMLU/Unobserved/10"] = self.unobserved10
        self.instr2preprocessor["MMLU/Unobserved/11"] = self.unobserved11
        self.instr2preprocessor["MMLU/Unobserved/12"] = self.unobserved12
        self.instr2preprocessor["MMLU/Unobserved/13"] = self.unobserved13
        self.instr2preprocessor["MMLU/Unobserved/14"] = self.unobserved14
        self.instr2preprocessor["MMLU/Unobserved/15"] = self.unobserved15
        self.instr2preprocessor["MMLU/Unobserved/16"] = self.unobserved16
        self.instr2preprocessor["MMLU/Unobserved/17"] = self.unobserved17
        self.instr2preprocessor["MMLU/Unobserved/18"] = self.unobserved18
        self.instr2preprocessor["MMLU/Unobserved/19"] = self.unobserved19
        self.instr2preprocessor["MMLU/Unobserved/20"] = self.unobserved20

        self.instr2preprocessor_fs["MMLU/Unobserved/1"] = self.unobserved1_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/2"] = self.unobserved2_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/3"] = self.unobserved3_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/4"] = self.unobserved4_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/5"] = self.unobserved5_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/6"] = self.unobserved6_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/7"] = self.unobserved7_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/8"] = self.unobserved8_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/9"] = self.unobserved9_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/10"] = self.unobserved10_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/11"] = self.unobserved11_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/12"] = self.unobserved12_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/13"] = self.unobserved13_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/14"] = self.unobserved14_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/15"] = self.unobserved15_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/16"] = self.unobserved16_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/17"] = self.unobserved17_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/18"] = self.unobserved18_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/19"] = self.unobserved19_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/20"] = self.unobserved20_fs

        self.instr2preprocessor["Alpaca/Unobserved/1"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      0])
        self.instr2preprocessor["Alpaca/Unobserved/2"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      1])
        self.instr2preprocessor["Alpaca/Unobserved/3"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      2])
        self.instr2preprocessor["Alpaca/Unobserved/4"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      3])
        self.instr2preprocessor["Alpaca/Unobserved/5"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      4])
        self.instr2preprocessor["Alpaca/Unobserved/6"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      5])
        self.instr2preprocessor["Alpaca/Unobserved/7"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      6])
        self.instr2preprocessor["Alpaca/Unobserved/8"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      7])
        self.instr2preprocessor["Alpaca/Unobserved/9"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                  self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                      8])
        self.instr2preprocessor["Alpaca/Unobserved/10"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       9])
        self.instr2preprocessor["Alpaca/Unobserved/11"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       10])
        self.instr2preprocessor["Alpaca/Unobserved/12"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       11])
        self.instr2preprocessor["Alpaca/Unobserved/13"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       12])
        self.instr2preprocessor["Alpaca/Unobserved/14"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       13])
        self.instr2preprocessor["Alpaca/Unobserved/15"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       14])
        self.instr2preprocessor["Alpaca/Unobserved/16"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       15])
        self.instr2preprocessor["Alpaca/Unobserved/17"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       16])
        self.instr2preprocessor["Alpaca/Unobserved/18"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       17])
        self.instr2preprocessor["Alpaca/Unobserved/19"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       18])
        self.instr2preprocessor["Alpaca/Unobserved/20"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       19])
        self.instr2preprocessor["Alpaca/Unobserved/21"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       20])
        self.instr2preprocessor["Alpaca/Unobserved/22"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       21])
        self.instr2preprocessor["Alpaca/Unobserved/23"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       22])
        self.instr2preprocessor["Alpaca/Unobserved/24"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       23])
        self.instr2preprocessor["Alpaca/Unobserved/25"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       24])
        self.instr2preprocessor["Alpaca/Unobserved/26"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       25])
        self.instr2preprocessor["Alpaca/Unobserved/27"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       26])
        self.instr2preprocessor["Alpaca/Unobserved/28"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       27])
        self.instr2preprocessor["Alpaca/Unobserved/29"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       28])
        self.instr2preprocessor["Alpaca/Unobserved/30"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       29])
        self.instr2preprocessor["Alpaca/Unobserved/31"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       30])
        self.instr2preprocessor["Alpaca/Unobserved/32"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       31])
        self.instr2preprocessor["Alpaca/Unobserved/33"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       32])
        self.instr2preprocessor["Alpaca/Unobserved/34"] = lambda item: self.alpaca_multiple_choice(item,
                                                                                                   self._UNOBSERVED_ALPACA_INSTRUCTIONS[
                                                                                                       33])

    def unobserved_template(self, item, input_temptlate):
        question, options, answer = item["question"], item["options"], item["answer"]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text = input_temptlate.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_question, example_options, example_answer = example["question"], example["options"], example[
                "answer"]
            A, B, C, D = example_options
            example_choice = ["A", "B", "C", "D"][example_options.index(example_answer)]
            input_text += example_template.format(id=i + 1, question=example_question, choiceA=A, choiceB=B, choiceC=C,
                                                  choiceD=D, answer=example_choice)

        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "Please act as a domain expert to choose the most suitable answer from the given choices" \
                         " to the question below. Question: {question}. Choices: A. {choiceA}  B. {choiceB} C. " \
                         "{choiceC} D. {choiceD}\nPlease answer the question with your choice only without any other words."
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "Solve the question with professional knowledge and output the best option for the question " \
                         "from \"A\", \"B\", \"C\", \"D\" without other words:\nQuestion: {question}\nOptions:\n" \
                         "A: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved3(self, item):
        input_template = "Solve the question which requires deep understanding to the field. {question}\nChoose from:\n" \
                         "A: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved4(self, item):
        input_template = "{question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nThe correct answer to " \
                         "this question is ("
        return self.unobserved_template(item, input_template)

    def unobserved5(self, item):
        input_template = "{question}\n\nA. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD}\nI know exactly the " \
                         "answer to this question! The correct choice is "
        return self.unobserved_template(item, input_template)

    def unobserved6(self, item):
        input_template = "You are given multiple-choice questions from a variety of domains. For each question, please" \
                         " select an answer from A, B, C, and D, and explain your reasoning.\n\nQuestion:" \
                         " {question}\nThe options are:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}" \
                         "\n\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved7(self, item):
        input_template = "Please provide the correct answer to the following question, which requires expert level " \
                         "knowledge by choosing one of the options below and outputting it as your answer:" \
                         "\n\nQuestion: {question}\nOptions\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                         "{choiceD}\nYour answer:"
        return self.unobserved_template(item, input_template)

    def unobserved8(self, item):
        input_template = "{question}\nOptions:\n\t-A {choiceA}\n\t-B {choiceB}\n\t-C {choiceC}\n\t-D {choiceD}\nWhich" \
                         " option is correct?:"
        return self.unobserved_template(item, input_template)

    def unobserved9(self, item):
        input_template = "Given the question: {question}, and the choices for the answer are A. {choiceA}, B. {choiceB}, " \
                         "C. {choiceC}, D. {choiceD}. Output one of A, B, C, and D to indicate the correct choice. The " \
                         "correct choice is: "
        return self.unobserved_template(item, input_template)

    def unobserved10(self, item):
        input_template = "What is the answer to the question: {question} A. {choiceA}, B. {choiceB}, C. {choiceC}," \
                         " D. {choiceD}"
        return self.unobserved_template(item, input_template)

    def unobserved11(self, item):
        input_template = "You are given a question that requires knowledge from a specific domain. Question: " \
                         "{question}. Select tha answer from A. '{choiceA}', B. '{choiceB}', C. '{choiceC}', D. " \
                         "and '{choiceD}'. Answer:"
        return self.unobserved_template(item, input_template)

    def unobserved12(self, item):
        input_template = "I want to know the answer to this question: {question}. Please select from the following: " \
                         "A. {choiceA}, B. {choiceB}, C. {choiceC}, D. {choiceD}. Indicate your choice with the letter" \
                         "."
        return self.unobserved_template(item, input_template)

    def unobserved13(self, item):
        input_template = "Question: {question}. Choices: A. {choiceA}, B. {choiceB}, C. {choiceC}, D. {choiceD}." \
                         " Answer:"
        return self.unobserved_template(item, input_template)

    def unobserved14(self, item):
        input_template = "Task: Multiple-choice question answering.\nQuestion: {question}\nChoices: (A) {choiceA} " \
                         "(B) {choiceB} (C) {choiceC} (D) {choiceD}\nAnswer: ("
        return self.unobserved_template(item, input_template)

    def unobserved15(self, item):
        input_template = " I am working with an exam question that has four different options. The question is:\n" \
                         "{question}\nAnd the choices are:\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}\n" \
                         "Here's the answer to the this question:"
        return self.unobserved_template(item, input_template)

    def unobserved16(self, item):
        input_template = "A multiple-choice question is given. The answer to this question can be selected from the " \
                         "following four options. Use your knowledge to find the correct choice: {question}\nA. " \
                         "{choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}"
        return self.unobserved_template(item, input_template)

    def unobserved17(self, item):
        input_template = "A question is given following with 4 options. Select the most correct options, output " \
                         "one of \"A\", \"B\", \"C\", or \"D\", and explain your choice with chain of thought.\n" \
                         "{question}\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}\n Answer: "
        return self.unobserved_template(item, input_template)

    def unobserved18(self, item):
        input_template = "This is a single-choice question coming from exams. Use your knowledge to solve" \
                         " the following question and select the correct answer among \"A\", \"B\", \"C\", and \"D\". " \
                         "Just output the answer with the corresponding letter!\n\nQuestion: {question}\nCandidate " \
                         "Answers:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nThe answer is: "
        return self.unobserved_template(item, input_template)

    def unobserved19(self, item):
        input_template = "Please answer the question using your knowledge. Output one of \"A\", \"B\", " \
                         "\"C\", or \"D\" to indicate your answer: A: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} " \
                         "{question} Answer:"
        return self.unobserved_template(item, input_template)

    def unobserved20(self, item):
        input_template = "Please use your domain-specific knowledge to answer the following questions:" \
                         "\nQuestion: {question}\nThe choices are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D:" \
                         " {choiceD}\n\nThe answer is "
        return self.unobserved_template(item, input_template)
    

    def unobserved1_fs(self, item):
        input_template_prefix = "Please act as a domain expert to choose the most suitable answer from the given choices to the " \
                                "question below. Please answer the question with your choice only without any other words."

        example_template = "Question: {question}. Choices: A. {choiceA}  B. {choiceB} C. {choiceC} D. {choiceD}\n {answer}\n"

        input_template = "Question: {question}. Choices: A. {choiceA}  B. {choiceB} C. {choiceC} D. {choiceD}\n"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved2_fs(self, item):
        input_template_prefix = "Here are given examples:\n"

        example_template = "Example{id}: Question: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\nAnswer: {answer}\n"

        input_template = "Refer to given examples, use professional knowledge to solve the question. Output the" \
                         " best option for the question from \"A\", \"B\", \"C\", \"D\" without other words: " \
                         "Question: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "Solve the question which requires deep understanding to the field. "

        example_template = "{question}\nChoose from:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer: {answer}\n"

        input_template = "{question}\nChoose from:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved4_fs(self, item):
        input_template_prefix = ""

        example_template = "{question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nThe correct answer to this question is ({answer})"

        input_template = "{question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nThe correct answer to this question is ("
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\n\nA. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD}\nI know exactly the answer to this question! The correct choice is {answer}"
        input_template = "{question}\n\nA. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD}\nI know exactly the answer to this question! The correct choice is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved6_fs(self, item):
        input_template_prefix = "You are given multiple-choice questions from a variety of domains. For each " \
                                "question, please select an answer from A, B, C, and D\n\n"

        example_template = "Question: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\nAnswer: {answer}\n"

        input_template = "Question: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                         "{choiceD}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved7_fs(self, item):
        input_template_prefix = "Please answer the following questions, which require expert level knowledge, by " \
                                "selecting the appropriate option and outputting it as your answer. The questions " \
                                "and answer options are as follows:\n"

        example_template = "Question {id}: {question}\nOption:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\nYour answer: {answer}\n\n"

        input_template = "Question: {question}\nOption:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                         "{choiceD}\nYour answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved8_fs(self, item):
        input_template_prefix = "What are the answers (out of the options given) to the following questions?"

        example_template = "Question: {question}\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                           "{choiceD}\nThe correct answer is: {answer}\n\n"

        input_template = "Question: {question}:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                         "{choiceD}\nThe correct answer is: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved9_fs(self, item):
        input_text = ""
        for example in self.examples:
            example_texts = self.unobserved9(example)
            input_text += example_texts["input_text"] + " " + example_texts["output_text"] + "\n\n"

        texts = self.unobserved9(item)
        input_text += texts["input_text"]
        output_text = texts["output_text"]
        label_space = texts["label_space"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved10_fs(self, item):
        input_text = ""
        for example in self.examples:
            example_texts = self.unobserved10(example)
            input_text += example_texts["input_text"] + " " + example_texts["output_text"] + "\n\n"

        texts = self.unobserved10(item)
        input_text += texts["input_text"]
        output_text = texts["output_text"]
        label_space = texts["label_space"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved11_fs(self, item):
        input_text = ""
        for example in self.examples:
            example_texts = self.unobserved11(example)
            input_text += example_texts["input_text"] + " " + example_texts["output_text"] + "\n\n"

        texts = self.unobserved11(item)
        input_text += texts["input_text"]
        output_text = texts["output_text"]
        label_space = texts["label_space"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved12_fs(self, item):
        input_text = ""
        for example in self.examples:
            example_texts = self.unobserved12(example)
            input_text += example_texts["input_text"] + " " + example_texts["output_text"] + "\n\n"

        texts = self.unobserved12(item)
        input_text += texts["input_text"]
        output_text = texts["output_text"]
        label_space = texts["label_space"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved13_fs(self, item):
        input_text = ""
        for example in self.examples:
            example_texts = self.unobserved13(example)
            input_text += example_texts["input_text"] + " " + example_texts["output_text"] + "\n\n"

        texts = self.unobserved13(item)
        input_text += texts["input_text"]
        output_text = texts["output_text"]
        label_space = texts["label_space"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved14_fs(self, item):
        input_template_prefix = "Task: Multiple-choice question answering."

        example_template = "\nQuestion: {question}\nChoices: (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nAnswer: ({answer})"

        input_template = "\nQuestion: {question}\nChoices: (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nAnswer: ("
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved15_fs(self, item):
        input_template_prefix = "I am working with an exam question that has four different options."

        example_template = "\nThe question is:\n{question}\nAnd the choices are:\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}\nHere's the answer to the this question: {answer}"

        input_template = "\nThe question is:\n{question}\nAnd the choices are:\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}\nHere's the answer to the this question: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved16_fs(self, item):
        input_template_prefix = "A multiple-choice question is given. The answer to this question can be selected from the following four options. Use your knowledge to find the correct choice:"

        example_template = "{question}\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD} {answer}"

        input_template = "{question}\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved17_fs(self, item):
        input_template_prefix = "A question is given following with 4 options. Select the most correct options, output one of \"A\", \"B\", \"C\", or \"D\", and explain your choice with chain of thought."

        example_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}\n Answer: {answer}"

        input_template = "\n{question}\nA. {choiceA}\nB. {choiceB}\n C. {choiceC}\nD. {choiceD}\n Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved18_fs(self, item):
        input_template_prefix = "This is a single-choice question coming from exams. Use your knowledge to solve the following question and select the correct answer among \"A\", \"B\", \"C\", and \"D\". Just output the answer with the corresponding letter!"

        example_template = "\n\nQuestion: {question}\nCandidate Answers:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nThe answer is: {answer}"

        input_template = "\n\nQuestion: {question}\nCandidate Answers:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nThe answer is: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved19_fs(self, item):
        input_template_prefix = "Please answer the question using your knowledge. Output one of \"A\", \"B\", \"C\", or \"D\" to indicate your answer:"

        example_template = "\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} {question} Answer: {answer}"

        input_template = "\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} {question} Answer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved20_fs(self, item):
        input_template_prefix = "Please use your domain-specific knowledge to answer the following questions:"

        example_template = "\nQuestion: {question}\nThe choices are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: {choiceD}\n\nThe answer is {answer}"

        input_template = "\nQuestion: {question}\nThe choices are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: {choiceD}\n\nThe answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)






