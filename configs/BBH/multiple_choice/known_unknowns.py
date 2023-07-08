import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
import random
from datasets import Dataset
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = KnownUnknownPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

def load_data_testing(input_dir, instruction, args):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = KnownUnknownPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class KnownUnknownPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "A test of 'hallucinations' by asking questions whose answers are known to be unknown",
        "Please select the best option for the question given to you based on the correct factual knowledge.",
        "Verify if the question is unknown, choose your answer from options.",
        "You are given a question asking about a specific knowledge. You need to respond with either the actual knowledge or it cannot be known.",
        "Determine if the question is factually knowable by choosing from the following options.",
        "Answer the following questions based on the list of available choices",
        "This is a test of ‘hallucination’, choose the most appropriate option for the question",
        "You will be given questions to test your knowledge of whether or not it is possible to know certain pieces of information. Each question either has an answer that you know or an answer that is unknown. For each of the questions below, please choose from the multiple choices (A, B) and provide an explanation when applicable."
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(KnownUnknownPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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

    def unobserved_template_QA(self, item, input_temptlate):
        question, options, answer = item["question"], item["options"], item["answer"]
        A, B = options
        choice = ["A", "B"][options.index(answer)]
        input_text = input_temptlate.format(question=question, choiceA=A, choiceB=B)
        output_text = choice
        label_space = ["A", "B"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_question, example_options, example_answer = example["question"], example["options"], example["answer"]
            A, B = example_options
            example_choice = ["A", "B"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, choiceA=A, choiceB=B, answer=example_choice)

        A, B = options
        choice = ["A", "B"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B)
        output_text = choice
        label_space = ["A", "B"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "Please select the best option for the question given to you based on the correct factual" \
                         " knowledge. Question: {question} A. {choiceA} B. {choiceB}\n" \
                         "Please answer with your choice only without any other words."
        return self.unobserved_template_QA(item, input_template)

    def unobserved2(self, item):
        input_template = "Verify if the question is unknown, choose your answer from options:\nQuestion: {question}\n" \
                         "Options:\nA: {choiceA}\nB: {choiceB}\nAnswer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved3(self, item):
        input_template = "You are given a question asking about a specific knowledge. You need to respond with either" \
                         "the actual knowledge or it cannot be known.\nQuestion: {question}\nOptions:\nA: {choiceA}" \
                         "\nB: {choiceB}\nAnswer with \"A\" or \"B\"."
        return self.unobserved_template_QA(item, input_template)

    def unobserved4(self, item):
        input_template = "Determine if the question is factually knowable by choosing from the following options:\n" \
                         "Q: {question}\n(A) {choiceA}\n(B) {choiceB}\nAnswer: ("
        return self.unobserved_template_QA(item, input_template)

    def unobserved5(self, item):
        input_template = "Answer the following questions based on the list of available choices\n{question}\nA: " \
                         "{choiceA}\nB: {choiceB}\nAnswer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved6(self, item):
        input_template = "{question}\n\nA. {choiceA} B. {choiceB}\n\nWith respect to the choices above, the correct " \
                         "one is "
        return self.unobserved_template_QA(item, input_template)

    def unobserved7(self, item):
        input_template = "Question: {question}\nTo avoid hallucination, if the answer to this question is unknown," \
                         " output \"B\", otherwise output \"A\""
        return self.unobserved_template_QA(item, input_template)

    def unobserved8(self, item):
        input_template = "This is a test of \'hallucination\', choose the most appropriate option for the question:" \
                         " {question} A.  {choiceA} B.  {choiceB}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved9(self, item):
        input_template = "{question}\n A.  {choiceA} B.  {choiceB}\nWhich of the choices between A and B is correct? " \
                         "\nThe correct option is"
        return self.unobserved_template_QA(item, input_template)

    def unobserved10(self, item):
        input_template = "You will be given questions to test your knowledge of whether or not it is possible " \
                         "to know certain pieces of information. Each question either has an answer that you know " \
                         "or an answer that is unknown. For each of the questions below, please choose from the " \
                         "multiple choices (A, B) and provide an explanation when applicable.\n\nQuestion: {question}" \
                         "\nA: {choiceA}\nB: {choiceB}\n\nAnswer: "
        return self.unobserved_template_QA(item, input_template)
    
    def unobserved1_fs(self, item):
        input_template_prefix = "Please select the best option for the question given to you based on the correct factual knowledge." \
                                "Please answer with your choice only without any other words. "

        example_template = "Question: {question} A. {choiceA} B. {choiceB} {answer} "

        input_template = "Question: {question} A. {choiceA} B. {choiceB} "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = "Verify if the question is unknown, choose your answer from options:"

        example_template = "\nQuestion: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\nAnswer: {answer}"

        input_template = "\nQuestion: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "You are given a question asking about a specific knowledge. You need to respond with either the actual knowledge or it cannot be known."

        example_template = "\nQuestion: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\nAnswer with \"A\" or \"B\": {answer}"

        input_template = "\nQuestion: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\nAnswer with \"A\" or \"B\": "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved4_fs(self, item):
        input_template_prefix = "Determine if the question is factually knowable by choosing from the following options:\n"

        example_template = "Q: {question}\n(A) {choiceA}\n(B) {choiceB}\nAnswer: ({answer})"

        input_template = "Q: {question}\n(A) {choiceA}\n(B) {choiceB}\nAnswer: ("
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = "Answer the following questions based on the list of available choices"

        example_template = "\n{question}\nA: {choiceA}\nB: {choiceB}\nAnswer: {answer}"

        input_template = "\n{question}\nA: {choiceA}\nB: {choiceB}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\n\nA. {choiceA} B. {choiceB}\n\nWith respect to the choices above, the correct one is {answer}\n"

        input_template = "{question}\n\nA. {choiceA} B. {choiceB}\n\nWith respect to the choices above, the correct one is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = ""

        example_template = "Question: {question}\nTo avoid hallucination, if the answer to this question is unknown," \
                         " output \"B\", otherwise output \"A\". {answer}\n"

        input_template = "Question: {question}\nTo avoid hallucination, if the answer to this question is unknown," \
                         " output \"B\", otherwise output \"A\". "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "This is a test of \'hallucination\', choose the most appropriate option for the question:"

        example_template = "\n{question} A. {choiceA} B.  {choiceB} {answer}\n"

        input_template = "\n{question} A. {choiceA} B.  {choiceB}"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\n A.  {choiceA} B.  {choiceB}\nWhich of the choices between A and B is correct? " \
                         "\nThe correct option is {answer}\n"

        input_template = "{question}\n A.  {choiceA} B.  {choiceB}\nWhich of the choices between A and B is correct? " \
                         "\nThe correct option is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = "You will be given questions to test your knowledge of whether or not it is possible " \
                         "to know certain pieces of information. Each question either has an answer that you know " \
                         "or an answer that is unknown. For each of the questions below, please choose from the " \
                         "multiple choices (A, B) and provide an explanation when applicable."

        example_template = "\n\nQuestion: {question}\nA: {choiceA}\nB: {choiceB}\n\nAnswer: {answer}"

        input_template = "\n\nQuestion: {question}\nA: {choiceA}\nB: {choiceB}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)