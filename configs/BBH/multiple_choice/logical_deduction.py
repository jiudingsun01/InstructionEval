import os.path
from configs.preprocessor import Preprocessor
import multiprocessing as mp
import random
from datasets import Dataset
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    input_dir = os.path.join(input_dir, "five_objects")

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = LogicalDeductionPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

def load_data_testing(input_dir, instruction, args):

    input_dir = os.path.join(input_dir, "five_objects")
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = LogicalDeductionPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

class LogicalDeductionPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\n",
        "You are given a paragraph that describes five objects arranged in order. Please select the best answer from A, B, C, D, and E which the answer contains a statement that is logically consistent with the paragraph.",
        "The most logically-correct answer, given this paraphraph?",
        "You are taking an exam where you will be given a paragraph of text describing five different objects in a sequence that are arranged in a fixed order. To answer the question correctly, you must keep track of where each object is in the sequence and then select the multiple choice answer that best corresponds to the correct answer from (\"A\", \"B\", \"C\", \"D\", \"E\"). Please carefully consider the information in the following paragraph and each of the answers before providing the right answer.",
        "Each of the following paragraphs describes a set of five objects arranged in a fixed order, and the statements in each paragraph are logically consistent. After reading the paragraph, select the best option that describes the arrangement of objects",
        "Given the following text describing the correct order of five objects, select the option from (A, B, C, D or E) that is consistent with the text.",
        "The following text describes the arrangement order of five objects. Please read the text and choose the one from the options that matches the logic of the text description. Your answer should be \"A\", \"B\", \"C\", \"D\" or \"E\".",
        "Deduce the order of the five objects and select the logically consistent statement from the given choices.",
        "Please decide which option is correct based on the descriptions in the following article. The article describes the order of the 5 objects, please output the correct option as your answer.",
        "You are given one passage, which sequentially gives a series of propositions. You task is to answer a given question based on the passage and select the correct answer from A, B, C, D, E.",
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

    def unobserved1(self, item):
        input_template = "You are given a paragraph that describes five objects arranged in order. Please select the" \
                         " best answer from A, B, C, D, and E which the answer contains a statement that is logically" \
                         " consistent with the paragraph.\n\nParaphraph:{question}{options_}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved2(self, item):
        input_template = "The most logically-correct answer, given this paraphraph? {question}{options_}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved3(self, item):
        input_template = "You are taking an exam where you will be given a paragraph of text describing five different " \
                     "objects in a sequence that are arranged in a fixed order. To answer the question correctly, you " \
                     "must keep track of where each object is in the sequence and then select the multiple choice " \
                     "answer that best corresponds to the correct answer from (\"A\", \"B\", \"C\", \"D\", \"E\"). Please " \
                     "carefully consider the information in the following paragraph and each of the answers before " \
                     "providing the right answer.\n\nParaphraph:{question}{options_}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved4(self, item):
        input_template = "Each of the following paragraphs describes a set of five objects arranged in a fixed order, " \
                     "and the statements in each paragraph are logically consistent. After reading the paragraph, " \
                     "select the best option that describes the arrangement of objects:\n{question}{options_}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved5(self, item):
        input_template = "Input\n\t- paragraph: {question}{options_}\nOutput\t- Answer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved6(self, item):
        input_template = "Given the following text describing the correct order of five objects, select the option " \
                         "from (A, B, C, D or E) that is consistent with the text.\n\ntext: {question}{options_}\n\n" \
                         "answer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved7(self, item):
        input_template = "The following text describes the arrangement order of five objects. Please read the text and " \
                         "choose the one from the options that matches the logic of the text description. Your answer " \
                         "should be \"A\", \"B\", \"C\", \"D\" or \"E\".\nText: {question}{options_} Answer:"
        return self.unobserved_template_QA(item, input_template)

    def unobserved8(self, item):
        input_template = "Deduce the order of the five objects and select the logically consistent statement from the" \
                         " given choices. {question}{options_} Answer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved9(self, item):
        input_template = "Please decide which option is correct based on the descriptions in the following article. " \
                         "The article describes the order of the 5 objects, please output the correct option as your " \
                         "answer.Article: {question}{options_}\nAnswer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved10(self, item):
        input_template = "You are given one passage, which sequentially gives a series of propositions. You task is " \
                         "to answer a given question based on the passage and select the correct answer from A, B, " \
                         "C, D, E.\n\nThe passage is: {question}\n\nThe candidate answers are: {options_}\n\nYou: " \
                         "The answer is obvious, I choose "
        return self.unobserved_template_QA(item, input_template)


    def unobserved1_fs(self, item):
        input_template_prefix = "You are given a paragraph that describes five objects arranged in order. Please select the best " \
                     "answer from A, B, C, D, and E which the answer contains a statement that is logically consistent" \
                     " with the paragraph.\n\n"

        example_template = "Paragraph: {question}{options_}\nAnswer: {answer}\n"

        input_template = "Paragraph: {question}{options_}\nAnswer: "
        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved2_fs(self, item):
        input_template_prefix = ""

        example_template = "The most logically-correct answer, given this paraphraph? {question}{options_}\n{answer}"

        input_template = "The most logically-correct answer, given this paraphraph? {question}{options_}"
        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved3_fs(self, item):
        input_template_prefix = "You are taking an exam where you will be given a paragraph of text describing five different " \
                     "objects in a sequence that are arranged in a fixed order. To answer the question correctly, you " \
                     "must keep track of where each object is in the sequence and then select the multiple choice " \
                     "answer that best corresponds to the correct answer from (\"A\", \"B\", \"C\", \"D\", \"E\"). Please " \
                     "carefully consider the information in the following paragraph and each of the answers before " \
                     "providing the right answer.\n\n"

        example_template = "Paragraph: {question}{options_}\nAnswer: {answer}\n"

        input_template = "Paragraph: {question}{options_}\nAnswer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved4_fs(self, item):
        input_template_prefix = "In this exercise, you will be given paragraphs that describe a set of five objects arranged in " \
                     "a fixed order. Each paragraph has a set of statements that are logically consistent within it. " \
                     "You will be asked to select the statement that is consistent with the description in the " \
                     "paragraph.\n"

        example_template = "Example {id}: {question} {options_} Answer: {answer}\n"

        input_template = "Question: {question} {options_} Answer: "
        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved5_fs(self, item):
        input_template_prefix = "Input:"

        example_template = "Here's another example:\n\t- paragraph: {question} {options_}\n\t- Answer: {answer}\n"

        input_template = "Now you do this:\n\t- paragraph: {question} {options_}\nOutput\n\t- Answer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved6_fs(self, item):
        input_template_prefix = "Given the following text describing the correct order of five objects, select the " \
                                "option from (A, B, C, D or E) that is consistent with the text.\n\n"

        example_template = "text: {question}{options_}\n\nanswer: {answer}\n"

        input_template = "text: {question}{options_}\n\nanswer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved7_fs(self, item):
        input_template_prefix = "The following texts each describe the arrangement order of five objects. Please read the texts," \
                     " for each text, choose the one from the corresponding options that matches the logic of the text" \
                     " description. Your answer should be \"A\", \"B\", \"C\", \"D\" or \"E\".\n"

        example_template = "Text: {question}{options_}\nAnswer: {answer}\n"

        input_template = "Text: {question}{options_}\nAnswer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved8_fs(self, item):
        input_template_prefix = "Deduce the order of the five objects and select the logically consistent statement " \
                                "from the given choices.\n"

        example_template = "{question}{options_} Answer: {answer}"

        input_template = "{question}{options_} Answer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved9_fs(self, item):
        input_template_prefix = "Please decide which option is correct based on the descriptions in the following " \
                                "article. The article describes the order of the 5 objects, please output the " \
                                "correct option as your answer.\n\nHere are some examples:\n\n"

        example_template = "Article: {question}{options_}\nAnswer: {answer}\n\n"

        input_template = "Now, you should answer the following question:\n\n{question}{options_} Answer: "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved10_fs(self, item):
        input_template_prefix = "The following practice aims to make logical deduction based on a paragraph with " \
                                "multiple statements. You need to combine all the statements to get the correct " \
                                "answer. The answer is selected among several candidates.\n\n"

        example_template = "Passage {id}: {question}\nSelect Among:\n{options_}\nThe answer is {answer}\n\n"

        input_template = "Passage: {question}\nSelect Among:\n{options_}\nThe answer is "

        return self.unobserved_template_QA_few_shot(item, input_template_prefix, input_template, example_template)
