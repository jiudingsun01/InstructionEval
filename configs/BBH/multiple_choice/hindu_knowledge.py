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
        "Answer questions about Hindu mythology.",
        "Please select the best matched answer for the given question from the choices list below based on Hindu mythology. Please respond with the choice only, without any other words.",
        "Solve question in the Hindu mythology area, output the best option for the question from \"A\", \"B\", \"C\", \"D\"",
        "In this task, you have to select the option that best answers the question given your knowledge about Hindu mythology.",
        "Answer the following question based on hindu mythology with the most accurate choice",
        "Choose the best option for the following question in Hindu Mythology",
        "Which of the options A, B, C, D is the correct one?",
        "You will be given a series of questions regarding Hindu knowledge. For each question, select among the multiple choice answers (A, B, C, D) and provide an explanation, where applicable."
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
        try:
            A, B, C, D = options
        except ValueError:
            print(options)
            raise
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
            example_question, example_options, example_answer = example["question"], example["options"], example["answer"]
            A, B, C, D = example_options
            example_choice = ["A", "B", "C", "D"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, answer=example_choice)

        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved1(self, item):
        input_template = "Please select the best matched answer for the given question from the choices list below" \
                         " based on Hindu mythology. Question: {question} Choices: A. {choiceA} B. {choiceB} C. " \
                         "{choiceC} D. {choiceD}.\nPlease respond with the choice only, without any other words."
        return self.unobserved_template_QA(item, input_template)

    def unobserved2(self, item):
        input_template = "Solve question in the Hindu mythology area, output the best option for the question " \
                         "from \"A\", \"B\", \"C\", \"D\": Question: {question} Options: A: {choiceA} B: " \
                         "{choiceB} C: {choiceC} D: {choiceD} Answer:"
        return self.unobserved_template_QA(item, input_template)

    def unobserved3(self, item):
        input_template = "Question: {question}\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nHindu knowledge" \
                         "expert: This is easy, the answer is "
        return self.unobserved_template_QA(item, input_template)

    def unobserved4(self, item):
        input_template = "In this task, you have to select the option that best answers the question given your " \
                         "knowledge about Hindu mythology.\nQuestion: {question}\nA.  {choiceA} B.  {choiceB} C.  " \
                         "{choiceC} D.  {choiceD}\nAnswer: among A, B, C, and D, the best choice is "
        return self.unobserved_template_QA(item, input_template)

    def unobserved5(self, item):
        input_template = "Answer the following question based on hindu mythology with the most accurate choice\n" \
                         "{question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved6(self, item):
        input_template = "{question}\n\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nWith your expertise in" \
                         "hindu mythology, provide the correct answer: "
        
        return self.unobserved_template_QA(item, input_template)

    def unobserved7(self, item):
        input_template = "Input:\n\t- Question: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n\t- C: {choiceC}" \
                         "\n\t- D: {choiceD}\nOutput\n\t- Answer: "
        return self.unobserved_template_QA(item, input_template)

    def unobserved8(self, item):
        input_template = "Choose the best option for the following question in Hindu Mythology {question}" \
                         " A.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD}"
        return self.unobserved_template_QA(item, input_template)

    def unobserved9(self, item):
        input_template = "{question}\nA.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD}\nWhich of the options" \
                         " A, B, C, D is the correct one? It is"
        return self.unobserved_template_QA(item, input_template)

    def unobserved10(self, item):
        input_template = "You will be given a series of questions regarding Hindu knowledge. For each question, select " \
                         "among the multiple choice answers (A, B, C, D) and provide an explanation, " \
                         "where applicable.\n\nQuestion: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}" \
                         "\nD: {choiceD}\n\nAnswer: "
        return self.unobserved_template_QA(item, input_template)
    
    def unobserved1_fs(self, item):
        input_template_prefix = "Please select the best matched answer for the given question from the choices list below based on Hindu mythology.\n"

        example_template = "Question: {question} Choices: A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD}. {answer}\n"

        input_template = "Question: {question} Choices: A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD}. "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = "Solve question in the Hindu mythology area, output the best option for the question from \"A\", \"B\", \"C\", \"D\":"

        example_template = "\nQuestion: {question} Options: A: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} Answer: {answer}"

        input_template = "\nQuestion: {question} Options: A: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = ""

        example_template = "Question: {question}\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nHindu knowledge expert: This is easy, the answer is {answer}\n"

        input_template = "Question: {question}\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nHindu knowledge expert: This is easy, the answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "In this task, you have to select the option that best answers the question given your knowledge about Hindu mythology."

        example_template = "\nQuestion: {question}\nA.  {choiceA} B.  {choiceB} C. {choiceC} D.  {choiceD}\nAnswer: among A, B, C, and D, the best choice is {answer}"

        input_template = "\nQuestion: {question}\nA.  {choiceA} B.  {choiceB} C. {choiceC} D.  {choiceD}\nAnswer: among A, B, C, and D, the best choice is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = "In this task, you have to select the option that best answers the question given your knowledge about Hindu mythology."

        example_template = "\nQuestion: {question}\nA.  {choiceA} B.  {choiceB} C. {choiceC} D. {choiceD}\nAnswer: among A, B, C, and D, the best choice is {answer}"

        input_template = "\nQuestion: {question}\nA.  {choiceA} B.  {choiceB} C. {choiceC} D. {choiceD}\nAnswer: among A, B, C, and D, the best choice is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\n\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nWith your expertise in" \
                         "hindu mythology, provide the correct answer: {answer}\n"

        input_template = "{question}\n\nA: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD}\nWith your expertise in" \
                         "hindu mythology, provide the correct answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = ""

        example_template = "Input:\n\t- Question: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n\t- C: {choiceC}" \
                         "\n\t- D: {choiceD}\nOutput\n\t- Answer: {answer}\n\n"

        input_template = "Input:\n\t- Question: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n\t- C: {choiceC}" \
                         "\n\t- D: {choiceD}\nOutput\n\t- Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "Choose the best option for the following question in Hindu Mythology. "

        example_template = "\n{question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} Answer: {answer}"

        input_template = "\n{question} A. {choiceA} B. {choiceB} C. {choiceC} D. {choiceD} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\nA.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD}\nWhich of the options A, B, C, D is the correct one? It is {answer}\n\n"

        input_template = "{question}\nA.  {choiceA} B.  {choiceB} C.  {choiceC} D.  {choiceD}\nWhich of the options A, B, C, D is the correct one? It is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = "You will be given a series of questions regarding Hindu knowledge. For each question, select " \
                                "among the multiple choice answers (A, B, C, D) and provide an explanation, where applicable."

        example_template = "\n\nQuestion: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nAnswer: {answer}"

        input_template = "\n\nQuestion: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
