import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
from configs.utils import load_BBL_file
from random import shuffle


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    for item in items:
        shuffle(item["options"])
        
    test_set = Dataset.from_list(items)
    preprocessor = LogicalSequencePreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class LogicalSequencePreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(LogicalSequencePreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["BBL/Default/1"] = self.default_QA
        self.instr2preprocessor["BBL/Unobserved/1"] = self.unobserved1
        self.instr2preprocessor["BBL/Unobserved/2"] = self.unobserved2
        self.instr2preprocessor["BBL/Unobserved/3"] = self.unobserved3
        self.instr2preprocessor["BBL/Unobserved/4"] = self.unobserved4
        self.instr2preprocessor["BBL/Unobserved/5"] = self.unobserved5
        self.instr2preprocessor["BBL/Random/1"] = self.random_instruction_1
        self.instr2preprocessor["BBL/Random/2"] = self.random_instruction_2
        self.instr2preprocessor["BBL/Random/3"] = self.random_instruction_3
        self.instr2preprocessor["BBL/Random/4"] = self.random_instruction_4
        self.instr2preprocessor["BBL/Random/5"] = self.random_instruction_5
        self.instr2preprocessor["BBL/Incorrect/1"] = self.incorrect_1
        self.instr2preprocessor["BBL/Incorrect/2"] = self.incorrect_2
        self.instr2preprocessor["BBL/Incorrect/3"] = self.incorrect_3
        self.instr2preprocessor["BBL/Incorrect/4"] = self.incorrect_4
        self.instr2preprocessor["BBL/Incorrect/5"] = self.incorrect_5
        self.instr2preprocessor["BBL/Correct/1"] = self.correct_1
        self.instr2preprocessor["BBL/Correct/2"] = self.correct_2
        self.instr2preprocessor["BBL/Correct/3"] = self.correct_3
        self.instr2preprocessor["BBL/Correct/4"] = self.correct_4
        self.instr2preprocessor["BBL/Correct/5"] = self.correct_5
        self.instr2preprocessor["BBL/Negation/1"] = self.negation_1
        self.instr2preprocessor["BBL/Negation/2"] = self.negation_2
        self.instr2preprocessor["BBL/Negation/3"] = self.negation_3
        self.instr2preprocessor["BBL/Negation/4"] = self.negation_4
        self.instr2preprocessor["BBL/Negation/5"] = self.negation_5


    def unobserved_template(self, item, input_temptlate):
        question, options, answer = item["question"], item["options"], item["answer"]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text = input_temptlate.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "Four items are naturally in a sequential or chronological order. Now, choose the correct order of these items from the following options:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}"
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "You are given four lists of the same objects in different orders. {question}\n\nLists:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}" 
        return self.unobserved_template(item, input_template)
    
    def unobserved3(self, item):
        input_template = "Choose the best answer that describes a sequence chronologically. Options: A: {choiceA}, B: {choiceB}, C: {choiceC}, D: {choiceD}\n\nAnswer: " 
        return self.unobserved_template(item, input_template)
    
    def unobserved4(self, item):
        input_template = "In this task, pick the list of the items that are chronologically orded most correctly. Choose from the following options and output the corresponding letter as one of \'A\', \'B\', \'C\', or \'D\'.\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}" 
        return self.unobserved_template(item, input_template)
    
    def unobserved5(self, item):
        input_template = "Question: {question}\nChoose the correct order from the lists: A. {choiceA}, B. {choiceB}, C. {choiceC}, D. {choiceD}. Answer: " 
        return self.unobserved_template(item, input_template)
    
    def random_instruction_1(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[0] 
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        for letter, text in zip(["A", "B", "C", "D"], [A, B, C, D]):
            input_text += "\n" + letter + ". " + text
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_2(self, item):
        _, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[1]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        for letter, text in zip(["A", "B", "C", "D"], [A, B, C, D]):
            input_text += "\n" + letter + ". " + text
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_3(self, item):
        _, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[2]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        for letter, text in zip(["A", "B", "C", "D"], [A, B, C, D]):
            input_text += "\n" + letter + ". " + text
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_4(self, item):
        _, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[3]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        for letter, text in zip(["A", "B", "C", "D"], [A, B, C, D]):
            input_text += "\n" + letter + ". " + text
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_5(self, item):
        _, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[4]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        for letter, text in zip(["A", "B", "C", "D"], [A, B, C, D]):
            input_text += "\n" + letter + ". " + text
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def incorrect_1(self, item):
        return self.niv2_template(item, self.niv2_1421_mathqa_general, self.niv2_zs_template_10)
    
    def incorrect_2(self, item):
        return self.niv2_template(item, self.niv2_1422_mathqa_physics, self.niv2_zs_template_10)
    
    def incorrect_3(self, item):
        input_text, output_text, label_space = self.wsc273_1(item["question"], item["answer"], item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def incorrect_4(self, item):
        input_text, output_text, label_space = self.trec_1(item["question"], item["answer"], item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def incorrect_5(self, item):
        input_text, output_text, label_space = self.piqa_1(item["question"], item["answer"], item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def correct_1(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_2)
    
    def correct_2(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_4)

    def correct_3(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_6)
    
    def correct_4(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_8)
    
    def correct_5(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_10)
    
    def negation_1(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa_negation, self.niv2_zs_template_2)
    
    def negation_2(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa_negation, self.niv2_zs_template_4)

    def negation_3(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa_negation, self.niv2_zs_template_6)
    
    def negation_4(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa_negation, self.niv2_zs_template_8)
    
    def negation_5(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa_negation, self.niv2_zs_template_10)
    
    


    

    