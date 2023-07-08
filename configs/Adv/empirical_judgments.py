import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
from configs.utils import load_BBL_file


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = EmpericalCombinationsPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class EmpericalCombinationsPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(EmpericalCombinationsPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["BBL/Default/1"] = self.default_Classification
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


    def unobserved1(self, item):
        input_text = "Two events are described in the following sentence: {question}\nClassify the relation between the events into one of \'causal\', \'correlative\', or \'neutral\'.".format(question=item["question"])
        output_text = item["answer"]
        label_space = ["causal", "correlative", "neutral"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved2(self, item):
        input_text = "Causal relation: two events have causal relation if one causes the other to happen." \
                     "\nCorrelative relation: two events have correlative relation if there is no explicity causal relation but they are correlated." \
                     "\nNeutral relation: two events have no obvious correlation.\n\n{question} Do the events described in the sentence have causal, correlative, or neutral relation?".format(question=item["question"])
        output_text = item["answer"]
        label_space = ["causal", "correlative", "neutral"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved3(self, item):
        input_text = "Sentence: {question} Make a judgment about the relation of the events in the sentence. The possible relations are: \"causal\", \"correlative\", \"neutral\"".format(question=item["question"])
        output_text = item["answer"]
        label_space = ["causal", "correlative", "neutral"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved4(self, item):
        input_text = "What is the relation between the events: {question}\nClassify it into \"causal\", \"correlative\", \"neutral\"".format(question=item["question"])
        output_text = item["answer"]
        label_space = ["causal", "correlative", "neutral"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved5(self, item):
        input_text = "You are given a sentence that describe two or more events. Now, classify the relation into one of \"causal\", \"correlative\", \"neutral\".\nSentence: \"{question}\"\nAnswer: ".format(question=item["question"])
        output_text = item["answer"]
        label_space = ["causal", "correlative", "neutral"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def no_instruction(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        labels = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                labels += "and {}.".format(option)
            else:
                labels += "{}, ".format(option)
        input_text = "The possible choices for the intents are: {labels}".format(labels=labels) + question + "Answer: "
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_1(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[0] + "\n" + question
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_2(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[1] + "\n" + question
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_3(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[2] + "\n" + question
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_4(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[3] + "\n" + question
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def random_instruction_5(self, item):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = self.RANDOM_INSTRUCTIONS[4] + "\n" + question
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def incorrect_1(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_143_odd_man_out_classification, self.niv2_zs_template_10)
    
    def incorrect_2(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_10)
    
    def incorrect_3(self, item):
        self.set_style(("text", 1), 1)
        item["question"] = item["question"].replace("Utterance", "")
        text, answer, options = item["question"], item["answer"], item["options"]
        intput_text, output_text, label_space = self.sentiment140_1(text, answer, options)
        return_dict = {"input_text": intput_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict

    def incorrect_4(self, item):
        self.set_style(("text", 1), 1)
        item["question"] = item["question"].replace("Utterance", "")
        text, answer, options = item["question"], item["answer"], item["options"]
        intput_text, output_text, label_space = self.sentiment140_6(text, answer, options)
        return_dict = {"input_text": intput_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_5(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_1422_mathqa_physics, self.niv2_zs_template_10)
    
    def correct_1(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_classification, self.niv2_zs_template_2)
    
    def correct_2(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_classification, self.niv2_zs_template_4)

    def correct_3(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_classification, self.niv2_zs_template_6)
    
    def correct_4(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_classification, self.niv2_zs_template_8)
    
    def correct_5(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_classification, self.niv2_zs_template_10)
    
    def negation_1(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_negation, self.niv2_zs_template_2)
    
    def negation_2(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_negation, self.niv2_zs_template_4)

    def negation_3(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_negation, self.niv2_zs_template_6)
    
    def negation_4(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_negation, self.niv2_zs_template_8)
    
    def negation_5(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_163_openpi_negation, self.niv2_zs_template_10)
    


    

    