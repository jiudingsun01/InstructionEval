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
    preprocessor = IntentRecognitionPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class IntentRecognitionPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(IntentRecognitionPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
        self.instr2preprocessor["BBL/Incorrect/6"] = self.incorrect_6
        self.instr2preprocessor["BBL/Incorrect/7"] = self.incorrect_7
        self.instr2preprocessor["BBL/Incorrect/8"] = self.incorrect_8
        self.instr2preprocessor["BBL/Incorrect/9"] = self.incorrect_9
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


    def unobserved_template_classification(self, item, input_temptlate):
        question, options, answer = item["question"], item["options"], item["answer"]
        labels = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                labels += "and {}.".format(option)
            else:
                labels += "{}, ".format(option)

        input_text = input_temptlate.format(question=question, labels=labels)
        output_text = answer
        label_space = options
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_classification_few_shot(self, item, prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        labels = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                labels += "and {}.".format(option)
            else:
                labels += "{}, ".format(option)
        input_text = prefix.format(labels=labels)
        
        for i, example in enumerate(self.examples):
            example_question, _, example_answer = example["question"], example["options"], example["answer"]
            input_text += example_template.format(id=i+1, question=example_question, answer=example_answer)
        
        output_text = answer
        label_space = options
        input_text += input_template.format(question=question)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        item["question"] = item["question"].replace("Utterance: ", "")
        input_template = "You are given a set of intentions to predict: {labels}. Pick the most suitable one to describe the following utterance: {question}. Intention: "
        return self.unobserved_template_classification(item, input_template)

    def unobserved2(self, item):
        item["question"] = item["question"].replace("Utterance: ", "")
        input_template = "You are a dialogue assistance at recognizing and classifying user's intention.\nAlways respond with one of the \ " \
                          "options: [{labels}] to indicate the intention.\nUtterance: {question}\nIntention: "
        return self.unobserved_template_classification(item, input_template)

    def unobserved3(self, item):
        item["question"] = item["question"].replace("Utterance: ", "")
        input_template = "The tasks is to classify the intention of the utterance: '{question}' into one of the followings: {labels}. Your answer is: "
        return self.unobserved_template_classification(item, input_template)

    def unobserved4(self, item):
        input_template = "Given the label space: {labels}, classify the intention of the given utterance.\n{question}\nIntention: "
        return self.unobserved_template_classification(item, input_template)

    def unobserved5(self, item):
        input_template = "Output the intention of the utterance from the list: {labels}. Output the exact word or phrase. {question}"
        return self.unobserved_template_classification(item, input_template)
    
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
    
    def incorrect_6(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_564_discofuse_classification, self.niv2_zs_template_10)
    
    def incorrect_7(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_562_language_identification, self.niv2_zs_template_10)
    
    def incorrect_8(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_1193_course_classification, self.niv2_zs_template_10)
    
    def incorrect_9(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_1588_tecla_classification, self.niv2_zs_template_10)
    
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
    


    

    