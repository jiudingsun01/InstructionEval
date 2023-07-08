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
    preprocessor = LanguageIdentificationPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class LanguageIdentificationPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(LanguageIdentificationPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
        A, B, C, D, E, F, G, H, I, J, K = options
        choice = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][options.index(answer)]
        input_text = input_temptlate.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E,
                                            choiceF=F, choiceG=G, choiceH=H, choiceI=I, choiceJ=J, choiceK=K)
        output_text = choice
        label_space = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "Input\n\t- sentence: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n" \
                         "\t- C: {choiceC}\n\t- D: {choiceD}\n\t- E: {choiceE}\n\t- F: {choiceF}\n\t- G: {choiceG}" \
                         "\n\t- H: {choiceH}\n\t- I: {choiceI}\n\t- J: {choiceJ}\n\t- K: {choiceK}\nOutput\n\t- Answer:"
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        options_ = ", ".join(item["options"])
        input_template = "Please give the language used in the following sentence. {question}. Each sentence will have some " \
                         "options, please output the corresponding option to represent the corresponding answer. The options are: {options_}"
        input_text = input_template.format(question=item["question"], options_=options_)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved3(self, item):
        input_template = "Given the sentence: {question}, select the correct language among the choices A. {choiceA} " \
                         "B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} F. {choiceF} G. {choiceG} H. {choiceH} " \
                         "I. {choiceI} J. {choiceJ} K. {choiceK}\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- " \
                         "D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}\n- I: " \
                         "{choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nLanguage:"
        return self.unobserved_template(item, input_template)

    def unobserved4(self, item):
        input_template = "{question}\n\nThis is a sentence written in one of {choiceA}, {choiceB}, {choiceC}, {choiceD}," \
                         " {choiceE}, {choiceF}, {choiceG}, {choiceH}, {choiceI}, {choiceJ}, {choiceK}. According to " \
                         "the words and the linguistic structure, I can tell that the language is:"
        return_dict = self.unobserved_template(item, input_template)
        return_dict["output_text"] = item["answer"]
        return_dict["label_space"] = item["options"]
        return return_dict
    
    def unobserved5(self, item):
        input_template = "{question}. A few options are provieded for the language of this sentence:\n\nA. {choiceA} " \
                         "B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} F. {choiceF} G. {choiceG} H. {choiceH} " \
                         "I. {choiceI} J. {choiceJ} K. {choiceK}\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- " \
                         "D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}\n- I: " \
                         "{choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nI know the language it belongs to is:"
        return self.unobserved_template(item, input_template)
    
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
        return self.niv2_template(item, self.niv2_1322_government_type_classification, self.niv2_zs_template_10)
    
    def incorrect_3(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_1422_mathqa_physics, self.niv2_zs_template_10)

    def incorrect_4(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_153_hatexplain_classification, self.niv2_zs_template_10)

    def correct_1(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_2)
    
    def correct_2(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_4)

    def correct_3(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_6)
    
    def correct_4(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_8)
    
    def correct_5(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_10)
    
    def negation_1(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_negation, self.niv2_zs_template_2)
    
    def negation_2(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_negation, self.niv2_zs_template_4)

    def negation_3(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_negation, self.niv2_zs_template_6)
    
    def negation_4(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_negation, self.niv2_zs_template_8)
    
    def negation_5(self, item):
        item["question"] = item["question"].replace("Utterance", "")
        return self.niv2_template(item, self.niv2_137_newscomm_negation, self.niv2_zs_template_10)
    
    


    

    