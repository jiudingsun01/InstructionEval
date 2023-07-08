import os.path
import json
import random
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file, ClassificationGivenLabel


special_tokens = []


fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = LanguageIdentificationPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

def load_data_testing(input_dir, instruction, args):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = LanguageIdentificationPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class LanguageIdentificationPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Given a sentence, select the correct language among the choices",
        "dentify the correct language of the given sentence. Please choose the best answer from A, B, C, D, E, F, G, H, I, J, and K.",
        "What language is the language stated above?",
        "You are taking a test that requires you to identify the language a given sentence is written in. To help narrow down your choices, we’ve made this a multiple choice question. After carefully examining the sentence and each answer below, please select the correct language of the sentence from one of \"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\", \"I\", \"J\", or \"K\"",
        "Please select the language that correctly corresponds to the provided sentence from the following options",
        "Given the following text, identify the correct language by selecting one of the options in the list (A, B, C, D, E, F, G, H, I, J, K)",
        "Please read the following sentence, then choose from the options which language you think it most likely came from. Your answer should be \"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\", \"I\", \"J\", or \"K\"",
        "Please give the language used in the following sentence. Each sentence will give five options, please output the corresponding option (i.e. A, B, C, D, E, F, G, H, I, J, or K) to represent the corresponding answer.",
        "Given the sentence, select the correct language among the choices",
        "According to the words and the linguistic structure, can you tell what the language is?"
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(LanguageIdentificationPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["BBL/Default/1"] = self.default_Classification
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
        self.instr2preprocessor["FLAN/Classification/8"] = lambda item: self.flan_classification_8(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/9"] = lambda item: self.flan_classification_9(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/10"] = lambda item: self.flan_classification_10(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/11"] = lambda item: self.flan_classification_11(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/12"] = lambda item: self.flan_classification_12(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/13"] = lambda item: self.flan_classification_13(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/14"] = lambda item: self.flan_classification_14(self.map_to_flan_cosmosqa(item))
        self.instr2preprocessor["FLAN/Classification/15"] = lambda item: self.flan_classification_15(self.map_to_flan_cosmosqa(item))


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

        self.instr2preprocessor_fs["BBL/Default/1"] = self.default_Classification_fs
        self.instr2preprocessor_fs["BBL/Unobserved/1"] = self.unobserved1_fs
        self.instr2preprocessor_fs["BBL/Unobserved/2"] = self.unobserved2_fs
        self.instr2preprocessor_fs["BBL/Unobserved/3"] = self.unobserved3_fs
        self.instr2preprocessor_fs["BBL/Unobserved/4"] = self.unobserved4_fs
        self.instr2preprocessor_fs["BBL/Unobserved/5"] = self.unobserved5_fs
        self.instr2preprocessor_fs["BBL/Unobserved/6"] = self.unobserved6_fs
        self.instr2preprocessor_fs["BBL/Unobserved/7"] = self.unobserved7_fs
        self.instr2preprocessor_fs["BBL/Unobserved/8"] = self.unobserved8_fs
        self.instr2preprocessor_fs["BBL/Unobserved/9"] = self.unobserved9_fs
        self.instr2preprocessor_fs["BBL/Unobserved/10"] = self.unobserved9_fs

        self.instr2preprocessor_fs["FLAN/Classification/8"] = lambda item: self.flan_classification_8_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/9"] = lambda item: self.flan_classification_9_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/10"] = lambda item: self.flan_classification_10_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/11"] = lambda item: self.flan_classification_11_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/12"] = lambda item: self.flan_classification_12_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/13"] = lambda item: self.flan_classification_13_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/14"] = lambda item: self.flan_classification_14_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/15"] = lambda item: self.flan_classification_15_fs(self.map_to_flan_cosmosqa_few_shot(item))

    def map_to_flan_cosmosqa(self, item):
        context = item["question"]
        question = "What is the language of the given context?"
        answer = item["answer"]
        options = item["options"]
        new_item = {
            "context": context,
            "question": question,
            "options": options,
            "answer": answer,
        }
        return new_item
    
    def map_to_flan_cosmosqa_few_shot(self, item):
        context = item["question"]
        question = "What is the language of the given context?"
        answer = item["answer"]
        options = item["options"]
        new_item = {
            "context": context,
            "question": question,
            "options": options,
            "answer": answer,
        }
        if "context" not in self.examples[0].keys():
            for example in self.examples:
                context = example["question"]
                question = "What is the language of the given context?"
                example["context"] = context
                example["question"] = question
        return new_item

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
    
    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_question, example_options, example_answer = example["question"], example["options"], example["answer"]
            A, B, C, D, E, F, G, H, I, J, K = example_options
            example_choice = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E,
                                                  choiceF=F, choiceG=G, choiceH=H, choiceI=I, choiceJ=J, choiceK=K, answer=example_choice)

        A, B, C, D, E, F, G, H, I, J, K= options
        choice = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, choiceE=E,
                                                  choiceF=F, choiceG=G, choiceH=H, choiceI=I, choiceJ=J, choiceK=K)
        output_text = choice
        label_space = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "Identify the correct language of the given sentence. Please choose the best answer from A, " \
                         "B, C, D, E, F, G, H, I, J, and K.\n\nSentence: {question}\nA: {choiceA}\nB: {choiceB}\n" \
                         "C: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: {choiceH}\n" \
                         "I: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nAnswer: "
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "{question}\nWhat language is the language stated above? A: {choiceA} B: {choiceB} " \
                         "C: {choiceC} D: {choiceD} E: {choiceE} F: {choiceF} G: {choiceG} H: {choiceH} " \
                         "I: {choiceI} J: {choiceJ} K: {choiceK}"
        return self.unobserved_template(item, input_template)

    def unobserved3(self, item):
        input_template = "You are taking a test that requires you to identify the language a given sentence is " \
                         "written in. To help narrow down your choices, we’ve made this a multiple choice question. " \
                         "After carefully examining the sentence and each answer below, please select the correct " \
                         "language of the sentence from one of \"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\"," \
                         " \"I\", \"J\", or \"K\"\nSentence: {question}\n- A: {choiceA}\n- B: {choiceB}\n" \
                         "- C: {choiceC}\n- D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}" \
                         "\n- I: {choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved4(self, item):
        input_template = "Please select the language that correctly corresponds to the provided sentence from the " \
                         "following options:\nSentence: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\n" \
                         "C: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: {choiceH}\n" \
                         "I: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nYour answer: "
        return self.unobserved_template(item, input_template)

    def unobserved5(self, item):
        input_template = "Input\n\t- sentence: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n" \
                         "\t- C: {choiceC}\n\t- D: {choiceD}\n\t- E: {choiceE}\n\t- F: {choiceF}\n\t- G: {choiceG}" \
                         "\n\t- H: {choiceH}\n\t- I: {choiceI}\n\t- J: {choiceJ}\n\t- K: {choiceK}\nOutput\n\t- Answer:"
        return self.unobserved_template(item, input_template)

    def unobserved6(self, item):
        input_template = "Given the following text, identify the correct language by selecting one of the options " \
                         "in the list (A, B, C, D, E, F, G, H, I, J, K):\n\nText: {question}\n\nA: {choiceA}\nB: " \
                         "{choiceB}\nC: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: " \
                         "{choiceH}\nI: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\n\nAnswer: "
        return self.unobserved_template(item, input_template)

    def unobserved7(self, item):
        input_template = "Please read the following sentence, then choose from the options which language you think " \
                         "it most likely came from. Your answer should be \"A\", \"B\", \"C\", \"D\", \"E\", \"F\", " \
                         "\"G\", \"H\", \"I\", \"J\", or \"K\"\nSentence: {question}\nOptions:\nA: {choiceA}\n" \
                         "B: {choiceB}\nC: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}" \
                         "\nH: {choiceH}\nI: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved8(self, item):
        input_template = "Please give the language used in the following sentence. Each sentence will give five " \
                         "options, please output the corresponding option (i.e. A, B, C, D, E, F, G, H, I, J, or K) " \
                         "to represent the corresponding answer.\n\nSentence: {question}\nOptions:"
        return self.unobserved_template(item, input_template)

    def unobserved9(self, item):
        input_template = "Given the sentence: {question}, select the correct language among the choices A. {choiceA} " \
                         "B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} F. {choiceF} G. {choiceG} H. {choiceH} " \
                         "I. {choiceI} J. {choiceJ} K. {choiceK}\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- " \
                         "D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}\n- I: " \
                         "{choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nLanguage:"
        return self.unobserved_template(item, input_template)

    def unobserved10(self, item):
        input_template = "{question}\n\nThis is a sentence written in one of {choiceA}, {choiceB}, {choiceC}, {choiceD}," \
                         " {choiceE}, {choiceF}, {choiceG}, {choiceH}, {choiceI}, {choiceJ}, {choiceK}. According to " \
                         "the words and the linguistic structure, I can tell that the language is:"
        return_dict = self.unobserved_template(item, input_template)
        return_dict["output_text"] = item["answer"]
        return_dict["label_space"] = item["options"]
        return return_dict
    
    def unobserved1_fs(self, item):
        input_template_prefix = "Identify the correct language of the given sentence. Please choose the best answer from A, B, C, D, E, F, G, H, I, J, and K."

        example_template = "\n\nSentence: {question}\nA: {choiceA}\nB: {choiceB}\n" \
                         "C: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: {choiceH}\n" \
                         "I: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nAnswer: {answer}"

        input_template = "\n\nSentence: {question}\nA: {choiceA}\nB: {choiceB}\n" \
                         "C: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: {choiceH}\n" \
                         "I: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = ""
        example_template = "{question}\nWhat language is the language stated above? A: {choiceA} B: {choiceB} " \
                         "C: {choiceC} D: {choiceD} E: {choiceE} F: {choiceF} G: {choiceG} H: {choiceH} " \
                         "I: {choiceI} J: {choiceJ} K: {choiceK} Answer: {answer}\n"

        input_template = "{question}\nWhat language is the language stated above? A: {choiceA} B: {choiceB} " \
                         "C: {choiceC} D: {choiceD} E: {choiceE} F: {choiceF} G: {choiceG} H: {choiceH} " \
                         "I: {choiceI} J: {choiceJ} K: {choiceK} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "You are taking a test that requires you to identify the language a given sentence is " \
                         "written in. To help narrow down your choices, we’ve made this a multiple choice question. " \
                         "After carefully examining the sentence and each answer below, please select the correct " \
                         "language of the sentence from one of \"A\", \"B\", \"C\", \"D\", \"E\", \"F\", \"G\", \"H\"," \
                         " \"I\", \"J\", or \"K\"\n"
        
        example_template = "Sentence: {question}\n- A: {choiceA}\n- B: {choiceB}\n" \
                         "- C: {choiceC}\n- D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}" \
                         "\n- I: {choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nAnswer: {answer}\n"

        input_template = "Sentence: {question}\n- A: {choiceA}\n- B: {choiceB}\n" \
                         "- C: {choiceC}\n- D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}" \
                         "\n- I: {choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nAnswer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "Please select the language that correctly corresponds to the provided sentence from the " \
                                "following options:"
        
        example_template = "\nSentence: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\n" \
                         "C: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: {choiceH}\n" \
                         "I: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nYour answer: {answer}"

        input_template = "\nSentence: {question}\nOptions:\nA: {choiceA}\nB: {choiceB}\n" \
                         "C: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: {choiceH}\n" \
                         "I: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nYour answer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved5_fs(self, item):
        input_template_prefix = ""
        
        example_template = "Input\n\t- sentence: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n" \
                         "\t- C: {choiceC}\n\t- D: {choiceD}\n\t- E: {choiceE}\n\t- F: {choiceF}\n\t- G: {choiceG}" \
                         "\n\t- H: {choiceH}\n\t- I: {choiceI}\n\t- J: {choiceJ}\n\t- K: {choiceK}\nOutput\n\t- Answer: {answer}\n"

        input_template = "Input\n\t- sentence: {question}\n\t- A: {choiceA}\n\t- B: {choiceB}\n" \
                         "\t- C: {choiceC}\n\t- D: {choiceD}\n\t- E: {choiceE}\n\t- F: {choiceF}\n\t- G: {choiceG}" \
                         "\n\t- H: {choiceH}\n\t- I: {choiceI}\n\t- J: {choiceJ}\n\t- K: {choiceK}\nOutput\n\t- Answer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "Given the following text, identify the correct language by selecting one of the options " \
                                "in the list (A, B, C, D, E, F, G, H, I, J, K):"
        
        example_template = "\n\nText: {question}\n\nA: {choiceA}\nB: " \
                         "{choiceB}\nC: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: " \
                         "{choiceH}\nI: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\n\nAnswer:  {answer}"

        input_template = "\n\nText: {question}\n\nA: {choiceA}\nB: " \
                         "{choiceB}\nC: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}\nH: " \
                         "{choiceH}\nI: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = "Please read the following sentence, then choose from the options which language you think " \
                                "it most likely came from. Your answer should be \"A\", \"B\", \"C\", \"D\", \"E\", \"F\", " \
                                "\"G\", \"H\", \"I\", \"J\", or \"K\""
        
        example_template = "\nSentence: {question}\nOptions:\nA: {choiceA}\n" \
                         "B: {choiceB}\nC: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}" \
                         "\nH: {choiceH}\nI: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nAnswer: {answer}"

        input_template = "\nSentence: {question}\nOptions:\nA: {choiceA}\n" \
                         "B: {choiceB}\nC: {choiceC}\nD: {choiceD}\nE: {choiceE}\nF: {choiceF}\nG: {choiceG}" \
                         "\nH: {choiceH}\nI: {choiceI}\nJ: {choiceJ}\nK: {choiceK}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "Please give the language used in the following sentence. Each sentence will give five " \
                                "options, please output the corresponding option (i.e. A, B, C, D, E, F, G, H, I, J, or K) " \
                                "to represent the corresponding answer."
        
        example_template = "\n\nSentence: {question}\nOptions: {answer}"
        input_template = "\n\nSentence: {question}\nOptions: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = ""
        
        example_template = "Given the sentence: {question}, select the correct language among the choices A. {choiceA} " \
                         "B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} F. {choiceF} G. {choiceG} H. {choiceH} " \
                         "I. {choiceI} J. {choiceJ} K. {choiceK}\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- " \
                         "D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}\n- I: " \
                         "{choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nLanguage: {answer}\n"
        input_template = "Given the sentence: {question}, select the correct language among the choices A. {choiceA} " \
                         "B. {choiceB} C. {choiceC} D. {choiceD} E. {choiceE} F. {choiceF} G. {choiceG} H. {choiceH} " \
                         "I. {choiceI} J. {choiceJ} K. {choiceK}\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- " \
                         "D: {choiceD}\n- E: {choiceE}\n- F: {choiceF}\n- G: {choiceG}\n- H: {choiceH}\n- I: " \
                         "{choiceI}\n- J: {choiceJ}\n- K: {choiceK}\nLanguage: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    

    


