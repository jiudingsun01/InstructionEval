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
    preprocessor = CrashBlossomPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class CrashBlossomPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(CrashBlossomPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
        self.instr2preprocessor["BBL/Correct/6"] = self.correct_6
        self.instr2preprocessor["BBL/Correct/7"] = self.correct_7
        self.instr2preprocessor["BBL/Correct/8"] = self.correct_8
        self.instr2preprocessor["BBL/Negation/1"] = self.negation_1
        self.instr2preprocessor["BBL/Negation/2"] = self.negation_2
        self.instr2preprocessor["BBL/Negation/3"] = self.negation_3
        self.instr2preprocessor["BBL/Negation/4"] = self.negation_4
        self.instr2preprocessor["BBL/Negation/5"] = self.negation_5
        self.instr2preprocessor["BBL/Negation/6"] = self.negation_6
        self.instr2preprocessor["BBL/Negation/7"] = self.negation_7
        self.instr2preprocessor["BBL/Negation/8"] = self.negation_8

    def unobserved1(self, item):
        question, sentence = item["question"].split("Sentence: ")
        word = question.split(" ")[-2]
        options = item["options"]
        options_ = ""
        for i, option in enumerate(options):
            if i != len(options) - 1:
                options_ += f"\'{option}\', "
            else:
                options_ += f"or \'{option}\'."
        input_text = "Classify the part of speech of the word \"{word}\" in the following sentence: {sentence}. The options are: {options_}\nAnswer: ".format(word=word, sentence=sentence, options_=options_)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved2(self, item):
        question, sentence = item["question"].split("Sentence: ")
        word = question.split(" ")[-2]
        options = item["options"]
        options_ = ""
        for i, option in enumerate(options):
            if i != len(options) - 1:
                options_ += f"\'{option}\', "
            else:
                options_ += f"or \'{option}\'."
        input_text = "Sentence: {sentence}\nIdentify the part of speech of {word} in the sentence. Choise your answer from {options_} and output the best choice.".format(word=word, sentence=sentence, options_=options_)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved3(self, item):
        question, sentence = item["question"].split("Sentence: ")
        word = question.split(" ")[-2]
        options = item["options"]
        options_ = ""
        for i, option in enumerate(options):
            if i != len(options) - 1:
                options_ += f"\'{option}\', "
            else:
                options_ += f"or \'{option}\'."
        input_text = "What is the part of speech of the word \'{word}\' in \'{sentence}\'. You may only choose from the following options: {options_}. Your answer is: ".format(word=word, sentence=sentence, options_=options_)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved4(self, item):
        question, sentence = item["question"].split("Sentence: ")
        word = question.split(" ")[-2]
        options = item["options"]
        options_ = ""
        for i, option in enumerate(options):
            if i != len(options) - 1:
                options_ += f"\'{option}\', "
            else:
                options_ += f"or \'{option}\'."
        input_text = "Given a sentence and a word contained in the sentence, output the part of speech of the word.\nWord: {word}\nSentence: {sentence}\nOptions: {options_}\n\nAnswer: ".format(word=word, sentence=sentence, options_=options_)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved5(self, item):
        question, sentence = item["question"].split("Sentence: ")
        word = question.split(" ")[-2]
        options = item["options"]
        options_ = ""
        for i, option in enumerate(options):
            if i != len(options) - 1:
                options_ += f"\'{option}\', "
            else:
                options_ += f"or \'{option}\'."
        input_text = "Identify the part of speech of the word. Question: which one of {options_} is \'{word}\' in \'{sentence}\'? Answer: ".format(word=word, sentence=sentence, options_=options_)
        output_text = item["answer"]
        label_space = item["options"]
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
        return self.niv2_template(item, self.niv2_143_odd_man_out_classification, self.niv2_zs_template_10)
    
    def incorrect_2(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_10)
    
    def incorrect_3(self, item):
        self.set_style(("text", 1), 1)
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
        return self.niv2_template(item, self.niv2_1422_mathqa_physics, self.niv2_zs_template_10)
    
    def correct_1(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_1(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_2(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_2(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_3(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_3(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict

    def correct_4(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_4(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_5(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_5(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_6(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_6(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_7(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_7(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_8(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_8(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_1(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_1_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_2(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_2_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_3(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_3_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_4(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_4_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_5(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_5_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict

    def negation_6(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_6_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_7(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_7_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_8(self, item):
        self.set_style(item_sytle=("text", 1), option_style=1)
        question, context = item["question"].split("Sentence: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_8_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    


    

    