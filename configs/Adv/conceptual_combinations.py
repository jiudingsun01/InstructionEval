import os.path
import json
from configs.preprocessor import Preprocessor
import multiprocessing as mp
from datasets import Dataset
from configs.utils import load_BBL_file


special_tokens = []

fewshot_examples = {
    "contradictions": [0],
    "emergent_properties": [],
    "fanciful_fictional_combinations": [],
    "homonyms": [],
    "invented_words": [],
    "surprising_uncommon_combinations": []
}


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):
    items, examples = [], []

    folders = [f for f in os.listdir(input_dir) if "." not in f]
    assert len(folders) == len(fewshot_examples)
    for folder in folders:
        new_items, new_examples, shot_count = load_BBL_file(os.path.join(input_dir, folder, "task.json"), fewshot_examples[folder], shot_count)
        items.extend(new_items)
        examples.extend(new_examples)

    test_set = Dataset.from_list(items)
    preprocessor = ConceptualCombinationPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set


class ConceptualCombinationPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(ConceptualCombinationPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
        self.instr2preprocessor["BBL/Incorrect/5"] = self.incorrect_2
        self.instr2preprocessor["BBL/Incorrect/6"] = self.incorrect_3
        self.instr2preprocessor["BBL/Incorrect/7"] = self.incorrect_4
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


    def unobserved_template(self, item, input_temptlate):
        options, answer = item["options"], item["answer"]
        context, question = item["question"].split("Question: ")
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text = input_temptlate.format(context=context, question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "{context} Question: {question}\nThe options are the following:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\n" \
                         "Use your common sense to output one of the letter \"A\", \"B\", \"C\", or \"D\" to indicate your answer. "
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "You are given a concept or a factual context. Answer the multiple choice question based on the context by choosing from the choices provided.\nContext: {context}\nQuestion: {question}\nChoices:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: " 
        return self.unobserved_template(item, input_template)

    def unobserved3(self, item):
        input_template = "You are a linguistic expert that knows most of the concepts and combinations of words. Now, answer the following question: " \
                         "{context} Question: {question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nYour answer is: "
        return self.unobserved_template(item, input_template)
    
    def unobserved4(self, item):
        input_template = "Answer the question about concepts combination. Specifically, you need to take contradictions, emergent properties, fanciful fictional combinations, homonyms, invented words, and surprising uncommon combinations into consideration. {context} Question: {question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nYour answer is: "
        return self.unobserved_template(item, input_template)
    
    def unobserved5(self, item):
        input_template = "Question: {question}\nThe options are: \nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nHere is a context to help you answer the question: {context}. Choose the best answer from \"A\", \"B\", \"C\", \"D\"."
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

    def correct_1(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_1(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_2(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_2(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_3(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_3(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_4(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_4(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_5(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_5(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_6(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_6(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_7(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_7(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def correct_8(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_8(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_1(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_1_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_2(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_2_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_3(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_3_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_4(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_4_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_5(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_5_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_6(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_6_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_7(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_7_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def negation_8(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        context, question = item["question"].split("Question: ")
        options, answer = item["options"], item["answer"]
        input_text, output_text, label_space = self.cosmos_qa_8_negated(context=context, question=question, answer=answer, options=options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_1(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        input_text, output_text, label_space = self.wsc273_2(context=item["question"], answer=item["answer"], options=item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_2(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        input_text, output_text, label_space = self.wsc273_9(context=item["question"], answer=item["answer"], options=item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_3(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        input_text, output_text, label_space = self.winogrande_3(context=item["question"], answer=item["answer"], options=item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_4(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        input_text, output_text, label_space = self.story_cloze_1(context=item["question"], answer=item["answer"], options=item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_5(self, item):
        self.set_style(item_sytle=("letter", 14), option_style=1)
        input_text, output_text, label_space = self.sentiment140_1(context=item["question"], answer=item["answer"], options=item["options"])
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_6(self, item):
        item["question"] = item["question"].replace("Question:", "")
        return self.niv2_template(item, self.niv2_1422_mathqa_physics, self.niv2_zs_template_10)
    
    def incorrect_7(self, item):
        item["question"] = item["question"].replace("Question:", "")
        return self.niv2_template(item, self.niv2_1297_qasc_question_answering, self.niv2_zs_template_10)

    
    


    

    