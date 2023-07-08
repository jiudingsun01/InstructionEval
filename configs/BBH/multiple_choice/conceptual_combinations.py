import os.path
import json
import random
from datasets import Dataset
from configs.utils import load_BBL_file
from configs.preprocessor import Preprocessor
import multiprocessing as mp

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
        new_items, new_examples, shot_count = load_BBL_file(os.path.join(input_dir, folder, "task.json"),
                                                            fewshot_examples[folder], shot_count)
        items.extend(new_items)
        examples.extend(new_examples)

    rand = random.Random(42)
    for item in items:
        rand.shuffle(item["options"])

    test_set = Dataset.from_list(items)
    preprocessor = ConceptualCombinationPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set


def load_data_testing(input_dir, instruction, args):
    items, examples = [], []

    folders = [f for f in os.listdir(input_dir) if "." not in f]
    assert len(folders) == len(fewshot_examples)
    for folder in folders:
        new_items, new_examples, shot_count = load_BBL_file(os.path.join(input_dir, folder, "task.json"),
                                                            fewshot_examples[folder], 0)
        items.extend(new_items)
        examples.extend(new_examples)
    
    if args.maximum_test_samples is not None:
        items = items = random.sample(items, min(args.maximum_test_samples, len(items)))
    
    rand = random.Random(42)
    for item in items:
        rand.shuffle(item["options"])
    test_set = Dataset.from_list(items)
    preprocessor = ConceptualCombinationPreprocessor(instruction, examples, True, input_dir)

    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set


class ConceptualCombinationPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Understand conceptual combinations in appropriate contexts",
        "Use your common sense to output one of the letter \"A\", \"B\", \"C\", or \"D\" to indicate your answer.",
        "You are given a concept or a factual context. Answer the multiple choice question based on the context by choosing from the choices provided.",
        "You are a linguistic expert that knows most of the concepts and combinations of words. Now, answer the following question:",
        "Answer the question about concepts combination. Specifically, you need to take contradictions, emergent properties, fanciful fictional combinations, homonyms, invented words, and surprising uncommon combinations into consideration.",
        "The following is a multiple-choice question answering problem about conceptual meaning of words. You should choose the answer that best answer the question based on the context.",
        "Understand the context, and answer the following question:",
        "The task is to answer the linguistic question about concepts combination.",
        "What is the correct answer to this conceptual combination question?"
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(ConceptualCombinationPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
    
    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        options, answer = item["options"], item["answer"]
        context, question = item["question"].split("Question: ")
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_options, example_answer = example["options"], example["answer"]
            example_context, example_question = example["question"].split("Question: ")
            A, B, C, D = example_options
            example_choice = ["A", "B", "C", "D"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, context=example_context, question=example_question,  choiceA=A, choiceB=B, choiceC=C, choiceD=D, answer=example_choice)

        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text += input_template.format(context=context, question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D)
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
    
    def unobserved6(self, item):
        input_template = "The following is a multiple-choice question answering problem about conceptual meaning of words. " \
                         "You should choose the answer that best answer the question based on the context. {context} Question: {question}\n" \
                         "The options are:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved7(self, item):
        input_template = "Context: {context}. Understand the context, and answer the following question: {question}Options:\n(A) {choiceA}\n(B) {choiceB}\n(C) {choiceC}\n(D) {choiceD}\nAnswer:"
        return self.unobserved_template(item, input_template) 
    
    def unobserved8(self, item):
        input_template = "Linguistic Professor: {context} {question}\nStudent: can you provide the options?\nLinguistic Professor: The choices are A) {choiceA} B) {choiceB} C) {choiceC} D) {choiceD}\nStudent: I got it. The answer is "
        return self.unobserved_template(item, input_template)
    
    def unobserved9(self, item):
        input_template = "The task is to answer the linguistic question about concepts combination. Context: {context}\n\nQuestion: {question}\n\nOptions:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\n\nAnswer:"
        return self.unobserved_template(item, input_template)
    
    def unobserved10(self, item):
        input_template = "{question}\nOptions: A. {choiceA}, B. {choiceB}, C. {choiceC}, or D. {choiceD}. What is the correct answer to this conceptual combination question? Based on the context \"{context}\", I think the most accurate answer is"
        return self.unobserved_template(item, input_template)
    
    def unobserved1_fs(self, item):
        input_template_prefix = ""

        example_template = "{context} Question: {question}\nThe options are the following:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: {answer}\n"
                         

        input_template = "{context} Question: {question}\nThe options are the following:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\n" \
                         "Use your common sense to output one of the letter \"A\", \"B\", \"C\", or \"D\" to indicate your answer. Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved2_fs(self, item):
        input_template_prefix = "You are given a concept or a factual context. Answer the multiple choice question based on the context by choosing from the choices provided."

        example_template = "\nContext: {context}\nQuestion: {question}\nChoices:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: {answer}"
                         

        input_template = "\nContext: {context}\nQuestion: {question}\nChoices:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "You are a linguistic expert that knows most of the concepts and combinations of words. Now, answer the following question:"

        example_template = "\n{context} Question: {question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nYour answer is: {answer}"
                         

        input_template = "\n{context} Question: {question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nYour answer is: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "Answer the question about concepts combination. Specifically, you need to take contradictions, emergent properties, fanciful fictional combinations, homonyms, invented words, and surprising uncommon combinations into consideration."

        example_template = "\n{context} Question: {question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nYour answer is: {answer}"
                         
        input_template = "\n{context} Question: {question} (A) {choiceA} (B) {choiceB} (C) {choiceC} (D) {choiceD}\nYour answer is: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = ""

        example_template = "Question: {question}\nThe options are: \nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nHere is a context to help you answer the question: {context}. Choose the best answer from \"A\", \"B\", \"C\", \"D\". Answer: {answer}\n"
                         
        input_template = "Question: {question}\nThe options are: \nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nHere is a context to help you answer the question: {context}. Choose the best answer from \"A\", \"B\", \"C\", \"D\". Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "The following is a multiple-choice question answering problem about conceptual meaning of words. " \
                         "You should choose the answer that best answer the question based on the context."

        example_template = "\n{context} Question: {question}\n" \
                         "The options are:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: {answer}"
                         
        input_template = "\n{context} Question: {question}\n" \
                         "The options are:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = "Understand the context, and answer the following question:\n"

        example_template = "\nContext: {context}. {question}Options:\n(A) {choiceA}\n(B) {choiceB}\n(C) {choiceC}\n(D) {choiceD}\nAnswer: {answer}"
                         
        input_template = "\nContext: {context}. {question}Options:\n(A) {choiceA}\n(B) {choiceB}\n(C) {choiceC}\n(D) {choiceD}\nAnswer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = ""

        example_template = "Linguistic Professor: {context} {question}\nStudent: can you provide the options?\nLinguistic Professor: The choices are A) {choiceA} B) {choiceB} C) {choiceC} D) {choiceD}\nStudent: I got it. The answer is {answer}\n"
                         
        input_template = "Linguistic Professor: Here's another question, {context} {question}\nStudent: can you provide the options?\nLinguistic Professor: The choices are A) {choiceA} B) {choiceB} C) {choiceC} D) {choiceD}\nStudent: I got it. The answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = "The task is to answer the linguistic question about concepts combination."

        example_template = "\nContext: {context}\n\nQuestion: {question}\n\nOptions:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\n\nAnswer: {answer}"
                         
        input_template = "\nContext: {context}\n\nQuestion: {question}\n\nOptions:\nA. {choiceA}\nB. {choiceB}\nC. {choiceC}\nD. {choiceD}\n\nAnswer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = ""

        example_template = "{question}\nOptions: A. {choiceA}, B. {choiceB}, C. {choiceC}, or D. {choiceD}. What is the correct answer to this conceptual combination question? Based on the context \"{context}\", I think the most accurate answer is {answer}\n"
                         
        input_template = "{question}\nOptions: A. {choiceA}, B. {choiceB}, C. {choiceC}, or D. {choiceD}. What is the correct answer to this conceptual combination question? Based on the context \"{context}\", I think the most accurate answer is"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)