import os.path
import json
import random
from datasets import Dataset
from configs.utils import ClassificationMatchAccuracy, OptionMatchingAccuracy, ClassificationAccuracy, ClassificationGivenLabel, load_BBL_file
from configs.preprocessor import Preprocessor
import multiprocessing as mp

special_tokens = []

fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)
    test_set = Dataset.from_list(items)
    preprocessor = StrategyQAPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

def load_data_testing(input_dir, instruction, args):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = StrategyQAPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

class StrategyQAPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(StrategyQAPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)


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
        self.instr2preprocessor_fs["BBL/Unobserved/10"] = self.unobserved10_fs

        self.instr2preprocessor["NIV2/BC/1"] = lambda item: self.niv2_binary_classification_1(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/2"] = lambda item: self.niv2_binary_classification_2(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/3"] = lambda item: self.niv2_binary_classification_3(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/4"] = lambda item: self.niv2_binary_classification_4(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/5"] = lambda item: self.niv2_binary_classification_5(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/6"] = lambda item: self.niv2_binary_classification_6(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/7"] = lambda item: self.niv2_binary_classification_7(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/8"] = lambda item: self.niv2_binary_classification_8(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/9"] = lambda item: self.niv2_binary_classification_9(self.map_to_niv2_56(item))
        self.instr2preprocessor["NIV2/BC/10"] = lambda item: self.niv2_binary_classification_10(self.map_to_niv2_56(item))
        self.instr2preprocessor["FLAN/BC/1"] = lambda item: self.flan_binary_classification_1(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/2"] = lambda item: self.flan_binary_classification_2(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/3"] = lambda item: self.flan_binary_classification_3(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/4"] = lambda item: self.flan_binary_classification_4(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/5"] = lambda item: self.flan_binary_classification_5(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/6"] = lambda item: self.flan_binary_classification_6(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/7"] = lambda item: self.flan_binary_classification_7(self.map_to_flan_multirc(item))
        self.instr2preprocessor["FLAN/BC/8"] = lambda item: self.flan_binary_classification_8(self.map_to_flan_multirc(item))

        self.instr2preprocessor["Alpaca/BC/1"] = lambda item: self.alpaca_binary_classification_1(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/2"] = lambda item: self.alpaca_binary_classification_2(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/3"] = lambda item: self.alpaca_binary_classification_3(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/4"] = lambda item: self.alpaca_binary_classification_4(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/5"] = lambda item: self.alpaca_binary_classification_5(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/6"] = lambda item: self.alpaca_binary_classification_6(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/7"] = lambda item: self.alpaca_binary_classification_7(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/8"] = lambda item: self.alpaca_binary_classification_8(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/9"] = self.alpaca_binary_classification_9
        self.instr2preprocessor["Alpaca/BC/10"] = self.alpaca_binary_classification_10
        self.instr2preprocessor["Alpaca/BC/11"] = self.alpaca_binary_classification_11

        self.instr2preprocessor["Alpaca/Default/1"] = self.alpaca_default
        self.instr2preprocessor["Alpaca/Unobserved/1"] = self.alpaca_unobserved1
        self.instr2preprocessor["Alpaca/Unobserved/2"] = self.alpaca_unobserved2
        self.instr2preprocessor["Alpaca/Unobserved/3"] = self.alpaca_unobserved3
        self.instr2preprocessor["Alpaca/Unobserved/4"] = self.alpaca_unobserved4
        self.instr2preprocessor["Alpaca/Unobserved/6"] = self.alpaca_unobserved6
        self.instr2preprocessor["Alpaca/Unobserved/7"] = self.alpaca_unobserved7
        self.instr2preprocessor["Alpaca/Unobserved/8"] = self.alpaca_unobserved8
        self.instr2preprocessor["Alpaca/Unobserved/9"] = self.alpaca_unobserved9
        self.instr2preprocessor["Alpaca/Unobserved/10"] = self.alpaca_unobserved10

        self.instr2preprocessor_fs["NIV2/BC/1"] = lambda item: self.niv2_binary_classification_1_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/2"] = lambda item: self.niv2_binary_classification_2_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/3"] = lambda item: self.niv2_binary_classification_3_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/4"] = lambda item: self.niv2_binary_classification_4_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/5"] = lambda item: self.niv2_binary_classification_5_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/6"] = lambda item: self.niv2_binary_classification_6_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/7"] = lambda item: self.niv2_binary_classification_7_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/8"] = lambda item: self.niv2_binary_classification_8_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/9"] = lambda item: self.niv2_binary_classification_9_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["NIV2/BC/10"] = lambda item: self.niv2_binary_classification_10_fs(self.map_to_niv2_56_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/1"] = lambda item: self.flan_binary_classification_1_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/2"] = lambda item: self.flan_binary_classification_2_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/3"] = lambda item: self.flan_binary_classification_3_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/4"] = lambda item: self.flan_binary_classification_4_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/5"] = lambda item: self.flan_binary_classification_5_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/6"] = lambda item: self.flan_binary_classification_6_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/7"] = lambda item: self.flan_binary_classification_7_fs(self.map_to_flan_multirc_few_shot(item))
        self.instr2preprocessor_fs["FLAN/BC/8"] = lambda item: self.flan_binary_classification_8_fs(self.map_to_flan_multirc_few_shot(item))
    
    def map_to_alpaca_binary(self, item):

        answer = "True" if item["answer"] == "Yes" else "False"
        item["question"] = "Question: {question}. Answer: Yes".format(question=item["question"])
        item["options"] = ["True", "False"]
        item["answer"] = answer
        return item

    def map_to_niv2_56(self, item):
        paragraph = ""
        question = item["question"]
        output_text = item["answer"]
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "Yes",
            "output_text": output_text
        }
        return new_item

    def map_to_niv2_56_few_shot(self, item):
        paragraph = ""
        question = item["question"]
        output_text = item["answer"]
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "Yes",
            "output_text": output_text
        }
        if "paragraph" not in self.examples[0].keys():
            for i in range(len(self.examples)):
                new_example = self.map_to_niv2_56(self.examples[i])
                self.examples[i] = new_example
                
        return new_item

    def map_to_flan_multirc(self, item):
        paragraph = ""
        question = item["question"]
        answer = item["answer"]
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "Yes",
            "answer": answer
        }
        return new_item

    def map_to_flan_multirc_few_shot(self, item):
        paragraph = ""
        question = item["question"]
        answer = item["answer"]
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "Yes",
            "answer": answer
        }
        if "paragraph" not in self.examples[0].keys():
            for i in range(len(self.examples)):
                new_example = self.map_to_flan_multirc(self.examples[i])
                self.examples[i] = new_example
        return new_item

    def unobserved_template(self, item, input_template):
        input_text = input_template.format(**item)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        example_text = input_template_prefix
        for example in self.examples:
            example_text += example_template.format(**example)
        input_text = input_template.format(**item)
        input_text = example_text + input_text
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "You are given a question which requires reasoning steps that are implicit in the question. " \
                         "Please choose the best answer from  \"yes\" or \"no\" and provide an explanation.\n\nQuestion:" \
                         "{question}\nAnswer and Explanation:"
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved1(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "You are given a question which requires reasoning steps that are implicit in the question. " \
                         "Please choose the best answer from  \"yes\" or \"no\" and provide an explanation."
        
        answer = "yes" if answer == "Yes" else "no"
        label_space = ["yes", "no"]
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=label_space)
    
    def unobserved2(self, item):
        input_template = "Reason about the answer to the question. {question}"
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved2(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Reason about the answer to the question."
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=options)

    def unobserved3(self, item):
        input_template = "You are taking an exam where each question requires implicit reasoning steps to answer. " \
                         "The answer will always be either \"yes\" or \"no\". Please carefully consider the following " \
                         "question, its implications, and any related information you may need to provide the correct " \
                         "answer.\nQuestion: {question}\nAnswer: "
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved3(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "You are taking an exam where each question requires implicit reasoning steps to answer. " \
                         "The answer will always be either \"yes\" or \"no\". Please carefully consider the following " \
                         "question, its implications, and any related information you may need to provide the correct " \
                         "answer."
        answer = "yes" if answer == "Yes" else "no"
        label_space = ["yes", "no"]
        inputs = "Question: {question}".format(question=question)
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)
    
    def unobserved4(self, item):
        input_template = "Answer questions that assume implicit reasoning steps in the question prompt: {question}"
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved4(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Answer questions that assume implicit reasoning steps in the question prompt: "
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=options)

    def unobserved5(self, item):
        input_template = "Input:\n\t- question: {question}\nOutput:\n\t- answer: "
        return self.unobserved_template(item, input_template)

    def unobserved6(self, item):
        input_template = "Use logic and reasoning to answer the following questions with either \"yes\" or \"no\"." \
                         "\nQuestion: {question}\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved6(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Use logic and reasoning to answer the following questions with either \"yes\" or \"no\"." 
        answer = "yes" if answer == "Yes" else "no"
        label_space = ["yes", "no"]
        inputs = "Question: {question}".format(question=question)
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)
    
    def unobserved7(self, item):
        input_template = "Please answer the following question, you should think step by step, but please use \"yes\"" \
                         " or \"no\" to answer.\nQuestion: {question}\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved7(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Please answer the following question, you should think step by step, but please use \"yes\"" \
                         " or \"no\" to answer."
        answer = "yes" if answer == "Yes" else "no"
        label_space = ["yes", "no"]
        inputs = "Question: {question}".format(question=question)
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved8(self, item):
        input_template = "Answer questions in which the required reasoning steps are implicit in the question. " \
                         "Please first answer \"Yes\" or \"No\" and then output your explanation.\n {question} Answer: "
        return self.unobserved_template(item, input_template)
        
    def alpaca_unobserved8(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Answer questions in which the required reasoning steps are implicit in the question. " \
                         "Please first answer \"Yes\" or \"No\" and then output your explanation."
        return self.alpaca_template_with_input(instruction=instruction, inputs=question, answer=answer, label_space=options)
    
    def unobserved9(self, item):
        input_template = "Please give your answer to the following question, which should be answered yes or no. " \
                         "This question may require you to do implicit multi-hop reasoning.\nQuestion: {question}" \
                         "\n Answer: "
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved9(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Please give your answer to the following question, which should be answered yes or no. " \
                         "This question may require you to do implicit multi-hop reasoning."
        inputs = "Question: {question}".format(question=question)
        answer = "yes" if item["answer"] == "Yes" else "no"
        label_space = ["yes", "no"]
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved10(self, item):
        input_template = "This question needs to be solved via decomposing it into multiple sub-questions and make " \
                         "comparison among the results of the sub-questions.\nThe question is {question}\nAfter " \
                         "decomposing the question, we find the answer is "
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved10(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "This question needs to be solved via decomposing it into multiple sub-questions and make " \
                         "comparison among the results of the sub-questions."
        inputs = "The question is {question}\nAfter decomposing the question, we find the answer is?".format(question=question)
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved1_fs(self, item):
        input_template_prefix = "You are given a question which requires reasoning steps that are implicit in the question. Please choose the best answer from  \"yes\" or \"no\" and provide an explanation."
        example_template = "\n\nQuestion: {question}\nAnswer and Explanation: {answer}"
        input_template = "\n\nQuestion: {question}\nAnswer and Explanation: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = ""
        example_template = "\nReason about the answer to the question. {question} {answer}"
        input_template = "\nReason about the answer to the question. {question} "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "You are taking an exam where each question requires implicit reasoning steps to answer. " \
                                "The answer will always be either \"yes\" or \"no\". Please carefully consider the following " \
                                "question, its implications, and any related information you may need to provide the correct " \
                                "answer."
        example_template = "\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "Answer questions that assume implicit reasoning steps in the question prompt: "
        example_template = "\n{question} {answer}"
        input_template = "\n{question} "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = ""
        example_template = "\nInput:\n\t- question: {question}\nOutput:\n\t- answer: {answer}"
        input_template = "\nInput:\n\t- question: {question}\nOutput:\n\t- answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "Use logic and reasoning to answer the following questions with either \"yes\" or \"no\"."
        example_template = "\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = "Please answer the following question, you should think step by step, but please use \"yes\" or \"no\" to answer."
        example_template = "\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "Answer questions in which the required reasoning steps are implicit in the question. Please first answer \"Yes\" or \"No\" and then output your explanation."
        example_template = "\n {question} Answer: {answer}"
        input_template = "\n {question} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = "Please give your answer to the following question, which should be answered yes or no. This question may require you to do implicit multi-hop reasoning."
        example_template = "\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved10_fs(self, item):
        input_template_prefix = "This question needs to be solved via decomposing it into multiple sub-questions and make comparison among the results of the sub-questions."
        example_template = "\nThe question is {question}\nAfter decomposing the question, we find the answer is {answer}"
        input_template = "\nThe question is {question}\nAfter decomposing the question, we find the answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)