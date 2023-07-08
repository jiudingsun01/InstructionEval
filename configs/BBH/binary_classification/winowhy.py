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
    preprocessor = PlayDialogPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

def load_data_testing(input_dir, instruction, args):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = PlayDialogPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

class PlayDialogPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(PlayDialogPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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

        self.instr2preprocessor["Alpaca/BC/1"] = lambda item: self.alpaca_binary_classification_1(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/2"] = lambda item: self.alpaca_binary_classification_2(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/3"] = lambda item: self.alpaca_binary_classification_3(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/4"] = lambda item: self.alpaca_binary_classification_4(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/5"] = lambda item: self.alpaca_binary_classification_5(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/6"] = lambda item: self.alpaca_binary_classification_6(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/7"] = lambda item: self.alpaca_binary_classification_7(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/8"] = lambda item: self.alpaca_binary_classification_8(self.map_to_alpaca(item))
        self.instr2preprocessor["Alpaca/BC/9"] = lambda item: self.alpaca_binary_classification_9(self.map_to_alpaca_yesno(item))
        self.instr2preprocessor["Alpaca/BC/10"] = lambda item: self.alpaca_binary_classification_10(self.map_to_alpaca_yesno(item))
        self.instr2preprocessor["Alpaca/BC/11"] = lambda item: self.alpaca_binary_classification_11(self.map_to_alpaca_yesno(item))

        self.instr2preprocessor["Alpaca/Default/1"] = self.alpaca_default
        self.instr2preprocessor["Alpaca/Unobserved/1"] = self.alpaca_unobserved1
        self.instr2preprocessor["Alpaca/Unobserved/2"] = self.alpaca_unobserved2
        self.instr2preprocessor["Alpaca/Unobserved/3"] = self.alpaca_unobserved3
        self.instr2preprocessor["Alpaca/Unobserved/4"] = self.alpaca_unobserved4
        self.instr2preprocessor["Alpaca/Unobserved/5"] = self.alpaca_unobserved5
        self.instr2preprocessor["Alpaca/Unobserved/6"] = self.alpaca_unobserved6
        self.instr2preprocessor["Alpaca/Unobserved/7"] = self.alpaca_unobserved7
        self.instr2preprocessor["Alpaca/Unobserved/8"] = self.alpaca_unobserved8
        self.instr2preprocessor["Alpaca/Unobserved/9"] = self.alpaca_unobserved9
        self.instr2preprocessor["Alpaca/Unobserved/10"] = self.alpaca_unobserved10
        
    def map_to_niv2_56(self, item):
        paragraph, question = item["question"].split("The \'", maxsplit=1)
        question = "The \'" + question
        output_text = "Yes" if item["answer"] == "correct" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "correct",
            "output_text": output_text
        }
        return new_item

    def map_to_niv2_56_few_shot(self, item):
        if "paragraph" not in self.examples[0].keys():
            for i in range(len(self.examples)):
                new_example = self.map_to_niv2_56(self.examples[i])
                self.examples[i] = new_example

        paragraph, question = item["question"].split("The \'", maxsplit=1)
        question = "The \'" + question
        output_text = "Yes" if item["answer"] == "correct" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "correct",
            "output_text": output_text
        }
        return new_item

    def map_to_flan_multirc(self, item):
        paragraph, question = item["question"].split("The \'", maxsplit=1)
        question = "The \'" + question
        answer = "Yes" if item["answer"] == "correct" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "correct",
            "answer": answer
        }
        return new_item

    def map_to_flan_multirc_few_shot(self, item):

        paragraph, question = item["question"].split("The \'", maxsplit=1)
        question = "The \'" + question
        answer = "Yes" if item["answer"] == "correct" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "correct",
            "answer": answer
        }

        if "paragraph" not in self.examples[0].keys():
            for i in range(len(self.examples)):
                new_example = self.map_to_flan_multirc(self.examples[i])
                self.examples[i] = new_example
        
        return new_item

    def map_to_alpaca(self, item):
        answer = "True" if item["answer"] == "correct" else "False"
        item["answer"] = answer
        item["options"] = ["True", "False"]
        return item
    
    def map_to_alpaca_yesno(self, item):
        answer = "Yes" if item["answer"] == "correct" else "No"
        item["answer"] = answer
        item["options"] = ["Yes", "No"]
        return item

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
        input_template = "In the sentence: {question}. Is the pronoun reasoning correct? Please answer with either " \
                         "\"correct\" or \"incorrect\". Do not include any other words."
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved1(self, item):
        inputs, answer, options = item["question"], item["answer"], item["options"]
        instruction = "In the sentence, is the pronoun reasoning correct? Please answer with either " \
                         "\"correct\" or \"incorrect\". Do not include any other words."
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved2(self, item):
        input_template = "Verify if the reasoning about which words certain pronouns refer to in the given words is " \
                         "right, choose one answer from \"correct\" and \"incorrect\":\nReasoning:{question}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved2(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Verify if the reasoning about which words certain pronouns refer to in the given words is " \
                         "right, choose one answer from \"correct\" and \"incorrect\":"
        inputs = f"Reasoning: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved3(self, item):
        input_template = "Given the context: {question}, determine if the co-reference resolution and the explanation" \
                         "is correct by output either \"correct\" or \"incorrect\". Answer: "
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved3(self, item):
        inputs, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Given the context, determine if the co-reference resolution and the explanation" \
                         "is correct by output either \"correct\" or \"incorrect\"."
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)
    
    def unobserved4(self, item):
        item["answer"] = "Yes" if item["answer"] == "correct" else "No"
        item["options"] = ["Yes", "No"]
        input_template = "Context: {question}\nQuestion: Is the pronoun referring to the correct object? Answer with" \
                         "\"Yes\" or \"No\"."
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved4(self, item):
        item["answer"] = "Yes" if item["answer"] == "correct" else "No"
        item["options"] = ["Yes", "No"]
        question = item["question"]
        instruction = f"Context: {question}\nQuestion: Is the pronoun referring to the correct object? Answer with" \
                         "\"Yes\" or \"No\"."
        return self.alpaca_template_without_input(instruction=instruction, answer=item["answer"], label_space=item["options"])
    
    def unobserved5(self, item):
        input_template = "Judge the correctness of the understanding of pronoun:\n{question}\nGive your answer as" \
                         "\"correct\" or \"incorrect\". Your answer: "
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved5(self, item):
        inputs, answer, options = item["question"], item["answer"], item["options"]
        instruction = "Judge the correctness of the understanding of pronoun, and give your answer as" \
                         "\"correct\" or \"incorrect\"."
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved6(self, item):
        item["answer"] = "Yes" if item["answer"] == "correct" else "No"
        item["options"] = ["Yes", "No"]
        input_template = "You are tested on your understanding of pronoun. Here is a sentence followed by the " \
                         "explanation: {question}\nOutput \"Yes\" if you think the explanation is correct; output \"No\"" \
                         "If the explanation is wrong."
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved6(self, item):
        item["answer"] = "Yes" if item["answer"] == "correct" else "No"
        item["options"] = ["Yes", "No"]
        instruction = "You are tested on your understanding of pronoun. Here is a sentence followed by the " \
                         "explanation. Output \"Yes\" if you think the explanation is correct; output \"No\"" \
                         "If the explanation is wrong."
        inputs, answer, options = item["question"], item["answer"], item["options"]
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)

    def unobserved7(self, item):
        input_template = "Read the following reasoning about who a particular pronoun refers to: {question}\n" \
                         "Is the reasoning correct?"
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved7(self, item):
        instruction = "Read the following reasoning about who a particular pronoun refers to Is the reasoning correct?"
        inputs, answer, options = item["question"], item["answer"], item["options"]
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)
    
    def unobserved8(self, item):
        input_template = "Read the following reasoning, and answer if its correct or incorrect. {question}\n"
        return self.unobserved_template(item, input_template)

    def alpaca_unobserved8(self, item):
        instruction = "Read the following reasoning, and answer if its correct or incorrect."
        inputs, answer, options = item["question"], item["answer"], item["options"]
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=options)
    
    def unobserved9(self, item):
        input_template = "{question} The reasoning stated above is \"correct\" or \"incorrect\"? It is"
        return self.unobserved_template(item, input_template)
    
    def alpaca_unobserved9(self, item):
        question = item["question"]
        instruction = f"{question} The reasoning stated above is \"correct\" or \"incorrect\"?"
        return self.alpaca_template_without_input(instruction=instruction, answer=item["answer"], label_space=item["options"])

    def unobserved10(self, item):
        context, explanation = item["question"].split("The \'", maxsplit=1)
        explanation = "The \'" + explanation
        input_text = "You will be given a sentence followed by an explanation of the use of pronouns in that " \
                         "sentence. Please answer if the explanation is correct or incorrect.\n\nSentence: {context}" \
                         "\nExplanation: {explanation}\nAnswer:".format(context=context, explanation=explanation)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def alpaca_unobserved10(self, item):
        context, explanation = item["question"].split("The \'", maxsplit=1)
        explanation = "The \'" + explanation
        input_text = "You will be given a sentence followed by an explanation of the use of pronouns in that " \
                         "sentence. Please answer if the explanation is correct or incorrect."
        inputs = "Sentence: {context}\nExplanation: {explanation}".format(context=context, explanation=explanation)
        answer = item["answer"]
        label_space = item["options"]
        return self.alpaca_template_with_input(instruction=input_text, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved1_fs(self, item):
        input_template_prefix = "In the sentence: {question}. Is the pronoun reasoning correct? Please answer with either \"correct\" or \"incorrect\". Do not include any other words."
        example_template = "\n{question} {answer}"
        input_template = "\n{question}"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved2_fs(self, item):
        input_template_prefix = "Verify if the reasoning about which words certain pronouns refer to in the given words is " \
                         "right, choose one answer from \"correct\" and \"incorrect\":"
        example_template = "\nReasoning: {question}\nAnswer: {answer}"
        input_template = "\nReasoning: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "Given the context, determine if the co-reference resolution and the explanation" \
                         "is correct by output either \"correct\" or \"incorrect\". Answer: "
        example_template = "\n{question} {answer}"
        input_template = "\n{question}"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        if self.examples[0]["options"] != ["Yes", "No"]:
            for i in range(len(self.examples)):
                self.examples[i]["answer"] = "Yes" if self.examples[i]["answer"] == "correct" else "No"
                self.examples[i]["options"] = ["Yes", "No"]

        input_template_prefix = ""
        example_template = "Context: {question}\nQuestion: Is the pronoun referring to the correct object? Answer with \"Yes\" or \"No\". {answer}\n"
        input_template = "Context: {question}\nQuestion: Is the pronoun referring to the correct object? Answer with \"Yes\" or \"No\". "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = "Judge the correctness of the understanding of pronoun. Give your answer as \"correct\" or \"incorrect\"."
        example_template = "\n{question} Your answer: {answer}"
        input_template = "\n{question} Your answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "You are tested on your understanding of pronoun. Here is a sentence followed by the " \
                         "explanation: {question}\nOutput \"correct\" if you think the explanation is correct; output \"incorrect\"" \
                         "If the explanation is wrong."
        example_template = "\n{question} {answer}"
        input_template = "\n{question} "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = "Read the following reasoning about who a particular pronoun refers to:"
        example_template = "\n{question}\nIs the reasoning correct? {answer}"
        input_template = "\n{question}\nIs the reasoning correct? {answer}"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "Read the following reasoning, and answer if its correct or incorrect. "
        example_template = "{question}\n{answer}\n"
        input_template = "{question}\n"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = ""
        example_template = "{question} The reasoning stated above is \"correct\" or \"incorrect\"? It is {answer}\n"
        input_template = "{question} The reasoning stated above is \"correct\" or \"incorrect\"? It is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        context, explanation = item["question"].split("The \'", maxsplit=1)
        new_item = item.copy()
        new_item["explanation"] = "The \'" + explanation
        new_item["question"] = context
        if "explanation" not in self.examples[0].keys():
            for example in self.examples:
                ex_context, ex_explanation = example["question"].split("The \'", maxsplit=1)
                example["explanation"] = "The \'" + ex_explanation
                example["question"] = ex_context

        input_template_prefix = "You will be given a sentence followed by an explanation of the use of pronouns in that " \
                                "sentence. Please answer if the explanation is correct or incorrect. "
        
        example_template = "\n\nSentence: {question}\nExplanation: {explanation}\nAnswer: {answer}"
        input_template = "\n\nSentence: {question}\nExplanation: {explanation}\nAnswer: "
        return self.unobserved_template_few_shot(new_item, input_template_prefix, input_template, example_template)