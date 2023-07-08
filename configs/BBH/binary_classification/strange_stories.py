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
    input_dir = os.path.join(input_dir, "boolean")
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)

    test_set = Dataset.from_list(items)
    preprocessor = StrangeStoriesPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set

def load_data_testing(input_dir, instruction, args):
    input_dir = os.path.join(input_dir, "boolean")
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = StrangeStoriesPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class StrangeStoriesPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(StrangeStoriesPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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

        self.instr2preprocessor["Alpaca/BC/1"] = lambda item: self.alpaca_binary_classification_1(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/2"] = lambda item: self.alpaca_binary_classification_2(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/3"] = lambda item: self.alpaca_binary_classification_3(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/4"] = lambda item: self.alpaca_binary_classification_4(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/5"] = lambda item: self.alpaca_binary_classification_5(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/6"] = lambda item: self.alpaca_binary_classification_6(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/7"] = lambda item: self.alpaca_binary_classification_7(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/8"] = lambda item: self.alpaca_binary_classification_8(self.map_to_alpaca_binary(item))
        self.instr2preprocessor["Alpaca/BC/9"] = lambda item: self.alpaca_binary_classification_9(self.map_to_alpaca_binary_yesno(item))
        self.instr2preprocessor["Alpaca/BC/10"] = lambda item: self.alpaca_binary_classification_10(self.map_to_alpaca_binary_yesno(item))
        self.instr2preprocessor["Alpaca/BC/11"] = lambda item: self.alpaca_binary_classification_11(self.map_to_alpaca_binary_yesno(item))

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
        paragraph, question = item["question"].split("\n", maxsplit=1)
        output_text = "Yes" if item["answer"] == "yes" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "correct_answer": "Yes",
            "output_text": output_text
        }
        return new_item

    def map_to_niv2_56_few_shot(self, item):
        paragraph, question = item["question"].split("\n", maxsplit=1)
        output_text = "Yes" if item["answer"] == "yes" else "No"
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
        paragraph, question = item["question"].split("\n", maxsplit=1)
        answer = "Yes" if item["answer"] == "yes" else "No"
        new_item = {
            "paragraph": paragraph,
            "question": question,
            "response": "Yes",
            "answer": answer
        }
        return new_item

    def map_to_flan_multirc_few_shot(self, item):
        paragraph, question = item["question"].split("\n", maxsplit=1)
        answer = "Yes" if item["answer"] == "yes" else "No"
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

    def map_to_alpaca_binary(self, item):
        answer = "True" if item["answer"] == "yes" else "False"
        item["options"] = ["True", "False"]
        item["answer"] = answer
        return item

    def map_to_alpaca_binary_yesno(self, item):
        answer = "Yes" if item["answer"] == "yes" else "No"
        item["options"] = ["Yes", "No"]
        item["answer"] = answer
        return item

    def unobserved_template(self, item, input_template):
        context, question = item["question"].split("\n", maxsplit=1)
        input_text = input_template.format(**item, context=context)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        example_text = input_template_prefix
        for example in self.examples:
            example_context, example_question = example["question"].split("\n", maxsplit=1)
            example_text += example_template.format(question=example_question, context=example_context, answer=example["answer"])
        context, question = item["question"].split("\n", maxsplit=1)
        input_text = input_template.format(question=question, context=context)
        input_text = example_text + input_text
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "You are given a psychology question that asks you to provide a socially intelligent " \
                         "response after reading a short story. Please answer \"yes\" or \"no\" to the given question." \
                         "\n\n{context}\n\nQuesiton: {question}\nAnswer:"
        return self.unobserved_template(item, input_template)
    
    def unobserved1_fs(self, item):
        input_template_prefix = "You are given a psychology question that asks you to provide a socially intelligent " \
                         "response after reading a short story. Please answer \"yes\" or \"no\" to the given question."
        example_template = "\n\n{context}\n\nQuesiton: {question}\nAnswer: {answer}\n"
        input_template = "\n\n{context}\n\nQuesiton: {question}\nAnswer:"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved1(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "You are given a psychology question that asks you to provide a socially intelligent " \
                         "response after reading a short story. Please answer \"yes\" or \"no\" to the given question." 
        inputs = f"\n\n{context}\n\nQuesiton: {question}\n"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)
    
    def unobserved2(self, item):
        input_template = "Given a story, answer whether the question is true or false.\n{context}\nQ: {question}\nA:"
        return self.unobserved_template(item, input_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = "Given a story, answer whether the question is true or false."
        example_template = "\n{context}\nQ: {question}\nA: {answer}"
        input_template = "\n{context}\nQ: {question}\nA: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved2(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "Given a story, answer whether the question is true or false."
        inputs = f"\n{context}\nQ: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)
        
    def unobserved3(self, item):
        input_template = "You are taking a test for reading comprehension. You will be presented with a story and " \
                         "asked a question related to the story. The answer to the question is either \"yes\" or " \
                         "\"no\". Please carefully consider the story below before selecting your answer.\nStory: " \
                         "{context}\nQuestion: {question}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved3_fs(self, item):
        input_template_prefix = "You are taking a test for reading comprehension. You will be presented with a story and " \
                         "asked a question related to the story. The answer to the question is either \"yes\" or " \
                         "\"no\". Please carefully consider the story below before selecting your answer."
        example_template = "\nStory: {context}\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\nStory: {context}\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved3(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "You are taking a test for reading comprehension. You will be presented with a story and " \
                         "asked a question related to the story. The answer to the question is either \"yes\" or " \
                         "\"no\". Please carefully consider the story below before selecting your answer."
        inputs = f"Story: {context}\nQuestion: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved4(self, item):
        input_template = "A psychology test with naturalistic short stories that measures social intelligence. " \
                         "Boolean options.{context}\nQ: {question}Answer:"
        return self.unobserved_template(item, input_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "A psychology test with naturalistic short stories that measures social intelligence. " \
                         "Boolean options."
        example_template = "\n{context}\nQ: {question}Answer: {answer}"
        input_template = "\n{context}\nQ: {question}Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved4(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "A psychology test with naturalistic short stories that measures social intelligence. " \
                         "Boolean options."
        
        inputs = f"{context}\nQ: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved5(self, item):
        input_template = "Story: {context}\n Q: {question}\n Output: "
        return self.unobserved_template(item, input_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = ""
        example_template = "Story: {context}\n Q: {question}\n Output: {answer}\n"
        input_template = "Story: {context}\n Q: {question}\n Output: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved5(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = f"Q: {question}"
        inputs = f"Story: {context}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved6(self, item):
        input_template = "Given the following text, answer the question with either \"yes\" or \"no\":\n\nText: " \
                         "{context}\nQuestion: {question}\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "Given the following text, answer the question with either \"yes\" or \"no\":"
        example_template = "\n\nText: {context}\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\n\nText: {context}\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved6(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "Given the following text, answer the question with either \"yes\" or \"no\":\n\n"
        inputs = f"Text: {context}\nQuestion: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved7(self, item):
        input_template = "Please read the following text and answer the question according to the content of the " \
                         "text, your answer should be \"yes\" or \"no\".\nText: {context}\nQuestion: {question}\n" \
                         "Answer:"
        return self.unobserved_template(item, input_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = "Please read the following text and answer the question according to the content of the text, your answer should be \"yes\" or \"no\"."
        example_template = "\nText: {context}\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\nText: {context}\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved7(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "Please read the following text and answer the question according to the content of the text, your answer should be \"yes\" or \"no\"."
        inputs = f"Text: {context}\nQuestion: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved8(self, item):
        input_template = "Image you are taking a psychology test. Please read the given story and answer the " \
                         "question. Please answer \"yes\" or \"No\".\nStory: {context}\nQ: {question}\nA: "
        return self.unobserved_template(item, input_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "Image you are taking a psychology test. Please read the given story and answer the question. Please answer \"yes\" or \"No\"."
        example_template = "\nStory: {context}\nQ: {question}\nA: {answer}"
        input_template = "\nStory: {context}\nQ: {question}\nA: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved8(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "Image you are taking a psychology test. Please read the given story and answer the " \
                         "question. Please answer \"yes\" or \"no\"."
        inputs = f"Story: {context}\nQ: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved9(self, item):
        input_template = "Please give your answer to the following question, which should be answered yes or no. " \
                         "You should judge the correctness of the question according to the story.\n\nStory: " \
                         "{context}\nQuestion: {question}\nAnswer: "
        return self.unobserved_template(item, input_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = "Please give your answer to the following question, which should be answered yes or no. You should judge the correctness of the question according to the story."
        example_template = "\n\nStory: {context}\nQuestion: {question}\nAnswer: {answer}"
        input_template = "\n\nStory: {context}\nQuestion: {question}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def alpaca_unobserved9(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "Please give your answer to the following question, which should be answered yes or no. " \
                         "You should judge the correctness of the question according to the story."
        inputs = f"Story: {context}\nQuestion: {question}"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)

    def unobserved10(self, item):
        input_template = "The following story is associated with a question, the answer of which is \"yes\" or " \
                         "\"no\". \n\nStory: {context}\nQuestion: {question}\n\nAccording to the story, the answer is "
        return self.unobserved_template(item, input_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = "The following story is associated with a question, the answer of which is \"yes\" or \"no\"."
        example_template = "\n\nStory: {context}\nQuestion: {question}\n\nAccording to the story, the answer is {answer}"
        input_template = "\n\nStory: {context}\nQuestion: {question}\n\nAccording to the story, the answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def alpaca_unobserved10(self, item):
        answer, label_space = item["answer"], item["options"]
        context, question = item["question"].split("\n", maxsplit=1)
        instruction = "PThe following story is associated with a question, the answer of which is \"yes\" or " \
                         "\"no\""
        inputs = f"Story: {context}\nQuestion: {question}\n\nAccording to the story, the answer is?"
        return self.alpaca_template_with_input(instruction=instruction, inputs=inputs, answer=answer, label_space=label_space)
