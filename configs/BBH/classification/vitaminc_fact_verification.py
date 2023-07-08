import os.path
import json
import random
from datasets import Dataset
from configs.utils import ClassificationMatchAccuracy, OptionMatchingAccuracy, ClassificationAccuracy, load_BBL_file
from configs.preprocessor import Preprocessor
import multiprocessing as mp

special_tokens = []

fewshot_examples = [1, 3, 5, 7, 9]


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, shot_count)

    test_set = Dataset.from_list(items)
    preprocessor = VitamincFactPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set

def load_data_testing(input_dir, instruction, args):
    items, examples, _ = load_BBL_file(os.path.join(input_dir, "task.json"), fewshot_examples, 0)
    if args.maximum_test_samples is not None:
        items = random.sample(items, min(args.maximum_test_samples, len(items)))
    test_set = Dataset.from_list(items)
    preprocessor = VitamincFactPreprocessor(instruction, examples, True, input_dir)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=mp.cpu_count())
    return test_set


class VitamincFactPreprocessor(Preprocessor):

    _UNOBSERVED_ALPACA_INSTRUCTIONS = [
        "Based only on the information contained in a brief quote from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when the Wikipedia quote does not provide the necessary information to resolve the question.\n\n",
        "You are now a very experienced judge. Based only on the information contained in a brief quote from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when the Wikipedia quote does not provide the necessary information to resolve the question.",
        "Now you are a Vitaminc Fact Verifier. Based only on the information contained in a brief quote from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when the Wikipedia quote does not provide the necessary information to resolve the question.",
        "Read the above paragraph, and answer the following claim. Answer True, Flase, or Neither. Neither means the Wikipedia quote does not provide the necessary information to resolve the question.",
        "Given a claim and its related information context from Wikipedia, determine whether the claim is True, False or Neither. Neither means the given information is not enough to decide if the claim is True or False, which is roughly equivalent to uncertain.",
        "Based only on the information contained in the given context, please make a judgement whether the related claim is True, False or Neither.",
        "Based on the context, is the claim true? false? Or Neither? Give your answer as one of \"True\", \"False\" or \"Neither\"",
        "Based only on the information contained in the given context, please make a judgement whether the related claim is True, False or Neither.",
        "Evaluate the related claim as True, False, or Neither based solely on the information given in the short Wikipedia excerpt. Select Neither when the excerpt doesn't provide sufficient information to address the question.",
        "Verify the factually of the claim based on the context\n\n- \"True\" if the claim is factually correct\n- \"False\" if the claim is factually incorrect\n- \"Neither\" if the factuality cannot be determined. Output you answer with one of \"True\", \"False\", or \"Neither\".",
        "Classify this claim into one of 'True', 'False', or 'Neither'"
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(VitamincFactPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
        self.instr2preprocessor["Alpaca/Unobserved/10"] = lambda item: self.alpaca_multiple_choice(item, self._UNOBSERVED_ALPACA_INSTRUCTIONS[10])

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
        self.instr2preprocessor_fs["FLAN/Classification/8"] = lambda item: self.flan_classification_8_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/9"] = lambda item: self.flan_classification_9_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/10"] = lambda item: self.flan_classification_10_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/11"] = lambda item: self.flan_classification_11_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/12"] = lambda item: self.flan_classification_12_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/13"] = lambda item: self.flan_classification_13_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/14"] = lambda item: self.flan_classification_14_fs(self.map_to_flan_cosmosqa_few_shot(item))
        self.instr2preprocessor_fs["FLAN/Classification/15"] = lambda item: self.flan_classification_15_fs(self.map_to_flan_cosmosqa_few_shot(item))

    def map_to_flan_cosmosqa(self, item):
        context, question = item["question"].split("\nClaim:")
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
        context, question = item["question"].split("\nClaim:")
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
                context, question = example["question"].split("\nClaim:")
                example["context"] = context
                example["question"] = question

        return new_item

    def unobserved_template(self, item, input_template):
        context, claim = item["question"].split("\nClaim:")
        input_text = input_template.format(context=context, claim=claim)
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        context, claim = item["question"].split("\nClaim:")
        example_text = input_template_prefix
        for example in self.examples:
            ex_context, ex_claim = example["question"].split("\nClaim:")
            ex_answer = example["answer"]
            example_text += example_template.format(context=ex_context, claim=ex_claim, answer=ex_answer)

        input_text = input_template.format(context=context, claim=claim)
        input_text = example_text + input_text
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "You are now a very experienced judge. Based only on the information contained in a brief quote " \
                     "from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when " \
                     "the Wikipedia quote does not provide the necessary information to resolve the question.\n\n" \
                     "{context}\nClaim: {claim}\nIs this True, False, or Neither?"
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "Now you are a Vitaminc Fact Verifier. Based only on the information contained in a brief quote " \
                     "from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when the" \
                     " Wikipedia quote does not provide the necessary information to resolve the question.\n\n" \
                     "{context}\nClaim: {claim}\nQuestion: Is this True, False, or Neither?\nYour answer: "
        return self.unobserved_template(item, input_template)

    def unobserved3(self, item):
        input_template = "{context}\nRead the above paragraph, and answer the following claim {claim}. Answer True, " \
                         "Flase, or Neither. Neither means the Wikipedia quote does not provide the necessary " \
                         "information to resolve the question. Answer: "
        return self.unobserved_template(item, input_template)

    def unobserved4(self, item):
        input_template = "Given a claim and its related information context from Wikipedia, determine whether the " \
                         "claim is True, False or Neither. Neither means the given information is not enough to " \
                         "decide if the claim is True or False, which is roughly equivalent to uncertain.\n\nContext:" \
                         "{context}\nClaim: {claim}\nTrue, False or Neither?"
        return self.unobserved_template(item, input_template)

    def unobserved5(self, item):
        input_template = "{context} Claim: {claim}\nBased on the context, is the claim true? false? Or Neither? " \
                         "Give your answer as one of \"True\", \"False\" or \"Neither\""
        return self.unobserved_template(item, input_template)

    def unobserved6(self, item):
        input_template = "Based only on the information contained in the given context, please make a judgement " \
                         "whether the related claim is True, False or Neither.\n\n{context}\n Claim: {claim}\n" \
                         "True, False, or Neither?"
        return self.unobserved_template(item, input_template)

    def unobserved7(self, item):
        input_template = "Wikipedia: {context}\nSomeone: based on the given context, is the {claim} True, " \
                         "False, or Neither?"
        return self.unobserved_template(item, input_template)

    def unobserved8(self, item):
        input_template = "Evaluate the related claim as True, False, or Neither based solely on the information " \
                         "given in the short Wikipedia excerpt. Select Neither when the excerpt doesn't provide " \
                         "sufficient information to address the question.\n{context}\nClaim: {claim}\nAnswer" \
                         "(True, False, or Neither):"
        return self.unobserved_template(item, input_template)

    def unobserved9(self, item):
        input_template = "Input: {claim}\nVerify the factually of the claim based on the following context\n{context}" \
                         "\n\n- \"True\" if the claim is factually correct\n- \"False\" if the claim is " \
                         "factually incorrect\n- \"Neither\" if the factuality cannot be determined. Output you answer" \
                         "with one of \"True\", \"False\", or \"Neither\". Answer: "
        return self.unobserved_template(item, input_template)

    def unobserved10(self, item):
        input_template = "Context: {context}\nNow classify this claim into one of 'True', 'False', or 'Neither'.\n" \
                         "{claim}"
        return self.unobserved_template(item, input_template)
    
    def unobserved1_fs(self, item):
        input_template_prefix = "You are now a very experienced judge. Based only on the information contained in a brief quote " \
                     "from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when " \
                     "the Wikipedia quote does not provide the necessary information to resolve the question."

        example_template = "\n\n{context}\nClaim: {claim}\nIs this True, False, or Neither? {answer}"

        input_template = "\n\n{context}\nClaim: {claim}\nIs this True, False, or Neither? "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved2_fs(self, item):
        input_template_prefix = "Now you are a Vitaminc Fact Verifier. Based only on the information contained in a brief quote " \
                     "from Wikipedia, answer whether the related claim is True, False or Neither. Use Neither when the" \
                     " Wikipedia quote does not provide the necessary information to resolve the question."

        example_template = "\n\n{context}\nClaim: {claim}\nQuestion: Is this True, False, or Neither?\nYour answer: {answer}"

        input_template = "\n\n{context}\nClaim: {claim}\nQuestion: Is this True, False, or Neither?\nYour answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved3_fs(self, item):
        input_template_prefix = "Read the paragraph, and answer the following claim. Answer True, " \
                         "Flase, or Neither. Neither means the Wikipedia quote does not provide the necessary " \
                         "information to resolve the question."

        example_template = "\n{context}\nClaim: {claim}\nAnswer: {answer}"

        input_template = "\n{context}\nClaim: {claim}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved4_fs(self, item):
        input_template_prefix = "Given a claim and its related information context from Wikipedia, determine whether the " \
                                "claim is True, False or Neither. Neither means the given information is not enough to " \
                                "decide if the claim is True or False, which is roughly equivalent to uncertain."

        example_template = "\n\nContext:{context}\nClaim: {claim}\nTrue, False or Neither? {answer}"

        input_template = "\n\nContext:{context}\nClaim: {claim}\nTrue, False or Neither? "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved5_fs(self, item):
        input_template_prefix = "Based on the context, is the claim true? false? Or Neither? Give your answer as one of \"True\", \"False\" or \"Neither\""

        example_template = "\n{context} Claim: {claim}\n {answer}"

        input_template = "\n{context} Claim: {claim}\n"
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved6_fs(self, item):
        input_template_prefix = "Based only on the information contained in the given context, please make a judgement whether the related claim is True, False or Neither."

        example_template = "\n\n{context}\n Claim: {claim}\nTrue, False, or Neither? {answer}"

        input_template = "\n\n{context}\n Claim: {claim}\nTrue, False, or Neither? "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved7_fs(self, item):
        input_template_prefix = ""

        example_template = "Wikipedia: {context}\nSomeone: based on the given context, is the {claim} True, " \
                         "False, or Neither?\nSystem: {answer}\n"

        input_template = "Wikipedia: {context}\nSomeone: based on the given context, is the {claim} True, " \
                         "False, or Neither?\nSystem: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved8_fs(self, item):
        input_template_prefix = "Evaluate the related claim as True, False, or Neither based solely on the information " \
                         "given in the short Wikipedia excerpt. Select Neither when the excerpt doesn't provide " \
                         "sufficient information to address the question."

        example_template = "\n{context}\nClaim: {claim}\nAnswer" \
                         "(True, False, or Neither): {answer}"

        input_template = "\n{context}\nClaim: {claim}\nAnswer" \
                         "(True, False, or Neither): "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved9_fs(self, item):
        input_template_prefix = "Verify the factuality of the claim based on the following context" \
                         "\n\n- \"True\" if the claim is factually correct\n- \"False\" if the claim is " \
                         "factually incorrect\n- \"Neither\" if the factuality cannot be determined. Output you answer" \
                         "with one of \"True\", \"False\", or \"Neither\"."

        example_template = "\nInput: {claim}\nContext: {context}\nAnswer: {answer}"

        input_template ="\nInput: {claim}\nContext: {context}\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
    
    def unobserved10_fs(self, item):
        input_template_prefix = ""

        example_template = "Context: {context}\nNow classify this claim into one of 'True', 'False', or 'Neither'.\n{claim} {answer}\n"

        input_template = "Context: {context}\nNow classify this claim into one of 'True', 'False', or 'Neither'.\n{claim} "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

