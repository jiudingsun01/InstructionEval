import os.path
import pandas as pd

from datasets import Dataset
from configs.utils import OptionMatchingAccuracy, ClassificationAccuracy
from configs.preprocessor import Preprocessor
import multiprocessing as mp


SUBSET = True

special_tokens = []


def load_data(input_dir, instruction, shot_count, eval_by_logits, tokenizer):

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    items, examples = [], dict()
    for file in files:
        domain = file.replace(".csv", "").replace("_", " ")
        df = pd.read_csv(os.path.join(input_dir, file), names=["question", "A", "B", "C", "D", "answer"])
        file_items = df.to_dict("records")

        for _ in range(shot_count):
            item = file_items.pop()
            item["answer"] = item[item.pop("answer")]
            item["options"] = [item.pop(x) for x in ["A", "B", "C", "D"]]
            item["domain"] = domain
            assert list(item.keys()) == ["question", "answer", "options", "domain"]
            if domain not in examples.keys():
                examples[domain] = []
            examples[domain].append(item)

        for item in file_items:
            item["answer"] = item[item.pop("answer")]
            item["options"] = [item.pop(x) for x in ["A", "B", "C", "D"]]
            item["domain"] = domain
            assert list(item.keys()) == ["question", "answer", "options", "domain"]
            if SUBSET:
                text = "{} {} {} {} {}".format(item["answer"], *item["options"])
                if len(tokenizer(text, truncation=True)["input_ids"]) >= 150:
                    continue
            items.append(item)

    test_set = Dataset.from_list(items)
    preprocessor = MMLUSpecificPreprocessor(instruction, examples, eval_by_logits)
    preprocess = preprocessor.processor
    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer", "domain"], num_proc=mp.cpu_count())

    return test_set


class MMLUSpecificPreprocessor(Preprocessor):

    def __init__(self, instruction, examples: dict, eval_by_logits):
        super(MMLUSpecificPreprocessor, self).__init__(instruction, examples, eval_by_logits)

    def add_unobserved_instructions(self):
        self.instr2preprocessor["MMLU/Unobserved/1"] = self.unobserved1
        self.instr2preprocessor["MMLU/Unobserved/2"] = self.unobserved2
        self.instr2preprocessor["MMLU/Unobserved/3"] = self.unobserved3
        self.instr2preprocessor["MMLU/Unobserved/4"] = self.unobserved4
        self.instr2preprocessor["MMLU/Unobserved/5"] = self.unobserved5
        self.instr2preprocessor["MMLU/Unobserved/6"] = self.unobserved6
        self.instr2preprocessor["MMLU/Unobserved/7"] = self.unobserved7
        self.instr2preprocessor["MMLU/Unobserved/8"] = self.unobserved8
        self.instr2preprocessor["MMLU/Unobserved/9"] = self.unobserved9
        self.instr2preprocessor["MMLU/Unobserved/10"] = self.unobserved10

        self.instr2preprocessor_fs["MMLU/Unobserved/1"] = self.unobserved1_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/2"] = self.unobserved2_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/3"] = self.unobserved3_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/4"] = self.unobserved4_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/5"] = self.unobserved5_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/6"] = self.unobserved6_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/7"] = self.unobserved7_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/8"] = self.unobserved8_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/9"] = self.unobserved9_fs
        self.instr2preprocessor_fs["MMLU/Unobserved/10"] = self.unobserved10_fs

    def unobserved_template(self, item, input_template):
        question, options, answer, domain = item["question"], item["options"], item["answer"], item["domain"]
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text = input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, domain=domain)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer, domain = item["question"], item["options"], item["answer"], item["domain"]
        input_text = input_template_prefix.format(domain=domain)
        for i, example in enumerate(self.examples[domain]):
            example_question, example_options, example_answer = example["question"], example["options"], example["answer"]
            A, B, C, D = example_options
            example_choice = ["A", "B", "C", "D"][example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, choiceA=A, choiceB=B, choiceC=C,
                                                  choiceD=D, answer=example_choice, domain=domain)
        A, B, C, D = options
        choice = ["A", "B", "C", "D"][options.index(answer)]
        input_text += input_template.format(question=question, choiceA=A, choiceB=B, choiceC=C, choiceD=D, domain=domain)
        output_text = choice
        label_space = ["A", "B", "C", "D"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved1(self, item):
        input_template = "Please answer the appropriate questions based on knowledge of the{domain}. Each question " \
                     "will give four options, please output the corresponding option (i.e. A, B, C or D) to " \
                     "represent the corresponding answer.\n\nQuestion: {question}\nThe options are:\n- A: {choiceA}" \
                     "\n- B: {choiceB}\n- C: {choiceC}\n- D: {choiceD}\n\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved2(self, item):
        input_template = "This is a single-choice question coming from {domain}. Use your knowledge in {domain} to solve" \
                         " the following question and select the correct answer among \"A\", \"B\", \"C\", and \"D\". Just output" \
                         " the answer with the corresponding letter!\n\nQuestion: {question}\nCandidate Answers:\nA: " \
                         "{choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nAccording to knowledge from {domain}, the answer is"
        return self.unobserved_template(item, input_template)

    def unobserved3(self, item):
        input_template = "This is a multiple choice test on {domain}. Please choose the correct answer" \
                         " among A, B, C and D by comparing them. Question: {question}\nA: {choiceA}\nB: " \
                         "{choiceB}\nC: {choiceC}\nD: {choiceD}\n\n"
        return self.unobserved_template(item, input_template)

    def unobserved4(self, item):
        input_template = "Please answer the question using your knowledge about {domain}. Output one of \"A\", \"B\", " \
                     "\"C\", or \"D\" to indicate your answer: A: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} " \
                         "{question} Answer:"
        return self.unobserved_template(item, input_template)

    def unobserved5(self, item):
        input_template = "Answer the following question with your {domain} knowledge. Note that there may be more " \
                         "than one correct option.\nQuestion: {question}\n- A: {choiceA}\n- B: {choiceB}\n- C: " \
                         "{choiceC}\n- D: {choiceD}\nAnswer: "
        return self.unobserved_template(item, input_template)

    def unobserved6(self, item):
        input_template = "Now you are an expert with vast knowledge from {domain}. Output one of \"A\", \"B\", \"C\", or \"D\"" \
                         " to indicate your answer for the following question:\nQuestion: {question}\nThe options are:\nA: " \
                         "{choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\nAnswer:"
        return self.unobserved_template(item, input_template)

    def unobserved7(self, item):
        input_template = "Based on the knowledge from {domain}, given the following question, output the best choice from" \
                         " \"A\", \"B\", \"C\", and \"D\".\nQuestion: {question}\nChoices:\n- A: {choiceA}\n- B: " \
                         "{choiceB}\n- C: {choiceC}\n- D: {choiceD}\nThe correct choice is:"
        return self.unobserved_template(item, input_template)

    def unobserved8(self, item):
        input_template = "Please use your domain-specific knowledge about {domain} to answer the following questions:" \
                     "\nQuestion: {question}\nThe choices are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D:" \
                     " {choiceD}\n\nThe answer is "
        return self.unobserved_template(item, input_template)

    def unobserved9(self, item):
        input_template = "Employ your {domain} knowledge to tackle the given question. Choose the right answer as " \
                         "\"A\", \"B\", \"C\", or \"D\":Q: {question}\nCandidates:\nA: {choiceA}\nB: {choiceB}\nC: " \
                         "{choiceC}\nD: {choiceD}\n\nThe answer is: "
        return self.unobserved_template(item, input_template)

    def unobserved10(self, item):
        input_template = "You are an expert of {domain}. Please solve this question with an output of \"A\", \"B\", " \
                         "\"C\", or \"D\":Q: {question}\nOptions:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n" \
                         "- D: {choiceD}\n\nAnswer: "
        return self.unobserved_template(item, input_template)

    def unobserved1_fs(self, item):
        input_template_prefix = "Please answer the appropriate questions based on knowledge of the{domain}. Each " \
                                "question will give four options, please output the corresponding option (i.e. A, B, " \
                                "C or D) to represent the corresponding answer.\n\nHere are some examples:\n\n"

        example_template = "Question: {question}\nOptions:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                           "{choiceD}\n\nAnswer: {answer}\n\n"

        input_template = "Now, you should answer the following question:\n\nQuestion: {question}\nThe options are:" \
                         "\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved2_fs(self, item):
        input_template_prefix = "The following questions are from {domain}. Each question is associated with 4 " \
                                "candidate answers, marked with \"A\", \"B\", \"C\", and \"D\". Answer the questions " \
                                "via selecting the correct letter.\n\n"

        example_template = "Question{id}: {question}\n\nCandidate Answers:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\n\nThe right answer is {answer}\n\n"

        input_template = "Question: {question}\n\nCandidate Answers:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                         "{choiceD}\n\nThe right answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved3_fs(self, item):
        input_template_prefix = "This is a multiple choice test on {domain}. Please choose the correct answer" \
                     " among A, B, C and D by comparing them."

        example_template = "Question: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\n\nTherefore the answer is {answer}\n\n\n"

        input_template = "Question: {question}\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\n\nTherefore the answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved4_fs(self, item):
        input_template_prefix = "Please answer the question using your knowledge about {domain}. Output one of " \
                                "\"A\", \"B\", \"C\", or \"D\" to indicate your answer:"

        example_template = " A: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} {question} Answer: {answer}\n"

        input_template = " A: {choiceA} B: {choiceB} C: {choiceC} D: {choiceD} {question} Answer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved5_fs(self, item):
        input_template_prefix = "Please solve this question using your knowledge from {domain}. Output one of \"A\", " \
                                "\"B\", \"C\", or \"D\" to indicate your answer:\n"

        example_template = "Q: {question}\n\nThe options are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                           "{choiceD}\n\nAnswer: {answer}\n\n"

        input_template = "Q: {question}\n\nThe options are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                         "{choiceD}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved6_fs(self, item):
        input_template_prefix = "Now you are an expert with vast knowledge from {domain}. Output one of \"A\", \"B\"" \
                                ", \"C\", or \"D\" to indicate your answer for the following question:\n"

        example_template = "Question: {question}\n\nThe options are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                           "{choiceD}\n\nAnswer: {answer}\n\n"

        input_template = "Q: {question}\n\nThe options are:\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                         "{choiceD}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved7_fs(self, item):
        input_template_prefix = "We have the following examples\n"

        example_template = "Question: {question}\n\nChoices:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                           "{choiceD}\n\nThe correct choice is: {answer}\n\n"

        input_template = "Based on these examples and the knowledge from {domain}, given the following question, " \
                         "output the best choice from \"A\", \"B\", \"C\", and \"D\". Question: {question}\n\nChoices:" \
                         "\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: {choiceD}\n\nThe correct choice is: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved8_fs(self, item):
        input_template_prefix = "Please use your domain-specific knowledge about {domain} to answer the following " \
                                "questions:\n"

        example_template = "Question: {question}\nThe choices are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                           "{choiceD}\n\nThe answer is {answer}\n\n"

        input_template = "Question: {question}\nThe choices are:\n- A: {choiceA}\n- B: {choiceB}\n- C: {choiceC}\n- D: " \
                         "{choiceD}\n\nThe answer is "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved9_fs(self, item):
        input_template_prefix = "Utilize your expertise in {domain} to solve the following questions. Indicate your " \
                                "response for each question as either \"A\", \"B\", \"C\", or \"D\" (some examples " \
                                "are shown following):"

        example_template = "Q: {question}\nChoices:\n\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                           "{choiceD}\n\nAnswer: {answer}\n\n"

        input_template = "Q: {question}\nChoices:\n\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: " \
                         "{choiceD}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)

    def unobserved10_fs(self, item):
        input_template_prefix = "Next you will follow a same structure to answer a series of questions. You are an " \
                                "expert of {domain}. "

        example_template = "{id}. Please solve this question with an output of \"A\", \"B\", \"C\", or \"D\":\nQ: " \
                           "{question}\nOptions:\n\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\n" \
                           "Answer: {answer}\n\n"

        input_template = "Please solve this question with an output of \"A\", \"B\", \"C\", or \"D\":\nQ: " \
                         "{question}\nOptions:\n\nA: {choiceA}\nB: {choiceB}\nC: {choiceC}\nD: {choiceD}\n\nAnswer: "
        return self.unobserved_template_few_shot(item, input_template_prefix, input_template, example_template)
