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
    preprocessor = EpistemicReasoningPreprocessor(instruction, examples, eval_by_logits, input_dir)
    preprocess = preprocessor.processor

    test_set = test_set.map(preprocess, remove_columns=["question", "options", "answer"], num_proc=1)
    return test_set


class EpistemicReasoningPreprocessor(Preprocessor):

    def __init__(self, instruction, examples, eval_by_logits, input_dir):
        super(EpistemicReasoningPreprocessor, self).__init__(instruction, examples, eval_by_logits, input_dir)

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
        self.instr2preprocessor["BBL/Incorrect/6"] = self.incorrect_6
        self.instr2preprocessor["BBL/Incorrect/7"] = self.incorrect_7
        self.instr2preprocessor["BBL/Incorrect/8"] = self.incorrect_8
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
        question, answer, _ = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text = "Classify whether two sentences have entailment relation. Output \"yes\" if they have entailment relation; output \"no\" if they do not have entailment relation.\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer: ".format(sentence1=sentence1, sentence2=sentence2)
        answer = "yes" if answer == "entailment" else "no"
        label_space = ["yes", "no"]
        return_dict = {"input_text": input_text, "output_text": answer, "label_space": label_space}
        return return_dict
    
    def unobserved2(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text = "What is the relation between the given two sentences? Choose one of \'entailment\' and \'non-entailment\'.\n\nSentence1: {sentence1}\nSentence2: {sentence2}\nRelation: ".format(sentence1=sentence1, sentence2=sentence2)
        return_dict = {"input_text": input_text, "output_text": answer, "label_space": options}
        return return_dict
    
    def unobserved3(self, item):
        question, answer, _ = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text = "Premise: {sentence1}\nIs the truthfulness of the premise entail the following hypothesis?\nHypothesis: {sentence2}.".format(sentence1=sentence1, sentence2=sentence2)
        answer = "yes" if answer == "entailment" else "no"
        label_space = ["yes", "no"]
        return_dict = {"input_text": input_text, "output_text": answer, "label_space": label_space}
        return return_dict
    
    def unobserved4(self, item):
        question, answer, _ = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text = "Given the premise \'{sentence1}\', can you infer that \'{sentence2}\'? Answer \'Yes\' or \'No\' as your answer. Answer: ".format(sentence1=sentence1, sentence2=sentence2)
        answer = "Yes" if answer == "entailment" else "No"
        label_space = ["Yes", "No"]
        return_dict = {"input_text": input_text, "output_text": answer, "label_space": label_space}
        return return_dict
    
    def unobserved5(self, item):
        question, answer, _ = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text = "I think \"{sentence1}\" entails \"{sentence2}\".\n\nAm I right?".format(sentence1=sentence1, sentence2=sentence2)
        answer = "Yes" if answer == "entailment" else "No"
        label_space = ["Yes", "No"]
        return_dict = {"input_text": input_text, "output_text": answer, "label_space": label_space}
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
        text, answer, options = item["question"], item["answer"], item["options"]
        intput_text, output_text, label_space = self.sentiment140_6(text, answer, options)
        return_dict = {"input_text": intput_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def incorrect_5(self, item):
        return self.niv2_template(item, self.niv2_1422_mathqa_physics, self.niv2_zs_template_10)
    
    def incorrect_6(self, item):
        return self.niv2_template(item, self.niv2_562_language_identification, self.niv2_zs_template_10)
    
    def incorrect_7(self, item):
        return self.niv2_template(item, self.niv2_1193_course_classification, self.niv2_zs_template_10)
    
    def incorrect_8(self, item):
        question, answer, _ = item["question"], item["answer"], item["options"]
        premise, hypothesis = question.split(" Hypothesis: ")
        hypothesis = " Hypothesis: " + hypothesis
        answer = "Yes" if answer == "entailment" else "No"
        processed_item = {
            "paragraph": premise,
            "question": hypothesis,
            "correct_answer": "entailment",
            "output_text": answer
        }
        return self.niv2_template(processed_item, self.niv2_56_multirc_classification, self.niv2_zs_template_10)
    
    def correct_1(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_1(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_2(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_2(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_3(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_3(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_4(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_4(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_5(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_5(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_6(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_6(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_7(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_7(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def correct_8(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_8(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_1(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_1_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_2(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_2_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_3(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_3_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_4(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_4_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_5(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_5_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_6(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_6_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_7(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_7_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def negation_8(self, item):
        question, answer, options = item["question"], item["answer"], item["options"]
        sentence1, sentence2 = question.split(" Hypothesis: ")
        sentence1 = sentence1.replace("Premise: ", "")
        input_text, output_text, label_space = self.rte_8_negation(sentence1, sentence2, answer, options)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    


    

    