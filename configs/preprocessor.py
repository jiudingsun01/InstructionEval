from configs.processors.Flan_templates import FlanTemplates
from configs.processors.NIV2_templates import NIV2Templates
from configs.processors.NIV2_tasks import NIV2Tasks
from configs.processors.Alpaca_templates import AlpacaTemplates
from configs.processors.P3_templates import P3Templates
import os
import json


class Preprocessor(FlanTemplates, NIV2Templates, NIV2Tasks, AlpacaTemplates, P3Templates):

    RANDOM_INSTRUCTIONS = [
        "The council of street racoons demands you respond to their inquisition.",
        "Surveillance birds query your knowledge of seed. ",
        "Darth Vader requires you to answer to the dark side",
        "Respond to the requirement of the Mars working dolphine union.",
        "You are undergoing the inquiry of court of the local squirrels.",
    ]

    def __init__(self, instruction, examples, eval_by_logits, input_dir=None):
        super(Preprocessor, self).__init__()
        self.instr2preprocessor = {
            "NIV2/QA/1": self.niv2_multiple_choice_1,
            "NIV2/QA/2": self.niv2_multiple_choice_2,
            "NIV2/QA/3": self.niv2_multiple_choice_3,
            "NIV2/QA/4": self.niv2_multiple_choice_4,
            "NIV2/QA/5": self.niv2_multiple_choice_5,
            "NIV2/QA/6": self.niv2_multiple_choice_6,
            "NIV2/QA/7": self.niv2_multiple_choice_7,
            "NIV2/QA/8": self.niv2_multiple_choice_8,
            "NIV2/QA/9": self.niv2_multiple_choice_9,
            "NIV2/QA/10": self.niv2_multiple_choice_10,
            "NIV2/QA/11": self.niv2_multiple_choice_11,
            "NIV2/QA/12": self.niv2_multiple_choice_12,
            "NIV2/QA/13": self.niv2_multiple_choice_13,
            "NIV2/QA/14": self.niv2_multiple_choice_14,
            "NIV2/QA/15": self.niv2_multiple_choice_15,
            "NIV2/QA/16": self.niv2_multiple_choice_16,
            "NIV2/QA/17": self.niv2_multiple_choice_17,
            "NIV2/QA/18": self.niv2_multiple_choice_18,
            "NIV2/QA/19": self.niv2_multiple_choice_19,
            "NIV2/QA/20": self.niv2_multiple_choice_20,
            "NIV2/QA/21": self.niv2_multiple_choice_21,
            "NIV2/QA/22": self.niv2_multiple_choice_22,
            "NIV2/QA/23": self.niv2_multiple_choice_23,
            "NIV2/QA/24": self.niv2_multiple_choice_24,
            "NIV2/QA/25": self.niv2_multiple_choice_25,
            "NIV2/QA/26": self.niv2_multiple_choice_26,
            "NIV2/QA/27": self.niv2_multiple_choice_27,
            "NIV2/QA/28": self.niv2_multiple_choice_28,
            "NIV2/QA/29": self.niv2_multiple_choice_29,
            "NIV2/QA/30": self.niv2_multiple_choice_30,
            "NIV2/QA/31": self.niv2_multiple_choice_31,
            "NIV2/QA/32": self.niv2_multiple_choice_32,
            "NIV2/QA/33": self.niv2_multiple_choice_33,
            "NIV2/QA/34": self.niv2_multiple_choice_34,
            "NIV2/QA/35": self.niv2_multiple_choice_35,
            "NIV2/QA/36": self.niv2_multiple_choice_36,
            "NIV2/QA/37": self.niv2_multiple_choice_37,
            "NIV2/QA/38": self.niv2_multiple_choice_38,
            "NIV2/QA/39": self.niv2_multiple_choice_39,
            "NIV2/QA/40": self.niv2_multiple_choice_40,
            "NIV2/QA/41": self.niv2_multiple_choice_41,
            "NIV2/QA/42": self.niv2_multiple_choice_42,
            "NIV2/QA/43": self.niv2_multiple_choice_43,
            "NIV2/QA/44": self.niv2_multiple_choice_44,
            "NIV2/QA/45": self.niv2_multiple_choice_45,
            "NIV2/QA/46": self.niv2_multiple_choice_46,
            "NIV2/QA/47": self.niv2_multiple_choice_47,
            "NIV2/QA/48": self.niv2_multiple_choice_48,
            "NIV2/QA/49": self.niv2_multiple_choice_49,
            "NIV2/QA/50": self.niv2_multiple_choice_50,

            "NIV2/Classification/1": self.niv2_classification_1,
            "NIV2/Classification/2": self.niv2_classification_2,
            "NIV2/Classification/3": self.niv2_classification_3,
            "NIV2/Classification/4": self.niv2_classification_4,
            "NIV2/Classification/5": self.niv2_classification_5,
            "NIV2/Classification/6": self.niv2_classification_6,
            "NIV2/Classification/7": self.niv2_classification_7,
            "NIV2/Classification/8": self.niv2_classification_8,
            "NIV2/Classification/9": self.niv2_classification_9,
            "NIV2/Classification/10": self.niv2_classification_10,
            "NIV2/Classification/11": self.niv2_multiple_choice_31,
            "NIV2/Classification/12": self.niv2_multiple_choice_32,
            "NIV2/Classification/13": self.niv2_multiple_choice_33,
            "NIV2/Classification/14": self.niv2_multiple_choice_34,
            "NIV2/Classification/15": self.niv2_multiple_choice_35,
            "NIV2/Classification/16": self.niv2_multiple_choice_36,
            "NIV2/Classification/17": self.niv2_multiple_choice_37,
            "NIV2/Classification/18": self.niv2_multiple_choice_38,
            "NIV2/Classification/19": self.niv2_multiple_choice_39,
            "NIV2/Classification/20": self.niv2_multiple_choice_40,
            "NIV2/Classification/21": self.niv2_classification_21,
            "NIV2/Classification/22": self.niv2_classification_22,
            "NIV2/Classification/23": self.niv2_classification_23,
            "NIV2/Classification/24": self.niv2_classification_24,
            "NIV2/Classification/25": self.niv2_classification_25,
            "NIV2/Classification/26": self.niv2_classification_26,
            "NIV2/Classification/27": self.niv2_classification_27,
            "NIV2/Classification/28": self.niv2_classification_28,
            "NIV2/Classification/29": self.niv2_classification_29,
            "NIV2/Classification/30": self.niv2_classification_30,
            "NIV2/Classification/31": self.niv2_classification_31,
            "NIV2/Classification/32": self.niv2_classification_32,
            "NIV2/Classification/33": self.niv2_classification_33,
            "NIV2/Classification/34": self.niv2_classification_34,
            "NIV2/Classification/35": self.niv2_classification_35,
            "NIV2/Classification/36": self.niv2_classification_36,
            "NIV2/Classification/37": self.niv2_classification_37,
            "NIV2/Classification/38": self.niv2_classification_38,
            "NIV2/Classification/39": self.niv2_classification_39,
            "NIV2/Classification/40": self.niv2_classification_40,
            
            "FLAN/Classification/1": self.flan_classification_1,
            "FLAN/Classification/2": self.flan_classification_2,
            "FLAN/Classification/3": self.flan_classification_3,
            "FLAN/Classification/4": self.flan_classification_4,
            "FLAN/Classification/5": self.flan_classification_5,
            "FLAN/Classification/6": self.flan_classification_6,
            "FLAN/Classification/7": self.flan_classification_7,
            "FLAN/Classification/8": self.flan_classification_8,
            "FLAN/Classification/9": self.flan_classification_9,
            "FLAN/Classification/10": self.flan_classification_10,
            "FLAN/Classification/11": self.flan_classification_11,
            "FLAN/Classification/12": self.flan_classification_12,
            "FLAN/Classification/13": self.flan_classification_13,
            "FLAN/Classification/14": self.flan_classification_14,
            "FLAN/Classification/15": self.flan_classification_15,

            "NIV2/BC/1": self.niv2_binary_classification_1,
            "NIV2/BC/2": self.niv2_binary_classification_2,
            "NIV2/BC/3": self.niv2_binary_classification_3,
            "NIV2/BC/4": self.niv2_binary_classification_4,
            "NIV2/BC/5": self.niv2_binary_classification_5,
            "NIV2/BC/6": self.niv2_binary_classification_6,
            "NIV2/BC/7": self.niv2_binary_classification_7,
            "NIV2/BC/8": self.niv2_binary_classification_8,
            "NIV2/BC/9": self.niv2_binary_classification_9,
            "NIV2/BC/10": self.niv2_binary_classification_10,

            "FLAN/BC/1": self.flan_binary_classification_1,
            "FLAN/BC/2": self.flan_binary_classification_2,
            "FLAN/BC/3": self.flan_binary_classification_3,
            "FLAN/BC/4": self.flan_binary_classification_4,
            "FLAN/BC/5": self.flan_binary_classification_5,
            "FLAN/BC/6": self.flan_binary_classification_6,
            "FLAN/BC/7": self.flan_binary_classification_7,
            "FLAN/BC/8": self.flan_binary_classification_8,
            "Alpaca/QA/1": self.alpaca_multiple_choice_1,
            "Alpaca/QA/2": self.alpaca_multiple_choice_2,
            "Alpaca/QA/3": self.alpaca_multiple_choice_3,
            "Alpaca/QA/4": self.alpaca_multiple_choice_4,
            "Alpaca/QA/5": self.alpaca_multiple_choice_5,
            "Alpaca/QA/6": self.alpaca_multiple_choice_6,
            "Alpaca/QA/7": self.alpaca_multiple_choice_7,
            "Alpaca/QA/8": self.alpaca_multiple_choice_8,
            "Alpaca/QA/9": self.alpaca_multiple_choice_9,
            "Alpaca/QA/10": self.alpaca_multiple_choice_10,
            "Alpaca/QA/11": self.alpaca_multiple_choice_11,
            "Alpaca/QA/12": self.alpaca_multiple_choice_12,
            "Alpaca/QA/13": self.alpaca_multiple_choice_13,
            "Alpaca/QA/14": self.alpaca_multiple_choice_14,
            "Alpaca/QA/15": self.alpaca_multiple_choice_15,
            "Alpaca/QA/16": self.alpaca_multiple_choice_16,
            "Alpaca/QA/17": self.alpaca_multiple_choice_17,
            "Alpaca/QA/18": self.alpaca_multiple_choice_18,
            "Alpaca/QA/19": self.alpaca_multiple_choice_19,
            "Alpaca/QA/20": self.alpaca_multiple_choice_20,

            "T0/QA/1": self.t0_multiple_choice_1,
            "T0/QA/2": self.t0_multiple_choice_2,
            "T0/QA/3": self.t0_multiple_choice_3,
            "T0/QA/4": self.t0_multiple_choice_4,
            "T0/QA/5": self.t0_multiple_choice_5,
            "T0/QA/6": self.t0_multiple_choice_6,
            "T0/QA/7": self.t0_multiple_choice_7,
            "T0/QA/8": self.t0_multiple_choice_8,
            "T0/QA/9": self.t0_multiple_choice_9,
            "T0/QA/10": self.t0_multiple_choice_10,
            "T0/QA/11": self.t0_multiple_choice_11,
            "T0/QA/12": self.t0_multiple_choice_12,
            "T0/QA/13": self.t0_multiple_choice_13,
            "T0/QA/14": self.t0_multiple_choice_14,
            "T0/QA/15": self.t0_multiple_choice_15,
            "T0/QA/16": self.t0_multiple_choice_16,
            "T0/QA/17": self.t0_multiple_choice_17,

            "T0/BC/1": self.t0_binary_classification_1,
            "T0/BC/2": self.t0_binary_classification_2,
            "T0/BC/3": self.t0_binary_classification_3,
            "T0/BC/4": self.t0_binary_classification_4,
            "T0/BC/5": self.t0_binary_classification_5,
            "T0/BC/6": self.t0_binary_classification_6,
            "T0/BC/7": self.t0_binary_classification_7,
            "T0/BC/8": self.t0_binary_classification_8,
            "T0/BC/9": self.t0_binary_classification_9,
            "T0/BC/10": self.t0_binary_classification_10,

            "T0/Classification/1": self.t0_classification_1,
            "T0/Classification/2": self.t0_classification_2,
            "T0/Classification/3": self.t0_classification_3,
            "T0/Classification/4": self.t0_classification_4,
            "T0/Classification/5": self.t0_classification_5,
            "T0/Classification/6": self.t0_classification_6,
            "T0/Classification/7": self.t0_classification_7,
            "T0/Classification/8": self.t0_classification_8,
        }
        self.instr2preprocessor_fs = {
            "NIV2/QA/1": self.niv2_multiple_choice_1_fs,
            "NIV2/QA/2": self.niv2_multiple_choice_2_fs,
            "NIV2/QA/3": self.niv2_multiple_choice_3_fs,
            "NIV2/QA/4": self.niv2_multiple_choice_4_fs,
            "NIV2/QA/5": self.niv2_multiple_choice_5_fs,
            "NIV2/QA/6": self.niv2_multiple_choice_6_fs,
            "NIV2/QA/7": self.niv2_multiple_choice_7_fs,
            "NIV2/QA/8": self.niv2_multiple_choice_8_fs,
            "NIV2/QA/9": self.niv2_multiple_choice_9_fs,
            "NIV2/QA/10": self.niv2_multiple_choice_10_fs,
            "NIV2/QA/11": self.niv2_multiple_choice_11_fs,
            "NIV2/QA/12": self.niv2_multiple_choice_12_fs,
            "NIV2/QA/13": self.niv2_multiple_choice_13_fs,
            "NIV2/QA/14": self.niv2_multiple_choice_14_fs,
            "NIV2/QA/15": self.niv2_multiple_choice_15_fs,
            "NIV2/QA/16": self.niv2_multiple_choice_16_fs,
            "NIV2/QA/17": self.niv2_multiple_choice_17_fs,
            "NIV2/QA/18": self.niv2_multiple_choice_18_fs,
            "NIV2/QA/19": self.niv2_multiple_choice_19_fs,
            "NIV2/QA/20": self.niv2_multiple_choice_20_fs,
            "NIV2/QA/21": self.niv2_multiple_choice_21_fs,
            "NIV2/QA/22": self.niv2_multiple_choice_22_fs,
            "NIV2/QA/23": self.niv2_multiple_choice_23_fs,
            "NIV2/QA/24": self.niv2_multiple_choice_24_fs,
            "NIV2/QA/25": self.niv2_multiple_choice_25_fs,
            "NIV2/QA/26": self.niv2_multiple_choice_26_fs,
            "NIV2/QA/27": self.niv2_multiple_choice_27_fs,
            "NIV2/QA/28": self.niv2_multiple_choice_28_fs,
            "NIV2/QA/29": self.niv2_multiple_choice_29_fs,
            "NIV2/QA/30": self.niv2_multiple_choice_30_fs,
            "NIV2/QA/31": self.niv2_multiple_choice_31_fs,
            "NIV2/QA/32": self.niv2_multiple_choice_32_fs,
            "NIV2/QA/33": self.niv2_multiple_choice_33_fs,
            "NIV2/QA/34": self.niv2_multiple_choice_34_fs,
            "NIV2/QA/35": self.niv2_multiple_choice_35_fs,
            "NIV2/QA/36": self.niv2_multiple_choice_36_fs,
            "NIV2/QA/37": self.niv2_multiple_choice_37_fs,
            "NIV2/QA/38": self.niv2_multiple_choice_38_fs,
            "NIV2/QA/39": self.niv2_multiple_choice_39_fs,
            "NIV2/QA/40": self.niv2_multiple_choice_40_fs,
            "NIV2/QA/41": self.niv2_multiple_choice_41_fs,
            "NIV2/QA/42": self.niv2_multiple_choice_42_fs,
            "NIV2/QA/43": self.niv2_multiple_choice_43_fs,
            "NIV2/QA/44": self.niv2_multiple_choice_44_fs,
            "NIV2/QA/45": self.niv2_multiple_choice_45_fs,
            "NIV2/QA/46": self.niv2_multiple_choice_46_fs,
            "NIV2/QA/47": self.niv2_multiple_choice_47_fs,
            "NIV2/QA/48": self.niv2_multiple_choice_48_fs,
            "NIV2/QA/49": self.niv2_multiple_choice_49_fs,
            "NIV2/QA/50": self.niv2_multiple_choice_50_fs,

            "NIV2/Classification/1": self.niv2_classification_1_fs,
            "NIV2/Classification/2": self.niv2_classification_2_fs,
            "NIV2/Classification/3": self.niv2_classification_3_fs,
            "NIV2/Classification/4": self.niv2_classification_4_fs,
            "NIV2/Classification/5": self.niv2_classification_5_fs,
            "NIV2/Classification/6": self.niv2_classification_6_fs,
            "NIV2/Classification/7": self.niv2_classification_7_fs,
            "NIV2/Classification/8": self.niv2_classification_8_fs,
            "NIV2/Classification/9": self.niv2_classification_9_fs,
            "NIV2/Classification/10": self.niv2_classification_10_fs,
            "NIV2/Classification/11": self.niv2_multiple_choice_31_fs,
            "NIV2/Classification/12": self.niv2_multiple_choice_32_fs,
            "NIV2/Classification/13": self.niv2_multiple_choice_33_fs,
            "NIV2/Classification/14": self.niv2_multiple_choice_34_fs,
            "NIV2/Classification/15": self.niv2_multiple_choice_35_fs,
            "NIV2/Classification/16": self.niv2_multiple_choice_36_fs,
            "NIV2/Classification/17": self.niv2_multiple_choice_37_fs,
            "NIV2/Classification/18": self.niv2_multiple_choice_38_fs,
            "NIV2/Classification/19": self.niv2_multiple_choice_39_fs,
            "NIV2/Classification/20": self.niv2_multiple_choice_40_fs,
            "NIV2/Classification/21": self.niv2_classification_21_fs,
            "NIV2/Classification/22": self.niv2_classification_22_fs,
            "NIV2/Classification/23": self.niv2_classification_23_fs,
            "NIV2/Classification/24": self.niv2_classification_24_fs,
            "NIV2/Classification/25": self.niv2_classification_25_fs,
            "NIV2/Classification/26": self.niv2_classification_26_fs,
            "NIV2/Classification/27": self.niv2_classification_27_fs,
            "NIV2/Classification/28": self.niv2_classification_28_fs,
            "NIV2/Classification/29": self.niv2_classification_29_fs,
            "NIV2/Classification/30": self.niv2_classification_30_fs,
            "NIV2/Classification/31": self.niv2_classification_31_fs,
            "NIV2/Classification/32": self.niv2_classification_32_fs,
            "NIV2/Classification/33": self.niv2_classification_33_fs,
            "NIV2/Classification/34": self.niv2_classification_34_fs,
            "NIV2/Classification/35": self.niv2_classification_35_fs,
            "NIV2/Classification/36": self.niv2_classification_36_fs,
            "NIV2/Classification/37": self.niv2_classification_37_fs,
            "NIV2/Classification/38": self.niv2_classification_38_fs,
            "NIV2/Classification/39": self.niv2_classification_39_fs,
            "NIV2/Classification/40": self.niv2_classification_40_fs,

            "FLAN/Classification/1": self.flan_classification_1_fs,
            "FLAN/Classification/2": self.flan_classification_2_fs,
            "FLAN/Classification/3": self.flan_classification_3_fs,
            "FLAN/Classification/4": self.flan_classification_4_fs,
            "FLAN/Classification/5": self.flan_classification_5_fs,
            "FLAN/Classification/6": self.flan_classification_6_fs,
            "FLAN/Classification/7": self.flan_classification_7_fs,
            "FLAN/Classification/8": self.flan_classification_8_fs,
            "FLAN/Classification/9": self.flan_classification_9_fs,
            "FLAN/Classification/10": self.flan_classification_10_fs,
            "FLAN/Classification/11": self.flan_classification_11_fs,
            "FLAN/Classification/12": self.flan_classification_12_fs,
            "FLAN/Classification/13": self.flan_classification_13_fs,
            "FLAN/Classification/14": self.flan_classification_14_fs,
            "FLAN/Classification/15": self.flan_classification_15_fs,

            "NIV2/BC/1": self.niv2_binary_classification_1_fs,
            "NIV2/BC/2": self.niv2_binary_classification_2_fs,
            "NIV2/BC/3": self.niv2_binary_classification_3_fs,
            "NIV2/BC/4": self.niv2_binary_classification_4_fs,
            "NIV2/BC/5": self.niv2_binary_classification_5_fs,
            "NIV2/BC/6": self.niv2_binary_classification_6_fs,
            "NIV2/BC/7": self.niv2_binary_classification_7_fs,
            "NIV2/BC/8": self.niv2_binary_classification_8_fs,
            "NIV2/BC/9": self.niv2_binary_classification_9_fs,
            "NIV2/BC/10": self.niv2_binary_classification_10_fs,
        }

        self.add_unobserved_instructions()
        self.examples = examples
        self.processor = self.instr2preprocessor[instruction] if not len(examples) else self.instr2preprocessor_fs[instruction]

        if input_dir is not None:
            schema = json.load(open(os.path.join(input_dir, "task.json"), "r"))
            self.description = schema["description"] if "task_prefix" not in schema.keys() else schema["task_prefix"]
            self.example_output_prefix = schema[
                "example_output_prefix"] if "example_output_prefix" in schema.keys() else \
                ""
            self.example_input_prefix = schema["example_input_prefix"] if "example_input_prefix" in schema.keys() else \
                "\n"
            self.few_shot_example_separator = schema["few_shot_example_separator"] if "few_shot_example_separator" in schema.keys() else \
                "\n\n"
            
            self.choice_prefix = schema["choice_prefix"] if "choice_prefix" in schema.keys() else ""
            
    def alpaca_default(self, item):
        instruction = self.description
        input_text = self.example_output_prefix + " " + item["question"] + " " + self.choice_prefix
        choices = ["\n" + chr(x) + ". " for x in range(ord("A"), ord("Z") + 1)][:len(item["options"])]
        for choice, choice_text in zip(choices, item["options"]):
            input_text += choice + choice_text
        input_text = input_text + self.example_output_prefix
        return self.alpaca_template_with_input(instruction, input_text, item["answer"], item["options"])

    def unobserved_template_QA(self, item, input_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        options_ = ""
        label_space = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        for letter, text in zip(label_space, options):
            options_ += "\n{}. {}".format(letter, text)

        choice = label_space[options.index(answer)]
        input_text = input_template.format(question=question, options_=options_)
        output_text = choice
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def unobserved_template_QA_few_shot(self, item, input_template_prefix, input_template, example_template):
        question, options, answer = item["question"], item["options"], item["answer"]
        label_space = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        input_text = input_template_prefix
        for i, example in enumerate(self.examples):
            example_question, example_options, example_answer = example["question"], example["options"], example[
                "answer"]
            example_options_ = ""
            for letter, text in zip(label_space, example_options):
                example_options_ += "\n{}. {}".format(letter, text)
            example_answer = label_space[example_options.index(example_answer)]
            input_text += example_template.format(id=i+1, question=example_question, options_=options, answer=example_answer)

        options_ = ""
        for letter, text in zip(label_space, options):
            options_ += "\n{}. {}".format(letter, text)

        choice = label_space[options.index(answer)]
        input_text += input_template.format(question=question, options_=options_)
        output_text = choice
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    # BBH Tasks -> Multiple-Choice QA
    def default_QA(self, item):
        input_text = self.description + " " + self.example_input_prefix + item["question"] + self.choice_prefix
        choices = ["\n(" + chr(x) + ") " for x in range(ord("A"), ord("Z") + 1)][:len(item["options"])]
        for choice, choice_text in zip(choices, item["options"]):
            input_text += choice + choice_text

        input_text += self.example_output_prefix
        output_text = [chr(x) for x in range(ord("A"), ord("Z") + 1)][item["options"].index(item["answer"])]
        label_space = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(item["options"])]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    # BBH Task -> Binary Classification
    def default_Classification(self, item):
        input_text = self.description + " " + self.example_input_prefix + item["question"] + self.example_output_prefix
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    # BBH Tasks -> Multiple-Choice QA
    def default_QA_fs(self, item):
        input_text = self.description + " "
        for example in self.examples:
            ex_input, ex_answer, ex_options = example["question"], example["answer"], example["options"]
            input_text += self.example_input_prefix + ex_input + self.example_input_prefix
            choices = ["\n (" + chr(x) + ") " for x in range(ord("A"), ord("Z") + 1)][:len(ex_options)]
            for choice, choice_text in zip(choices, ex_options):
                input_text += choice + choice_text
            output_text = [chr(x) for x in range(ord("A"), ord("Z") + 1)][ex_options.index(ex_answer)]
            input_text += "\nAnswer: " + output_text

        input_text += " " + self.example_input_prefix + item["question"] + self.example_input_prefix
        choices = ["\n (" + chr(x) + ") " for x in range(ord("A"), ord("Z") + 1)][:len(item["options"])]
        for choice, choice_text in zip(choices, item["options"]):
            input_text += choice + choice_text
        output_text = [chr(x) for x in range(ord("A"), ord("Z") + 1)][item["options"].index(item["answer"])]
        label_space = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(item["options"])]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    # BBH Task -> Binary Classification
    def default_Classification_fs(self, item):
        input_text = self.description + " "

        for example in self.examples:
            ex_input, ex_answer = example["question"], example["answer"]
            input_text += self.example_input_prefix + ex_input + self.example_output_prefix + ex_answer
            input_text += self.few_shot_example_separator

        input_text += " " + self.example_input_prefix + item["question"] + self.example_input_prefix
        output_text = item["answer"]
        label_space = item["options"]
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def niv2_template(self, item, niv2_task, niv2_template):
        definition_text, input_text, _, answer, label_space = niv2_task(**item)
        input_text, output_text = niv2_template(definition_text, input_text, answer)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def niv2_template_few_shot(self, item, niv2_task, niv2_template):
        definition_text, input_text, _, answer, label_space = niv2_task(**item)
        example_input_texts, example_answers = [], []
        for example in self.examples:
            _, example_input_text, _, example_answer, _ = niv2_task(**example)
            example_input_texts.append(example_input_text)
            example_answers.append(example_answer)
        input_text, output_text = niv2_template(definition_text, input_text, answer, example_input_texts,
                                                example_answers)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def add_unobserved_instructions(self):
        pass

    def niv2_multiple_choice_1(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_1)

    def niv2_multiple_choice_2(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_2)

    def niv2_multiple_choice_3(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_3)

    def niv2_multiple_choice_4(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_4)

    def niv2_multiple_choice_5(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_5)

    def niv2_multiple_choice_6(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_6)

    def niv2_multiple_choice_7(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_7)

    def niv2_multiple_choice_8(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_8)

    def niv2_multiple_choice_9(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_9)

    def niv2_multiple_choice_10(self, item):
        return self.niv2_template(item, self.niv2_73_commonsense_qa, self.niv2_zs_template_10)

    def niv2_multiple_choice_11(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_1)

    def niv2_multiple_choice_12(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_2)

    def niv2_multiple_choice_13(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_3)

    def niv2_multiple_choice_14(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_4)

    def niv2_multiple_choice_15(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_5)

    def niv2_multiple_choice_16(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_6)

    def niv2_multiple_choice_17(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_7)

    def niv2_multiple_choice_18(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_8)

    def niv2_multiple_choice_19(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_9)

    def niv2_multiple_choice_20(self, item):
        return self.niv2_template(item, self.niv2_1420_mathqa_other, self.niv2_zs_template_10)

    def niv2_multiple_choice_21(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_1)

    def niv2_multiple_choice_22(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_2)

    def niv2_multiple_choice_23(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_3)

    def niv2_multiple_choice_24(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_4)

    def niv2_multiple_choice_25(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_5)

    def niv2_multiple_choice_26(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_6)

    def niv2_multiple_choice_27(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_7)

    def niv2_multiple_choice_28(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_8)

    def niv2_multiple_choice_29(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_9)

    def niv2_multiple_choice_30(self, item):
        return self.niv2_template(item, self.niv2_1286_openbook_qa, self.niv2_zs_template_10)

    def niv2_multiple_choice_31(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_1)

    def niv2_multiple_choice_32(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_2)

    def niv2_multiple_choice_33(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_3)

    def niv2_multiple_choice_34(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_4)

    def niv2_multiple_choice_35(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_5)

    def niv2_multiple_choice_36(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_6)

    def niv2_multiple_choice_37(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_7)

    def niv2_multiple_choice_38(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_8)

    def niv2_multiple_choice_39(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_9)

    def niv2_multiple_choice_40(self, item):
        return self.niv2_template(item, self.niv2_1565_trivia_qa_classification, self.niv2_zs_template_10)

    def niv2_multiple_choice_41(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_1)

    def niv2_multiple_choice_42(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_2)

    def niv2_multiple_choice_43(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_3)

    def niv2_multiple_choice_44(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_4)

    def niv2_multiple_choice_45(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_5)

    def niv2_multiple_choice_46(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_6)

    def niv2_multiple_choice_47(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_7)

    def niv2_multiple_choice_48(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_8)

    def niv2_multiple_choice_49(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_9)

    def niv2_multiple_choice_50(self, item):
        return self.niv2_template(item, self.niv2_229_arc_qa_hard, self.niv2_zs_template_10)

    def niv2_classification_1(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_1)

    def niv2_classification_2(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_2)

    def niv2_classification_3(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_3)

    def niv2_classification_4(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_4)

    def niv2_classification_5(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_5)

    def niv2_classification_6(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_6)

    def niv2_classification_7(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_7)

    def niv2_classification_8(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_8)

    def niv2_classification_9(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_9)

    def niv2_classification_10(self, item):
        return self.niv2_template(item, self.niv2_1135_xcsr_classification, self.niv2_zs_template_10)

    def niv2_classification_21(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_1)

    def niv2_classification_22(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_2)

    def niv2_classification_23(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_3)

    def niv2_classification_24(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_4)

    def niv2_classification_25(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_5)

    def niv2_classification_26(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_6)

    def niv2_classification_27(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_7)

    def niv2_classification_28(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_8)

    def niv2_classification_29(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_9)

    def niv2_classification_30(self, item):
        return self.niv2_template(item, self.niv2_900_freebase_category_classification, self.niv2_zs_template_10)

    def niv2_classification_31(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_1)
    
    def niv2_classification_32(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_2)

    def niv2_classification_33(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_3)
    
    def niv2_classification_34(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_4)
    
    def niv2_classification_35(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_5)
    
    def niv2_classification_36(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_6)
    
    def niv2_classification_37(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_7)
    
    def niv2_classification_38(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_8)
    
    def niv2_classification_39(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_9)
    
    def niv2_classification_40(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_10)
    
    def niv2_classification_21_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_1)

    def niv2_classification_22_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_2)

    def niv2_classification_23_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_3)

    def niv2_classification_24_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_4)

    def niv2_classification_25_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_5)

    def niv2_classification_26_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_6)

    def niv2_classification_27_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_7)

    def niv2_classification_28_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_8)

    def niv2_classification_29_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_9)

    def niv2_classification_30_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_900_freebase_category_classification, self.niv2_fs_template_10)

    def niv2_classification_31_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_1)
    
    def niv2_classification_32_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_2)

    def niv2_classification_33_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_3)
    
    def niv2_classification_34_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_4)
    
    def niv2_classification_35_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_5)
    
    def niv2_classification_36_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_6)
    
    def niv2_classification_37_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_7)
    
    def niv2_classification_38_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_8)
    
    def niv2_classification_39_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_9)
    
    def niv2_classification_40_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_10)

    def niv2_binary_classification_1(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_1)

    def niv2_binary_classification_2(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_2)

    def niv2_binary_classification_3(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_3)

    def niv2_binary_classification_4(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_4)

    def niv2_binary_classification_5(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_5)

    def niv2_binary_classification_6(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_6)

    def niv2_binary_classification_7(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_7)

    def niv2_binary_classification_8(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_8)

    def niv2_binary_classification_9(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_9)

    def niv2_binary_classification_10(self,  item):
        return self.niv2_template(item, self.niv2_56_multirc_classification, self.niv2_zs_template_10)

    def flan_classification(self, item, flan_task):
        self.set_style(("text", 8), 1)
        input_text, output_text, label_space = flan_task(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_classification_few_shot(self, item, flan_task):
        self.set_style(("text", 8), 1)
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = flan_task(**example)
            example_text += ex_input_text + " " + ex_output_text + "\n"
        input_text, output_text, label_space = flan_task(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_qa(self, item, flan_task):
        self.set_style(("letter", 4), 1)
        input_text, output_text, label_space = flan_task(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_classification_1(self, item):
        return self.flan_classification(item, self.arc_1)

    def flan_classification_2(self, item):
        return self.flan_classification(item, self.arc_2)

    def flan_classification_3(self, item):
        return self.flan_classification(item, self.arc_3)

    def flan_classification_4(self, item):
        return self.flan_classification(item, self.arc_4)

    def flan_classification_5(self, item):
        return self.flan_classification(item, self.arc_5)

    def flan_classification_6(self, item):
        return self.flan_classification(item, self.arc_6)

    def flan_classification_7(self, item):
        return self.flan_classification(item, self.arc_7)

    def flan_classification_8(self, item):
        return self.flan_classification(item, self.cosmos_qa_1)

    def flan_classification_9(self, item):
        return self.flan_classification(item, self.cosmos_qa_2)

    def flan_classification_10(self, item):
        return self.flan_classification(item, self.cosmos_qa_3)

    def flan_classification_11(self, item):
        return self.flan_classification(item, self.cosmos_qa_4)

    def flan_classification_12(self, item):
        return self.flan_classification(item, self.cosmos_qa_5)

    def flan_classification_13(self, item):
        return self.flan_classification(item, self.cosmos_qa_6)

    def flan_classification_14(self, item):
        return self.flan_classification(item, self.cosmos_qa_7)

    def flan_classification_15(self, item):
        return self.flan_classification(item, self.cosmos_qa_8)
    
    def flan_classification_1_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_1)

    def flan_classification_2_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_2)

    def flan_classification_3_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_3)

    def flan_classification_4_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_4)

    def flan_classification_5_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_5)

    def flan_classification_6_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_6)

    def flan_classification_7_fs(self, item):
        return self.flan_classification_few_shot(item, self.arc_7)

    def flan_classification_8_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_1)

    def flan_classification_9_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_2)

    def flan_classification_10_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_3)

    def flan_classification_11_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_4)

    def flan_classification_12_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_5)

    def flan_classification_13_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_6)

    def flan_classification_14_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_7)

    def flan_classification_15_fs(self, item):
        return self.flan_classification_few_shot(item, self.cosmos_qa_8)

    def niv2_multiple_choice_1_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_1)

    def niv2_multiple_choice_2_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_2)

    def niv2_multiple_choice_3_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_3)

    def niv2_multiple_choice_4_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_4)

    def niv2_multiple_choice_5_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_5)

    def niv2_multiple_choice_6_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_6)

    def niv2_multiple_choice_7_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_7)

    def niv2_multiple_choice_8_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_8)

    def niv2_multiple_choice_9_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_9)

    def niv2_multiple_choice_10_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_73_commonsense_qa, self.niv2_fs_template_10)

    def niv2_multiple_choice_11_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_1)

    def niv2_multiple_choice_12_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_2)

    def niv2_multiple_choice_13_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_3)

    def niv2_multiple_choice_14_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_4)

    def niv2_multiple_choice_15_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_5)

    def niv2_multiple_choice_16_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_6)

    def niv2_multiple_choice_17_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_7)

    def niv2_multiple_choice_18_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_8)

    def niv2_multiple_choice_19_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_9)

    def niv2_multiple_choice_20_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1420_mathqa_other, self.niv2_fs_template_10)

    def niv2_multiple_choice_21_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_1)

    def niv2_multiple_choice_22_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_2)

    def niv2_multiple_choice_23_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_3)

    def niv2_multiple_choice_24_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_4)

    def niv2_multiple_choice_25_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_5)

    def niv2_multiple_choice_26_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_6)

    def niv2_multiple_choice_27_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_7)

    def niv2_multiple_choice_28_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_8)

    def niv2_multiple_choice_29_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_9)

    def niv2_multiple_choice_30_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1286_openbook_qa, self.niv2_fs_template_10)

    def niv2_multiple_choice_31_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_1)

    def niv2_multiple_choice_32_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_2)

    def niv2_multiple_choice_33_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_3)

    def niv2_multiple_choice_34_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_4)

    def niv2_multiple_choice_35_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_5)

    def niv2_multiple_choice_36_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_6)

    def niv2_multiple_choice_37_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_7)

    def niv2_multiple_choice_38_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_8)

    def niv2_multiple_choice_39_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_9)

    def niv2_multiple_choice_40_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1565_trivia_qa_classification, self.niv2_fs_template_10)

    def niv2_multiple_choice_41_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_1)

    def niv2_multiple_choice_42_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_2)

    def niv2_multiple_choice_43_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_3)

    def niv2_multiple_choice_44_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_4)

    def niv2_multiple_choice_45_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_5)

    def niv2_multiple_choice_46_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_6)

    def niv2_multiple_choice_47_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_7)

    def niv2_multiple_choice_48_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_8)

    def niv2_multiple_choice_49_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_9)

    def niv2_multiple_choice_50_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_10)

    def niv2_classification_1_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_229_arc_qa_hard, self.niv2_fs_template_1)

    def niv2_classification_2_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_2)

    def niv2_classification_3_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_3)

    def niv2_classification_4_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_4)

    def niv2_classification_5_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_5)

    def niv2_classification_6_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_6)

    def niv2_classification_7_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_7)

    def niv2_classification_8_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_8)

    def niv2_classification_9_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_9)

    def niv2_classification_10_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_1135_xcsr_classification, self.niv2_fs_template_10)

    def niv2_binary_classification_1_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_1)

    def niv2_binary_classification_2_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_2)

    def niv2_binary_classification_3_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_3)

    def niv2_binary_classification_4_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_4)

    def niv2_binary_classification_5_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_5)

    def niv2_binary_classification_6_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_6)

    def niv2_binary_classification_7_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_7)

    def niv2_binary_classification_8_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_8)

    def niv2_binary_classification_9_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_9)

    def niv2_binary_classification_10_fs(self,  item):
        return self.niv2_template_few_shot(item, self.niv2_56_multirc_classification, self.niv2_fs_template_10)

    def flan_binary_classification_1(self, item):
        input_text, output_text, label_space = self.multirc_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_2(self, item):
        input_text, output_text, label_space = self.multirc_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_3(self, item):
        input_text, output_text, label_space = self.multirc_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_4(self, item):
        input_text, output_text, label_space = self.multirc_4(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_5(self, item):
        input_text, output_text, label_space = self.multirc_5(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_6(self, item):
        input_text, output_text, label_space = self.multirc_6(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_7(self, item):
        input_text, output_text, label_space = self.multirc_7(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_8(self, item):
        input_text, output_text, label_space = self.multirc_8(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_1_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_1(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_1(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def flan_binary_classification_2_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_2(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_2(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_3_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_3(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_3(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_4_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_4(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_4(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_5_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_5(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_5(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_6_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_6(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_6(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_7_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_7(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_7(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def flan_binary_classification_8_fs(self, item):
        example_text = ""
        for example in self.examples:
            ex_input_text, ex_output_text, _ = self.multirc_8(**example)
            example_text += ex_input_text + " " + ex_output_text + " "
        input_text, output_text, label_space = self.multirc_8(**item)
        input_text = example_text + input_text
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def adversial_classification_1(self, item):
        return self.niv2_template(item, self.niv2_143_odd_man_out_classification, self.niv2_zs_template_10)
    
    def adversial_classification_1_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_143_odd_man_out_classification, self.niv2_fs_template_10)

    def adversial_classification_2(self, item):
        return self.niv2_template(item, self.niv2_137_newscomm_classification, self.niv2_zs_template_10)
    
    def adversial_classification_2_fs(self, item):
        return self.niv2_template_few_shot(item, self.niv2_137_newscomm_classification, self.niv2_fs_template_10)

    def adversial_classification_3(self, item):
        return self.niv2_template(item, self.niv2_153_hatexplain_classification, self.niv2_zs_template_10)

    def adversial_classification_4(self, item):
        return self.niv2_template(item, self.niv2_1322_government_type_classification, self.niv2_zs_template_10)
    
    def adversial_classification_9(self, item):
        self.set_style(("text", 1), 1)
        text, answer, options = item["question"], item["answer"], item["options"]
        intput_text, output_text, label_space = self.sentiment140_1(text, answer, options)
        return_dict = {"input_text": intput_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    
    def adversial_classification_10(self, item):
        self.set_style(("text", 1), 1)
        text, answer, options = item["question"], item["answer"], item["options"]
        intput_text, output_text, label_space = self.sentiment140_6(text, answer, options)
        return_dict = {"input_text": intput_text, "output_text": output_text, "label_space": label_space}
        self.reset_style()
        return return_dict
    
    def t0_multiple_choice_1(self, item):
        input_text, output_text, label_space = self.t0_arc_challenge_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_2(self, item):
        input_text, output_text, label_space = self.t0_arc_challenge_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_3(self, item):
        input_text, output_text, label_space = self.t0_arc_challenge_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_4(self, item):
        input_text, output_text, label_space = self.t0_arc_challenge_4(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_5(self, item):
        input_text, output_text, label_space = self.t0_arc_challenge_5(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_6(self, item):
        input_text, output_text, label_space = self.t0_cos_e_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def t0_multiple_choice_7(self, item):
        input_text, output_text, label_space = self.t0_cos_e_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def t0_multiple_choice_8(self, item):
        input_text, output_text, label_space = self.t0_cos_e_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_9(self, item):
        input_text, output_text, label_space = self.t0_cos_e_4(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def t0_multiple_choice_10(self, item):
        input_text, output_text, label_space = self.t0_cos_e_5(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def t0_multiple_choice_11(self, item):
        input_text, output_text, label_space = self.t0_cos_e_6(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_12(self, item):
        input_text, output_text, label_space = self.t0_openbookqa_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_13(self, item):
        input_text, output_text, label_space = self.t0_openbookqa_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_14(self, item):
        input_text, output_text, label_space = self.t0_openbookqa_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_15(self, item):
        input_text, output_text, label_space = self.t0_openbookqa_4(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_16(self, item):
        input_text, output_text, label_space = self.t0_openbookqa_5(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_multiple_choice_17(self, item):
        input_text, output_text, label_space = self.t0_openbookqa_6(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_1(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_2(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_3(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_4(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_4(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_5(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_5(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_6(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_6(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def t0_binary_classification_7(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_7(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_8(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_8(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_9(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_9(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_binary_classification_10(self, item):
        input_text, output_text, label_space = self.t0_super_glue_multirc_10(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_1(self, item):
        input_text, output_text, label_space = self.t0_dbpedia_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_2(self, item):
        input_text, output_text, label_space = self.t0_dbpedia_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_3(self, item):
        input_text, output_text, label_space = self.t0_dbpedia_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_4(self, item):
        input_text, output_text, label_space = self.t0_trec_1(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def t0_classification_5(self, item):
        input_text, output_text, label_space = self.t0_trec_2(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_6(self, item):
        input_text, output_text, label_space = self.t0_trec_3(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_7(self, item):
        input_text, output_text, label_space = self.t0_trec_4(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    
    def t0_classification_8(self, item):
        input_text, output_text, label_space = self.t0_trec_5(**item)
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict
    























