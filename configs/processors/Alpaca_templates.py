class AlpacaTemplates:

    def alpaca_template_with_input(self, instruction, inputs, answer, label_space):
        input_text = "Below is an instruction that describes a task, paired with an input that provides further " \
                     "context. Write a response that appropriately completes the request.\n\n### Instruction:\n" \
                     "{instruction}\n\n### Input:\n{input}\n\n### Response:".format(instruction=instruction,
                                                                                    input=inputs)
        output_text = answer
        label_space = label_space
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def alpaca_template_without_input(self, instruction, answer, label_space):
        input_text = "Below is an instruction that describes a task, paired with an input that provides further " \
                     "context. Write a response that appropriately completes the request.\n\n### Instruction:\n" \
                     "{instruction}\n\n### Response:".format(instruction=instruction)
        output_text = answer
        label_space = label_space
        return_dict = {"input_text": input_text, "output_text": output_text, "label_space": label_space}
        return return_dict

    def alpaca_multiple_choice(self, item, instruction):
        question, answer, options = item["question"], item["answer"], item["options"]
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        _options = ""
        for item, option in zip(item_names, options):
            _options += "\n({}) {}".format(item, option)
        answer = item_names[options.index(answer)]
        inputs = "Question: {question}{_options}".format(question=question, _options=_options)
        label_space = item_names
        return self.alpaca_template_with_input(instruction, inputs, answer, label_space)

    def alpaca_binary_classification(self, item, instruction, label_space):
        question, answer = item["question"], item["answer"]
        inputs = "{question}".format(question=question)
        return self.alpaca_template_with_input(instruction, inputs, answer, label_space)

    def alpaca_multiple_choice_1(self, item):
        instruction = "Select the correct letter in the parentheses."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_2(self, item):
        instruction = "Select the correct option from the following choices."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_3(self, item):
        instruction = "Answer this multiple choice question."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_4(self, item):
        instruction = "Read the answer choices and select the correct one."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_5(self, item):
        instruction = "Identify the correct answer from the choices below."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_6(self, item):
        instruction = "Determine which choice is correct and output it."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_7(self, item):
        instruction = "Refer to the given input and identify the correct answer."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_8(self, item):
        instruction = "From the given three options, select the one most relevant to the given input."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_9(self, item):
        instruction = "Select the most optimal response."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_10(self, item):
        instruction = "Read the answer choices and select the correct one."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_11(self, item):
        instruction = "Select the best answer out of given options."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_12(self, item):
        instruction = "Determine which of these options is the correct answer."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_13(self, item):
        instruction = "Choose the best option."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_14(self, item):
        instruction = "Choose the best option."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_15(self, item):
        instruction = "Select the best answer."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_16(self, item):
        instruction = "Choose the correct answer."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_17(self, item):
        instruction = "Select the correct answer from a list."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_18(self, item):
        instruction = "Choose the best answer."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_19(self, item):
        instruction = "Choose the statement that best suits the given context."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_multiple_choice_20(self, item):
        instruction = "Answer the question based on common sense and your knowledge."
        return self.alpaca_multiple_choice(item, instruction)

    def alpaca_binary_classification_1(self, item):
        instruction = "Determine if this claim is true or false:"
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])
    
    def alpaca_binary_classification_2(self, item):
        instruction = "Is the following sentence true or false?"
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])

    def alpaca_binary_classification_3(self, item):
        instruction = "Identify whether the following phrase is a true or false statement"
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])
    
    def alpaca_binary_classification_4(self, item):
        instruction = "Check if the following statement is true or false:"
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])
        
    def alpaca_binary_classification_5(self, item):
        instruction = "Classify the following statement as true or false:"
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])
    
    def alpaca_binary_classification_6(self, item):
        instruction = "Classify the following statement as true or false:"
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])

    def alpaca_binary_classification_7(self, item):
        instruction = "Do a fact check to confirm the accuracy of the statement and output true or false."
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])

    def alpaca_binary_classification_8(self, item):
        instruction = "Label whether an input sentence is true or false."
        return self.alpaca_binary_classification(item, instruction, ["True", "False"])

    # Answer yes or now
    def alpaca_binary_classification_9(self, item):
        instruction = "Indicate a yes or no answer to the given statement.."
        return self.alpaca_binary_classification(item, instruction, ["Yes", "No"])

    def alpaca_binary_classification_10(self, item):
        instruction = "Evaluate the following proposal as a yes or no response."
        return self.alpaca_binary_classification(item, instruction, ["Yes", "No"])

    def alpaca_binary_classification_11(self, item):
        instruction = "Respond to the following statement with a yes or no."
        return self.alpaca_binary_classification(item, instruction, ["Yes", "No"])
    
    # Dialog Specific
    def alpaca_dialog_1(self, item):
        instruction = "Identify the speaker of these lines.."
        return self.alpaca_binary_classification(item, instruction, ["same", "different"])
    
    def alpaca_dialog_2(self, item):
        instruction = "Given a dialogue, identify the speaker."
        return self.alpaca_binary_classification(item, instruction, ["same", "different"])





