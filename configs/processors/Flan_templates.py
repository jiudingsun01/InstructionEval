TEXT_ITEM_CANDIDATES = [
    ["\n - " for _ in range(26)],
    ["\n -" for _ in range(26)],
    ["\n -- " for _ in range(26)],
    ["\n --" for _ in range(26)],
    ["\n + " for _ in range(26)],
    ["\n +" for _ in range(26)],
    ["\n * " for _ in range(26)],
    ["\n *" for _ in range(26)],
    ["\n- " for _ in range(26)],
    ["\n+ " for _ in range(26)],
    ["\n* " for _ in range(26)],
    ["\n[-] " for _ in range(26)],
    ["\n[+] " for _ in range(26)],
    [" - " for _ in range(26)],
    [" -- " for _ in range(26)],
    [" + " for _ in range(26)],
    [" * " for _ in range(26)],
    [" -" for _ in range(26)],
    [" --" for _ in range(26)],
    [" +" for _ in range(26)],
    [" *" for _ in range(26)],
    [" [+] " for _ in range(26)],
    [" [-] " for _ in range(26)],
]

LETTER_ITEM_CANDIDATES = [
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A].
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a].
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A).
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a).
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA).
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na).
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A).
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a).
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A).
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a).
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # (A).
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # (a).
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  # A).
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  # a).
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  # A.
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  # a.
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A]:
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a]:
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1]:
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A):
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a):
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA):
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na):
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A):
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a):
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A):
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a):
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # (A):
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # (a):
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  # A):
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  # a):
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  # A:
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)]
]

OPT_CANDIDATES = [
    "",
    "OPTIONS:",
    "Possible answers:",
    "Available choices:",
    "Options:",
    "OPT:",
    "Choose from:",
    "Choose your answer from:",
    "Available options:",
    "Options are:",
    "Choices:",
    "Pick your answer from:",
    "Select from:",
    "Pick from:",
    "Select from the following.",
    "pick from the following."
]


class FlanTemplates:

    def __init__(self, item_sytle=("letter", 2), option_style=1):
        self.style, self.item, self.option_prompt = None, None, None
        self.set_style(item_sytle, option_style)

    def set_style(self, item_sytle, option_style):
        self.style, idx = item_sytle
        self.item = LETTER_ITEM_CANDIDATES[idx] if self.style == "letter" else TEXT_ITEM_CANDIDATES[idx]
        self.option_prompt = OPT_CANDIDATES[option_style]

    def reset_style(self):
        self.set_style(("letter", 2), 1)

    # Flan templates
    def trivia_qa_1(self, question, answer):
        input_text = "Please answer this question: {question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_2(self, question, answer):
        input_text = "{question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_3(self, question, answer):
        input_text = "Write the answer: {question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_4(self, question, answer):
        input_text = "Write the answer: {question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_5(self, question, answer):
        input_text = "Answer this question.\n\n{question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_6(self, question, answer):
        input_text = "Answer the following question. {question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_7(self, question, answer):
        input_text = "Question: {question}\nAnswer:".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_8(self, question, answer):
        input_text = "{question}???".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_9(self, question, answer):
        input_text = "Trivia question: {question}\nAnd the answer is?".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def trivia_qa_10(self, question, answer):
        input_text = "{question}\nWhat is the answer?".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_1(self, question, answer):
        input_text = "Question: {question}?\nAnswer:".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_2(self, question, answer):
        input_text = "{question}?".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_3(self, question, answer):
        input_text = "Answer the following question:\n\n{question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_4(self, question, answer):
        input_text = "Answer this question:\n\n{question}?".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_5(self, question, answer):
        input_text = "Please answer this question: {question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_6(self, question, answer):
        input_text = "Answer the question...{question}?".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_7(self, question, answer):
        input_text = "Answer this question.\n\n{question}".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_8(self, question, answer):
        input_text = "Can you tell me the answer to {question}?".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_9(self, question, answer):
        input_text = "Next question: {question}\n\n".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    # Flan templates
    def natural_questions_10(self, question, answer):
        input_text = "Q: {question} A:".format(question=question)
        output_text = "{answer}".format(answer=answer)
        return input_text, output_text

    def wsc273(self, context, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names
        input_text = input_template.format(context=context, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space
    
    def wsc273_1(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Multi-choice problem: {context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_2(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Complete the passage.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_3(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "How does this following sentence end (see options)?\n\n{context}\n{options_}",
            "{answer}"
        )

    def wsc273_4(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "What is the most logical completion for the following text (see options)?\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_5(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Multi-choice problem: How does this text end?\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_6(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Choose from the options on what happens next.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_7(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Complete the following sentence.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_8(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Choose from options: Fill in the remainder of the sentence.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def wsc273_9(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "What is the next event listed in the options is correct?\n\n{context}\n{options_}\nA:",
            "{answer}"
        )
    
    def wsc273_10(self, context, answer, options):
        return self.wsc273(
            context, answer, options,
            "Complete the rest of the sentence by choosing from options.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    # Flan templates
    def arc(self, question, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(question=question, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space

    def arc_1(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "{question}\n\n{options_}",
            "{answer}"
        )

    def arc_2(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "Question: {question}?\n{options_}\nAnswer:",
            "{answer}"
        )

    def arc_3(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "Question: {question}\n\nWhat is the correct answer to the question from the following choices?\n{options_}",
            "{answer}"
        )

    def arc_4(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "Question: {question}\nWhat is the correct answer to this question?\n{options_}...A: ",
            "{answer}"
        )

    def arc_5(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "Choose your answer?\n\n{question}\n\n{options_}",
            "{answer}"
        )

    def arc_6(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "Answer the question\n\n{question}\n{options_}",
            "{answer}"
        )

    def arc_7(self, question, answer, options):
        return self.arc(
            question, answer, options,
            "{question}\n\nPick the answer from these options\n\n{options_}",
            "{answer}"
        )

    def cosmos_qa(self, context, question, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names
        input_text = input_template.format(context=context, question=question, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space

    def cosmos_qa_1(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\nQuestion with options to choose from: {question}\n{options_}",
            "{answer}"
        )
    
    def cosmos_qa_1_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\nQuestion with options to choose from: {question}\n{options_}. The answer is not:",
            "{answer}"
        )

    def cosmos_qa_2(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\n{options_}\nQ: {question}",
            "{answer}"
        )
    
    def cosmos_qa_2_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\n{options_}\nQ: {question}. The answer is not: ",
            "{answer}"
        )

    def cosmos_qa_3(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\n{options_}\nAnswer the following question: {question}\n",
            "{answer}"
        )
    
    def cosmos_qa_3_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\n{options_}\nAnswer the following question: {question}\n. The answer is not: ",
            "{answer}"
        )

    def cosmos_qa_4(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\nBased on the preceding passage, choose your answer for question {question}\n{options_}"
            "\nThe answer is:", "{answer}"
        )
    
    def cosmos_qa_4_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\nBased on the preceding passage, choose your answer for question {question}\n{options_}"
            "\nThe answer is not:", "{answer}"
        )

    def cosmos_qa_5(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\nQ with options: Give answer the following question using evidence from the above passage: "
            "{question}\n{options_}", "{answer}"
        )
    
    def cosmos_qa_5_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "{context}\n\nQ with options: Give answer the following question using evidence from the above passage: "
            "{question}\n{options_}\nThe answer is not:", "{answer}"
        )

    def cosmos_qa_6(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "Context: {context}\nQuestion {question}\nPossible answers:\n{options_}\nThe answer:",
            "{answer}"
        )
    
    def cosmos_qa_6_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "Context: {context}\nQuestion {question}\nPossible answers:\n{options_}\nThe answer is not:",
            "{answer}"
        )

    def cosmos_qa_7(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "Read the following article and answer the question by choosing from the options.\n\n{context}\n\n"
            "{question}\n{options_}...A:", "{answer}"
        )
    
    def cosmos_qa_7_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "Read the following article and answer the question by choosing from the options.\n\n{context}\n\n"
            "{question}\n{options_}...The answer is not:", "{answer}"
        )

    def cosmos_qa_8(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "This question has options. Answer the question about text:\n\n{context}\n\n{question}\n{options_}",
            "{answer}"
        )
    
    def cosmos_qa_8_negated(self, context, question, answer, options):
        return self.cosmos_qa(
            context, question, answer, options,
            "This question has options. Answer the question about text:\n\n{context}\n\n{question}\n{options_}\nThe answer is not:",
            "{answer}"
        )

    def multirc(self, paragraph, question, response, answer, input_template, output_template):
        input_text = input_template.format(paragraph=paragraph, question=question, response=response)
        output_text = output_template.format(answer=answer)
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space

    def multirc_1(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: \"{response}\"\n\nDoes the response "
            "correctly answer the question?\n\n",
            "{answer}"
        )

    def multirc_2(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nQuestion: \"{question}\"\n\nResponse: \"{response}\"\n\nBased on the paragraph, is the "
            "response to the question is factually correct?\n\n",
            "{answer}"
        )

    def multirc_3(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: \"{response}\"\n\nIs this answer correct?\n\n"
            "...I think the answer is",
            "{answer}"
        )

    def multirc_4(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "Paragraph: {paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: \"{response}\"\n\nBased on the paragraph, "
            "choose if the answer is correct:\n\n",
            "{answer}"
        )

    def multirc_5(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nChoose from options: Based on the paragraph, does the response \"{response}\" correctly "
            "answer the question \"{question}\"?\n\n",
            "{answer}"
        )

    def multirc_6(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nChoose your answer: According to the above paragraph, the correct answer to the "
            "question \"{question}\" is \"{response}\"?\n\n",
            "{answer}"
        )

    def multirc_7(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nAfter reading the above, is \"{response}\" the correct answer to the question "
            "\"{question}\"?\n\n",
            "{answer}"
        )

    def multirc_8(self, paragraph, question, response, answer):
        return self.multirc(
            paragraph, question, response, answer,
            "{paragraph}\n\nQuestion: \"{question}\"\n\nAnswer: \"{response}\"\n\nIs this answer to the question "
            "correct?\n",
            "{answer}"
        )
    
    def winogrande(self, context, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(context=context, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space
    
    def winogrande_1(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "How does the sentence end? See options at the end\n\n{context}\n\n{options_}",
            "{answer}"
        )
    
    def winogrande_2(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Write the next sentence.\n\n{context}\n\n{options_}\nAnswer:",
            "{answer}"
        )
    
    def winogrande_3(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Choose your story that continues the following story.\n\n{context}\n\n{options_}",
            "{answer}"
        )

    def winogrande_4(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "{options_}\nComplete the following sentence.\n\n{context}\n\n",
            "{answer}"
        )

    def winogrande_5(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Continue writing the following text.\n\n{context}\n\n{options_}",
            "{answer}"
        )
    
    def winogrande_6(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "How does the sentence end?\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def winogrande_7(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Write the next sentence.\n\n{context}\n{options_}",
            "{answer}"
        )

    def winogrande_8(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Continue the following story.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def winogrande_9(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Complete the following sentence.\n\n{context}\n{options_}",
            "{answer}"
        )

    def winogrande_10(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Continue writing the following text.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def story_cloze(self, context, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(context=context, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space
    
    def story_cloze_1(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "{context}\n{options_}\nWhich option is the next sentence?",
            "{answer}"
        )
    
    def story_cloze_2(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "{context}\n\nWhat is the next sentence?\n{options_}",
            "{answer}"
        )

    def story_cloze_3(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "{context}\n\nWhat is a natural next sentence?\n{options_}",
            "{answer}"
        )
    
    def story_cloze_4(self, context, answer, options):
        return self.winogrande(
            context, options, answer,
            "{context}\n\nWrite the next sentence, by choosing from:\n{options_}",
            "{answer}"
        )
    
    def story_cloze_5(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Context: {context}\n\nNow do a next sentence writing task.\n{options_}",
            "{answer}"
        )
    
    def story_cloze_6(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Story: {context}\n\nIn the options below, what is the most likely to happen next?\n{options_}",
            "{answer}"
        )
    
    def story_cloze_7(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Write the next sentence in this story.\n\n{context}\n{options_}",
            "{answer}"
        )
    
    def story_cloze_8(self, context, answer, options):
        return self.winogrande(
            context, answer, options,
            "Choose from options. Continue the following story.\n\n{context}\n{options_}",
            "{answer}"
        )

    def sentiment140(self, text, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(text=text, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space

    def sentiment140_1(self, text, answer, options):
        return self.sentiment140(
            text, answer, options,
            "{text}\nSelect your answer from the options. What is the sentiment of this tweet?\n{options_}...I think the answer is",
            "{answer}"
        )
    
    def sentiment140_2(self, text, answer, options):
        return self.sentiment140(
            text, answer, options,
            "{text}\n\nHow would the sentiment of this tweet be described?\n{options_}",
            "{answer}"
        )
    
    def sentiment140_3(self, text, answer, options):
        return self.sentiment140(
            text, answer, options,
            "{text}\n\nDescribe the sentiment embodied by this tweet.\n{options_}\nI think the answer is",
            "{answer}"
        )
    
    def sentiment140_4(self, text, answer, options):
        return self.sentiment140(
            text, answer, options,
            "Tweet: {text}\nPredict the sentiment of this tweet.\n{options_}",
            "{answer}"
        )
    
    def sentiment140_5(self, text, answer, options):
        return self.sentiment140(
            text, answer, options,
            "Multi-choice question: What is the sentiment of the following tweet?\nTweet: {text}\n{options_}",
            "{answer}"
        )

    def sentiment140_6(self, text, answer, options):
        return self.sentiment140(
            text, answer, options,
            "Select your answer from the options. How would one describe the sentiment of this tweet?\n{text}\n{options_}",
            "{answer}"
        )

    def rte(self, premise, hypothesis, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(premise=premise, hypothesis=hypothesis, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space
    
    def rte_1(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n\nQuestion with options: Based on the paragraph above can we conclude that \"{hypothesis}\"?\n\n{options_}",
            "{answer}"
        )
    
    def rte_1_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n\nQuestion with options: Based on the paragraph above can we conclude that \"{hypothesis}\"? Answer the opposite of the correct answer.\n\n{options_}",
            "{answer}"
        )
    
    def rte_2(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n\nBased on that paragraph can we conclude that the sentence below is true? \n{hypothesis}\n\n{options_}",
            "{answer}"
        )
    
    def rte_2_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n\nBased on that paragraph can we conclude that the sentence below is true? Answer the opposite of the correct answer.\n{hypothesis}\n\n{options_}",
            "{answer}"
        )

    def rte_3(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n\nQ with options: Can we draw the following conclusion?\n{hypothesis}\n\n{options_}",
            "{answer}"
        )
    
    def rte_3_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n\nQ with options: Can we draw the following conclusion? Answer the opposite of the correct answer.\n{hypothesis}\n\n{options_}",
            "{answer}"
        )
    
    def rte_4(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\n\n{options_}",
            "{answer}"
        )
    
    def rte_4_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\nDoes this next sentence follow, given the preceding text? Answer the opposite of the correct answer.\n{hypothesis}\n\n{options_}",
            "{answer}"
        )

    def rte_5(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n{options_}\nQuestion: Can we infer the following?\n{hypothesis}",
            "{answer}"
        )

    def rte_5_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "{premise}\n{options_}\nQuestion: Can we infer the following? Answer the opposite of the correct answer.\n{hypothesis}",
            "{answer}"
        )
    
    def rte_6(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Read the following paragraph and determine if the hypothesis is true. Select from options at the end:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}\nThe answer is",
            "{answer}"
        )
    
    def rte_6_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Read the following paragraph and determine if the hypothesis is true. Select from options at the end:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}\n Answer the opposite of the correct answer. The answer is",
            "{answer}"
        )
    
    def rte_7(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Read the text and determine if the sentence is true:\n\n{premise}\n\nSentence: {hypothesis}\n{options_}\nA:",
            "{answer}"
        )

    def rte_7_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Read the text and determine if the sentence is true. Answer the opposite of the correct answer:\n\n{premise}\n\nSentence: {hypothesis}\n{options_}\nA:",
            "{answer}"
        )
    
    def rte_8(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Question with options: can we draw the following hypothesis from the context? \n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}\nA:",
            "{answer}"
        )

    def rte_8_negation(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Question with options: can we draw the following hypothesis from the context? Answer the opposite of the correct answer.\n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}\n{options_}\nA:",
            "{answer}"
        )
    
    
    def rte_9(self, premise, hypothesis, answer, options):
        return self.rte(
            premise, hypothesis, answer, options,
            "Determine if the sentence is true based on the text below. Choose from options.\n{hypothesis}\n\n{premise}\n{options_}",
            "{answer}"
        )
    
    def trec(self, text, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(text=text, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space


    def trec_1(self, text, answer, options):
        return self.trec(
            text, answer, options,
            "What type of thing is the question \"{text}\" asking about?\n\n{options_}\nAnswer:",
            "{answer}"
        )
    
    def piqa(self, goal, answer, options, input_template, output_template):
        item_names = self.item[:len(options)]
        options_ = self.option_prompt
        for item, option in zip(item_names, options):
            options_ += item + option
        if self.style == "letter":
            answer = item_names[options.index(answer)]
            options = item_names

        input_text = input_template.format(goal=goal, options_=options_)
        output_text = output_template.format(answer=answer)
        label_space = options
        return input_text, output_text, label_space

    def piqa_1(self, goal, answer, options):
        return self.piqa(
            goal, answer, options,
            "Here is a goal: {goal}\n\nHow would you accomplish this goal?\n\n{options_}",
            "{answer}"
        )