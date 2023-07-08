class P3Templates:

    def t0_arc_challenge_1(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        for item, option in zip(item_names, options):
            options_ += "\n- {}: {}".format(item, option)
        answer = item_names[options.index(answer)]
        input_text = "Here's a problem to solve: {question}\n\nAmong the 4 following options, which " \
                     "is the correct answer?{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = item_names
        return input_text, output_text, label_space
    
    def t0_arc_challenge_2(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}\n\nOptions:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_arc_challenge_3(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "I am hesitating between 4 options to answer the following question, which option should I " \
                     "choose?\nQuestion: {question}\nPossibilities:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_arc_challenge_4(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "I gave my students this multiple choice question: {question}\n\nOnly one answer is correct " \
                     "among these 4 choices:{options_}\n\nCould you tell me which one is correct?".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_arc_challenge_5(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        for item, option in zip(item_names, options):
            options_ += "\n- {}: {}".format(item, option)
        answer = item_names[options.index(answer)]
        input_text = "Pick the most correct option to answer the following question.\n\n{question}\n\nOptions:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = item_names
        return input_text, output_text, label_space
    
    def t0_cos_e_1(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}\nChoose the most suitable option to answer the above question.\nOptions:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_cos_e_2(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        for item, option in zip(item_names, options):
            options_ += "\n{}. {}".format(item, option)
        answer = item_names[options.index(answer)]
        input_text = "{question}\nChoose the most suitable option to answer the above question.\nOptions{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = item_names
        return input_text, output_text, label_space

    def t0_cos_e_3(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}{options_}\nThe best answer is:".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_cos_e_4(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        for item, option in zip(item_names, options):
            options_ += "\n{}. {}".format(item, option)
        answer = item_names[options.index(answer)]
        input_text = "Pick the option in line with common sense to answer the question.\nQuestion: {question}\nOptions:{options_}\nThe best answer is:".format(question=question, options_=options_)
        output_text = answer
        label_space = item_names
        return input_text, output_text, label_space
    
    def t0_cos_e_5(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        for item, option in zip(item_names, options):
            options_ += "\n{}. {}".format(item, option)
        answer = item_names[options.index(answer)]
        input_text = "Pick the option in line with common sense to answer the question.\nQuestion: {question}\nOptions:{options_}\n".format(question=question, options_=options_)
        output_text = answer
        label_space = item_names
        return input_text, output_text, label_space
    
    def t0_cos_e_6(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "Pick the option in line with common sense to answer the question.\nQuestions: {question}\nOptions:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_openbookqa_1(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}\n\nChoose an answer from this list:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_openbookqa_2(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}\n\nWhich is the correct answer?{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_openbookqa_3(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        prompt_options = ""
        for i, (item, option) in enumerate(zip(item_names, options)):
            options_ += "\n{} -> {}".format(item, option)
            if i == len(options) - 1:
                prompt_options += " or {}".format(item)
            else:
                prompt_options += ", {}".format(item)
        prompt_options = prompt_options[2:]

        answer = item_names[options.index(answer)]
        input_text = "{question}{options_}\nIs the right answer \"{prompt_options}\" ?".format(question=question, options_=options_, prompt_options=prompt_options)
        output_text = answer
        label_space = item_names
        return input_text, output_text, label_space
    
    def t0_openbookqa_4(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}\n\nChoices:{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_openbookqa_5(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}{options_}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_openbookqa_6(self, question, answer, options):
        options_ = ""
        for option in options:
            options_ += "\n- {}".format(option)
        input_text = "{question}{options_}\n\nWhich is the correct answer?".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    

    # Binary Classification

    def t0_super_glue_multirc_1(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\n\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or no?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_2(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\nBased on the previous passage, {question}\nIs \"{answer}\" a correct answer?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_3(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\nQuestion: {question}\n\nI am grading my students' exercises. Is the answer \"{answer}\" correct?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_4(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\n{question}\nWould it be good to answer \"{answer}\"?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_5(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\nQuestion: {question}\nIs it \"{answer}\"?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_6(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\n\nDecide whether\"{answer}\" is a valid answer to the following question:\n{question}\nAnswer yes or no.".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_7(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\nQuestion: {question}\nIs the correct answer \"{answer}\"?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_8(self, paragraph, question, answer, correct_answer):
        input_text = "Is \"{answer}\" a correct answer to the following question?\nQuestion: {question}\n\nRely on the following text: {paragraph}".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_9(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\n\nQuestion: {question}\nI think \"{answer}\" is a valid answer. Could you confirm? Yes or no?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    
    def t0_super_glue_multirc_10(self, paragraph, question, answer, correct_answer):
        input_text = "{paragraph}\n{question}\nI was going to say \"{answer}\". Does that sound right?".format(paragraph=paragraph, question=question, answer=answer)
        output_text = "Yes" if answer == correct_answer else "No"
        label_space = ["Yes", "No"]
        return input_text, output_text, label_space
    

    # Classification
    def t0_dbpedia_1(self, question, answer, options):
        options_ = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                options_ += "or {}".format(option)
            else:
                options_ += ", {}".format(option)
        options_ = "\"" + options_[2:] + "\""
        input_text = "{question} Given a list of categories: {options_}, what category does the paragraph belong to?".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_dbpedia_2(self, question, answer, options):
        options_ = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                options_ += "or {}".format(option)
            else:
                options_ += ", {}".format(option)
        options_ = options_[2:]
        input_text = "Pick one category for the following text. The options are - {options_}. {question}".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_dbpedia_3(self, question, answer, options):
        options_ = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                options_ += "or {}".format(option)
            else:
                options_ += ", {}".format(option)
        options_ = options_[2:]
        input_text = "{question} Given a choice of categories {categories}, the text refers to which one?".format(question=question, options_=options_)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_trec_1(self, question, answer, options):
        categories = ", ".join(options)
        input_text = "Categories: {categories}\n\nWhat category best describes: {question}\nAnswer:".format(categories=categories, question=question)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_trec_2(self, question, answer, options):
        categories = ", ".join(options)
        input_text = "Question: {question}\n\nDescriptors: {categories}\n\nBest Descriptor?".format(categories=categories, question=question)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_trec_3(self, question, answer, options):
        categories = ", ".join(options)
        input_text = "Which category best describes the following question: {question}\n\nChoose from the following list:\n{categories}".format(categories=categories, question=question)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_trec_4(self, question, answer, options):
        categories = ", ".join(options)
        input_text = "{question}Is this asking about {categories}?".format(categories=categories, question=question)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space
    
    def t0_trec_5(self, question, answer, options):
        categories = ", ".join(options)
        input_text = "Is the following question asking about  {categories}?\n\n{question}".format(categories=categories, question=question)
        output_text = answer
        label_space = options
        return input_text, output_text, label_space

    