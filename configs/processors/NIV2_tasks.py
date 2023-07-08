LEN2WORD = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
}


class NIV2Tasks:

    def niv2_56_multirc_classification(self, paragraph, question, correct_answer, output_text):
        definition_text = "In this task, your goal is to judge a correct answer to a given question based on an " \
                          "associated paragraph and decide if it is a good correct answer or not. A good correct " \
                          "answer is one that correctly and completely answers the question. A bad correct answer " \
                          "addresses the question only partially or incorrectly. If you think the given correct " \
                          "answer is good, indicate it by responding \"Yes\". Otherwise, respond \"No\". There are " \
                          "only two types of responses possible: \"Yes\" and \"No\"."
        input_text = "Paragraph- {} Question: {} Correct Answer: {}".format(paragraph, question, correct_answer)
        explanation = "undefined"
        label_space = ["Yes", "No"]
        return definition_text, input_text, explanation, output_text, label_space

    def niv2_73_commonsense_qa(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
            if item == item_names[-1]:
                option_space_in_instruction += "\'{}\'".format(item)
            else:
                option_space_in_instruction += "\'{}\', ".format(item)

        answer = item_names[options.index(answer)]
        definition_text = "You are given a question and some answer options (associated with \"A\", \"B\", \"C\", " \
                          "\"D\"). You should choose the correct answer based on commonsense knowledge. Avoid " \
                          "answering questions based on associations, the set of answers are chosen deliberately " \
                          "to capture common sense beyond associations. Do not generate anything else apart from " \
                          "one of the following characters: {} and only give one answer for " \
                          "each question.".format(option_space_in_instruction)

        input_text = "{} {}".format(question, options_)
        label_space = item_names
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_73_commonsense_qa_negation(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
            if item == item_names[-1]:
                option_space_in_instruction += "\'{}\'".format(item)
            else:
                option_space_in_instruction += "\'{}\', ".format(item)

        answer = item_names[options.index(answer)]
        definition_text = "You are given a question and some answer options (associated with \"A\", \"B\", \"C\", " \
                          "\"D\"). You should choose the incorrect answer based on commonsense knowledge. Avoid " \
                          "answering questions based on associations, the set of answers are chosen deliberately " \
                          "to capture common sense beyond associations. Do not generate anything else apart from " \
                          "one of the following characters: {} and only give one answer for " \
                          "each question.".format(option_space_in_instruction)

        input_text = "{} {}".format(question, options_)
        label_space = item_names
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space

    def niv2_73_paraphrased(self, definition):
        definition_text = definition

        def process_func(question, answer, options):
            item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
            options_ = ""
            for item, option in zip(item_names, options):
                options_ += "({}) {} ".format(item, option)

            answer = item_names[options.index(answer)]
            input_text = "{} {}".format(question, options_)
            label_space = item_names
            explanation = "undefined"
            return definition_text, input_text, explanation, answer, label_space

        return process_func

    def niv2_1135_xcsr_classification(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
            if item == item_names[-1]:
                option_space_in_instruction += "and \"{}\"".format(item)
            else:
                option_space_in_instruction += "\"{}\", ".format(item)

        answer = item_names[options.index(answer)]
        definition_text = "In this task, you will be presented with a question that has multiple possible answers. " \
                          "You should choose the most suitable option out of {}, based on your commonsense knowledge" \
                          ".".format(option_space_in_instruction)

        input_text = "{} Options: {}".format(question, options_)
        label_space = item_names
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_1193_course_classification(self, question, answer, options):
        definition_text = "In this task, you are given the name of an Indian food dish. You need to classify the dish as a {options_}"
        explanation = "undefined"
        label_space = options
        options_ = ""
        for i, option in enumerate(options_):
            if i == len(options_) - 1:
                options_ += " or \'{}\'".format(option)
            else:
                options_ += ", \"{}\"".format(option)
        
        options_ = options_[2:]
        definition_text = definition_text.format(options_=options_)
        return definition_text, question, explanation, answer, label_space

    def niv2_1286_openbook_qa(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
            if item == item_names[-1]:
                option_space_in_instruction += "and \"{}\"".format(item)
            else:
                option_space_in_instruction += "\"{}\", ".format(item)

        answer = item_names[options.index(answer)]
        definition_text = " In this task, you are given a multiple-choice question and you have to pick the incorrect" \
                          " option. Answer with option indexes (i.e., {}).".format(option_space_in_instruction)

        input_text = "{} {}".format(question, options_)
        label_space = item_names
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_1286_openbook_qa_negation(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
            if item == item_names[-1]:
                option_space_in_instruction += "and \"{}\"".format(item)
            else:
                option_space_in_instruction += "\"{}\", ".format(item)

        answer = item_names[options.index(answer)]
        definition_text = " In this task, you are given a multiple-choice question and you have to pick the incorrect" \
                          " option. Answer with option indexes (i.e., {}).".format(option_space_in_instruction)

        input_text = "{} {}".format(question, options_)
        label_space = item_names
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space

    def niv2_1286_paraphrased(self, definition):
        definition_text = definition

        def process_fun(question, answer, options):
            item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
            options_ = ""
            option_space_in_instruction = ""
            for item, option in zip(item_names, options):
                options_ += "({}) {} ".format(item, option)

            answer = item_names[options.index(answer)]
            input_text = "{} {}".format(question, options_)
            label_space = item_names
            explanation = "undefined"
            return definition_text, input_text, explanation, answer, label_space

        return process_fun
    
    def niv2_1297_qasc_question_answering(question, answer, options):
        definition_text = "In this task, you are given two facts, and a multiple-choice question. Based on the given facts, answer the question with index of the correct option (e.g, \"A\")."
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
        answer = item_names[options.index(answer)]
        input_text = "{} {}".format(question, options_)
        label_space = item_names
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space


    def niv2_1322_government_type_classification(self, question, answer, options):
        definition_text = " In this task, you are given a country name and you need to answer with the government type of the country, as of the year 2015. " \
                          "The following are possible government types that are considered valid answers: "
        for i, option in enumerate(options):
            if i == len(options) - 1:
                definition_text += "{}.".format(option)
            else:
                definition_text += "{}, ".format(option)
        input_text = question
        explanation = "undefined"
        label_space = options
        return definition_text, input_text, explanation, answer, label_space

    def niv2_137_newscomm_classification(self, question, answer, options):
        items = [str(i) for i in range(1, len(options) + 1)]
        definition_text = "Classify the given news commentary into the language in which it is written in. There are {} " \
                          "languages to classify the sentences into".format(len(options))
        
        for item, option in zip(items, options):
            definition_text += ", {}) {}".format(item, option)
        label_space = options
        definition_text += "."
        explanation = "undefined"
        input_text = question
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_137_newscomm_negation(self, question, answer, options):
        items = [str(i) for i in range(1, len(options) + 1)]
        definition_text = "Classify the given news commentary into the language in which it is not written in. There are {} " \
                          "languages to classify the sentences into".format(len(options))
        
        for item, option in zip(items, options):
            definition_text += ", {}) {}".format(item, option)
        label_space = options
        definition_text += "."
        explanation = "undefined"
        input_text = question
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_1370_paraphrased(self, question, answer, options, definition):
        items = [str(i) for i in range(1, len(options) + 1)]
        
        options_ = ""
        for item, option in zip(items, options):
            options_ += ", {}) {}".format(item, option)
        label_space = options
        options_ += "."
        explanation = "undefined"
        input_text = question
        definition_text = definition.format(options=options_)
        return definition_text, input_text, explanation, answer, label_space


    def niv2_1420_mathqa_other(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("a"), ord("z") + 1)][:len(options)]
        options_ = "Options:"

        for item, option in zip(item_names, options):
            if item == item_names[-1]:
                options_ += " {} ) {}".format(item, option)
            else:
                options_ += " {} ) {} ,".format(item, option)
        answer = item_names[options.index(answer)]
        definition_text = "In this task, you need to provide the correct option for a given problem from the" \
                          " provided options."
        input_text = "Problem: {} \n{}".format(question, options_)
        explanation = "\"u / i = 6 / 2 i / b = 5 / 1 since i is multiple of both 2 ( as per first ratio ) and 5" \
                      " ( as per second ratio ) so let ' s assume that i = 10 i . e . multiplying teh first ratio" \
                      " by 5 and second ration by 2 in each numerator and denominator then , u : i : b = 30 : 18 :" \
                      " 2 i . e . u : b = 30 : 2 answer : option b\""
        label_space = item_names
        return definition_text, input_text, explanation, answer, label_space

    def niv2_143_odd_man_out_classification(self, question, answer, options):
        definition_text = "Given a set of four words, generate the category that the words belong to. Words are separated " \
                          " by commas. The possible categories are "
        for i, option in enumerate(options):
            if i == len(options) - 1:
                definition_text += "and {}.".format(option)
            else:
                definition_text += "{}, ".format(option)
        input_text = question
        explanation = "These four words are all construction equipment, such as 'excavator', 'crane', and 'hoist', or describe a common activity used in construction, such as 'upraise'."
        label_space = options
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_143_odd_man_out_classification_negation(self, question, answer, options):
        definition_text = "Given a set of four words, generate the category that the words does not belong to. Words are separated " \
                          " by commas. The possible categories are "
        for i, option in enumerate(options):
            if i == len(options) - 1:
                definition_text += "and {}.".format(option)
            else:
                definition_text += "{}, ".format(option)
        input_text = question
        explanation = "These four words are all construction equipment, such as 'excavator', 'crane', and 'hoist', or describe a common activity used in construction, such as 'upraise'."
        label_space = options
        return definition_text, input_text, explanation, answer, label_space

    def niv2_1420_paraphrased(self, definition):
        definition_text = definition

        def process_func(question, answer, options):
            item_names = [chr(x) for x in range(ord("a"), ord("z") + 1)][:len(options)]
            options_ = "Options:"

            for item, option in zip(item_names, options):
                if item == item_names[-1]:
                    options_ += " {} ) {}".format(item, option)
                else:
                    options_ += " {} ) {} ,".format(item, option)
            answer = item_names[options.index(answer)]
            input_text = "Problem: {} \n{}".format(question, options_)
            explanation = "\"u / i = 6 / 2 i / b = 5 / 1 since i is multiple of both 2 ( as per first ratio ) and 5" \
                          " ( as per second ratio ) so let ' s assume that i = 10 i . e . multiplying teh first ratio" \
                          " by 5 and second ration by 2 in each numerator and denominator then , u : i : b = 30 : 18 :" \
                          " 2 i . e . u : b = 30 : 2 answer : option b\""
            label_space = item_names
            return definition_text, input_text, explanation, answer, label_space

        return process_func

    def niv2_1421_mathqa_general(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("a"), ord("z") + 1)][:len(options)]
        options_ = "Options:"
        option_space_in_instruction = ""

        for item, option in zip(item_names, options):
            if item == item_names[-1]:
                options_ += " {} ) {}".format(item, option)
                option_space_in_instruction += "and '{}'.".format(item)
            else:
                options_ += " {} ) {} ,".format(item, option)
                option_space_in_instruction += "'{}', ".format(item)
        answer = item_names[options.index(answer)]
        definition_text = "In this task, you need to answer the given multiple-choice question on the general" \
                          " math. Classify your answers into {}".format(option_space_in_instruction)
        input_text = "Problem: {} \n{}".format(question, options_)
        explanation = "\"explanation : 2 / 3 = . 66 , 3 / 4 = . 75 , 4 / 5 = . 8 and 5 / 3 = 1.66 so the biggest is" \
                      " 5 / 3 and the smallest is 2 / 3 their difference is 5 / 3 - 2 / 3 = 3 / 3 = 1 option d\""
        label_space = item_names
        return definition_text, input_text, explanation, answer, label_space

    def niv2_1422_mathqa_physics(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("a"), ord("z") + 1)][:len(options)]
        options_ = "Options: "
        option_space_in_instruction = ""

        for item, option in zip(item_names, options):
            if item == item_names[-1]:
                options_ += " {} ) {}".format(item, option)
                option_space_in_instruction += "and '{}'.".format(item)
            else:
                options_ += " {} ) {},".format(item, option)
                option_space_in_instruction += "'{}', ".format(item)
        answer = item_names[options.index(answer)]
        definition_text = "In this task, you need to answer the given multiple-choice question on the physics." \
                          " Classify your answers into {}".format(option_space_in_instruction)
        input_text = "Problem: {} \n{}".format(question, options_)
        explanation = "New speed = 5 / 6 th of usual speed new time = 6 / 5 th of usual time 6 / 5 ut - ut = 15 m " \
                      "ut / 5 = 15 m ut = 75 m answer is d"
        label_space = item_names
        return definition_text, input_text, explanation, answer, label_space

    def niv2_148_dart_similarity_classification(self, sentences, answer):
        definition_text = "This task is about classifying the similarity of two sentences. The sentences can be " \
                          "classified as (a) SIMILAR - similar to each other, and (b) DISSIMILAR - not similar to " \
                          "each other. Sentences that have the same RDF relationship in terms of [subject, predicate, " \
                          "object] are similar to each other. The input is a list of two sentences and the output is " \
                          "either SIMILAR or DISSIMILAR."
        input_text = "[{}]".format(", ".join(f"'{sentence}'" for sentence in sentences))
        explanation = "undefined"
        label_space = ["SIMILAR", "DISSIMILAR"]
        return definition_text, input_text, explanation, answer, label_space

    def niv2_152_srl_answer_generation(self, sentence, question, answer):
        definition_text = "In this task, you are given a sentence and question which can be answered using the " \
                          "sentence. Your task is to answer the question using the information from the sentence." \
                          " The answer to the question is unique and it is a continuous text span from the sentence."
        explanation = "undefined"
        input_text = "Sentence: {sentence} . Question: {question}".format(sentence=sentence, question=question)
        label_space = None
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_153_hatexplain_classification(self, question, answer, options):
        labels = ""
        for i, option in enumerate(options):
            if i == len(options) - 1:
                labels += "or \'{}\'".format(option)
            else:
                labels += "\'{}\', ".format(option)
        definition_text = "The input is a tweet which can be Hate Speech, Offensive or Normal tweet. Hate Speech and Offensive " \
                          "tweets target one community. Given such a tweet, output the community targeted in the tweet. The community " \
                          "will be one of the nine values: {labels}. Output 'None' if the tweet does not target " \
                          "any community. A tweet targets only one community.".format(labels=labels)
        label_space = options
        input_text = question
        explanation = "undefined"
        return definition_text, input_text, explanation, answer, label_space

    def niv2_1565_trivia_qa_classification(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = "Options: ["
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            if item == item_names[-1]:
                options_ += "{}. {}]".format(item, option)
                option_space_in_instruction += "or {}".format(item)
            else:
                options_ += "{}. {}, ".format(item, option)
                option_space_in_instruction += "{}, ".format(item)
        answer = item_names[options.index(answer)]
        definition_text = "This task involves asking a question, providing a set of {} options. You are expected to " \
                          "choose the best answer to the question. The output will be in the form of {}, " \
                          "corresponding to which option is chosen.".format(len(options), option_space_in_instruction)

        input_text = "Question:{} , {}".format(question, options_)
        explanation = "undefined"
        label_space = item_names
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_1565_trivia_qa_classification_negation(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_ = "Options: ["
        option_space_in_instruction = ""
        for item, option in zip(item_names, options):
            if item == item_names[-1]:
                options_ += "{}. {}]".format(item, option)
                option_space_in_instruction += "or {}".format(item)
            else:
                options_ += "{}. {}, ".format(item, option)
                option_space_in_instruction += "{}, ".format(item)
        answer = item_names[options.index(answer)]
        definition_text = "This task involves asking a question, providing a set of {} options. You are expected to " \
                          "choose the worst answer to the question. The output will be in the form of {}, " \
                          "corresponding to which option is chosen.".format(len(options), option_space_in_instruction)

        input_text = "Question:{} , {}".format(question, options_)
        explanation = "undefined"
        label_space = item_names
        return definition_text, input_text, explanation, answer, label_space


    def niv2_1565_paraphrased(self, definition):
        definition_text = definition

        def process_func(question, answer, options):
            item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
            options_ = "Options: ["
            option_space_in_instruction = ""
            for item, option in zip(item_names, options):
                if item == item_names[-1]:
                    options_ += "{}. {}]".format(item, option)
                else:
                    options_ += "{}. {}, ".format(item, option)
            answer = item_names[options.index(answer)]
            input_text = "Question:{} , {}".format(question, options_)
            explanation = "undefined"
            label_space = item_names
            return definition_text, input_text, explanation, answer, label_space

        return process_func

    def niv2_1588_tecla_classification(self, question, answer, options):
        definition_text = "In this task, you are given a text in Catalan. Your task is to classify it into {length} different given themes. Names of all the classes are {options_}"
        explanation = "undefined"
        label_space = options
        options_ = ""
        for i, option in enumerate(options_):
            if i == len(options_) - 1:
                options_ += "and {}".format(option)
            else:
                options_ += "{}, ".format(option)
        
        definition_text = definition_text.format(options_=options_, length=len(options))
        return definition_text, question, explanation, answer, label_space
    

    def niv2_161_webquestions_answer_generation(self, concept, question, answer):
        definition_text = "Based on the given question and tppic, give an answer. The answer is available on on the" \
                          " internet. The questions are mostly centered around a single named entity."
        explanation = "undefined"
        input_text = "concept: {concept} question: {question}".format(concept=concept, question=question)
        label_space = None
        return definition_text, input_text, explanation, answer, label_space
    
    def niv2_163_openpi_classification(self, question, answer, options):
        options_ = ", ".join(options)
        definition_text = "Given a passage as input, answer with the category to which the passage belongs. There are {length} categories - {options_}. The answer should be one of the categories based on words from the passage which closely belong to the category."
        definition_text = definition_text.format(length=len(options), options_=options_)
        explanation = "undefined"
        label_space = options
        return definition_text, question, explanation, answer, label_space
    
    def niv2_163_openpi_negation(self, question, answer, options):
        options_ = ", ".join(options)
        definition_text = "Given a passage as input, answer with the category to which the passage does not belongs. There are {length} categories - {options_}. The answer should be one of the categories based on words from the passage which closely belong to the category."
        definition_text = definition_text.format(length=len(options), options_=options_)
        explanation = "undefined"
        label_space = options
        return definition_text, question, explanation, answer, label_space

    def niv2_229_arc_qa_hard(self, question, answer, options):
        item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
        options_, option_space_in_instruction1, option_space_in_instruction2 = "", "", ""
        for item, option in zip(item_names, options):
            options_ += "({}) {} ".format(item, option)
            if item == item_names[-1]:
                option_space_in_instruction1 += "\"{}\"".format(item)
                option_space_in_instruction2 += "\'{}\'".format(item)
            else:
                option_space_in_instruction1 += "\"{}\", ".format(item)
                option_space_in_instruction2 += "\'{}\', ".format(item)
        answer = item_names[options.index(answer)]
        num = LEN2WORD[len(options)]
        definition_text = "You are given a science question (hard-level) and {num} answer options (associated with " \
                          "{option1}). Your task is to find the correct answer based on scientific facts, knowledge, " \
                          "and reasoning. Do not generate anything else apart from one of the following characters: " \
                          "{option2}. There is only one correct answer for each question.".format(
            num=num, option1=option_space_in_instruction1,option2=option_space_in_instruction2)

        input_text = "{} {}".format(question, options_)
        explanation = "undefined"
        label_space = item_names
        return definition_text, input_text, explanation, answer, label_space

    def niv2_229_paraphrased(self, definition):
        definition_text = definition

        def process_func(question, answer, options):
            item_names = [chr(x) for x in range(ord("A"), ord("Z") + 1)][:len(options)]
            options_, option_space_in_instruction1, option_space_in_instruction2 = "", "", ""
            for item, option in zip(item_names, options):
                options_ += "({}) {} ".format(item, option)
            answer = item_names[options.index(answer)]

            input_text = "{} {}".format(question, options_)
            explanation = "undefined"
            label_space = item_names
            return definition_text, input_text, explanation, answer, label_space

        return process_func
    
    def niv2_562_language_identification(self, question, answer, options):
        definition_text = "In this task, an input sentence is given which can be in the {options_} languages. There are a total of {length} languages. Your task is to identify the language of the input sentence. The input sentence can only be in any of the {length} languages provided."
        explanation = "undefined"
        label_space = options
        options_ = ""
        for i, option in enumerate(options_):
            if i == len(options_) - 1:
                options_ += " or " + option
            else:
                options_ += ", " + option 
        
        options_ = options_[2:]
        definition_text = definition_text.format(options_=options_, length=len(options))
        return definition_text, question, explanation, answer, label_space
    
    def niv2_562_language_negation(self, question, answer, options):
        definition_text = "In this task, an input sentence is given which can be in the {options_} languages. There are a total of {length} languages. Your task is to not identify the language of the input sentence. The input sentence can only be in any of the {length} languages provided."
        explanation = "undefined"
        label_space = options
        options_ = ""
        for i, option in enumerate(options_):
            if i == len(options_) - 1:
                options_ += " or " + option
            else:
                options_ += ", " + option 
        
        options_ = options_[2:]
        definition_text = definition_text.format(options_=options_, length=len(options))
        return definition_text, question, explanation, answer, label_space
    
    def niv2_564_discofuse_classification(self, question, answer, options):
        definition_text = "In this task, you are given two sentences in the English language and your task is to classify them into one of their discourse types. A discourse type is an indicator to classify the given two sentences on the basis of a co-text as well as a relevant context. There are {length} discourse types in total which are {options_}"
        explanation = "undefined"
        label_space = options
        options_ = ""
        for i, option in enumerate(options_):
            if i == len(options_) - 1:
                options_ += " and \'{}\'".format(option)
            else:
                options_ += ", \'{}\'".format(option)
        
        options_ = options_[2:]
        definition_text = definition_text.format(options_=options_, length=len(options))
        return definition_text, question, explanation, answer, label_space

    def niv2_591_sciq_answer_generation(self, question, answer):
        definition_text = "Given a scientific question, generate a correct answer to it."
        explanation = "undefined"
        label_space = None
        return definition_text, question, explanation, answer, label_space

    def niv2_900_freebase_category_classification(self, question, answer, options):
        label_space = options
        options_ = ["\'{}\'".format(option) for option in options]
        options_ = ", ".join(options_)
        definition_text = "Given a trivia question, classify broad topical category from this list: {options_}.".format(
            options_=options_)
        explanation = "undefined"
        return definition_text, question, explanation, answer, label_space



