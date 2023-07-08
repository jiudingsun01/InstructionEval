class NIV2Templates:


    def niv2_zs_template_1(self, definition, inputs, output):
        input_text = "{Definition}\n\n{input}".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_2(self, definition, inputs, output):
        input_text = "You will be given a definition of a task first, then some input of the task.\n" \
                     "{Definition}\n\n{input}\nOutput:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_3(self, definition, inputs, output):
        input_text = "Definition: {Definition}\nInput: {input}\nOutput:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_4(self, definition, inputs, output):
        input_text = "Instructions: {Definition}\nInput: {input}\nOutput:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_5(self, definition, inputs, output):
        input_text = "{Definition}\nQ: {input}\nA: ".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_6(self, definition, inputs, output):
        input_text = "Given the task definition and input, reply with output. {Definition}\n\n" \
                     "{input}\n".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_7(self, definition, inputs, output):
        input_text = "Teacher:{Definition}\nTeacher: Now, understand the problem? Solve this instance:" \
                     " {input}\nStudent:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_8(self, definition, inputs, output):
        input_text = "Q: {Definition}\n{input}\nA:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_9(self, definition, inputs, output):
        input_text = "Detailed Instructions: {Definition}\nProblem:{input}\n" \
                     "Solution:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_zs_template_10(self, definition, inputs, output):
        input_text = "Detailed Instructions: {Definition}\nQ: {input}\nA:".format(Definition=definition, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text
    
    def niv2_fs_template_1(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "{ex_input}\nSolution: {ex_output}\nWhy? undefined\n\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "You will be given a definition of a task first, then an example. Follow the example to solve" \
                     " a new instance of the task.\n{Definition}\n\n{ex_input_text}New input: {input}\nSolution:".format(
            Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_2(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "Example: {ex_input}\nOutput: {ex_output}\n\n".format(
                ex_input=ex_input, ex_output=ex_output)

        input_text = "Given the task definition, example input & output, solve the new input case.\n{Definition}" \
                     "\n{ex_input_text}New input case for you: {input}\nOutput:".format(
            Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_3(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "{ex_input}\nSolution: {ex_output}\nReason: undefined\n\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "Teacher: {Definition}\nTeacher: Now, understand the problem? If you are still confused, see " \
                     "the following example:\n{ex_input_text}Now, solve this instance: {input}\nStudent:".format(
            Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_4(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "Example input: {ex_input}\nExample output: {ex_output}\nExample explanation: undefined\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "{Definition}\n\n{ex_input_text}Q: {input}\nA:".format(
            Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_5(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "Problem: {ex_input}\nSolution: {ex_output}\nExplanation: undefined\n\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "Detailed Instructions: {Definition}\nSee one example below:\n{ex_input_text}Problem: " \
                     "{input}\nSolution:".format(Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_6(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "Problem: {ex_input}\nSolution: {ex_output}\nExplanation: undefined\n\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "Detailed Instructions: {Definition}\nSee one example below:\n{ex_input_text}Problem: " \
                     "{input}\nSolution:".format(Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_7(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "One example: {ex_input}\nSolution is here: {ex_output}\nExplanation: undefined\n\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "{Definition}\n{ex_input_text}Now, solve this: {input}\n" \
                     "Solution:".format(Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_8(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "Example\n{ex_input}\nAnswer: {ex_output}\nExplanation: undefined\n".format(
                ex_input=ex_input, ex_output=ex_output)
        input_text = "Part 1. Definition\n{Definition}\nPart 2. {ex_input_text}Part 3. Exercise" \
                     "\n{input}\nAnswer:".format(Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_9(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "Let me give you an example: {ex_input}\nThe answer to this example can be:" \
                             " {ex_output}\nHere is why: undefined\n\n".format(ex_input=ex_input, ex_output=ex_output)
        input_text = "{Definition}\n\n{ex_input_text}OK. solve this:\n{input}\n" \
                     "Answer:".format(Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text

    def niv2_fs_template_10(self, definition, inputs, output, ex_inputs, ex_outputs):
        ex_input_text = ""
        for ex_input, ex_output in zip(ex_inputs, ex_outputs):
            ex_input_text += "One example is below.\nQ: {ex_input}\nA: {ex_output}\nRationale: undefined\n".format(ex_input=ex_input, ex_output=ex_output)
        input_text = "{Definition}\n{ex_input_text}Q: {input}\nA:".format(Definition=definition, ex_input_text=ex_input_text, input=inputs)
        output_text = "{output}".format(output=output)
        return input_text, output_text






