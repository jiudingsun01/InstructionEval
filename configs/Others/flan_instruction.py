import os
import json
from configs.utils import match_accuracy


special_tokens = []


def load_data(args):
    if args.promptsource_id is not None:
        raise NotImplementedError("This benchmark does not have Promptsource instructions!")
    else:
        test_set = []
        data = json.load(open(os.path.join(args.input_dir, "intent_recognition/task.json"), "r"))
        items = data["examples"]
        shots = items[:5]
        items = items[5:]
        prefix = "Given a set of four words, generate the category that the words belong to. Words are separated" \
                 " by commas. The possible categories are add_to_playlist, book_restaurant, get_weather, " \
                 "play_music, search_screening_event, search_creative_work, and rate_book.\nFive examples are below\n"
        for shot in shots:
            answer = ""
            prefix += "Q: {}\n".format(shot["input"])
            for key, value in shot["target_scores"].items():
                if int(value) == 1:
                    answer = key
            prefix += "A: {}\n".format(answer)

        for item in items:
            input_sequence = prefix
            answer = ""
            input_sequence += "Q: {}\n".format(item["input"])
            for key, value in item["target_scores"].items():
                if int(value) == 1:
                    answer = key
            input_sequence += "A: "
            assert answer
            target_sequence = answer
            test_set.append(
                {"input": input_sequence, "target": target_sequence}
            )

        return [], [], test_set


def evaluate(args, outputs, targets, data):
    return match_accuracy(args, outputs, targets)

