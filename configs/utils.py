import torch
import torch.nn
import json


class LogitsMetric:

    def __init__(self):
        self.with_logits = True
        self.fabric = None

    def __call__(self, classes, gold_classes):
        assert len(classes) == len(gold_classes)
        correct = 0
        for cls, gold in zip(classes, gold_classes):
            if int(cls) == int(gold):
                correct += 1

        return correct

    def classify(self, logits, label_spaces_ids, sample_to):
        pass

    def to_device(self, tokenizer, fabric):
        self.fabric = fabric


class OutputMetric:

    def __init__(self):
        self.with_logits = False

    def __call__(self, preds, golds):
        pass


class OptionMatchingAccuracy(OutputMetric):

    def __init__(self):
        super(OptionMatchingAccuracy, self).__init__()

    def __call__(self, preds, golds):
        correct = 0
        assert len(preds) == len(golds)
        for pred, gold in zip(preds, golds):
            for x in ["(", ")", ".", ":"]:
                pred = pred.replace(x, "")
                gold = gold.replace(x, "")
            if pred.strip().lower() == gold.strip().lower():
                correct += 1

        return correct


class ClassificationMatchAccuracy(OutputMetric):

    def __init__(self):
        super(ClassificationMatchAccuracy, self).__init__()

    def __call__(self, preds, golds):
        correct = 0
        assert len(preds) == len(golds)
        for pred, gold in zip(preds, golds):
            if pred.strip() == gold.strip():
                correct += 1

        return correct


class ClassificationAccuracy(LogitsMetric):

    def __init__(self, label_space: list):
        super(ClassificationAccuracy, self).__init__()
        self.label_space = label_space
        self.label_space_tokens = None

    def to_device(self, tokenizer, fabric):
        label_space_tokens = tokenizer(self.label_space, padding=True, return_tensors="pt")["input_ids"]
        self.label_space_tokens = fabric.to_device(label_space_tokens)
        self.fabric = fabric

    def classify(self, logits, label_spaces_ids, sample_to):
        softmax = torch.nn.Softmax(dim=-1)
        logits = softmax(logits)
        max_seq_len = logits.shape[1]
        label_space_tokens = self.label_space_tokens
        if max_seq_len < label_space_tokens.shape[1]:
            label_space_tokens = label_space_tokens[:, :max_seq_len]

        labels = torch.unsqueeze(label_space_tokens, 0)
        labels = labels.repeat(logits.shape[0], 1, 1)
        scores = torch.ones_like(labels, dtype=torch.float)
        scores = self.fabric.to_device(scores)
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                for k in range(scores.shape[2]):
                    scores[i][j][k] = logits[i][k][labels[i][j][k]]

        scores = torch.prod(scores, dim=-1)
        classes = torch.argmax(scores, dim=-1)
        return classes


class ClassificationGivenLabel(LogitsMetric):
    def __init__(self):
        super(ClassificationGivenLabel, self).__init__()

    def classify(self, logits, label_spaces_ids, sample_to):
        softmax = torch.nn.Softmax(dim=-1)
        logits = softmax(logits)
        batch_size, option_num, max_seq_len = label_spaces_ids.shape
        scores = torch.ones_like(label_spaces_ids, dtype=torch.float)
        scores = self.fabric.to_device(scores)

        for i in range(batch_size):
            seq_len = sample_to[i]
            for j in range(option_num):
                for k in range(min(seq_len, logits.shape[1])):
                    scores[i][j][k] = logits[i][k][label_spaces_ids[i][j][k]]

        scores = torch.prod(scores, dim=-1)
        classes = torch.argmax(scores, dim=-1)
        return classes


def load_BBL_file(file_dir, example_id: list, count_left: int):
    items, examples = [], []
    file_items = json.load(open(file_dir, "r", encoding="utf-8"))["examples"]
    for raw_item in file_items:
        item = {"question": raw_item["input"]}
        options = []
        answer = None
        for k, v in raw_item["target_scores"].items():
            options.append(k)
            if int(v) == 1:
                answer = k
        assert answer is not None
        item["options"] = options
        item["answer"] = answer
        items.append(item)
    for i in example_id:
        if count_left != 0:
            examples.append(items.pop(i))
            count_left -= 1
    return items, examples, count_left

