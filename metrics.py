import torch


class LogitsMetric:

    def __init__(self, fabric):
        self.fabric = fabric

    def __call__(self, classes, gold_classes):
        assert len(classes) == len(gold_classes)
        correct = torch.tensor(0, dtype=torch.int32, device=self.fabric.device)
        length = torch.tensor(len(classes), dtype=torch.int32, device=self.fabric.device)
        for cls, gold in zip(classes, gold_classes):
            if int(cls) == int(gold):
                correct += 1

        return correct, length

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


class OutputMetric:

    def __init__(self):
        return

    def __call__(self, preds, golds):
        correct = torch.tensor(0, dtype=torch.int32, device=self.fabric.device)
        length = torch.tensor(len(preds), dtype=torch.int32, device=self.fabric.device)
        assert len(preds) == len(golds)
        for pred, gold in zip(preds, golds):
            for x in ["(", ")", ".", ":"]:
                pred = pred.replace(x, "")
                gold = gold.replace(x, "")
            if pred.strip().lower() == gold.strip().lower():
                correct += 1

        return correct, length

