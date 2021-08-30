import numpy as np

from typing import List, Dict
from tabulate import tabulate
from collections import defaultdict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


class EpochStats:
    def __init__(self):
        self.sizes = [] # number of elements per step
        self.losses = []
        self.ner_losses = []
        self.lm_losses = []

        self.probs = []
        self.preds = []
        self.golds = []

    def loss_step(self, loss: float, ner_loss: float, lm_loss: float, batch_size: int):
        self.losses.append(loss)
        self.ner_losses.append(ner_loss)
        self.lm_losses.append(lm_loss)
        self.sizes.append(batch_size)

    def step(self, scores, target, mask, loss, ner_loss, lm_loss):
        self.loss_step(loss, ner_loss, lm_loss, len(scores))

        probs, classes = scores.max(dim=2)

        for i in range(len(scores)):
            prob_i = probs[i][mask[i] == 1].cpu().tolist()
            pred_i = classes[i][mask[i] == 1].cpu().tolist()
            gold_i = target[i][mask[i] == 1].cpu().tolist()

            self.preds.append(pred_i) # self.preds.extend(pred_i)
            self.golds.append(gold_i) # self.golds.extend(gold_i)
            self.probs.append(prob_i) # self.probs.extend(prob_i)

    def loss(self, loss_type: str = ''):
        if loss_type == 'ner':
            losses = self.ner_losses
        elif loss_type == 'lm':
            losses = self.lm_losses
        else:
            losses = self.losses
        return np.mean([l for l, s in zip(losses, self.sizes) for _ in range(s)]), np.min(losses), np.max(losses)

    def _map_to_labels(self, index2label):
        # Predictions should have been as nested list to separate predictions
        # Since we store the predictions across epochs during training, we need to wrap up this in a try except
        # so that it handles the flattened lists in case they are not nested. New runs will be nested
        try:
            golds = [[index2label[j] for j in i] for i in self.golds]
            preds = [[index2label[j] for j in i] for i in self.preds]
        except TypeError:
            golds = [index2label[i] for i in self.golds]
            preds = [index2label[i] for i in self.preds]
        return golds, preds

    def metrics(self, index2label: [List[str], Dict[int, str]]):
        golds, preds =self._map_to_labels(index2label)

        f1 = f1_score(golds, preds)
        p = precision_score(golds, preds)
        r = recall_score(golds, preds)

        return f1, p, r

    def get_classification_report(self, index2label: [List[str], Dict[int, str]]):
        golds, preds = self._map_to_labels(index2label)

        cr = classification_report(golds, preds, digits=5)
        return report2dict(cr)

    def print_classification_report(self, index2label: [List[str], Dict[int, str]] = None, report = None):
        assert index2label is not None or report is not None

        if report is None:
            report = self.get_classification_report(index2label)

        printcr(report)


def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x.strip() for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


def printcr(report, classes=None, sort_by_support=False):
    headers = ['classes', 'precision', 'recall', 'f1-score', 'support']

    if classes is None:
        classes = [k for k in report.keys() if k not in {'macro avg', 'micro avg'}]

        if sort_by_support:
              classes = sorted(classes, key=lambda c: report[c]['support'], reverse=True)
        else: classes = sorted(classes)

    if 'macro avg' not in classes: classes.append('macro avg')
    if 'micro avg' not in classes: classes.append('micro avg')

    table = []
    for c in classes:
        if c == 'macro avg':
            table.append([])
        row = [c]
        for h in headers:
            if h not in report[c]:
                continue
            if h in {'precision', 'recall', 'f1-score'}:
                  row.append(report[c][h] * 100)
            else: row.append(report[c][h])
        table.append(row)
    print(tabulate(table, headers=headers, floatfmt=(".3f", ".3f", ".3f", ".3f")))
    print()

