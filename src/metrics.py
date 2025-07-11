"""
src.metrics.py
Torchmetrics for image prediction
BoMeyering 2025
"""
import torch
from torchmetrics.classification import F1Score, Accuracy, JaccardIndex, Precision, Recall
from torchmetrics.segmentation import MeanIoU

class MetricLogger:
    """ Class to log metrics during an epoch """
    def __init__(self, num_classes: int, device: str):
        """
        Initialize the MetricLogger

        Args:
            num_classes (int): total number of classes to track
            device (str): Where the computation will be taking place
        """
        self.avg_metrics = {
            'f1_score': F1Score(num_classes=num_classes, task='multiclass').to(device),
            'jaccard_index': JaccardIndex(num_classes=num_classes, task='multiclass').to(device),
            'mIOU': MeanIoU(num_classes=num_classes, include_background=True, per_class=False, input_format='index').to(device)
            # 'accuracy': Accuracy(num_classes=num_classes, task='multiclass').to(device),
            # 'precision': Precision(num_classes=num_classes, task='multiclass').to(device),
            # 'recall': Recall(num_classes=num_classes, task='multiclass').to(device)
        }
        self.mc_metrics = {
            'f1_score': F1Score(num_classes=num_classes, task='multiclass', average='none').to(device),
            'jaccard_index': JaccardIndex(num_classes=num_classes, task='multiclass', average='none').to(device),
            'mIOU': MeanIoU(num_classes=num_classes, include_background=True, per_class=True, input_format='index').to(device)
            # 'accuracy': Accuracy(num_classes=num_classes, task='multiclass', average='none').to(device),
            # 'precision': Precision(num_classes=num_classes, task='multiclass', average='none').to(device),
            # 'recall': Recall(num_classes=num_classes, task='multiclass', average='none').to(device)
        }
        self.batch_results = {
            'avg': {},
            'mc': {}
        }
    
    def update(self, preds: torch.tensor, targets: torch.tensor, verbose: bool=False):
        # update avg metrics
        for key, metric in self.avg_metrics.items():
            self.batch_results['avg'][key] = metric(preds, targets)
        
        # update multiclass metrics
        for key, metric in self.mc_metrics.items():
            self.batch_results['mc'][key] = metric(preds, targets)
        
        if verbose:
            self.print_metrics('both')
        
    def compute(self):
        try:
            avg_metrics = {k: metric.compute() for k, metric in self.avg_metrics.items()}
            mc_metrics = {k: metric.compute() for k, metric in self.mc_metrics.items()}
        except Exception as e:
            print(e)
            avg_metrics, mc_metrics = None, None
        return avg_metrics, mc_metrics
    
    def print_metrics(self, type: str):
        if type=='avg':
            print(self.batch_results['avg'])
        elif type=='mc':
            print(self.batch_results['mc'])
        elif type=='both':
            print(self.batch_results)

    def reset(self):
        for k, metric in self.avg_metrics.items():
            metric.reset()
        for k, metric in self.mc_metrics.items():
            metric.reset()


if __name__ == '__main__':
    batches = 20
    num_classes = 5
    metrics = MetricLogger(num_classes)

    for i in range(batches):
        preds = torch.randn(10, 5, 20, 20)
        targets = torch.randint(num_classes, (10, 20, 20))
        # targets = torch.argmax(preds.softmax(dim=1), dim=1)

        metrics.update(preds=preds, targets=targets, verbose=False)

    avg, mc = metrics.compute()

    print(avg)
    print(mc)

    # metrics.reset()
    # avg, mc = metrics.compute()

    # print(avg)
    # print(mc)
    
