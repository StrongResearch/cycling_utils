from collections import defaultdict

import torch
import torch.distributed as dist


class MetricsTracker:
    """
    This is a general purpose MetricsTracker to assist with recording metrics from
    a disributed cluster.

    The MetricsTracker is initialised without any prior knowledge of the metrics
    to be tracked.

    >>> metrics = MetricsTracker()

    Metrics can be accumulated as required, for example after each batch is procesed
    by the model, by passing a dictionary with metrics to be updated, then reduced
    accross all nodes. Metric values are stored in a defaultdict.

    >>> preds = model(input)
    >>> loss = loss_fn(preds, targs)
    >>> metrics.update({"images_seen": len(images), "loss": loss.item()})
    >>> metrics.reduce()

    Metrics are assumed to be summable scalar values. After calling reduce(), the
    metrics.local object contains the sum of corresponding metrics from all nodes
    which can be used for intermediate reporting or logging.

    >>> writer = SummaryWriter()
    >>> for metric,val in metrics.local.items():
    >>>     writer.add_scalar(metric, val, step)
    >>> writer.flush()
    >>> writer.close()

    Once all processing of the current batch has been completed, the MetricsTracker
    can be prepared for the next batch using reset_local().

    >>> metrics.reset_loca()

    Metrics are also accumulated for consecutive batches in the metrics.agg object.
    At the end of an epoch the MetricsTracker can be reset using end_epoch().

    >>> metrics.end_epoch()

    The MetricsTracker saves a copy of the accumulated metrics (metrics.agg) for
    each epoch which can be stored within a checkpoint and accessed later.

    >>> metrics_history = metrics.epoch_reports
    >>> loss_history = [epoch["loss"] for epoch in metrics_history]
    """

    def __init__(self):
        self.local = defaultdict(float)
        self.agg = defaultdict(float)
        self.epoch_reports = []

    def update(self, metrics: dict):
        for m, v in metrics.items():
            self.local[m] += v
        return self

    def reduce(self):
        names, local = zip(*self.local.items())
        local = torch.tensor(
            local, dtype=torch.float32, requires_grad=False, device="cuda"
        )
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        self.local = defaultdict(float, zip(names, local.tolist()))
        for k in self.local:
            self.agg[k] += self.local[k]
        return self

    def reset_local(self):
        self.local = defaultdict(float)
        return self

    def end_epoch(self):
        self.epoch_reports.append(dict(self.agg))
        self.agg = defaultdict(float)
        self.reset_local()
        return self

    def state_dict(self):
        # Note local is not saved as local results will not be rank consistent
        return {"epoch_reports": self.epoch_reports, "agg": self.agg}

    def load_state_dict(self, state_dict):
        self.epoch_reports = state_dict["epoch_reports"]
        self.agg = state_dict["agg"]
        self.reset_local()
        return self
