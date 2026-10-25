"""Shared application state."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from . import data as D
from . import engine as E


class AppState(QObject):
    raw_changed = Signal()
    prepared_changed = Signal()
    config_changed = Signal()
    result_changed = Signal()
    training_changed = Signal()

    def __init__(self):
        super().__init__()
        self.raw: D.RawDataset | None = None
        self.prepared: D.PreparedData | None = None
        self.config = E.TrainConfig()
        self.result: E.RunResult | None = None
        self.worker: E.TrainingWorker | None = None

    # -- raw data ----------------------------------------------------------- #
    def set_raw(self, raw: D.RawDataset) -> None:
        self.raw = raw
        self.prepared = None
        self.raw_changed.emit()

    def set_prepared(self, prepared: D.PreparedData) -> None:
        self.prepared = prepared
        self.raw = D.RawDataset(
            name=prepared.name,
            frame=prepared.frame,
            path=prepared.source_path,
            source="prepared",
        ) if self.raw is None else self.raw
        self.prepared_changed.emit()

    # -- config/results ----------------------------------------------------- #
    def set_config(self, config: E.TrainConfig) -> None:
        self.config = config
        self.config_changed.emit()

    def set_result(self, result: E.RunResult | None) -> None:
        self.result = result
        self.result_changed.emit()

    def set_worker(self, worker: E.TrainingWorker | None) -> None:
        self.worker = worker
        self.training_changed.emit()

    @property
    def training_active(self) -> bool:
        return self.worker is not None and self.worker.isRunning()
