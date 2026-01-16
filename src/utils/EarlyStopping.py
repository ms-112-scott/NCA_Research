# src/utils.py


class EarlyStopping:
    """
    早停機制 (Early Stopping)。
    若 Loss 在 patience 步數內未改善超過 min_delta，則觸發停止。
    """

    def __init__(self, patience: int = 1000, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        檢查是否需要停止訓練。

        Returns:
            bool: True 表示需要停止訓練。
        """
        # 檢查 Loss 是否有顯著改善
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置計數器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
