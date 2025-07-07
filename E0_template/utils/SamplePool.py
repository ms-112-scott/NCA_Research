import numpy as np
from typing import Dict, Any, List, Optional

class SamplePool:
    def __init__(self, data: Dict[str, np.ndarray], 
                 parent: Optional['SamplePool'] = None,
                 parent_idx: Optional[np.ndarray] = None):
        """
        建立 SamplePool，管理多個同長度的欄位資料（slot）

        Args:
            data: Dict[str, np.ndarray]，每個欄位一組資料（如 x, y 等）
            parent: 若為 sample 子集，則為父 pool
            parent_idx: 抽樣子集在原 pool 中對應的 index
        """
        self._parent = parent
        self._parent_idx = parent_idx
        self._slot_names = list(data.keys())
        
        lengths = [len(v) for v in data.values()]
        assert len(set(lengths)) == 1, "All slots must have same length."
        self._size = lengths[0]
        
        # 將所有欄位轉為屬性
        for name, array in data.items():
            setattr(self, name, np.asarray(array))

    def sample(self, n: int) -> 'SamplePool':
        """
        隨機抽取 n 筆資料，回傳新的 SamplePool 子集。
        """
        idx = np.random.choice(self._size, n, replace=False)
        sample_data = {
            name: getattr(self, name)[idx] for name in self._slot_names
        }
        return SamplePool(sample_data, parent=self, parent_idx=idx)

    def commit(self):
        """
        將此 SamplePool 的 slot 資料回寫至父層。
        """
        if self._parent is None or self._parent_idx is None:
            raise ValueError("No parent pool to commit to.")
        
        for name in self._slot_names:
            parent_array = getattr(self._parent, name)
            parent_array[self._parent_idx] = getattr(self, name)
