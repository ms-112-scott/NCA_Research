class OmniConfig(dict):
    """
    萬能 Config 物件：同時支援 '字典操作' 與 '屬性操作'。
    解決：AttributeError: 'dict' object has no attribute 'seed'
    """

    def __getattr__(self, name):
        # 允許用 .key 存取
        if name in self:
            return self[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name, value):
        # 允許用 .key = value 設定
        self[name] = value
