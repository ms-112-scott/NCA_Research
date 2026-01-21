import torch
import unittest
from src.nca.model import BaseTextureNCA

class TestBaseTextureNCA(unittest.TestCase):

    def test_seed_creation(self):
        """
        測試 seed() 方法是否能產生正確形狀和設備的張量。
        """
        config = {'chn': 12, 'img_size': 64}
        model = BaseTextureNCA(config)
        
        n_seeds = 4
        seed_tensor = model.seed(n_seeds)
        
        # 驗證形狀
        self.assertEqual(seed_tensor.shape, (n_seeds, config['chn'], config['img_size'], config['img_size']))
        
        # 驗證設備
        self.assertEqual(seed_tensor.device, model.w2.weight.device)

    def test_forward_pass(self):
        """
        測試模型的前向傳播是否能順利執行並回傳正確形狀的張量。
        """
        config = {'chn': 16, 'img_size': 32, 'hidden_n': 64}
        model = BaseTextureNCA(config)
        
        seed_tensor = model.seed(2)
        output_tensor = model(seed_tensor)
        
        # 驗證輸出形狀是否與輸入相同
        self.assertEqual(output_tensor.shape, seed_tensor.shape)

if __name__ == '__main__':
    unittest.main()
