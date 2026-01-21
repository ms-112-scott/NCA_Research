import torch
import unittest
from src.nca.model import BaseTextureNCA
from src.nca.trainer import NCATrainer
from src.nca.loss import MSELoss

class TestNCATrainer(unittest.TestCase):

    def test_single_train_step(self):
        """
        測試 NCATrainer 的 train_step() 是否能整合 Model 和 Loss Function 順利執行一步訓練。
        """
        # 1. 準備設定
        config = {
            'chn': 12,
            'img_size': 32,
            'hidden_n': 32,
            'batch_size': 2,
            'lr': 1e-3,
            'step_min': 8,
            'step_max': 16,
            'pool_reset_freq': 4,
        }
        
        # 2. 建立整合測試的物件
        model = BaseTextureNCA(config)
        target_img = torch.rand(1, 3, config['img_size'], config['img_size'])
        loss_fn = MSELoss(target_img)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        pool = model.seed(16)

        trainer = NCATrainer(
            model=model,
            pool=pool,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=loss_fn,
            config=config,
            device=torch.device('cpu')
        )

        # 記錄訓練前的模型參數
        param_before = model.w2.weight.clone()

        # 3. 執行單步訓練
        stats = trainer.train_step(current_step=0)

        # 記錄訓練後的模型參數
        param_after = model.w2.weight.clone()

        # 4. 驗證結果
        # 驗證 loss 是否為有效的數字
        self.assertFalse(torch.isnan(stats['loss']))
        self.assertFalse(torch.isinf(stats['loss']))
        
        # 驗證模型參數是否被更新 (梯度下降應該會改變參數)
        self.assertFalse(torch.equal(param_before, param_after))
        
        # 驗證回傳的 batch_x 是否有正確的形狀
        self.assertEqual(stats['batch_x'].shape, (config['batch_size'], config['chn'], config['img_size'], config['img_size']))


if __name__ == '__main__':
    unittest.main()
