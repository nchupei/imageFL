import torch
import torch.nn as nn
import torch.nn.functional as F

def build_cnn(input_dim, output_dim):
    """構建真正的CNN模型，用於網絡攻擊檢測"""
    class NetworkAttackCNN(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_rate=0.3):
            super().__init__()
            
            # 🚀 真正的CNN架構：使用1D卷積處理網絡流量特徵
            # 將輸入重塑為適合卷積的格式: (batch, channels, sequence_length)
            
            # 特徵重塑層：將78維特徵重塑為2D格式
            self.input_reshape = nn.Linear(input_dim, 128)  # 78 -> 128
            
            # 簡化的卷積層組1
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),  # 1D卷積
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.3),
                nn.MaxPool1d(kernel_size=2, stride=2)  # 128 -> 64
            )
            
            # 簡化的卷積層組2
            self.conv_block2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.4),
                nn.MaxPool1d(kernel_size=2, stride=2)  # 64 -> 32
            )
            
            # 簡化的卷積層組3
            self.conv_block3 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.AdaptiveAvgPool1d(8)  # 自適應池化到固定長度
            )
            
            # 簡化的全連接層
            self.classifier = nn.Sequential(
                nn.Linear(64 * 8, 128),  # 64 * 8 = 512
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                
                nn.Linear(64, output_dim)
            )
            
            # 初始化權重
            self._initialize_weights()
        
        def _initialize_weights(self):
            """使用Xavier初始化"""
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # 輸入重塑: (batch, 78) -> (batch, 128)
            x = self.input_reshape(x)
            
            # 重塑為卷積格式: (batch, 128) -> (batch, 1, 128)
            x = x.unsqueeze(1)
            
            # 卷積特徵提取
            x = self.conv_block1(x)  # (batch, 16, 64)
            x = self.conv_block2(x)  # (batch, 32, 32)
            x = self.conv_block3(x)  # (batch, 64, 8)
            
            # 展平: (batch, 64, 8) -> (batch, 64*8)
            x = x.view(x.size(0), -1)
            
            # 分類
            x = self.classifier(x)
            
            return x
    
    return NetworkAttackCNN(input_dim, output_dim)

class AdvancedServerClassifier(nn.Module):
    """改進的服務器端分類器"""
    
    def __init__(self, input_dim=64, num_classes=4, dropout_rate=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(32, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用更穩定的權重初始化 - 極度修復版本"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 🚨 極度修復：使用極度保守的初始化，解決梯度爆炸
                nn.init.xavier_uniform_(m.weight, gain=1.0)  # 恢復正常gain值
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)

class FocalLoss(nn.Module):
    """改進的Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_balanced_sampler(labels):
    """創建改進的平衡採樣器"""
    from torch.utils.data import WeightedRandomSampler
    
    # 計算類別權重
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )