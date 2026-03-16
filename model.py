import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarCNN(nn.Module):
    def __init__(self):
        super(LidarCNN, self).__init__()
        
        # 入力: (バッチサイズ, 2, 64, 430)
        # 1層目の畳み込み (2チャンネル -> 16チャンネルに特徴を抽出)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        # 2層目 (16 -> 32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 3層目 (32 -> 64)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 画像サイズを半分にするプーリング層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- 画像サイズの計算 ---
        # 初期: 64 x 430
        # Pool1後: 32 x 215
        # Pool2後: 16 x 107
        # Pool3後: 8 x 53
        # 最終的な特徴量の数: 64(チャンネル) * 8(縦) * 53(横) = 27136
        
        # 全結合層（抽出した特徴から3クラスに分類する）
        self.fc1 = nn.Linear(64 * 8 * 53, 128)
        self.fc2 = nn.Linear(128, 3) # 出力は「3クラス」

    def forward(self, x):
        # 畳み込み -> 活性化関数(ReLU) -> プーリング のセットを3回繰り返す
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 2次元のデータを1次元の長い配列に平坦化（Flatten）
        x = x.view(-1, 64 * 8 * 53)
        
        # 全結合層を通して分類
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 最終出力
        return x

# 動作確認用のテストコード
if __name__ == "__main__":
    print("モデルの構造テストを開始します...")
    
    # モデルの実体化
    model = LidarCNN()
    
    # data_loaderで確認したのと同じ、ダミーの入力データ(バッチ32)を作成
    dummy_input = torch.randn(32, 2, 64, 430)
    
    # モデルにデータを流し込む
    output = model(dummy_input)
    
    print("入力データの形:", dummy_input.shape)
    print("出力データの形:", output.shape)
    
    if output.shape == (32, 3):
        print("大成功！モデルの次元計算は完璧です。")