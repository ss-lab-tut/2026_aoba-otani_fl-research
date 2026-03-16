import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. PyTorch用のデータセット定義
class LidarDataset(Dataset):
    def __init__(self, x_data, y_data):
        # [N, 64, 430, 2] -> [N, 2, 64, 430] に変換
        x_data = np.transpose(x_data, (0, 3, 1, 2))
        
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. FLのクライアントごとにデータを分割して読み込む関数
def load_fl_data(client_id: int, num_clients: int = 2):
    print(f"クライアント {client_id} のデータを読み込みます...")
    
    # 改良版データの読み込み（4クラスのラベルを読み込む）
    x_train_all = np.load("X_train.npy")       
    y_train_sub_all = np.load("y_train_sub.npy") 
    
    # ★共同研究者さんの仕様に合わせてラベルを3クラスに変換★
    # 0: sit はそのまま
    # 1: walk_stop はそのまま (1)
    # 2: walk_sit は 1 (walk系) に統合
    # 3: fall は 2 (fall) にスライド
    y_train_3class = np.copy(y_train_sub_all)
    y_train_3class[y_train_sub_all == 2] = 1 
    y_train_3class[y_train_sub_all == 3] = 2 
    
    # データをクライアント数で均等に分割
    data_per_client = len(x_train_all) // num_clients
    start_idx = client_id * data_per_client
    end_idx = start_idx + data_per_client
    
    x_client = x_train_all[start_idx:end_idx]
    y_client = y_train_3class[start_idx:end_idx]
    
    dataset = LidarDataset(x_client, y_client)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"クライアント {client_id}: {len(dataset)} 件のデータをセットアップ完了！")
    return dataloader

# 動作確認用のテストコード
if __name__ == "__main__":
    train_loader = load_fl_data(client_id=0, num_clients=2)
    
    for images, labels in train_loader:
        print("入力データの形:", images.shape) # (32, 2, 64, 430)
        print("ラベルデータの形:", labels.shape) # (32,)
        print("ラベルの中身（0, 1, 2の3分類になっているか）:", labels[:10].tolist())
        break