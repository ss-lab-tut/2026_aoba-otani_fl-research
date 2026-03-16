import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import flwr as fl

# 先ほど作ったファイルから、データ読み込み関数とCNNモデルを呼び出す
from data_loader import load_fl_data
from model import LidarCNN

# ターミナルからクライアント番号（0か1）を受け取る仕組み
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# デバイスの設定（PCにNVIDIAのGPUがあれば使い、なければCPUを使う）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        # AIモデルをセット
        self.model = LidarCNN().to(device)
        # 自分の番号(cid)に応じた担当データを読み込む
        self.train_loader = load_fl_data(client_id=cid, num_clients=2)
        # 誤差の計算方法（3クラス分類用）と、重みの更新方法（Adam）を設定
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # サーバーから受け取った最新の重みをモデルにセット
        self.set_parameters(parameters)
        self.model.train()
        
        print(f"\n>> [クライアント {client_id}] 手元のデータで学習を開始します...")
        total_loss = 0.0
        correct = 0
        total = 0
        
        # PyTorchの本格的な学習ループ（データを見ながら重みを更新）
        for images, labels in self.train_loader:
            images, labels = images.to(device), labels.to(device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f">> [クライアント {client_id}] 学習完了! 精度(Accuracy): {accuracy*100:.2f}%")
        
        # 学習で賢くなった重みをサーバーに返す
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {"accuracy": 0.0}

if __name__ == "__main__":
    print(f"--- クライアント {client_id} を起動し、サーバーに接続します ---")
    fl.client.start_client(
        server_address="127.0.0.1:8080", 
        client=FlowerClient(cid=client_id).to_client()
    )