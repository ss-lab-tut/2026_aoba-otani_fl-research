import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict

# 1. 超シンプルなダミーモデル（後でここにTime-C GANなどを組み込みます）
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

# 2. Flowerクライアントの定義
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = DummyModel()

    # サーバーへ現在のモデルの重みを送る
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # サーバーから受け取った新しい重みをモデルにセットする
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # クライアント側での学習処理（今回はプリント文のみのダミー）
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(">> クライアントで学習(fit)が実行されました")
        # 本来はここで手元のデータを使って学習ループを回します
        return self.get_parameters(config=config), 10, {} # 10はダミーのデータサンプル数

    # クライアント側での評価処理
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print(">> クライアントで評価(evaluate)が実行されました")
        return 0.0, 10, {"accuracy": 1.0}

# 3. クライアントを生成する関数
def client_fn(cid: str):
    return FlowerClient().to_client()

# 4. シミュレーションの実行
if __name__ == "__main__":
    print("ローカルFLシミュレーションを開始します...")
    # 1台のPC内で、2つのクライアントを立ち上げて2ラウンドの学習をシミュレーション
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=2),
    )