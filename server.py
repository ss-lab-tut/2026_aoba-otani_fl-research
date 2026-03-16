import flwr as fl

if __name__ == "__main__":
    print("サーバーを起動します...")
    # 2ラウンドの学習を指示するサーバー
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=2)
    )