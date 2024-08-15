# VSCode Remote 設定ガイド
1. VSCode Remote インストール
2. アクセス元のPCで VSCode インストール
3. アクセス元のPCのVSCodeで Remote Development 拡張をインストール
4. アクセス元のPCの `.ssh/config` に以下を追加
    ```
    ForwardAgent yes
    StrictHostKeyChecking no
    TCPKeepAlive yes
    ServerAliveInterval 30
    ServerAliveCountMax 5

    Host sbir
      Hostname [作ったインスタンスの の public IP]
      User ubuntu
    ```    
5. アクセス先 `~/.ssh/authorized_keys` にアクセス元の `cat ~/.ssh/id_rsa.pub` の結果を追加
  - アクセス元の `cat ~/.ssh/id_rsa.pub` の結果が何も出ない場合、
    - `ssh-keygen` で `~/.ssh/id_rsa.pub` 作成
6. アクセス元のPCのVSCodeで Ctrl(Cmd) + Shift + P を押して、以下を入力
    ```
    Remote-SSH Connect
    ```
7. sbir を選択して アクセス先VSCodeに切り替え
8. open を選択して bldg-lod2-tool フォルダーを選択
9. アクセス先VSCodeで Ctrl(Cmd) + Shift + P を押して、以下を入力
    ```
    Python: Select Interpreter
    ```
10. バージョン 3.9.19 の Python を選択
11. アクセス先VSCodeで Pylance 拡張をインストール
12. アクセス先VSCodeで Pylint 拡張をインストール
13. アクセス先VSCodeで autopep8 拡張をインストール
