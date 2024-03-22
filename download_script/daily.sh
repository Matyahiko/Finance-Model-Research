#!/bin/bash

# tmuxセッションの名前
SESSION_NAME="daily_download"

# tmuxセッションが存在するかチェック
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    # セッションが存在しない場合は新しいセッションを作成
    tmux new-session -d -s $SESSION_NAME
fi

# 既存のセッションにアタッチ
tmux attach-session -t $SESSION_NAME

# 新しいウィンドウを作成し、daily_download.pyスクリプトを実行
tmux new-window -t $SESSION_NAME:1 -n "daily_download" "python3 /root/src/download_script/download.py"

# デタッチ
tmux detach-client