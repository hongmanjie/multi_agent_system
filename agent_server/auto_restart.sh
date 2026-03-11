设置服务自动重启

----retrieval_agent服务------
1. 编辑服务配置文件
   sudo vim /etc/systemd/system/retrieval_agent.service

2. 文件内容为
    [Unit]
    Description=Retrieval Agent Service
    After=network.target

    [Service]
    # 运行脚本的用户
    User=xtxk
   # Python脚本所在目录
    WorkingDirectory=/data_ssd/smart_catalog_release_v1.1/agent_server

   # 使用shell执行重定向
    ExecStart=/bin/bash -c '/home/xtxk/miniconda3/envs/ljy/bin/python retrieval_agent.py >> logs/retrieval_agent.log 2>&1'

   # 注意：systemd自己管理进程，不需要nohup和&
   # 如果进程退出，总是自动重启
    Restart=always
    RestartSec=10

   # 设置日志处理方式（可选）
    StandardOutput=journal
    StandardError=journal

    [Install]
    WantedBy=multi-user.target

3. 重新加载 systemd 配置
   sudo systemctl daemon-reload

4. 启用并启动服务
   sudo systemctl enable retrieval_agent.service
   sudo systemctl start retrieval_agent.service

5. 验证服务状态
   sudo systemctl status retrieval_agent.service
   确保 `Active: active (running)` 显示服务正在运行，且 `Restart:` 显示为 `always`。

