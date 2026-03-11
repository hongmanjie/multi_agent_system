设置服务自动重启

----qwen3_vl服务------
1. 编辑服务配置文件
   sudo vim /etc/systemd/system/qwen3_vl.service

2. 文件内容为
    [Unit]
    Description=Qwen3_vl Service
    After=network.target

    [Service]
    # 运行脚本的用户
    User=xtxk
   # Python脚本所在目录
    WorkingDirectory=/data_ssd/smart_catalog_release_v1.1/qwen3_video_server

   # 使用shell执行重定向
    ExecStart=/bin/bash -c '/home/xtxk/miniconda3/envs/zsc/bin/python qwen3vl_video_service.py >> logs/qwen3vl_video_service.log 2>&1'

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
   sudo systemctl enable qwen3_vl.service
   sudo systemctl start qwen3_vl.service

5. 验证服务状态
   sudo systemctl status qwen3_vl.service
   确保 `Active: active (running)` 显示服务正在运行，且 `Restart:` 显示为 `always`。

