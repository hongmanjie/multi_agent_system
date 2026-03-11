## 回调服务，用于推送编目结果

nohup python task_callback_v3.py > logs/task_callback_v3.log 2>&1 &
日志文件为 logs/task_callback_v3.log



## 编目任务队列服务，用于管理视频编目任务

nohup python catalog_server_v3.py > logs/catalog_server_v3.log 2>&1 &
日志文件为 logs/catalog_server_v3.log


## 编目流程

1. 客户端提交编目任务到队列
2. 队列服务从队列中取出任务
3. 队列服务将任务发送给回调服务
4. 回调服务收到任务后，调用编目服务进行编目
5. 编目服务完成编目后，将结果发送给回调服务
6. 回调服务收到结果后，将结果发送给客户端
