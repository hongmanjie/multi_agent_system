sudo docker run -itd --gpus=all --shm-size 8G --restart=always  \
  -v /data_ssd/smart_catalog_release_v1.1/smart_maas_server:/workspace \
  -v /data_ssd/maas_models/weight:/weight \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -w /workspace \
  -p 8010:8010 \
  --name maas_server \
  maas_server:v1.1 \
  /bin/bash -c "python3 api_server.py"