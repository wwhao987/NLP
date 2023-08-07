FROM registry.cn-shanghai.aliyuncs.com/gerry_ai/eas-deploy:v1.0

# 指定工作文件夹
WORKDIR /data/eas

# 安装依赖的相关库
RUN ENV/bin/pip install tqdm flask

# 将主机/当前机器的代码文件copy到镜像内
COPY . /data/eas/

# 给定docker容器的启动命令
CMD ENV/bin/python /data/eas/intention_identification_run.py