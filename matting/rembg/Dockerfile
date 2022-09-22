#Dockerfile
#based on nvcr.io/nvidia/tensorrt:20.09-py3
#change python version from 3.6 to 3.7 for pip install rembg[gpu]
FROM host[harbor]/dmcv/lianshuailong/tensorrt:rembg_trt_v1
ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com A4B469963BF863CC 
RUN sed -i "s#archive.ubuntu.com#mirrors.aliyun.com#g" /etc/apt/sources.list && \
    sed -i "s#security.ubuntu.com#mirrors.aliyun.com#g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsm6 \
        python-opencv \
        lrzsz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /matting

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --upgrade pip
# RUN pip install rembg[gpu]
# RUN mkdir ~/.u2net
# COPY /data3/lianshuailong/models/rembg/* ~/.u2net
COPY ./*.py /matting/
COPY ./*.jpg /matting/ 
RUN chmod -R 777 /matting




#ADD ./ /root/
#RUN chmod +x /root/start_service.sh
#EXPOSE 8002

#WORKDIR /root
#ENTRYPOINT ./start_service.sh
