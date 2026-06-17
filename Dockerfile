FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --use-deprecated=legacy-resolver

# 复制代码
COPY "app (2).py" .

# 暴露端口 (Railway/Render 会自动读取 PORT 环境变量)
EXPOSE 8088

CMD ["python", "app (2).py"]
