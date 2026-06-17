# 实时量化看板

A股主板实时量化分析工具，支持东方财富真实数据、技术指标、每日推荐、策略回测、Claude AI 对话。

## 功能

- **大盘情绪**：涨跌家数、涨跌停统计、情绪评分
- **智慧选股**：全市场主板扫描，量比/换手率/均线多头排列筛选
- **卖出信号**：持仓代码一键检查技术卖点
- **个股分析**：RSI、MACD、MA均线、K线图
- **策略回测**：双均线策略，收益/回撤/夏普率
- **Claude AI**：基于实时数据的智能问答

-----

## 部署到 Railway（推荐，免费）

### 第一步：上传代码到 GitHub

```bash
git init
git add .
git commit -m "init: 实时量化看板"
git branch -M main
git remote add origin https://github.com/你的用户名/quant-dashboard.git
git push -u origin main
```

### 第二步：Railway 部署

1. 访问 [railway.app](https://railway.app) → 用 GitHub 登录
1. 点击 **New Project** → **Deploy from GitHub repo** → 选择你的仓库
1. Railway 自动识别 Dockerfile，点击 **Deploy**
1. 部署成功后，进入项目 → **Settings** → **Generate Domain**，获得永久域名

### 第三步：配置环境变量

在 Railway 项目页面 → **Variables** 标签，添加：

|变量名                |值           |说明                                                  |
|-------------------|------------|----------------------------------------------------|
|`ANTHROPIC_API_KEY`|`sk-ant-...`|Claude AI 密钥（[获取地址](https://console.anthropic.com/)）|
|`USE_REAL`         |`true`      |使用东财真实数据（false = 模拟数据）                              |


> ⚠️ API Key **只在环境变量里填**，绝不要写进代码或上传到 GitHub！

-----

## 部署到 Render（备选，免费）

1. 访问 [render.com](https://render.com) → 用 GitHub 登录
1. **New** → **Web Service** → 选择仓库
1. Runtime 选 **Docker**，Free Plan
1. **Environment Variables** 里填 `ANTHROPIC_API_KEY`
1. 点击 **Create Web Service**

> 注意：Render 免费版 15 分钟无访问会休眠，首次请求需等待约 30 秒唤醒。

-----

## 本地运行

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-你的密钥
python app.py
# 打开 http://localhost:8088
```

-----

## 免责声明

本工具为算法量化输出，所有分析结果**不构成投资建议**。股市有风险，入市需谨慎。