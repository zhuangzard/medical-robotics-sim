# 🚀 最简单的启动方法

## 方案：直接从 GitHub 打开（推荐）⭐

**无需上传！无需授权！**

### 一步打开

点击这个链接：
```
https://colab.research.google.com/github/zhuangzard/medical-robotics-sim/blob/main/notebooks/week1_training_colab.ipynb
```

**或者**手动构造：
```
https://colab.research.google.com/github/<username>/<repo>/blob/<branch>/notebooks/<notebook-name>.ipynb
```

### 设置和运行

1. **第一次打开**可能提示 "Connect to GitHub"
   - 点击 "Connect"
   - 授权 Colab 访问你的 GitHub（一次性）

2. **Runtime → Change runtime type**
   - Hardware accelerator: GPU
   - GPU type: V100（推荐）或 A100

3. **Runtime → Run all**
   - 第一次会警告 "This notebook was not authored by Google"
   - 点击 "Run anyway"

4. **等待训练完成** (~8-10 小时)

5. **查看结果** (自动保存到 Drive)
   - `/MyDrive/medical-robotics-results/20260205_HHMMSS/`

---

## 为什么这个方法最好？

✅ **零配置** - 不需要 Google Cloud 项目  
✅ **不需要上传** - 直接从 GitHub 加载  
✅ **自动同步** - Git push 后立即可用  
✅ **私有仓库支持** - 用你的 GitHub 账号登录就能访问  
✅ **可分享** - 发链接给团队成员（如果他们有 repo 权限）

---

## 完整 URL 模板

```
https://colab.research.google.com/github/zhuangzard/medical-robotics-sim/blob/main/notebooks/week1_training_colab.ipynb
```

**如果更新 notebook**:
1. Git commit + push 到 GitHub
2. 刷新 Colab 页面
3. 自动加载最新版本 ✅

---

## 进度监控

训练会自动写进度到 Drive：
```
/MyDrive/medical-robotics-progress/training_progress.json
```

查看：
```bash
cat ~/Google\ Drive/medical-robotics-progress/training_progress.json
```

---

## 其他方案（如果 GitHub 方案不工作）

### 方案 B: 手动上传

1. 下载 notebook: `medical-robotics-sim/notebooks/week1_training_colab.ipynb`
2. 打开 https://drive.google.com
3. 上传文件
4. 右键 → Open with → Google Colaboratory

### 方案 C: Drive API 自动上传

需要一次性设置（15分钟）:
1. 创建 Google Cloud 项目
2. 启用 Drive API
3. 下载 OAuth 凭证
4. 运行 `setup_drive_auth.py`

详见 `UPLOAD_GUIDE.md`

---

**最简单**: 用 GitHub 方案 ⭐  
**最快**: 5 秒打开，无需任何设置 🚀
