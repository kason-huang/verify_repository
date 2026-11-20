# 3DVLM 评测

- [图片评测](image_eval/README.md)判断黑色托盘是否为空，并统计托盘上的物品。入口为 `image_eval/evaluate.py`。
- [视频评测](video_eval/README.md)按时间窗口判断任务进度、结果和是否需要干预。

## 查看视频评测报告

在仓库根目录运行以下命令，并保持服务运行。

```sh
python3 video_eval/serve.py --port 8766 --output results/video-astra-v3-warmup --motvlm-output results/video-motvlm-v1
```

在浏览器打开[评测首页](http://127.0.0.1:8766/)，通过顶部导航栏选择 **Astra 离线报告**、**MoTVLM 离线报告**、**对比报告**或 **实时监控**。打开离线报告后，选择 episode，点击时间线即可回看判断、耗时和证据帧。

如果只想查看已有报告，可以直接在浏览器打开仓库根目录的 [index.html](index.html)。首页汇集图片和视频的对比报告，无需启动服务。运行视频评测的其他命令见[视频评测说明](video_eval/README.md)。
