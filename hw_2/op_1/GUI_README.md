# GUI_README

## 文件说明
- `gui.py`：GUI 主体实现，提供 `SVDCWindow` 类与 `SVDCWindow.run()` 入口。
- `main.py`：仅负责启动程序。

## 运行方式
1. 进入项目根目录。
2. 使用 `conda activate USTC_MM_26` 激活环境。
3. 运行 `python main.py`。

## 界面结构
- 主窗口：顶部菜单栏 + 下方活动背景区域。
- 顶部菜单：`File -> Load image...`。
- 每次成功加载图片会创建一个可独立操作的 `Target` 窗口（允许重复创建）。

## Target 窗口
- 窗口名为图片文件名。
- 菜单 `Action -> SVDC`：
  - 若当前没有子窗口，则创建 `Target-SVDC`。
  - 若已存在，则只激活已有子窗口，不重复创建。
- 关闭 `Target` 时会同步关闭其 `Target-SVDC` 子窗口。

## Target-SVDC 窗口
- 左侧：图像预览区，支持鼠标滚轮等比缩放、左键拖拽平移。
- 右侧：工具栏。

### 工具栏功能
- 第一行：`Save` / `or` / `Save as`
  - `Save`：保存到 `./out/out_r%d_%Y%m%d_%H%M%S.png`。
  - `Save as`：使用“文本路径 + 扩展名下拉框”拼接出的完整路径保存。
- 第二行：路径文本框（默认 `./out/out`）和扩展名下拉框（默认 `.png`，可选 `.jpg`、`.bmp`）。
- 第三行：`rank = [输入框] [Compute]`
  - 默认 rank 为原图 `min(width, height)`。
  - 非数字输入会弹窗：`rank should be a number!`。
  - rank 会被裁剪到 `[0, min_wh]`。
  - 点击 `Compute` 后调用 `SVDImage.svdmatrix2image(rank=rank)` 并刷新左侧图像。

## 日志输出
关键创建/关闭/计算/保存活动会按以下格式写入终端：
- `[%H:%M:%S] <module name> : <info>`

## 依赖说明
- 依赖 `Pillow` 与项目内 `svdimage.py`。
- `Target-SVDC` 创建时会构造对应 `SVDImage` 对象；关闭时释放引用。
