# 期中作业--中国象棋
#### 文件介绍
* `main.py`是main函数的程序，直接运行这个文件可以实现两个AI博弈对抗。
* `Man2ai.py`是人机对弈的文件。
* 其他`.py`文件都是程序运行所需要的类，包括`ChessBoard`、`Game`等。
* `MyAI.py`关于用于博弈的AI的算法。
* `images`文件夹是可视化界面所需的图片。
* 对手AI在`ChessAI.py`中实现，对手AI类已被`pyarmor`加密，需要安装`pyarmor`库才能运行此py文件。
#### 代码运行
建议使用`python3.7或python3.6`运行代码
需要安装`pygame`、`numpy`、`pyarmor`库：
开始程序的命令：
``` python
# 在terminal中运行：
python main.py
python Man2ai.py
# 在pycharm或vscode中运行：
main.py
Man2ai.py
```
## BUG

* 重复走棋子（已解决）：重复走子，判输
* 和棋（已解决）：如果30个回合没有棋子被吃，判和
* “将”图片显示（已解决）
* GUI显示（不要注释else里面的第一个for循环）

注意:
在main会实现交换先后手，但是未改变颜色,黑棋会先走