import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from PIL import Image, ImageTk
import datetime

# 导入predict.py中的预测功能
try:
    from predict import predict_image, VGGNet
    has_predict_module = True
except ImportError as e:
    print(f"导入predict模块失败: {e}")
    has_predict_module = False

class MongolianRecognitionApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.minsize(900, 600)  # 设置最小窗口尺寸
        self.master.title("蒙古语手写体识别工具")
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("Header.TLabel", font=("SimHei", 12, "bold"))
        # 图片相关状态变量
        self.selected_image_path = None  # 选中的图片路径
        self.image_preview = None  # 预览图片对象（避免GC回收）
        # 文件相关状态变量
        self.current_result_file = None  # 识别结果文件路径
        # 检查predict模块是否成功导入
        if not has_predict_module:
            self.show_import_error()
        else:
            self.create_widgets()
            self.create_layout()

    def show_import_error(self):
        """显示导入错误信息"""
        error_frame = ttk.Frame(self)
        error_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(error_frame, text="导入识别模块失败", style="Header.TLabel").pack(pady=20)
        ttk.Label(error_frame, text=f"无法导入LanguageRecognition/predict.py模块，请检查文件是否存在").pack(pady=10)
        ttk.Label(error_frame, text=f"错误: {sys.exc_info()[1]}").pack(pady=10)
        ttk.Button(error_frame, text="退出", command=self.master.quit).pack(pady=20)

    def create_widgets(self):
        """创建菜单和其他UI组件"""
        # 创建主菜单栏
        menubar = tk.Menu(self.master)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="选择图片", accelerator="Ctrl+O",
                              command=self.select_image, compound=tk.LEFT)
        file_menu.add_command(label="保存识别结果", accelerator="Ctrl+S",
                              command=self.save_recognition_result, compound=tk.LEFT)
        file_menu.add_separator()
        file_menu.add_command(label="退出", accelerator="Ctrl+Q",
                              command=self.master.quit, compound=tk.LEFT)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        help_menu.add_command(label="使用帮助", command=self.show_help)

        # 添加菜单到菜单栏
        menubar.add_cascade(label="文件", menu=file_menu)
        menubar.add_cascade(label="帮助", menu=help_menu)

        # 设置菜单栏
        self.master.config(menu=menubar)

        # 绑定快捷键
        self.master.bind("<Control-o>", lambda e: self.select_image())
        self.master.bind("<Control-s>", lambda e: self.save_recognition_result())
        self.master.bind("<Control-q>", lambda e: self.master.quit())

    def create_layout(self):
        """创建主界面布局"""
        # 创建主容器，使用网格布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------------- 左侧：手写体图片导入与预览区域 ----------------------
        left_frame = ttk.LabelFrame(main_frame, text="蒙古语手写体图片导入与预览", padding=10)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # 1. 图片选择按钮
        self.select_img_btn = ttk.Button(left_frame, text="选择手写体图片",
                                         command=self.select_image, padding=5)
        self.select_img_btn.pack(side=tk.TOP, pady=5)

        # 2. 图片预览容器（带边框，固定比例适配）
        self.preview_frame = ttk.Frame(left_frame, borderwidth=1, relief=tk.SUNKEN)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        # 预览提示文本（初始状态）
        self.preview_label = ttk.Label(self.preview_frame, text="未选择图片\n请点击上方按钮选择",
                                       font=("SimHei", 11), justify=tk.CENTER)
        self.preview_label.pack(expand=True)

        # ---------------------- 右侧：识别结果输出区域 ----------------------
        right_frame = ttk.LabelFrame(main_frame, text="蒙古语手写体识别结果", padding=10)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # 结果文本框及滚动条
        output_frame = ttk.Frame(right_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.text_output = tk.Text(output_frame, wrap=tk.WORD, font=("SimHei", 11),
                                   borderwidth=1, relief=tk.SUNKEN, state=tk.DISABLED)
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 初始提示文本
        self.update_output_text("识别结果将显示在这里\n1. 点击左侧「选择手写体图片」\n2. 点击下方「开始识别」按钮")

        output_scroll = ttk.Scrollbar(output_frame, command=self.text_output.yview)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=output_scroll.set)

        # ---------------------- 底部：功能按钮与状态区域 ----------------------
        bottom_frame = ttk.Frame(main_frame, padding=10)
        bottom_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # 1. 开始识别按钮（核心功能触发）
        self.recognize_btn = ttk.Button(bottom_frame, text="开始蒙古语手写体识别",
                                        command=self.start_handwriting_recognition, padding=5, state=tk.DISABLED)
        self.recognize_btn.pack(side=tk.LEFT, padx=10)

        # 2. 清空按钮（清空图片与结果）
        self.clear_btn = ttk.Button(bottom_frame, text="清空内容",
                                    command=self.clear_all_content, padding=5)
        self.clear_btn.pack(side=tk.LEFT, padx=10)

        # 3. 状态标签（显示当前操作状态）
        self.status_var = tk.StringVar()
        self.status_var.set("就绪：请选择手写体图片")
        status_label = ttk.Label(bottom_frame, textvariable=self.status_var, anchor=tk.E)
        status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

        # 配置网格权重：让左右区域等比例拉伸，上下区域自适应
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def select_image(self):
        """选择蒙古语手写体图片并显示预览"""
        # 打开图片选择对话框（支持常见图片格式）
        image_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")],
            title="选择蒙古语手写体图片"
        )

        if not image_path:  # 用户取消选择
            return

        try:
            # 1. 验证图片文件有效性
            if not os.path.exists(image_path):
                raise FileNotFoundError("图片文件不存在")

            # 2. 读取图片并调整尺寸（适配预览区域，保持宽高比）
            original_img = Image.open(image_path)
            # 获取预览框架尺寸
            preview_width, preview_height = self.preview_frame.winfo_width() - 20, self.preview_frame.winfo_height() - 20
            
            # 如果预览框架还没有实际尺寸（刚启动时），使用默认尺寸
            if preview_width < 10 or preview_height < 10:
                preview_width, preview_height = 300, 300
            
            # 按比例缩放（避免拉伸）
            original_img.thumbnail((preview_width, preview_height), Image.Resampling.LANCZOS)

            # 3. 显示图片预览（替换初始提示文本）
            self.image_preview = ImageTk.PhotoImage(original_img)
            self.preview_label.config(text="", image=self.image_preview)  # 清空文本，显示图片

            # 4. 更新状态变量
            self.selected_image_path = image_path
            self.recognize_btn.config(state=tk.NORMAL)  # 启用识别按钮
            self.status_var.set(f"已选择图片：{os.path.basename(image_path)}")

        except Exception as e:
            messagebox.showerror("图片加载错误", f"无法加载图片：{str(e)}")
            self.status_var.set("图片加载失败，请重新选择")

    def start_handwriting_recognition(self):
        """开始蒙古语手写体识别，调用predict.py中的实际识别功能"""
        if not self.selected_image_path:
            messagebox.showwarning("无图片", "请先选择手写体图片再执行识别")
            return

        try:
            # 1. 识别前状态更新
            self.status_var.set("正在识别...请稍候")
            self.master.update_idletasks()  # 强制刷新UI，显示“识别中”状态

            # 2. 调用predict.py中的实际识别功能
            recognition_result = predict_image(self.selected_image_path)

            # 3. 显示识别结果
            if recognition_result:
                result_content = f"识别完成！\n\n图片路径：{os.path.basename(self.selected_image_path)}\n\n识别结果（蒙古文）：\n{recognition_result}\n\n识别时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                self.update_output_text(result_content)
                self.status_var.set("识别成功！可保存结果")
            else:
                error_msg = "识别失败：无法识别图片内容或模型文件缺失"
                self.update_output_text(error_msg)
                messagebox.showerror("识别错误", error_msg)
                self.status_var.set("识别失败，请重试")

        except Exception as e:
            error_msg = f"识别失败：{str(e)}"
            self.update_output_text(error_msg)
            messagebox.showerror("识别错误", error_msg)
            self.status_var.set("识别失败，请重试")

    def update_output_text(self, content):
        """更新右侧结果文本框（处理文本框只读状态）"""
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)  # 清空原有内容
        self.text_output.insert(tk.END, content)
        self.text_output.config(state=tk.DISABLED)

    def save_recognition_result(self):
        """保存识别结果到文本文件"""
        result_content = self.text_output.get("1.0", tk.END).strip()
        if not result_content or "识别结果将显示在这里" in result_content:
            messagebox.showwarning("无结果可保存", "暂无识别结果，请先执行识别")
            return

        # 选择保存路径
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="保存蒙古语手写体识别结果"
        )

        if not save_path:
            return

        try:
            # 写入结果（包含图片路径和识别信息）
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"蒙古语手写体识别结果\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"识别时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"图片路径：{self.selected_image_path if self.selected_image_path else '未选择'}\n")
                
                # 提取识别结果内容
                if "识别结果（蒙古文）：" in result_content and "识别时间：" in result_content:
                    result_part = result_content.split("识别结果（蒙古文）：")[1].split("识别时间：")[0].strip()
                else:
                    result_part = result_content
                
                f.write(f"识别结果：\n{result_part}\n")
                f.write(f"=" * 50 + "\n")

            self.current_result_file = save_path
            self.status_var.set(f"结果已保存：{os.path.basename(save_path)}")
            messagebox.showinfo("保存成功", f"识别结果已保存到：\n{save_path}")

        except Exception as e:
            messagebox.showerror("保存错误", f"保存结果失败：{str(e)}")
            self.status_var.set("保存结果失败")

    def clear_all_content(self):
        """清空图片预览和识别结果"""
        # 清空图片预览（恢复提示文本）
        self.preview_label.config(text="未选择图片\n请点击上方按钮选择", image="")
        self.selected_image_path = None
        self.image_preview = None  # 释放图片对象

        # 清空结果文本
        self.update_output_text("识别结果将显示在这里\n1. 点击左侧「选择手写体图片」\n2. 点击下方「开始识别」按钮")

        # 禁用识别按钮，更新状态
        self.recognize_btn.config(state=tk.DISABLED)
        self.status_var.set("内容已清空：请重新选择图片")

    def show_about(self):
        """显示关于对话框"""
        about_window = tk.Toplevel(self.master)
        about_window.title("关于")
        about_window.geometry("300x220")
        about_window.resizable(False, False)
        about_window.transient(self.master)  # 子窗口
        about_window.grab_set()  # 模态窗口

        ttk.Label(about_window, text="蒙古语手写体识别工具", style="Header.TLabel").pack(pady=10)
        ttk.Label(about_window, text="版本: 1.0.0").pack(pady=3)
        ttk.Label(about_window, text="功能：识别蒙古语手写体图片").pack(pady=3)
        ttk.Label(about_window, text="支持格式：PNG/JPG/JPEG/BMP").pack(pady=3)
        ttk.Label(about_window, text="输出：蒙古文识别结果与保存").pack(pady=3)

        ttk.Button(about_window, text="确定", command=about_window.destroy).pack(pady=10)

    def show_help(self):
        """显示使用帮助对话框"""
        help_window = tk.Toplevel(self.master)
        help_window.title("使用帮助")
        help_window.geometry("550x400")
        help_window.transient(self.master)
        help_window.grab_set()

        help_text = tk.Text(help_window, wrap=tk.WORD, font=("SimHei", 10),
                            borderwidth=1, relief=tk.SUNKEN, state=tk.DISABLED)
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scroll = ttk.Scrollbar(help_text, command=help_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        help_text.config(yscrollcommand=scroll.set)

        # 帮助内容
        content = "蒙古语手写体识别工具使用指南\n\n"
        content += "1. 启动程序\n"
        content += "   - 双击运行MongolianRecognitionApp.py\n"
        content += "   - 或通过命令行运行: python MongolianRecognitionApp.py\n\n"
        content += "2. 基本操作流程\n"
        content += "   a. 点击左侧的「选择手写体图片」按钮，选择要识别的蒙古语手写体图片\n"
        content += "   b. 图片将显示在左侧预览区域\n"
        content += "   c. 点击底部的「开始蒙古语手写体识别」按钮\n"
        content += "   d. 识别结果将显示在右侧的文本区域\n"
        content += "   e. 可点击「保存识别结果」按钮将结果保存为文本文件\n\n"
        content += "3. 快捷键\n"
        content += "   - Ctrl+O: 选择图片\n"
        content += "   - Ctrl+S: 保存识别结果\n"
        content += "   - Ctrl+Q: 退出程序\n\n"
        content += "4. 支持的图片格式\n"
        content += "   - PNG、JPG、JPEG、BMP等常见图片格式\n\n"
        content += "5. 注意事项\n"
        content += "   - 图片质量越好，识别准确率越高\n"
        content += "   - 确保图片中蒙古文字符清晰可见\n"
        content += "   - 程序需要LanguageRecognition目录下的predict.py模块和相关模型文件\n"

        # 插入帮助内容
        help_text.config(state=tk.NORMAL)
        help_text.insert(tk.END, content)
        help_text.config(state=tk.DISABLED)

        # 添加关闭按钮
        ttk.Button(help_window, text="关闭", command=help_window.destroy).pack(pady=10)

# 主函数
if __name__ == '__main__':
    root = tk.Tk()
    app = MongolianRecognitionApp(master=root)
    root.mainloop()

