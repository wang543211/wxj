import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import torch
import numpy as np
from dotenv import load_dotenv
import os
import time
import threading
from datetime import datetime
import json

# LangChain 相关导入
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --------------------------
# 阿里云百炼配置
# --------------------------
load_dotenv()
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_MODEL = "qwen-plus"

class AircraftBleedAirDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✈️ Aircraft Bleed Air System - 智能故障诊断系统")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # 航空蓝主题配色
        self.colors = {
            'bg_primary': '#0A1628',      # 深蓝黑 - 航空主题
            'bg_secondary': '#152238',    # 次深蓝
            'bg_card': '#1A2D47',         # 卡片背景
            'bg_input': '#0D1B2A',        # 输入框背景
            'primary_blue': '#1E88E5',    # 航空蓝
            'accent_blue': '#1565C0',     # 强调蓝
            'hover_blue': '#0D47A1',      # 悬停蓝
            'text_primary': '#B0BEC5',    # 主文字 - 银灰
            'text_secondary': '#78909C',  # 次要文字
            'text_bright': '#FFFFFF',     # 亮色文字
            'border': '#263850',          # 边框
            'success': '#4CAF50',         # 成功绿
            'warning': '#FF9800',         # 警告橙
            'danger': '#F44336',          # 危险红
            'mode_diagnosis': '#FF6F00',  # 诊断模式 - 琥珀色（航空警告色）
            'mode_chat': '#00BCD4',       # 聊天模式 - 青色
        }
        
        # 初始化数据
        self.struct_data_df = None  # 结构化数据 (10,10)
        self.voxel_data = None      # 体素图像数据 (10,10,10)
        self.flight_data = None     # 航班时序数据 (N步×4传感器)
        self.model = None           # 故障诊断模型
        self.degradation_model = None  # 性能衰退模型
        self.llm = None
        self.messages = []
        self.diagnosis_history = []
        
        # 当前模式
        self.current_mode = tk.StringVar(value="diagnosis")
        
        # 配置窗口
        self.configure_window()
        
        # 初始化 LangChain 组件
        self.init_langchain_components()
        
        # 加载深度学习模型
        self.load_torch_model()
        
        # 创建界面
        self.create_aviation_ui()
    
    def configure_window(self):
        """配置窗口样式"""
        self.root.configure(bg=self.colors['bg_primary'])
    
    def init_langchain_components(self):
        """初始化 LangChain 组件"""
        try:
            # 1. 初始化 LLM
            self.llm = ChatOpenAI(
                model=BAILIAN_MODEL,
                openai_api_key=BAILIAN_API_KEY,
                openai_api_base=BAILIAN_BASE_URL,
                temperature=0.7
            )
            print("✅ LangChain LLM 初始化成功")
            
            # 2. 创建飞机引气系统诊断 Prompt（简洁版）
            self.diagnosis_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是专业的飞机引气系统（Bleed Air System）故障诊断专家。

【研究背景】
本系统用于飞机引气系统故障诊断。引气系统从发动机压气机或APU提取高压高温空气，用于机舱空调、增压、防冰等关键功能。

【8个故障类别定义】
索引0: 5级引气正常
索引1: 5级未超温低压
索引2: 5级超温低压
索引3: 5级超温未低压
索引4: 9级低压
索引5: 9级引气正常
索引6: 地面引气正常
索引7: 地面慢车低压

【输出要求】
请基于小模型的预测结果，生成简洁的诊断报告。必须使用中文输出，只包含以下三个部分：

**一、故障类别**
- 故障名称：[根据预测索引对应的故障类别名称]
- 预测索引：[0-7的索引值]

**二、置信度**
- 置信度数值：[具体数值，保留2位小数]
- 可信度评估：[高/中/低]

**三、维修维护计划**
- 紧急程度：[AOG紧急停场 / 可派遣排故 / 例行维护]
- 建议措施：[简要说明需要检查的部件和操作]
- 预计工时：[估计维修时间]

【重要】请保持输出简洁明了，只包含上述三个部分，不要添加其他冗余内容。"""),
                ("human", """【小模型预测结果】
预测故障类别索引: {fault_index}
故障类别名称: {fault_name}
置信度: {confidence}
结构化数据贡献: {struct_weight}
图像数据贡献: {img_weight}

【10×10 结构化监测数据】（压力、温度、流量等参数）
{raw_data}

请生成简洁的诊断报告。""")
            ])
            
            # 3. 创建性能衰退分析 Prompt
            self.degradation_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是专业的飞机引气系统性能衰退分析专家。

【研究背景】
本系统用于分析飞机引气系统在整个航班过程中的性能衰退趋势。通过分析传感器数据序列，评估系统健康状态的变化。

【分析任务】
基于模型预测的压力和温度序列，分析整个航班的性能衰退情况。

【输出要求】
请生成简洁的性能衰退分析报告。必须使用中文输出，只包含以下三个部分：

**一、衰退概况**
- 航班时长：[总步数/时间点数量]
- 初始压力：[起始压力值]
- 最终压力：[结束压力值]
- 压力变化量：[差值]
- 初始温度：[起始温度值]
- 最终温度：[结束温度值]
- 温度变化量：[差值]

**二、衰退趋势分析**
- 衰退类型：[线性衰退/指数衰退/波动衰退/稳定]
- 衰退速率：[单位时间的变化率]
- 关键转折点：[如有，指出性能明显下降的时间点]
- 健康评估：[良好/轻微衰退/中度衰退/严重衰退]

**三、维护建议**
- 监控建议：[是否需要加强监控]
- 维护时机：[建议下次检查时间]
- 风险提示：[潜在风险说明]

【重要】请保持输出简洁明了，只包含上述三个部分，不要添加其他冗余内容。"""),
                ("human", """【性能衰退分析数据】
总时间步数: {total_steps}
压力序列统计:
  - 均值: {pressure_mean}
  - 标准差: {pressure_std}
  - 最小值: {pressure_min}
  - 最大值: {pressure_max}
  - 起始值: {pressure_start}
  - 结束值: {pressure_end}
  - 变化量: {pressure_change}

温度序列统计:
  - 均值: {temperature_mean}
  - 标准差: {temperature_std}
  - 最小值: {temperature_min}
  - 最大值: {temperature_max}
  - 起始值: {temperature_start}
  - 结束值: {temperature_end}
  - 变化量: {temperature_change}

请生成性能衰退分析报告。""")
            ])
            
            # 4. 创建普通聊天模式 Prompt
            self.chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是飞机引气系统（Bleed Air System）的专业AI助手。
你可以回答关于引气系统的问题，包括：
- 系统原理和工作方式
- 故障诊断和排故
- 维护程序和标准
- 航空技术知识

请用专业但易懂的中文回答，保持友好态度。""")
            ])
            
            # 5. 创建 Output Parser
            self.output_parser = StrOutputParser()
            
            print("✅ 引气系统诊断 Prompts 初始化成功")
            
        except Exception as e:
            print(f"❌ LangChain 初始化失败: {e}")
    
    def load_torch_model(self):
        """加载深度学习模型"""
        # 加载故障诊断模型
        model_path = "best_cross_attention_model.pth"
        if not os.path.exists(model_path):
            print(f"⚠️ 故障诊断模型文件不存在，将使用模拟模式")
            self.model = None
        else:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loaded_object = torch.load(model_path, map_location=device, weights_only=False)
                
                if isinstance(loaded_object, dict):
                    print(f"⚠️ 故障诊断模型加载的是字典类型，将使用模拟模式")
                    self.model = None
                    return
                
                if hasattr(loaded_object, 'eval'):
                    self.model = loaded_object
                    self.model.eval()
                    print(f"✅ 故障诊断模型已加载到设备: {device}")
                else:
                    self.model = None
                    
            except Exception as e:
                print(f"❌ 故障诊断模型加载失败: {e}")
                self.model = None
        
        # 加载性能衰退模型
        degradation_model_path = "best_model.pth"
        if not os.path.exists(degradation_model_path):
            print(f"⚠️ 性能衰退模型文件不存在，将使用模拟模式")
            self.degradation_model = None
        else:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loaded_object = torch.load(degradation_model_path, map_location=device, weights_only=False)
                
                if isinstance(loaded_object, dict):
                    print(f"⚠️ 性能衰退模型加载的是字典类型，将使用模拟模式")
                    self.degradation_model = None
                    return
                
                if hasattr(loaded_object, 'eval'):
                    self.degradation_model = loaded_object
                    self.degradation_model.eval()
                    print(f"✅ 性能衰退模型已加载到设备: {device}")
                else:
                    self.degradation_model = None
                    
            except Exception as e:
                print(f"❌ 性能衰退模型加载失败: {e}")
                self.degradation_model = None
    
    def create_aviation_ui(self):
        """创建航空主题UI"""
        # 主容器 - 左右分栏
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # ===== 左侧边栏 =====
        sidebar = tk.Frame(main_container, bg=self.colors['bg_secondary'], width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 0))
        sidebar.pack_propagate(False)
        
        # Logo 区域 - 飞机图标
        logo_frame = tk.Frame(sidebar, bg=self.colors['bg_secondary'])
        logo_frame.pack(fill=tk.X, pady=(20, 15), padx=20)
        
        # 飞机图标 + 标题
        aircraft_icon = tk.Label(logo_frame,
                                text="✈️",
                                font=("Segoe UI Emoji", 36),
                                bg=self.colors['bg_secondary'])
        aircraft_icon.pack()
        
        logo_label = tk.Label(logo_frame,
                             text="Bleed Air System",
                             font=("Microsoft YaHei UI", 14, "bold"),
                             fg=self.colors['text_bright'],
                             bg=self.colors['bg_secondary'])
        logo_label.pack()
        
        subtitle_label = tk.Label(logo_frame,
                                 text="智能故障诊断系统",
                                 font=("Microsoft YaHei UI", 9),
                                 fg=self.colors['primary_blue'],
                                 bg=self.colors['bg_secondary'])
        subtitle_label.pack(pady=(3, 0))
        
        # 分隔线
        separator1 = tk.Frame(sidebar, height=2, bg=self.colors['primary_blue'])
        separator1.pack(fill=tk.X, padx=20, pady=(10, 15))
        
        # ===== 模式切换区域 =====
        mode_frame = tk.LabelFrame(sidebar, text="🎛️ Mode Selection",
                                  font=("Microsoft YaHei UI", 10, "bold"),
                                  fg=self.colors['text_primary'],
                                  bg=self.colors['bg_secondary'],
                                  bd=0)
        mode_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # 诊断模式按钮
        self.diagnosis_mode_btn = tk.Button(mode_frame,
                                           text="🔧 故障诊断",
                                           font=("Microsoft YaHei UI", 11, "bold"),
                                           fg=self.colors['text_bright'],
                                           bg=self.colors['mode_diagnosis'],
                                           activebackground='#E65100',
                                           activeforeground=self.colors['text_bright'],
                                           relief=tk.FLAT,
                                           cursor="hand2",
                                           command=self.set_diagnosis_mode)
        self.diagnosis_mode_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 性能衰退分析模式按钮
        self.degradation_mode_btn = tk.Button(mode_frame,
                                             text="📉 性能衰退",
                                             font=("Microsoft YaHei UI", 11, "bold"),
                                             fg=self.colors['text_bright'],
                                             bg='#2A3B4C',
                                             activebackground='#3A4B5C',
                                             activeforeground=self.colors['text_bright'],
                                             relief=tk.FLAT,
                                             cursor="hand2",
                                             command=self.set_degradation_mode)
        self.degradation_mode_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 普通聊天模式按钮
        self.chat_mode_btn = tk.Button(mode_frame,
                                      text="💬 技术咨询",
                                      font=("Microsoft YaHei UI", 11, "bold"),
                                      fg=self.colors['text_bright'],
                                      bg='#2A3B4C',
                                      activebackground='#3A4B5C',
                                      activeforeground=self.colors['text_bright'],
                                      relief=tk.FLAT,
                                      cursor="hand2",
                                      command=self.set_chat_mode)
        self.chat_mode_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 当前模式指示器
        self.mode_indicator = tk.Label(mode_frame,
                                      text="● 当前: 诊断模式",
                                      font=("Microsoft YaHei UI", 9, "bold"),
                                      fg=self.colors['mode_diagnosis'],
                                      bg=self.colors['bg_secondary'])
        self.mode_indicator.pack(pady=10)
        
        separator_mode = tk.Frame(sidebar, height=1, bg=self.colors['border'])
        separator_mode.pack(fill=tk.X, padx=20, pady=(10, 15))
        
        # 系统状态
        system_frame = tk.LabelFrame(sidebar, text="📊 System Status",
                                    font=("Microsoft YaHei UI", 9, "bold"),
                                    fg=self.colors['text_primary'],
                                    bg=self.colors['bg_secondary'],
                                    bd=0)
        system_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        llm_status = tk.Label(system_frame,
                             text=f"✓ LLM: {BAILIAN_MODEL}",
                             font=("Microsoft YaHei UI", 9),
                             fg=self.colors['success'],
                             bg=self.colors['bg_secondary'])
        llm_status.pack(anchor=tk.W, padx=5, pady=3)
        
        prompt_status = tk.Label(system_frame,
                                text="✓ Prompt: 引气系统专用",
                                font=("Microsoft YaHei UI", 9),
                                fg=self.colors['success'],
                                bg=self.colors['bg_secondary'])
        prompt_status.pack(anchor=tk.W, padx=5, pady=3)
        
        separator2 = tk.Frame(sidebar, height=1, bg=self.colors['border'])
        separator2.pack(fill=tk.X, padx=20, pady=(10, 15))
        
        # Operation 区域
        operation_frame = tk.LabelFrame(sidebar, text="🔨 Operations",
                                       font=("Microsoft YaHei UI", 9, "bold"),
                                       fg=self.colors['text_primary'],
                                       bg=self.colors['bg_secondary'],
                                       bd=0)
        operation_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # 加载示例数据按钮
        sample_btn = tk.Button(operation_frame,
                              text="️ Load Sample Data",
                              font=("Microsoft YaHei UI", 10, "bold"),
                              fg=self.colors['text_bright'],
                              bg=self.colors['success'],
                              activebackground='#388E3C',
                              activeforeground=self.colors['text_bright'],
                              relief=tk.FLAT,
                              cursor="hand2",
                              command=self.load_sample_data)
        sample_btn.pack(fill=tk.X, padx=5, pady=5)
        
        separator_op = tk.Frame(operation_frame, height=1, bg=self.colors['border'])
        separator_op.pack(fill=tk.X, padx=5, pady=5)
        
        # History 按钮
        history_btn = tk.Button(operation_frame,
                               text="📜 Maintenance Log",
                               font=("Microsoft YaHei UI", 10, "bold"),
                               fg=self.colors['text_bright'],
                               bg=self.colors['accent_blue'],
                               activebackground=self.colors['hover_blue'],
                               activeforeground=self.colors['text_bright'],
                               relief=tk.FLAT,
                               cursor="hand2",
                               command=self.show_history)
        history_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Clear chat 按钮
        clear_btn = tk.Button(operation_frame,
                             text="🗑️ Clear Records",
                             font=("Microsoft YaHei UI", 10, "bold"),
                             fg=self.colors['text_bright'],
                             bg=self.colors['accent_blue'],
                             activebackground=self.colors['hover_blue'],
                             activeforeground=self.colors['text_bright'],
                             relief=tk.FLAT,
                             cursor="hand2",
                             command=self.clear_chat)
        clear_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # ===== 右侧主区域 =====
        main_area = tk.Frame(main_container, bg=self.colors['bg_primary'])
        main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 15), pady=15)
        
        # 顶部标题
        title_frame = tk.Frame(main_area, bg=self.colors['bg_primary'])
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(title_frame,
                              text="✈️ Aircraft Bleed Air System - Fault Diagnosis & Technical Support",
                              font=("Microsoft YaHei UI", 13, "bold"),
                              fg=self.colors['text_bright'],
                              bg=self.colors['bg_primary'])
        title_label.pack(anchor=tk.W)
        
        subtitle_info = tk.Label(title_frame,
                                text="引气系统故障诊断与技术支援平台",
                                font=("Microsoft YaHei UI", 9),
                                fg=self.colors['text_secondary'],
                                bg=self.colors['bg_primary'])
        subtitle_info.pack(anchor=tk.W, pady=(3, 0))
        
        # Chat 显示区域
        chat_container = tk.LabelFrame(main_area, text="📡 Communication Log",
                                      font=("Microsoft YaHei UI", 11, "bold"),
                                      fg=self.colors['text_primary'],
                                      bg=self.colors['bg_card'],
                                      bd=1,
                                      relief=tk.FLAT)
        chat_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 聊天内容显示
        self.chat_display = scrolledtext.ScrolledText(chat_container,
                                                      bg=self.colors['bg_card'],
                                                      fg=self.colors['text_primary'],
                                                      font=("Consolas", 10),
                                                      wrap=tk.WORD,
                                                      borderwidth=0,
                                                      highlightthickness=0)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)
        
        # 配置文本标签
        self.chat_display.tag_configure("system_content",
                                       foreground=self.colors['text_primary'],
                                       font=("Consolas", 10))
        self.chat_display.tag_configure("highlight",
                                       foreground=self.colors['primary_blue'],
                                       font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("diagnosis_mode",
                                       foreground=self.colors['mode_diagnosis'],
                                       font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("chat_mode",
                                       foreground=self.colors['mode_chat'],
                                       font=("Consolas", 10, "bold"))
        self.chat_display.tag_configure("warning",
                                       foreground=self.colors['warning'],
                                       font=("Consolas", 10, "bold"))
        
        # 底部输入区域
        task_container = tk.Frame(main_area, bg=self.colors['bg_card'])
        task_container.pack(fill=tk.X)
        
        task_label = tk.Label(task_container,
                             text="📝 Message Input",
                             font=("Microsoft YaHei UI", 10, "bold"),
                             fg=self.colors['text_primary'],
                             bg=self.colors['bg_card'])
        task_label.pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        # System data 输入行
        self.system_data_frame = tk.Frame(task_container, bg=self.colors['bg_input'])
        self.system_data_frame.pack(fill=tk.X, padx=15, pady=5)
        
        system_data_label = tk.Label(self.system_data_frame,
                                    text="📊 System Data",
                                    font=("Microsoft YaHei UI", 9, "bold"),
                                    fg=self.colors['text_primary'],
                                    bg=self.colors['bg_input'])
        system_data_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        hint_label = tk.Label(self.system_data_frame,
                             text="Upload sensor data (Pressure, Temperature, Flow rate)",
                             font=("Microsoft YaHei UI", 8),
                             fg=self.colors['text_secondary'],
                             bg=self.colors['bg_input'])
        hint_label.pack(side=tk.LEFT, padx=5)
        
        select_btn = tk.Button(self.system_data_frame,
                              text="📂 Upload CSV",
                              font=("Microsoft YaHei UI", 9, "bold"),
                              fg=self.colors['text_bright'],
                              bg=self.colors['primary_blue'],
                              activebackground=self.colors['hover_blue'],
                              activeforeground=self.colors['text_bright'],
                              relief=tk.FLAT,
                              cursor="hand2",
                              command=self.select_data)
        select_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 消息输入行
        prompt_frame = tk.Frame(task_container, bg=self.colors['bg_input'])
        prompt_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.prompt_entry = tk.Entry(prompt_frame,
                                    font=("Microsoft YaHei UI", 10),
                                    fg=self.colors['text_primary'],
                                    bg=self.colors['bg_input'],
                                    insertbackground=self.colors['text_bright'],
                                    relief=tk.FLAT,
                                    borderwidth=0)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=8)
        self.prompt_entry.insert(0, "输入诊断指令或技术问题...")
        self.prompt_entry.bind('<Return>', self.send_message)
        self.prompt_entry.bind('<FocusIn>', self.on_entry_focus_in)
        self.prompt_entry.bind('<FocusOut>', self.on_entry_focus_out)
        
        enter_btn = tk.Button(prompt_frame,
                             text="🚀 Send",
                             font=("Microsoft YaHei UI", 9, "bold"),
                             fg=self.colors['text_bright'],
                             bg=self.colors['primary_blue'],
                             activebackground=self.colors['hover_blue'],
                             activeforeground=self.colors['text_bright'],
                             relief=tk.FLAT,
                             cursor="hand2",
                             command=self.send_message)
        enter_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 状态显示
        self.status_label = tk.Label(main_area,
                                    text="⚪ 未加载数据",
                                    font=("Microsoft YaHei UI", 9),
                                    fg=self.colors['text_secondary'],
                                    bg=self.colors['bg_primary'])
        self.status_label.pack(anchor=tk.W, padx=5, pady=(5, 0))
    
    def set_diagnosis_mode(self):
        """切换到故障诊断模式"""
        self.current_mode.set("diagnosis")
        self.mode_indicator.config(text="● 当前: 故障诊断", fg=self.colors['mode_diagnosis'])
        self.diagnosis_mode_btn.config(bg=self.colors['mode_diagnosis'])
        self.degradation_mode_btn.config(bg='#2A3B4C')
        self.chat_mode_btn.config(bg='#2A3B4C')
        self.system_data_frame.pack(fill=tk.X, padx=15, pady=5)
        self.append_to_chat("system", "✈️ 已切换到【故障诊断模式】- 可上传传感器数据进行故障分析", "diagnosis_mode")
    
    def set_degradation_mode(self):
        """切换到性能衰退分析模式"""
        self.current_mode.set("degradation")
        self.mode_indicator.config(text="● 当前: 性能衰退", fg='#9C27B0')
        self.degradation_mode_btn.config(bg='#9C27B0')
        self.diagnosis_mode_btn.config(bg='#2A3B4C')
        self.chat_mode_btn.config(bg='#2A3B4C')
        self.system_data_frame.pack(fill=tk.X, padx=15, pady=5)
        self.append_to_chat("system", "📉 已切换到【性能衰退分析模式】- 可上传航班时序数据", "warning")
    
    def set_chat_mode(self):
        """切换到技术咨询模式"""
        self.current_mode.set("chat")
        self.mode_indicator.config(text="● 当前: 技术咨询", fg=self.colors['mode_chat'])
        self.chat_mode_btn.config(bg=self.colors['mode_chat'])
        self.diagnosis_mode_btn.config(bg='#2A3B4C')
        self.degradation_mode_btn.config(bg='#2A3B4C')
        self.system_data_frame.pack_forget()
        self.append_to_chat("system", "💬 已切换到【技术咨询模式】- 可询问引气系统相关问题", "chat_mode")
    
    def on_entry_focus_in(self, event):
        """输入框获得焦点时清除提示文字"""
        if self.prompt_entry.get() == "输入诊断指令或技术问题...":
            self.prompt_entry.delete(0, tk.END)
    
    def on_entry_focus_out(self, event):
        """输入框失去焦点时显示提示文字"""
        if self.prompt_entry.get() == "":
            self.prompt_entry.insert(0, "输入诊断指令或技术问题...")
    
    def select_data(self):
        """选择数据文件"""
        print("📂 打开文件选择对话框...")
        file_path = filedialog.askopenfilename(
            title="选择传感器数据CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if file_path:
            print(f"📄 选择的文件: {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)
                
                import chardet
                result = chardet.detect(raw_data)
                detected_encoding = result['encoding']
                print(f"🔍 检测到编码: {detected_encoding}")
                
                encodings_to_try = []
                if detected_encoding:
                    encodings_to_try.append(detected_encoding)
                encodings_to_try.extend(['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'utf-8-sig'])
                encodings_to_try = list(dict.fromkeys(encodings_to_try))
                
                df = None
                success_encoding = None
                
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, header=None, encoding=encoding)
                        success_encoding = encoding
                        print(f"✅ 使用 {encoding} 编码成功读取文件")
                        break
                    except:
                        continue
                
                if df is None:
                    messagebox.showerror("编码错误", "无法识别文件编码！")
                    return
                
                # 根据当前模式判断数据类型
                if self.current_mode.get() == "degradation":
                    # 性能衰退模式：需要N×4的时序数据
                    if df.shape[1] == 4:
                        self.flight_data = df
                        self.status_label.config(text=f"● 已加载: {os.path.basename(file_path)} ({df.shape[0]}步×4传感器)",
                                               fg=self.colors['success'])
                        self.append_to_chat("system", f"✅ 航班数据加载成功: {os.path.basename(file_path)} ({df.shape[0]}步×4传感器)")
                        messagebox.showinfo("成功", f"✅ 航班数据加载成功！\n文件: {os.path.basename(file_path)}\n维度: {df.shape[0]}步×4传感器")
                    else:
                        messagebox.showerror("错误", f"数据维度错误: 当前 {df.shape}, 性能衰退模式需要 N×4 的时序数据")
                else:
                    # 故障诊断模式：需要10×10的数据
                    if df.shape == (10, 10):
                        self.struct_data_df = df
                        self.status_label.config(text=f"● 已加载: {os.path.basename(file_path)}",
                                               fg=self.colors['success'])
                        self.append_to_chat("system", f"✅ 传感器数据加载成功: {os.path.basename(file_path)} (10×10)")
                        messagebox.showinfo("成功", f"✅ 数据加载成功！\n文件: {os.path.basename(file_path)}\n维度: 10×10")
                    else:
                        messagebox.showerror("错误", f"数据维度错误: 当前 {df.shape}, 需要 (10, 10)")
            except Exception as e:
                messagebox.showerror("错误", f"解析失败: {e}")
    
    def load_sample_data(self):
        """加载示例数据"""
        if self.current_mode.get() == "degradation":
            # 性能衰退模式：生成航班时序数据 (500步×4传感器)
            steps = 500
            # 模拟4个传感器数据，带有轻微衰退趋势
            time_axis = np.linspace(0, 1, steps)
            sensor1 = 100 + 10 * np.sin(2 * np.pi * time_axis) - 5 * time_axis  # 带衰退趋势
            sensor2 = 80 + 8 * np.cos(2 * np.pi * time_axis) - 3 * time_axis
            sensor3 = 60 + 5 * np.sin(4 * np.pi * time_axis) - 2 * time_axis
            sensor4 = 90 + 7 * np.cos(3 * np.pi * time_axis) - 4 * time_axis
            
            # 添加噪声
            noise = np.random.normal(0, 0.5, (steps, 4))
            data = np.column_stack([sensor1, sensor2, sensor3, sensor4]) + noise
            
            self.flight_data = pd.DataFrame(data)
            self.status_label.config(text="● 已加载: 示例航班数据", fg=self.colors['success'])
            self.append_to_chat("system", f"✅ 示例航班数据已加载 ({steps}步×4传感器)")
            messagebox.showinfo("成功", f"✅ 示例航班数据已加载\n维度: {steps}步×4传感器")
        else:
            # 故障诊断模式：生成结构化数据和体素数据
            base = np.arange(100).reshape(10, 10).astype(float)
            noise = np.random.normal(0, 5, base.shape)
            self.struct_data_df = pd.DataFrame(base + noise)
            
            # 模拟体素图像数据 (10×10×10)
            self.voxel_data = np.random.rand(10, 10, 10).astype(np.float32)
            
            self.status_label.config(text="● 已加载: 示例传感器数据", fg=self.colors['success'])
            self.append_to_chat("system", "✅ 示例数据已加载 (结构化: 10×10, 体素: 10×10×10)")
            messagebox.showinfo("成功", "✅ 示例数据已加载")
    
    def show_history(self):
        """显示历史记录"""
        if not self.diagnosis_history:
            messagebox.showinfo("Maintenance Log", "暂无诊断记录")
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("📜 维修记录")
        history_window.geometry("900x600")
        history_window.configure(bg=self.colors['bg_primary'])
        
        title_label = tk.Label(history_window,
                              text="✈️ Aircraft Bleed Air System - Maintenance Log",
                              font=("Microsoft YaHei UI", 12, "bold"),
                              fg=self.colors['text_bright'],
                              bg=self.colors['bg_primary'])
        title_label.pack(pady=10)
        
        history_text = scrolledtext.ScrolledText(history_window,
                                                bg=self.colors['bg_card'],
                                                fg=self.colors['text_primary'],
                                                font=("Consolas", 10))
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, record in enumerate(self.diagnosis_history, 1):
            history_text.insert(tk.END, f"\n{'='*70}\n")
            history_text.insert(tk.END, f"🔧 Record #{i} - {record['timestamp']}\n")
            history_text.insert(tk.END, f"⚠️ Fault Score: {record['score']}\n")
            history_text.insert(tk.END, f"\n{record['report']}\n")
    
    def clear_chat(self):
        """清空聊天"""
        self.messages = []
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.append_to_chat("system", "🗑️ 通信记录已清空")
    
    def call_fault_model(self, struct_df, voxel_data=None):
        """调用多模态小模型进行故障诊断
        
        模型返回5个值:
        - logits: [1, 8] 8个故障类别的原始分数
        - two_modal_weights: [1, 2] 双模态权重 [结构化, 图像]
        - three_modal_weights: [1, 3] 三模态权重 [结构化, 图像, 跨模态融合]
        - struct_attn: [1, 10, 10] 结构化数据注意力矩阵
        - img_attn: [1, 100, 10] 图像侧交叉注意力矩阵
        """
        # 定义8个故障类别名称
        fault_classes = [
            "5级引气正常",
            "5级未超温低压", 
            "5级超温低压",
            "5级超温未低压",
            "9级低压",
            "9级引气正常",
            "地面引气正常",
            "地面慢车低压"
        ]
        
        if self.model is None:
            # 模拟模式：随机生成结果
            time.sleep(1)
            fault_index = np.random.randint(0, 8)
            confidence = np.random.uniform(0.7, 0.99)
            struct_weight = np.random.uniform(0.6, 0.9)
            img_weight = round(1 - struct_weight, 2)
            
            return {
                "fault_index": int(fault_index),
                "fault_name": fault_classes[fault_index],
                "confidence": round(float(confidence), 2),
                "struct_weight": round(float(struct_weight), 2),
                "img_weight": round(float(img_weight), 2),
                "mode": "simulation"
            }
        
        try:
            # 真实模型推理
            struct_np = struct_df.values.astype(np.float32)
            struct_tensor = torch.tensor(struct_np).unsqueeze(0)  # [1, 10, 10]
            
            if voxel_data is not None:
                voxel_tensor = torch.tensor(voxel_data).unsqueeze(0)  # [1, 10, 10, 10]
                
                with torch.no_grad():
                    # 模型返回5个值
                    outputs = self.model(struct_tensor, voxel_tensor)
            else:
                with torch.no_grad():
                    outputs = self.model(struct_tensor)
                
            # 解包输出: logits, two_modal_weights, three_modal_weights, struct_attn, img_attn
            if isinstance(outputs, tuple) and len(outputs) >= 5:
                logits = outputs[0]  # [1, 8]
                two_modal_weights = outputs[1]  # [1, 2]
                three_modal_weights = outputs[2]  # [1, 3]
                struct_attn = outputs[3]  # [1, 10, 10]
                img_attn = outputs[4]  # [1, 100, 10]
                
                # 使用三模态权重
                struct_weight = float(three_modal_weights[0, 0].item())
                img_weight = float(three_modal_weights[0, 1].item())
            elif isinstance(outputs, tuple) and len(outputs) >= 2:
                logits = outputs[0]  # [1, 8]
                two_modal_weights = outputs[1]  # [1, 2]
                
                struct_weight = float(two_modal_weights[0, 0].item())
                img_weight = float(two_modal_weights[0, 1].item())
            else:
                # 如果输出不是元组，假设是logits
                logits = outputs
                struct_weight = 0.8
                img_weight = 0.2
            
            # 计算预测类别和置信度
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probabilities, dim=-1)
            
            fault_index = int(predicted_idx.item())
            confidence_value = float(confidence.item())
            
            return {
                "fault_index": fault_index,
                "fault_name": fault_classes[fault_index],
                "confidence": round(confidence_value, 2),
                "struct_weight": round(struct_weight, 2),
                "img_weight": round(img_weight, 2),
                "mode": "real_model"
            }
                
        except Exception as e:
            print(f"❌ 模型推理错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                "fault_index": 0,
                "fault_name": fault_classes[0],
                "confidence": 0.0,
                "struct_weight": 0.0,
                "img_weight": 0.0,
                "mode": "error"
            }
    
    def call_degradation_model(self, flight_df):
        """调用性能衰退模型分析整个航班数据
        
        模型输入: 100步 × 4传感器
        模型输出: 压力 + 温度 (2个值)
        
        处理流程:
        1. 将航班数据拆分成多个100步的窗口
        2. 对每个窗口进行预测
        3. 拼接所有预测结果
        4. 计算衰退量
        """
        if self.degradation_model is None:
            # 模拟模式
            time.sleep(1)
            total_steps = len(flight_df)
            
            # 生成模拟的压力和温度序列
            time_axis = np.linspace(0, 1, total_steps)
            pressure = 100 - 10 * time_axis + np.random.normal(0, 0.5, total_steps)
            temperature = 80 - 8 * time_axis + np.random.normal(0, 0.4, total_steps)
            
            return {
                "pressure": pressure,
                "temperature": temperature,
                "total_steps": total_steps,
                "mode": "simulation"
            }
        
        try:
            data = flight_df.values.astype(np.float32)
            total_steps = len(data)
            window_size = 100
            stride = 50  # 滑动窗口步长
            
            pressure_list = []
            temperature_list = []
            timestamps = []
            
            # 滑动窗口处理
            for i in range(0, total_steps - window_size + 1, stride):
                window = data[i:i+window_size]  # [100, 4]
                window_tensor = torch.tensor(window).unsqueeze(0)  # [1, 100, 4]
                
                with torch.no_grad():
                    output = self.degradation_model(window_tensor)
                
                # 假设输出是 [1, 2]，分别是压力和温度
                if isinstance(output, tuple):
                    output = output[0]
                
                if output.shape[-1] == 2:
                    p, t = output[0].cpu().numpy()
                    pressure_list.append(p)
                    temperature_list.append(t)
                    timestamps.append(i + window_size // 2)  # 记录窗口中心时间点
            
            # 如果数据不足100步，直接处理
            if len(pressure_list) == 0 and total_steps < window_size:
                window_tensor = torch.tensor(data).unsqueeze(0)
                with torch.no_grad():
                    output = self.degradation_model(window_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                if output.shape[-1] == 2:
                    p, t = output[0].cpu().numpy()
                    pressure_list.append(p)
                    temperature_list.append(t)
                    timestamps.append(total_steps // 2)
            
            # 插值生成完整序列
            if len(timestamps) > 1:
                from scipy.interpolate import interp1d
                pressure_interp = interp1d(timestamps, pressure_list, kind='linear', fill_value='extrapolate')
                temperature_interp = interp1d(timestamps, temperature_list, kind='linear', fill_value='extrapolate')
                
                full_timestamps = np.arange(total_steps)
                pressure = pressure_interp(full_timestamps)
                temperature = temperature_interp(full_timestamps)
            else:
                # 如果只有一个点，使用常数
                pressure = np.full(total_steps, pressure_list[0] if pressure_list else 0)
                temperature = np.full(total_steps, temperature_list[0] if temperature_list else 0)
            
            return {
                "pressure": pressure,
                "temperature": temperature,
                "total_steps": total_steps,
                "mode": "real_model"
            }
                
        except Exception as e:
            print(f"❌ 性能衰退模型推理错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回模拟数据作为后备
            total_steps = len(flight_df)
            time_axis = np.linspace(0, 1, total_steps)
            pressure = 100 - 10 * time_axis + np.random.normal(0, 0.5, total_steps)
            temperature = 80 - 8 * time_axis + np.random.normal(0, 0.4, total_steps)
            
            return {
                "pressure": pressure,
                "temperature": temperature,
                "total_steps": total_steps,
                "mode": "error_fallback"
            }
    
    def send_message(self, event=None):
        """发送消息"""
        user_input = self.prompt_entry.get().strip()
        
        if not user_input or user_input == "输入诊断指令或技术问题...":
            return
        
        # 清空输入框
        self.prompt_entry.delete(0, tk.END)
        
        # 添加用户消息
        self.append_to_chat("user", user_input)
        self.messages.append({"role": "user", "content": user_input})
        
        # 根据当前模式执行不同逻辑
        if self.current_mode.get() == "diagnosis":
            if self.struct_data_df is None:
                self.append_to_chat("system", "⚠️ 请先上传传感器数据或加载示例数据！", "warning")
                return
            
            thread = threading.Thread(target=self.run_diagnosis)
            thread.daemon = True
            thread.start()
        elif self.current_mode.get() == "degradation":
            if self.flight_data is None:
                self.append_to_chat("system", "⚠️ 请先上传航班时序数据或加载示例数据！", "warning")
                return
            
            thread = threading.Thread(target=self.run_degradation_analysis)
            thread.daemon = True
            thread.start()
        else:
            thread = threading.Thread(target=self.run_chat)
            thread.daemon = True
            thread.start()
    
    def run_degradation_analysis(self):
        """执行性能衰退分析"""
        try:
            self.root.after(0, lambda: self.append_to_chat("system", "📉 正在分析航班性能衰退趋势..."))
            
            # 1. 调用性能衰退模型
            degradation_result = self.call_degradation_model(self.flight_data)
            
            pressure = degradation_result['pressure']
            temperature = degradation_result['temperature']
            total_steps = degradation_result['total_steps']
            
            # 2. 计算统计信息
            pressure_stats = {
                'mean': float(np.mean(pressure)),
                'std': float(np.std(pressure)),
                'min': float(np.min(pressure)),
                'max': float(np.max(pressure)),
                'start': float(pressure[0]),
                'end': float(pressure[-1]),
                'change': float(pressure[-1] - pressure[0])
            }
            
            temperature_stats = {
                'mean': float(np.mean(temperature)),
                'std': float(np.std(temperature)),
                'min': float(np.min(temperature)),
                'max': float(np.max(temperature)),
                'start': float(temperature[0]),
                'end': float(temperature[-1]),
                'change': float(temperature[-1] - temperature[0])
            }
            
            # 3. 构建分析 Chain
            degradation_chain = self.degradation_prompt | self.llm | self.output_parser
            
            chain_input = {
                "total_steps": str(total_steps),
                "pressure_mean": f"{pressure_stats['mean']:.2f}",
                "pressure_std": f"{pressure_stats['std']:.2f}",
                "pressure_min": f"{pressure_stats['min']:.2f}",
                "pressure_max": f"{pressure_stats['max']:.2f}",
                "pressure_start": f"{pressure_stats['start']:.2f}",
                "pressure_end": f"{pressure_stats['end']:.2f}",
                "pressure_change": f"{pressure_stats['change']:.2f}",
                "temperature_mean": f"{temperature_stats['mean']:.2f}",
                "temperature_std": f"{temperature_stats['std']:.2f}",
                "temperature_min": f"{temperature_stats['min']:.2f}",
                "temperature_max": f"{temperature_stats['max']:.2f}",
                "temperature_start": f"{temperature_stats['start']:.2f}",
                "temperature_end": f"{temperature_stats['end']:.2f}",
                "temperature_change": f"{temperature_stats['change']:.2f}"
            }
            
            # 4. 调用大模型生成分析报告
            analysis_result = degradation_chain.invoke(chain_input)
            
            # 5. 格式化输出
            formatted_output = f"""
✈️ Aircraft Bleed Air System - Performance Degradation Analysis
{'='*60}

【研究背景】飞机引气系统性能衰退分析

{analysis_result}

{'='*60}
⚠️ Note: 此分析基于AI模型预测，请结合实际飞行数据确认
            """
            
            # 6. 保存分析记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "type": "degradation",
                "total_steps": total_steps,
                "pressure_change": pressure_stats['change'],
                "temperature_change": temperature_stats['change'],
                "report": analysis_result
            }
            self.diagnosis_history.append(record)
            
            # 7. 显示结果
            self.root.after(0, lambda: self.append_to_chat("assistant", formatted_output))
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 分析失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, lambda: self.append_to_chat("system", f"❌ 分析失败: {str(e)}"))
    
    def run_diagnosis(self):
        """执行引气系统诊断"""
        try:
            self.root.after(0, lambda: self.append_to_chat("system", "🔍 正在分析引气系统传感器数据..."))
            
            # 1. 多模态小模型推理
            fault_result = self.call_fault_model(self.struct_data_df, self.voxel_data)
            
            # 2. 构建诊断 Chain
            diagnosis_chain = self.diagnosis_prompt | self.llm | self.output_parser
            
            # 准备输入参数（精简有用输出）
            chain_input = {
                "fault_index": str(fault_result['fault_index']),
                "fault_name": fault_result['fault_name'],
                "confidence": str(fault_result['confidence']),
                "struct_weight": str(fault_result['struct_weight']),
                "img_weight": str(fault_result['img_weight']),
                "raw_data": str(self.struct_data_df.values.tolist()[:3]) + "... (共100个数据点)"
            }
            
            # 3. 调用大模型生成诊断报告
            diagnosis_result = diagnosis_chain.invoke(chain_input)
            
            # 4. 格式化输出
            formatted_output = f"""
✈️ Aircraft Bleed Air System - Diagnostic Report
{'='*60}

【研究背景】飞机引气系统故障诊断

{diagnosis_result}

{'='*60}
⚠️ Note: 此报告基于AI分析，请结合实际排故手册确认
            """
            
            # 5. 保存诊断记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "fault_index": fault_result['fault_index'],
                "fault_name": fault_result['fault_name'],
                "confidence": fault_result['confidence'],
                "report": diagnosis_result
            }
            self.diagnosis_history.append(record)
            
            # 6. 显示结果
            self.root.after(0, lambda: self.append_to_chat("assistant", formatted_output))
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 诊断失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, lambda: self.append_to_chat("system", f"❌ 诊断失败: {str(e)}"))
    
    def run_chat(self):
        """执行技术咨询"""
        try:
            self.root.after(0, lambda: self.append_to_chat("system", "🤖 AI助手思考中..."))
            
            # 构建聊天 Chain（包含对话历史）
            messages = [SystemMessage(content="你是飞机引气系统（Bleed Air System）的专业AI助手。")]
            
            # 添加历史对话（保留最近10条）
            for msg in self.messages[-10:]:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            
            # 调用 LLM
            response = self.llm.invoke(messages)
            reply = response.content
            
            self.root.after(0, lambda: self.append_to_chat("assistant", reply))
            
        except Exception as e:
            self.root.after(0, lambda: self.append_to_chat("system", f"❌ 回答失败: {str(e)}"))
    
    def append_to_chat(self, role, content, mode_tag=None):
        """追加消息到聊天显示"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if role == "system":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] ", "system_content")
            if mode_tag:
                self.chat_display.insert(tk.END, f"{content}\n", mode_tag)
            else:
                self.chat_display.insert(tk.END, f"{content}\n", "system_content")
        elif role == "user":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] 🧑✈️ ", "system_content")
            self.chat_display.insert(tk.END, f"{content}\n", "highlight")
        else:
            self.chat_display.insert(tk.END, f"\n[{timestamp}] 🤖 ", "system_content")
            self.chat_display.insert(tk.END, f"{content}\n", "system_content")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)


def main():
    root = tk.Tk()
    app = AircraftBleedAirDiagnosisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
