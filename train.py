from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from datasets import Dataset
import json
from transformers import DataCollatorForLanguageModeling

model_name = "./qwen"  # 或本地路径
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 节省显存
    device_map="auto",
    use_cache=False  # 训练时建议关闭
)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

def load_and_prepare_data(file_path):
    # 读取原始JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    SYSTEM_PROMPT = """你是一个细粒度中文仇恨言论识别专家。请严格按照以下规则处理输入文本：
1. 输出格式：仇恨四元组格式为「评论对象 | 论点 | 目标群体 | 是否仇恨 [END]」
2. 多组处理：当存在多个评论对象时，用「[SEP]」分隔多个四元组
3. 元素顺序：必须按「Target | Argument | Targeted Group | Hateful」顺序排列
4. 分隔符规范：元素间用「 | 」分割（前后各一个空格）
5. 评论对象(Target)：
   - 提取被评述的具体对象（个人/群体）
   - 无具体目标时设为「NULL」
6. 论点(Argument)：提取对评论目标的关键负面描述片段
7. 目标群体(Targeted Group)：仅限以下5类：
   - 地域（Regional）
   - 种族（Racism）
   - 性别（Sexism）
   - LGBTQ（LGBTQ）
   - 其他（others）
   - 非仇恨时设为「non-hate」
8. 是否仇恨(Hateful)：
   - 构成仇恨言论：「hate」
   - 非仇恨/一般攻击：「non-hate」
9. 特殊处理：
   - 非仇恨文本仍需抽取Target/Argument
   - 无特定群体时目标群体设为「non-hate」
   - 每个四元组必须以「[END]」结尾

示例：
输入："老黑我是真的讨厌，媚黑的还倒贴"
输出："老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]"

输入："你可真是头蠢驴"
输出："你 | 蠢驴 | non-hate | non-hate [END]"

输入："某些地区的风俗真恶心"
输出："某些地区的风俗 | 真恶心 | Regional | hate [END]"

现在请严格按上述规则处理输入""".strip()
    # 转换为HuggingFace数据集格式
    formatted_data = []
    for item in raw_data:
        formatted_data.append({
            "system": SYSTEM_PROMPT,
            "input": item["content"],
            "output": item["output"]
        })
    
    return Dataset.from_list(formatted_data)

def format_data(example):
    # 构建输入文本
    text = f"<|system|>\n{example['system']}</s>\n<|user|>\n{example['input']}</s>\n<|assistant|>\n"
    
    # 构建完整文本（输入+输出）
    full_text = text + example['output'] + tokenizer.eos_token
    
    # 使用tokenizer一次性处理
    return tokenizer(
        full_text,
        max_length=768,  # 增加最大长度
        truncation=True,
        padding=False,  # 不在预处理阶段填充
        return_tensors=None,  # 返回字典而非张量
        add_special_tokens=False
    )
dataset = load_and_prepare_data("dataset/train.json")
tokenized_dataset = dataset.map(
    format_data, 
    remove_columns=dataset.column_names,
    batched=False  # 确保每个样本单独处理
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言建模
    pad_to_multiple_of=8  # GPU优化
)

# print("===== 原始样本示例 =====")
# print(dataset[0])

# print("\n===== 处理后的特征示例 =====")
# sample = tokenized_dataset[0]
# print(f"Input IDs长度: {len(sample['input_ids'])}")
# print(f"Labels长度: {len(sample['labels'])}")
# print(f"Input IDs: {sample['input_ids'][:10]}...")
# print(f"Labels: {sample['labels'][:10]}...")
# 配置 LoRA
lengths = [len(item["input_ids"]) for item in tokenized_dataset]
print(f"最小长度: {min(lengths)}, 最大长度: {max(lengths)}, 平均长度: {sum(lengths)/len(lengths)}")
lora_config = LoraConfig(
    r=64,                   # LoRA 秩
    lora_alpha=128,          # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对 Qwen2 的注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 检查可训练参数占比

training_args = TrainingArguments(
    output_dir="./logs/qwen2-7b-hate-speech",
    num_train_epochs=3,              # 根据数据量和效果调整
    per_device_train_batch_size=2,     # 根据 GPU 显存调整 (A100可到8-16)
    gradient_accumulation_steps=4,     # 模拟更大 batch size
    learning_rate=1e-5,                # 常用学习率范围 1e-5 到 5e-5
    optim="paged_adamw_8bit",          # 节省显存优化器
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="no" ,  # 训练时不进行评估
    fp16=True,                         # 或 bf16=True (Ampere+ GPU)
    report_to="none"                # 可选监控
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
trainer.train()
trainer.save_model("./qwen2-7b-hate-speech-lora")  # 保存 LoRA 权重或合并模型