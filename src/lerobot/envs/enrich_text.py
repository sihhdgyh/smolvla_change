from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import numpy as np

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-2B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


def get_enriched_text(image,text):
    from PIL import Image
    import torch
    import numpy as np

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # --- 核心 Prompt 工程：自动生成逻辑 ---
    # 这里的 Meta-Prompt 是为了让模型无论遇到什么指令都能自动套用“空间分析”逻辑
    meta_rules = (
        "Task: Based on the image, transform the original brief instruction into a highly detailed execution plan. \n"
        "Requirements:\n"
        "1. Visual Grounding: Mention the specific visual features (color, texture) and the exact spatial location of the target (e.g., 'at the far back of the counter', 'left-hand side').\n"
        "2. Relative Landmarks: Describe the target's position relative to other fixed objects in the scene to provide depth cues.\n"
        "3. Step-by-Step Breakdown: Divide into two sub-tasks: Sub-task 1 for precise grasping and Sub-task 2 for stable placement.\n"
        "4. Action Specificity: Use precise verbs like 'Center the gripper over', 'Align horizontally', 'Maintain clearance'.\n\n"
        f"Original Instruction: {text}\n"
        "Enriched Instruction (Start directly with Sub-task 1):"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": meta_rules},
            ],
        }
    ]

    # --- 推理过程 ---
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        # max_new_tokens 设为 256 以确保描述足够详细
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False  # 训练数据的指令生成必须保证确定性
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()