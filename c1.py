# create_preprocessor_config.py
import json
import os

preprocessor_config = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "feature_extractor_type": "ViTFeatureExtractor",
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": 2,
    "size": 224
}

save_path = "./models/photo-finetuned/preprocessor_config.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(preprocessor_config, f, indent=2)

print(f"preprocessor_config.json 已创建到: {save_path}")