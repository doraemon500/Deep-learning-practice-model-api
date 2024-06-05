import os
import io
import torch
from fastapi import FastAPI
from fastapi import UploadFile
from starlette.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTFeatureExtractor, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch.nn.functional as F

os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

origins = [
    "*"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
  title="Model Server",
  version="1.0",
  description="A simple API server using huggingface's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/which_tag")
async def get_tag(text: str):
    SAVE_PATH = r"./adaptors"
    MODEL_ID = "beomi/KcELECTRA-base-v2022"
    # categories = ['자유', '학사', '장학/행정', '학교생활', '수업', '수업/이과', '수업/문과' , '캠퍼스', '취업/진로' ,'일상생활', '음식점/카페', '취미/여가', '인간관계', '병역']
    categories = ['학사/졸업' , '장학/행정', '학교생활', '수업/이과', '수업/문과' , '캠퍼스' ,'일상생활', '취미/여가', '인간관계', '취업/진로']

    tokenizer = AutoTokenizer.from_pretrained(
      MODEL_ID
    )

    target_modules = "query,value,output.dense,intermediate.dense,classifier.dense,classifier.out_proj".split(",")

    lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=target_modules,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model.to(device)
    model = get_peft_model(model, lora_config)

    tokens =tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128
    )    

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    pred_labels = []

    for id in range(1, 11):
        model_path = SAVE_PATH + f'/model-with-LoRA-category-id-' + str(id) + '-LoRA-adaptors'
        lora_weights = torch.load(model_path, map_location=torch.device('cpu'))
        for name, param in model.named_parameters():
            if 'lora' in name and name in lora_weights:
                param.data = lora_weights[name]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(logits, dim=1)

        if predicted_label == 1:
            pred_labels.append(categories[id - 1])

    return {"message": pred_labels}
    

@app.post("/which_pokemon")  
async def upload_photo(file: UploadFile):
    MODEL_ID = "IAmFlyingMonkey/pokemon_classifier"
    
    model = ViTForImageClassification.from_pretrained(MODEL_ID).to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_ID)

    img = await file.read()
    img = PILImage.open(io.BytesIO(img))
    img = img.convert('RGB')
    extracted = feature_extractor(images=img, return_tensors='pt').to(device)

    predicted_logits = model(**extracted).logits
    predicted_id = predicted_logits.argmax(-1).item()
    softmax_probs = F.softmax(predicted_logits, dim=1)
    predicted_score, _ = torch.max(softmax_probs, dim=1)
    predicted_score = predicted_score.item() 
    predicted_pokemon = model.config.id2label[predicted_id]

    return {"pokemon": predicted_pokemon, "score": predicted_score}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
