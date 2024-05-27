import os
import io
import torch
from fastapi import FastAPI
from fastapi import UploadFile
from transformers import ViTForImageClassification, ViTFeatureExtractor, pipeline
from PIL import Image as PILImage

os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

MODEL_ID = "IAmFlyingMonkey/pokemon_classifier"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTForImageClassification.from_pretrained(MODEL_ID).to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_ID)

app = FastAPI(
  title="Model Server",
  version="1.0",
  description="A simple API server using huggingface's Runnable interfaces",
)

@app.post("/which_pokemon")  
async def upload_photo(file: UploadFile): 
    img = await file.read()
    img = PILImage.open(io.BytesIO(img))
    img = img.convert('RGB')
    extracted = feature_extractor(images=img, return_tensors='pt').to(device)
    predicted_id = model(**extracted).logits.argmax(-1).item()
    predicted_pokemon = model.config.id2label[predicted_id]
    return {"message": predicted_pokemon}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
