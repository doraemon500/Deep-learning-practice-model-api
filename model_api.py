import os
import io
import torch
from fastapi import FastAPI
from fastapi import UploadFile
from transformers import ViTForImageClassification, ViTFeatureExtractor, pipeline
from PIL import Image as PILImage
import torch.nn.functional as F

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
    # print(file.filename)
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
