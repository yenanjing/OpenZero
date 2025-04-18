from fastapi import FastAPI
from pydantic import BaseModel
from similarities import BertSimilarity
import uvicorn

# 初始化模型
model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-chinese")

app = FastAPI()


class SimilarityRequest(BaseModel):
    ground_truth: str
    answer_text: str


@app.post("/similarity")
async def calculate_similarity(request: SimilarityRequest):
    # 计算相似度
    sentence_similarity = model.similarity(request.ground_truth, request.answer_text)
    semantic_score = float(sentence_similarity)

    return {
        "ground_truth": request.ground_truth,
        "answer_text": request.answer_text,
        "similarity_score": semantic_score
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)