import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

app = FastAPI()

NVIDIA_API_KEY = "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"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    brand_name: str
    topic: str
    tone: str
    platform: str
    language: str
    image_base64: Optional[str] = None

@app.post("/generate")
async def generate_content(data: GenerateRequest):
    try:
        image_description = "No image provided."
        
        # පියවර 1: Vision AI එකෙන් පින්තූරය විස්තර කරගැනීම (If image exists)
        if data.image_base64:
            vision_completion = client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe every single detail of this product image, colors, texture, and brand quality. Just describe the image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data.image_base64}"}}
                    ]
                }],
                max_tokens=300
            )
            image_description = vision_completion.choices[0].message.content

        # පියවර 2: ලොකු AI එකෙන් (Llama 3.3-70b) ලංකාවේ වයිබ් එකට Caption එක ලියවා ගැනීම
        final_prompt = f"""
        YOU ARE THE NO.1 AD AGENCY IN THE WORLD.
        IMAGE ANALYSIS: {image_description}
        OBJECTIVE: {data.topic}
        BRAND: {data.brand_name}
        PLATFORM: {data.platform}
        TONE: {data.tone}

        STRICT SINHALA RULES:
        - Use "Elite Colloquial Sinhala" only. (Next level, පිස්සුවක්, ගින්දර, සුපිරිම, අතේ දුරින්, අමතක කරලා දාන්න බැරි).
        - No formal words (පැමිණේ, රථය, පවතී are strictly BANNED).
        - Target Audience: High-end Sri Lankans.

        RETURN ONLY VALID JSON:
        {{
          "caption": "A long, persuasive, and incredibly viral caption in {data.language}",
          "hashtags": "15 premium hashtags",
          "ideas": ["3 revolutionary marketing moves"]
        }}
        """

        final_completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=1.0,
            max_tokens=2048
        )

        raw_result = final_completion.choices[0].message.content
        
        # Clean JSON
        json_content = raw_result[raw_result.find("{"):raw_result.rfind("}")+1]
        return json.loads(json_content)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)