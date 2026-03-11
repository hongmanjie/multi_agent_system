from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

# 创建FastAPI应用
app = FastAPI(
    title="ES搜索服务",
    description="基于FastAPI的ES搜索接口封装",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class SearchRequest(BaseModel):
    keyword: str

# 定义响应模型
class VideoInfo(BaseModel):
    name: str
    url: str
    cover: str
    size: int = 0
    upload_time: str = ""

class SearchResponse(BaseModel):
    success: bool
    keyword: str
    videos: list[VideoInfo]
    count: int
    message: str = ""

# 定义ES搜索端点
@app.post("/search/es", response_model=SearchResponse)
async def search_es(request: SearchRequest):
    """
    通过ES搜索视频
    
    Args:
        request: 包含搜索关键词的请求体
        
    Returns:
        包含搜索结果的响应
    """
    try:
        keyword = request.keyword.strip()
        
        if not keyword:
            raise HTTPException(status_code=400, detail="关键词不能为空")
        
        # 调用ES服务
        es_service_url = "http://192.168.86.113:8082/search/keyword"
        response = requests.post(
            es_service_url,
            headers={'Content-Type': 'application/json'},
            json={'keyword': keyword}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ES服务调用失败: {response.status_code}")
        
        # 解析ES服务返回的结果
        result = response.json()
        
        if result.get('code') != 200:
            raise HTTPException(status_code=500, detail=f"ES搜索失败: {result.get('message', '未知错误')}")
        
        # 处理ES搜索结果，转换为前端需要的格式
        es_videos = []
        records = result.get('data', {}).get('records', [])
        
        for record in records:
            video_info = VideoInfo(
                name=record.get('videoName', ''),
                url=record.get('filePath', ''),
                cover=record.get('videoUrl', ''),
                size=0,  # ES接口没有返回文件大小
                upload_time=""
            )
            
            # 处理时间格式
            create_time = record.get('createTime')
            if create_time:
                try:
                    # 尝试解析ISO格式的时间字符串
                    if isinstance(create_time, str):
                        # 处理可能的时间格式
                        if 'T' in create_time:
                            # ISO格式: 2026-02-10T03:25:16.797693210Z
                            from datetime import datetime
                            dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                            video_info.upload_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"解析时间失败: {e}")
            
            es_videos.append(video_info)
        
        return SearchResponse(
            success=True,
            keyword=keyword,
            videos=es_videos,
            count=len(es_videos)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ES搜索API错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 定义健康检查端点
@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    Returns:
        健康状态
    """
    return {"status": "healthy", "service": "ES搜索服务"}

# 定义根端点
@app.get("/")
async def root():
    """
    根端点
    
    Returns:
        服务信息
    """
    return {
        "message": "ES搜索服务",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search/es",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)
