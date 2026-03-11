from fastmcp import FastMCP
import requests
import json


# 创建FastMCP实例
mcp = FastMCP(name="ES搜索服务")


@mcp.tool()
async def search_es(keyword: str) -> dict:
    """
    通过ES搜索视频
    
    Args:
        keyword: 搜索关键词
        
    Returns:
        包含搜索结果的字典
    """
    if not keyword or not keyword.strip():
        raise ValueError("关键词不能为空")
    
    keyword = keyword.strip()
    
    try:
        # 调用ES服务
        es_service_url = "http://192.168.86.113:8082/search/keyword"
        response = requests.post(
            es_service_url,
            headers={'Content-Type': 'application/json'},
            json={'keyword': keyword},
            timeout=30
        )
        
        if response.status_code != 200:
            raise ValueError(f"ES服务调用失败: {response.status_code}")
        
        # 解析ES服务返回的结果
        result = response.json()
        
        if result.get('code') != 200:
            raise ValueError(f"ES搜索失败: {result.get('message', '未知错误')}")
        
        # 处理ES搜索结果，转换为更友好的格式
        es_videos = []
        records = result.get('data', {}).get('records', [])
        
        for record in records:
            video_info = {
                "name": record.get('videoName', ''),
                "url": record.get('filePath', ''),
                "cover": record.get('videoUrl', ''),
                "size": 0,
                "upload_time": ""
            }
            
            # 处理时间格式
            create_time = record.get('createTime')
            if create_time:
                try:
                    if isinstance(create_time, str):
                        if 'T' in create_time:
                            from datetime import datetime
                            dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                            video_info["upload_time"] = dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"解析时间失败: {e}")
            
            es_videos.append(video_info)
        
        return {
            "success": True,
            "keyword": keyword,
            "videos": es_videos,
            "count": len(es_videos),
            "message": "搜索成功"
        }
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"搜索失败: {str(e)}")


if __name__ == "__main__":
    # 使用HTTP传输启动MCP服务器
    mcp.run(transport="http", host="0.0.0.0", port=8889)
