from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# 创建FastAPI应用
app = FastAPI(
    title="检索智能体",
    description="基于FastAPI和LangChain的检索智能体，用于识别用户搜索意图",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化MaaS服务配置
MAAS_SERVER_URL = "http://localhost:8010"
QWEN3_MODEL_PATH = "/nlp/qwen3_4b/predict"

# 调用MaaS服务中的Qwen3模型
def call_qwen3_model(prompt: str) -> str:
    """
    调用smart_maas_server中的Qwen3模型
    
    Args:
        prompt: 输入提示词
        
    Returns:
        模型生成的响应
    """
    try:
        print(f"开始调用Qwen3模型...")
        url = f"{MAAS_SERVER_URL}{QWEN3_MODEL_PATH}"
        print(f"调用URL: {url}")
        payload = {
            "request_id": f"retrieval_agent_{int(time.time())}",
            "prompt": prompt,
            "temperature": 0.7,
            "max_new_tokens": 512
        }
        headers = {
            "Content-Type": "application/json"
        }
        print(f"发送请求到MaaS服务...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"收到响应，状态码: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        print(f"响应结果: {result}")
        
        # 处理响应结果
        prediction = result.get("prediction", "")
        print(f"模型预测结果: {prediction}")
        
        # 检查prediction的类型
        if isinstance(prediction, dict):
            # 如果是字典，尝试获取gen_text字段
            gen_text = prediction.get("gen_text", "")
            print(f"从prediction字典中提取gen_text: {gen_text}")
            return gen_text
        else:
            # 如果不是字典，直接返回
            return prediction
    except Exception as e:
        print(f"调用Qwen3模型失败: {e}")
        # 失败时返回空字符串，会触发备用方法
        return ""

# 创建意图分析提示模板
intent_template = """
你是一个专业的搜索意图分析器，负责分析用户的搜索查询并识别其意图。

请分析以下用户搜索查询：

查询：{query}

可能的意图类型包括：
- 人物
- 音乐
- 电影
- 剧集
- 体育
- 新闻
- 动物
- 美食
- 旅游
- 科技
- 教育
- 游戏
- 汽车
- 自然
- 历史
- 艺术
- 其他

请按照以下JSON格式返回分析结果：
{{
  "intent": "识别出的意图类型",
  "confidence": 置信度（0-1之间的数字）,
  "keywords": ["提取的关键词1", "提取的关键词2"],
  "enhanced_query": "增强后的查询"
}}

要求：
1. 意图类型可以是预定义列表中的类型，也可以是根据查询内容自动生成的类型
2. 当查询涉及新兴领域、专业话题或预定义意图无法准确描述时，强烈建议自动生成更具体的意图类型
3. 自动生成的意图应该简洁明了，能够准确反映查询的核心主题
4. 置信度要合理反映识别的确定程度
5. 关键词要能准确反映查询的核心内容
6. 增强后的查询要能提高搜索准确性
7. 增强后的查询应该简洁明了，去除冗余词汇，只保留核心搜索词
8. 对于复杂查询，如"帮我找包含...的视频"，应提取核心关键词作为增强查询
"""

# 创建提示模板
intent_prompt = PromptTemplate(
    template=intent_template,
    input_variables=["query"]
)

# 创建输出解析器
output_parser = StrOutputParser()

# 分析用户意图的函数
def analyze_intent_with_qwen3(query: str) -> dict:
    """
    使用Qwen3模型分析用户意图
    
    Args:
        query: 用户查询文本
        
    Returns:
        包含意图分析结果的字典
    """
    try:
        print("=== 开始使用LangChain进行意图分析 ===")
        
        # 使用LangChain的PromptTemplate构建完整的提示词
        print(f"1. 开始使用LangChain的PromptTemplate处理查询: {query}")
        try:
            full_prompt = intent_prompt.format(query=query)
            print(f"   ✓ 成功生成提示词: {full_prompt[:100]}...")
        except Exception as prompt_error:
            print(f"   ✗ LangChain PromptTemplate处理失败: {prompt_error}")
            # 使用备用方法
            return analyze_intent_fallback(query)
        
        # 调用Qwen3模型
        print("2. 开始调用Qwen3模型...")
        result_str = call_qwen3_model(full_prompt)
        
        # 使用LangChain的StrOutputParser解析结果
        if result_str:
            print(f"   ✓ 模型调用成功，获取响应")
            print(f"3. 开始使用LangChain的StrOutputParser解析结果...")
            
            try:
                # 使用LangChain的StrOutputParser解析
                parsed_output = output_parser.parse(result_str)
                print(f"   ✓ 成功使用LangChain解析输出")
                print(f"   解析结果: {parsed_output[:100]}...")
            except Exception as parse_error:
                print(f"   ✗ LangChain StrOutputParser解析失败: {parse_error}")
                # 直接使用原始输出
                parsed_output = result_str
                print(f"   使用原始输出: {parsed_output[:100]}...")
            
            # 进一步处理解析后的输出
            print("4. 处理解析后的输出...")
            processed_output = parsed_output.strip()
            if processed_output.startswith('```json'):
                processed_output = processed_output[7:]
                print("   移除JSON代码块标记")
            if processed_output.endswith('```'):
                processed_output = processed_output[:-3]
                print("   移除代码块结束标记")
            processed_output = processed_output.strip()
            print(f"   处理后的JSON字符串: {processed_output}")
            
            # 解析JSON
            print("5. 解析JSON结果...")
            try:
                result = json.loads(processed_output)
                print(f"   ✓ JSON解析成功")
                print(f"   解析后的结果: {result}")
                print("=== LangChain意图分析完成 ===")
                return result
            except json.JSONDecodeError as json_error:
                print(f"   ✗ JSON解析失败: {json_error}")
                # 解析失败，使用备用方法
                return analyze_intent_fallback(query)
        else:
            # 模型调用失败，使用备用方法
            print("   ✗ 模型返回空结果")
            return analyze_intent_fallback(query)
    except Exception as e:
        print(f"Qwen3意图分析失败，使用备用方法: {e}")
        # 解析失败或其他错误，使用备用方法
        return analyze_intent_fallback(query)

# 定义请求模型
class AgentRequest(BaseModel):
    query: str

# 定义响应模型
class IntentInfo(BaseModel):
    intent: str
    confidence: float
    keywords: list[str]
    enhanced_query: str

class AgentResponse(BaseModel):
    success: bool
    code: int
    original_query: str
    data: IntentInfo
    message: str = ""

# 定义检索智能体端点
@app.post("/agent/analyze", response_model=AgentResponse)
async def analyze_query(request: AgentRequest):
    """
    分析用户搜索查询，识别意图并增强查询
    
    Args:
        request: 包含用户查询的请求体
        
    Returns:
        包含意图分析结果的响应
    """
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="查询不能为空")
        
        # 使用Qwen3模型进行意图分析
        try:
            result = analyze_intent_with_qwen3(query)
        except Exception as e:
            print(f"Qwen3意图分析失败，使用备用方法: {e}")
            # 备用方法：使用规则匹配
            result = analyze_intent_fallback(query)
        
        # 后处理：提取核心关键词，确保增强查询更适合ES搜索
        # 1. 提取核心关键词
        core_keywords = extract_core_keywords(query, result["intent"])
        # 2. 生成优化的增强查询
        optimized_query = generate_optimized_query(core_keywords, result["intent"])
        
        # 更新结果
        result["keywords"] = core_keywords
        result["enhanced_query"] = optimized_query
        
        # 构建响应
        return AgentResponse(
            success=True,
            code=200,
            original_query=query,
            data=IntentInfo(
                intent=result["intent"],
                confidence=result["confidence"],
                keywords=result["keywords"],
                enhanced_query=result["enhanced_query"]
            ),
            message=""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"意图分析错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 定义搜索增强端点
@app.post("/agent/search", response_model=dict)
async def enhanced_search(request: AgentRequest):
    """
    增强搜索：先分析意图，再执行搜索
    
    Args:
        request: 包含用户查询的请求体
        
    Returns:
        包含搜索结果的响应
    """
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="查询不能为空")
        
        # 1. 分析用户意图
        try:
            intent_result = analyze_intent_with_qwen3(query)
        except Exception as e:
            print(f"Qwen3意图分析失败，使用备用方法: {e}")
            intent_result = analyze_intent_fallback(query)
        
        # 后处理：提取核心关键词，确保增强查询更适合ES搜索
        # 1. 提取核心关键词
        core_keywords = extract_core_keywords(query, intent_result["intent"])
        # 2. 生成优化的增强查询
        optimized_query = generate_optimized_query(core_keywords, intent_result["intent"])
        
        # 更新结果
        intent_result["keywords"] = core_keywords
        intent_result["enhanced_query"] = optimized_query
        
        # 2. 使用增强后的查询执行ES搜索
        es_service_url = "http://localhost:8089/search/es"
        response = requests.post(
            es_service_url,
            headers={'Content-Type': 'application/json'},
            json={'keyword': intent_result["enhanced_query"]}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ES服务调用失败: {response.status_code}")
        
        search_result = response.json()
        
        # 3. 整合意图分析结果和搜索结果
        return {
            "success": True,
            "code": 200,
            "original_query": query,
            "data": intent_result,
            "search_result": search_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"增强搜索错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 备用意图分析函数（当LangChain失败时使用）
def analyze_intent_fallback(query: str) -> dict:
    """
    备用意图分析函数，使用规则匹配
    
    Args:
        query: 用户查询文本
        
    Returns:
        包含意图分析结果的字典
    """
    # 小写处理
    query_lower = query.lower()
    
    # 意图识别规则
    intents = {
        "人物": ["人物", "演员", "明星", "角色", "人"],
        "音乐": ["音乐", "歌曲", "歌", "歌手", "演唱会", "mv"],
        "电影": ["电影", "影片", "电影名", "电影片段"],
        "剧集": ["电视剧", "剧集", "电视剧集", "网剧"],
        "体育": ["体育", "足球", "篮球", "比赛", "赛事"],
        "新闻": ["新闻", "资讯", "报道", "新闻联播"],
        "动物": ["动物", "宠物", "猫", "狗", "鸟", "鱼", "野生动物", "家畜"],
        "美食": ["美食", "食物", "菜", "菜谱", "烹饪", "餐厅", "小吃", "甜点"],
        "旅游": ["旅游", "旅行", "景点", "景区", "度假", "酒店", "民宿"],
        "科技": ["科技", "技术", "人工智能", "ai", "机器人", "互联网", "软件", "硬件"],
        "教育": ["教育", "学习", "课程", "教学", "培训", "考试", "学校", "大学"],
        "游戏": ["游戏", "电竞", "手游", "网游", "主机游戏", "游戏直播"],
        "汽车": ["汽车", "车辆", "车", "驾驶", "交通", "赛车", "电动车"],
        "自然": ["自然", "风景", "山水", "风景", "景观", "环境", "气候", "天气"],
        "历史": ["历史", "古代", "文物", "古迹", "博物馆", "文化", "传统"],
        "艺术": ["艺术", "绘画", "书法", "雕塑", "表演", "舞蹈", "戏剧"],
        "其他": []
    }
    
    # 识别意图
    intent_scores = {}
    for intent, keywords in intents.items():
        if intent == "其他":
            continue
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            intent_scores[intent] = score
    
    # 确定最可能的意图
    if intent_scores:
        max_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        confidence = min(intent_scores[max_intent] / len(query.split()), 1.0)
    else:
        # 对于无法匹配预定义意图的查询，尝试自动生成意图
        # 简单实现：使用核心关键词作为意图
        core_words = extract_core_keywords(query, "其他")
        if core_words:
            # 使用第一个核心关键词作为意图
            max_intent = core_words[0]
        else:
            max_intent = "其他"
        confidence = 0.5
    
    # 提取关键词
    extracted_keywords = []
    for word in query.split():
        if len(word) > 1:
            extracted_keywords.append(word)
    
    # 增强查询
    enhanced_query = query
    
    # 根据意图增强查询
    if max_intent == "人物":
        # 人物意图：保留原始查询
        pass
    elif max_intent == "音乐":
        # 音乐意图：可能需要添加相关关键词
        if "歌曲" not in query_lower and "歌" not in query_lower:
            enhanced_query = f"{query} 歌曲"
    elif max_intent == "电影":
        # 电影意图：可能需要添加相关关键词
        if "电影" not in query_lower:
            enhanced_query = f"{query} 电影"
    
    return {
        "intent": max_intent,
        "confidence": confidence,
        "keywords": extracted_keywords,
        "enhanced_query": enhanced_query
    }

def extract_core_keywords(query: str, intent: str) -> list[str]:
    """
    提取查询中的核心关键词
    
    Args:
        query: 用户查询文本
        intent: 识别出的意图类型
        
    Returns:
        核心关键词列表
    """
    # 小写处理
    query_lower = query.lower()
    
    # 移除常见的查询前缀
    prefixes = ["帮我找", "帮我搜索", "我想找", "我想搜索", "找", "搜索", "查询", "帮我", "我要", "包含", "含有", "关于", "有关", "我想看"]
    for prefix in prefixes:
        if query_lower.startswith(prefix):
            query_lower = query_lower[len(prefix):].strip()
            break
    
    # 移除常见的查询后缀
    suffixes = ["的视频", "的电影", "的歌曲", "的新闻", "的资讯", "的内容", "的信息", "的资料", "视频", "电影", "歌曲", "新闻", "资讯", "内容", "信息", "资料"]
    for suffix in suffixes:
        if query_lower.endswith(suffix):
            query_lower = query_lower[:-len(suffix)].strip()
            break
    
    # 移除中间的虚词和无意义词汇
    虚词 = ["包含", "含有", "关于", "有关", "里面", "中", "里", "跟", "和", "与", "或", "是", "在", "有", "为", "以", "几个", "一些", "很多", "查找", "搜索", "查询", "找", "帮我", "我要", "我想", "我想看", "我想找", "的资料", "的内容", "的信息"]
    for word in 虚词:
        query_lower = query_lower.replace(word, " ").strip()
    
    # 分词并提取关键词
    # 这里使用简单的规则分词，实际应用中可以使用更复杂的NLP工具
    words = []
    current_word = ""
    for char in query_lower:
        if char.isalnum():
            current_word += char
        else:
            if current_word:
                words.append(current_word)
                current_word = ""
    if current_word:
        words.append(current_word)
    
    # 过滤短词和停用词
    stop_words = set(["的", "了", "和", "与", "或", "是", "在", "有", "为", "以", "我们", "你们", "他们", "个", "些", "多", "少", "我想", "我想看", "我想找", "查找", "搜索", "查询", "找", "帮我", "我要", "包含", "含有", "关于", "有关", "里面", "中", "里", "跟", "几个", "一些", "很多", "资料", "内容", "信息"])
    core_words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    # 特殊处理：如果意图是新闻，且没有提取到关键词，直接返回["新闻"]
    if intent == "新闻" and not core_words:
        return ["新闻"]
    
    # 如果没有提取到关键词，返回原始查询的主要部分
    if not core_words:
        return [query_lower[:20]]  # 限制长度
    
    return core_words

def generate_optimized_query(keywords: list[str], intent: str) -> str:
    """
    生成优化的搜索查询
    
    Args:
        keywords: 核心关键词列表
        intent: 识别出的意图类型
        
    Returns:
        优化后的搜索查询
    """
    # 特殊处理：如果意图是新闻，直接返回"新闻"
    if intent == "新闻":
        return "新闻"
    
    # 如果有多个关键词，使用空格连接
    if len(keywords) > 0:
        optimized_query = " ".join(keywords)
    else:
        optimized_query = ""
    
    # 根据意图添加相关关键词
    intent_keywords = {
        "人物": [],
        "音乐": ["音乐"],
        "电影": ["电影"],
        "剧集": ["电视剧"],
        "体育": ["体育"],
        "新闻": ["新闻"],
        "动物": ["动物"],
        "美食": ["美食"],
        "旅游": ["旅游"],
        "科技": [],
        "教育": ["教育"],
        "游戏": ["游戏"],
        "汽车": ["汽车"],
        "自然": ["自然"],
        "历史": ["历史"],
        "艺术": ["艺术"],
        "其他": []
    }
    
    # 处理预定义意图
    if intent in intent_keywords:
        for keyword in intent_keywords[intent]:
            if keyword not in optimized_query:
                optimized_query = f"{optimized_query} {keyword}"
    else:
        # 处理自动生成的意图
        # 对于自动生成的意图，将其作为关键词添加到查询中
        if intent and intent not in optimized_query:
            optimized_query = f"{optimized_query} {intent}"
    
    return optimized_query.strip()

# 定义健康检查端点
@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    Returns:
        健康状态信息
    """
    return {"status": "healthy", "service": "检索智能体"}

# 定义根端点
@app.get("/")
async def root():
    """
    根端点
    
    Returns:
        服务信息
    """
    return {
        "service": "检索智能体",
        "version": "2.0.0",
        "description": "基于FastAPI和LangChain的检索智能体，用于识别用户搜索意图",
        "endpoints": {
            "analyze": "/agent/analyze",
            "search": "/agent/search",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9990)