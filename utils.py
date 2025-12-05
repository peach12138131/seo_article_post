import requests
import json
from datetime import datetime
import os
import random
from typing import List, Dict, Any, Optional,Generator,Tuple
from collections import defaultdict
import time

import re 
from config import claude_key,openai_key,tavily_key


def query_gpt_model(prompt: str, article: str, api_key: str=claude_key, 
                          base_url: str = "https://api.anthropic.com/v1", 
                          model: str = "claude-sonnet-4-5-20250929", 
                          max_tokens: int = 10240, 
                          temperature: float = 0.0, 
                          json_schema: dict = None) -> Generator[Tuple[str, Optional[str]], None, None]:
 
    
    url = f"{base_url}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    # 构建 payload
    if json_schema:
        user_content = f"{prompt}\n\nIMPORTANT: Output ONLY valid JSON in this exact format, with no markdown code blocks, no explanations, no extra text:\n{str(json_schema)}\n\nData to process:\n{article}"
    else:
        user_content = f"{prompt}\n\n{article}"
    
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True  # 流式传输
    }

    full_response = ""  # 🔑 收集完整内容
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                
                if not line_text.startswith('data: '):
                    continue
                
                json_str = line_text[6:]
                
                if json_str == '[DONE]':
                    break
                
                try:
                    event_data = json.loads(json_str)
                    
                    if event_data.get('type') == 'content_block_delta':
                        delta = event_data.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            text_chunk = delta.get('text', '')
                            if text_chunk:
                                full_response += text_chunk  # 累积完整内容
                                yield (text_chunk, None)  # 🔑 流式返回片段，完整内容为 None
                                
                except json.JSONDecodeError:
                    continue
        
       
        if full_response:
            if json_schema:
                # 清理 JSON
                json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', full_response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1).strip()
                else:
                    json_text = full_response.strip()
                
                json_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_text)
                yield ("", json_text)  # 🔑 最后返回完整 JSON
            else:
                yield ("", full_response)  # 🔑 最后返回完整文本
        else:
            yield ("", None)
                    
    except requests.exceptions.RequestException as e:
        print(f"API请求异常: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API错误响应: {e.response.text}")
        yield ("", None)



def query_openai_model(prompt: str, article: str, api_key: str=openai_key, base_url: str = "https://api.openai.com/v1", 
                       model: str = "gpt-5-chat-latest", max_tokens: int = 10240, 
                       temperature: float = 0.8,json_schema: dict = None) -> Optional[str]:
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{prompt}\n \n{article}"}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema
        }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            text_content = response_json["choices"][0]["message"]["content"]
            return text_content
        else:
            print("API返回内容格式异常")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API请求异常: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API错误响应: {e.response.text}")
        return None
    


def search_news(tavily_api_key: str=tavily_key, query: str="", max_results: int = 15, days: int = 3, 
                search_depth: str = "basic", include_answer: bool = True) -> List[Dict[str, Any]]:
    """
    使用Tavily API搜索新闻
    
    Args:
        tavily_api_key: Tavily API密钥
        query: 搜索查询词
        max_results: 最大返回结果数 (默认10)
        days: 搜索时间范围(天) (默认7天)
        search_depth: 搜索深度 "basic" 或 "advanced" (默认basic)
        include_answer: 是否包含AI生成的答案摘要 (默认True)
    
    Returns:
        包含搜索结果的字典列表
    """
    url = "https://api.tavily.com/search"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {tavily_api_key}'
    }
    
    data = {
        "query": query,
        "topic": "news",  # 专门搜索新闻
        "search_depth": search_depth,  # 搜索深度
        "chunks_per_source": 3,  # 每个来源的内容块数
        "max_results": max_results,  # 最大结果数
        "time_range": None,  # 时间范围(null表示使用days参数)
        "days": days,  # 搜索最近N天的新闻
        "include_answer": include_answer,  # 包含AI生成的答案摘要
        "include_raw_content": False,  # 不包含原始HTML内容
        "include_images": False,  # 包含图片
        "include_image_descriptions": False,  # 不包含图片描述
        "include_domains": [],  # 包含的域名列表(空表示不限制)
        "exclude_domains": [],  # 排除的域名列表
        "country": None  # 国家限制(null表示全球)
    }
    
    try:
        print(f"正在搜索新闻: {query}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        result = response.json()
        
        # 检查API响应状态
        if 'results' not in result:
            print(f"⚠️ API响应异常: {result}")
            return []
        
        results = result.get("results", [])
        answer = result.get("answer", "")  # AI生成的答案摘要
        
        print(f"找到 {len(results)} 条新闻结果\n")
        # print(result)
        
        
        # 如果包含AI答案摘要，打印出来
        if include_answer and answer:
            pass
            # print(f"AI摘要: {answer[:]}...")
        
        return answer, results
        
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请检查网络连接")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误详情: {e.response.text}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")
        return []
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return []



def chose_keywords(keyword_list, n=1):
   
    if n >= len(keyword_list):
        return keyword_list
    
    max_rank = max(item.get('real_time_rank', 20) for item in keyword_list)
    
   
    weighted_items = []
    for item in keyword_list:
        rank = item.get('real_time_rank', 20)
        weight = (max_rank + 1) - rank
        weighted_items.append((item, weight))
    
  
    weighted_pool = []
    for item, weight in weighted_items:
        weighted_pool.extend([item] * weight)
    
    
    selected = []
    remaining_pool = weighted_pool.copy()
    
    while len(selected) < n and remaining_pool:
        chosen = random.choice(remaining_pool)
        if chosen not in selected:
            selected.append(chosen)
       
        remaining_pool = [item for item in remaining_pool if item != chosen]
    
    return selected

def get_news_seo_articles(keywords:str =None,company_describe=None):
    from news_seo_prompts import news_keywords,news_search_prompt,news_schema,extract_prompt,rewrite_prompt,seo_keywords,seo_metadata,seo_rewrite_prompt 
    date_str = datetime.now().strftime("%Y%m%d")
    news_pool=[]


    if keywords:
        keyword_str=keywords
        all_keyword=keywords
        category=""
        
        ai_summary,news_results=search_news(query=news_search_prompt.format(category=category,keyword=keyword_str))
        print(ai_summary)
        yield f"**News Search Results**\n{ai_summary}\n\n"
        all_news_results = news_results  
        
    else:
        
        keywords_list=chose_keywords(news_keywords,n=1)
        all_news_results = [] 
        all_keyword=""
        for i, news in enumerate(keywords_list): 
            category = news.get("category", "")
            keyword_str = news.get("keyword_en", "")
            all_keyword +=keyword_str 
            ai_summary,news_result=search_news(query=news_search_prompt.format(category=category,keyword=keyword_str))
            yield f"**News Search Results**\n{ai_summary}\n\n"
            print(ai_summary)
            
            # print(news_result)
            all_news_results.extend(news_result) 

  
    print("正在整理新闻")
    yield "**📰 Organizing news...**\n\n"
    for chunk, complete in query_gpt_model(prompt=extract_prompt.format(date_str=date_str,keywords=all_keyword),article=str(all_news_results),json_schema=news_schema):
        
        if chunk:
            yield chunk
            print(chunk, end='', flush=True)  # 实时显示给用户
        if complete is not None:
            extract_news = complete  # 收集完整结果
    # print(extract_news)
    extract_news=json.loads(extract_news)
    news_pool.append(extract_news["news_list"])

    #去对news_pool进行去重
    all_news = []
    for news_list in news_pool:
        all_news.extend(news_list)

    # 用字典去重，URL作为key
    unique_news_dict = {}
    for news in all_news:
        url = news.get('url')
        if url:
            unique_news_dict[url] = news  # 如果URL重复，会被覆盖

    news_pool = list(unique_news_dict.values())
   


    # for chunk, complete in query_gpt_model(prompt=rewrite_prompt.format(str(news_pool)),article=""):
    
    #     if chunk:
    #         print(chunk, end='', flush=True)  # 实时显示给用户
    #     if complete is not None:
    #         rewritten_article = complete  # 收集完整结果

    print("智能分析关键词")
    yield "\n\n**🔍 Analyzing keywords intelligently...**\n\n"
    for chunk, complete in query_gpt_model(prompt=seo_keywords.format(company=company_describe,news=str(news_pool)),article=""):
        
        if chunk:
            yield chunk
            print(chunk, end='', flush=True)  # 实时显示给用户
        if complete is not None:
            extract_keywords = complete  # 收集完整结果
    print("构建文章结构")
    yield "\n\n**📋 Building article structure...**\n\n"
    for chunk, complete in query_gpt_model(prompt=seo_metadata.format(company=company_describe,keywords=extract_keywords,news=str(news_pool)),article=""):
       
        if chunk:
            yield chunk
            print(chunk, end='', flush=True)  # 实时显示给用户
        if complete is not None:
            metadata = complete  # 收集完整结果
    print("专业seo文章生成中")
    yield "\n\n**✍️ Generating professional SEO article...**\n\n"
    for chunk, complete in query_gpt_model(prompt=seo_rewrite_prompt.format(news=str(news_pool),keywords=extract_keywords,metadata=metadata),article=""):
       
        if chunk:
            yield chunk
            print(chunk, end='', flush=True)  # 实时显示给用户
        if complete is not None:
            seo_article = complete  # 收集完整结果









if __name__ == "__main__":
#   
    get_news_seo_articles( keywords="what about Beijing private jet")