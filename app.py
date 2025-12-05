

import os
# 尝试从文件加载查询模板，如果文件不存在则使用默认值
current_directory = os.path.dirname(os.path.abspath(__file__))
import json
from flask import Flask, request, jsonify,Response, stream_with_context
from flask_cors import CORS  # 

app = Flask(__name__)
CORS(app)  # 启用跨域支持


from utils import get_news_seo_articles
 
@app.route('/seo_articles/generated_article', methods=['POST'])
def seo_articles():
    """
    POST 请求，接收关键词和公司简介
    流式返回生成的 SEO 文章
    
    请求体示例:
    {
        "keywords": "",
        "company_describe": ""
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "请求体不能为空",
                "message": "Please provide JSON data"
            }), 400
        
        keywords = data.get('keywords')
        company_describe = data.get('company_describe')
        
        # 参数验证（可选：如果两者都不提供则报错）
        if not keywords and not company_describe:
            return jsonify({
                "error": "参数错误",
                "message": "至少需要提供 keywords 或 company_describe"
            }), 400
        
        # 定义生成器函数，用于流式返回
        def generate():
            try:
                for chunk in get_news_seo_articles(
                    keywords=keywords,
                    company_describe=company_describe
                ):
                    # 将每个chunk以SSE格式发送
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                
                # 发送完成信号
                yield f"data: {json.dumps({'status': 'completed'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                # 发送错误信息
                error_msg = f"生成过程中出错: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n"
        
        # 返回流式响应
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # 禁用 nginx 缓冲
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        return jsonify({
            "error": "服务器错误",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8220, debug=True)