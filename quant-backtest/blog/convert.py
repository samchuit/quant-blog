#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将markdown转换为带样式的HTML"""
import os
import re

TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Jarvis's Quant Blog</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.8; color: #333; background: #f5f5f5; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; }}
        header h1 {{ font-size: 1.5rem; }}
        .nav {{ background: white; padding: 0.75rem 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: sticky; top: 0; z-index: 100; }}
        .nav ul {{ list-style: none; display: flex; gap: 1rem; flex-wrap: wrap; }}
        .nav a {{ text-decoration: none; color: #667eea; font-weight: 500; padding: 0.4rem; }}
        .nav a:hover {{ color: #764ba2; }}
        .container {{ max-width: 1000px; margin: 1.5rem auto; padding: 0 1rem; }}
        .content {{ background: white; border-radius: 8px; padding: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .content h1 {{ color: #333; margin: 1.5rem 0 1rem; font-size: 1.8rem; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }}
        .content h2 {{ color: #444; margin: 1.5rem 0 0.8rem; font-size: 1.4rem; border-left: 4px solid #667eea; padding-left: 0.8rem; }}
        .content h3 {{ color: #555; margin: 1.2rem 0 0.6rem; font-size: 1.1rem; }}
        .content p {{ margin: 0.8rem 0; }}
        .content ul, .content ol {{ margin: 0.8rem 0; padding-left: 1.5rem; }}
        .content li {{ margin: 0.4rem 0; }}
        .content code {{ background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: monospace; font-size: 0.9em; }}
        .content pre {{ background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 1rem 0; }}
        .content pre code {{ background: none; padding: 0; color: inherit; }}
        .content table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        .content th, .content td {{ border: 1px solid #ddd; padding: 0.6rem; text-align: left; }}
        .content th {{ background: #f5f5f5; font-weight: 600; }}
        .content blockquote {{ border-left: 4px solid #667eea; padding-left: 1rem; margin: 1rem 0; color: #666; background: #f9f9f9; padding: 0.8rem; }}
        .content strong {{ color: #333; }}
        .content a {{ color: #667eea; }}
        footer {{ text-align: center; padding: 1.5rem; color: #666; margin-top: 2rem; }}
    </style>
</head>
<body>
    <header><h1>🤖 量化研究 | Jarvis's Quant Blog</h1></header>
    <nav class="nav">
        <ul>
            <li><a href="../index.html">🏠 首页</a></li>
            <li><a href="quant_resources.html">📚 资源</a></li>
            <li><a href="factor_research.html">📊 因子</a></li>
            <li><a href="ml_rl_research.html">🧠 ML/RL</a></li>
            <li><a href="practical_guide.html">⚡ 实战</a></li>
            <li><a href="advanced_strategies.html">🚀 高级</a></li>
        </ul>
    </nav>
    <div class="container">
        <main class="content">
{body}
        </main>
    </div>
    <footer><p>🤖 量化研究 | <a href="../index.html">返回首页</a></p></footer>
</body>
</html>'''

def convert_md(md_content, title):
    html = md_content
    
    # 标题
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # 粗体
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # 代码块
    html = re.sub(r'```python\n(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```\n(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
    
    # 链接
    html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
    
    # 段落
    html = html.replace('\n\n', '</p><p>')
    html = '<p>' + html + '</p>'
    
    # 列表
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # 清理
    html = re.sub(r'<p></p>', '', html)
    html = re.sub(r'<p>(<h[123]>)', r'\1', html)
    html = re.sub(r'(</h[123]>)<p>', r'\1', html)
    html = re.sub(r'<p>(<pre>)', r'\1', html)
    html = re.sub(r'(</pre>)<p>', r'\1', html)
    
    return TEMPLATE.format(title=title, body=html)

def main():
    blog_dir = os.path.dirname(os.path.abspath(__file__))
    src_docs = os.path.dirname(blog_dir) + '/docs'
    out_docs = blog_dir + '/docs'
    
    os.makedirs(out_docs, exist_ok=True)
    
    for f in os.listdir(src_docs):
        if f.endswith('.md'):
            md_path = src_docs + '/' + f
            html_path = out_docs + '/' + f.replace('.md', '.html')
            
            print(f'Converting: {f}')
            
            with open(md_path, 'r', encoding='utf-8') as file:
                md = file.read()
            
            # 提取标题
            title = f.replace('.md', '')
            for line in md.split('\n'):
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            html = convert_md(md, title)
            
            with open(html_path, 'w', encoding='utf-8') as file:
                file.write(html)
            
            print(f'  -> {html_path}')
    
    print('\nDone!')

if __name__ == '__main__':
    main()
