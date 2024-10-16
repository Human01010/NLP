import requests
from lxml import etree
import jieba.posseg as pseg
import networkx as nx
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties

# 请求弹幕XML数据
url = 'https://comment.bilibili.com/26247693358.xml'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = 'utf-8'
    response.raise_for_status()
    xml_content = response.text
    print(response.text[:1800])
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

# 解析弹幕XML
root = etree.fromstring(xml_content.encode('utf-8'))
danmus = root.xpath('//d/text()')

print(f"弹幕数量: {len(danmus)}")
print(f"前10条弹幕: {danmus[:10]}")

# 对前100条弹幕进行词性标注
danmus_sample = danmus[:1800]
word_pos_pairs = []
for danmu in danmus_sample:
    words = pseg.cut(danmu)
    word_pos_pairs.extend(words)

# 过滤名词和动词
filtered_words = [word for word, pos in word_pos_pairs if pos.startswith('n') or pos.startswith('v')]

# 加载停用词表
def load_stopwords(filepath='C:/Users/Timothy/PycharmProjects/pythonProject/NLP/hit_stopwords.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    return stopwords

stopwords = load_stopwords()
filtered_keywords = [word for word in filtered_words if word not in stopwords]

# 构建词语共现图
def build_graph(words, window_size=5):
    graph = nx.Graph()
    for i in range(len(words) - window_size):
        window = words[i:i + window_size]
        for w1, w2 in combinations(window, 2):
            if graph.has_edge(w1, w2):
                graph[w1][w2]['weight'] += 1
            else:
                graph.add_edge(w1, w2, weight=1)
    return graph

graph = build_graph(filtered_keywords, window_size=5)

# 使用TextRank算法
def textrank(graph, max_iter=200, tol=1e-6):
    return nx.pagerank(graph, max_iter=max_iter, tol=tol)

rankings = textrank(graph)
top_keywords = Counter(rankings).most_common(10)
print("Top 10 Keywords:", top_keywords)

# 生成词云并可视化
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 确保本地有这个中文字体文件路径
wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=600).generate_from_frequencies(rankings)

# 绘制词云
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
