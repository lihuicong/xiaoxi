# -*- coding: utf-8 -*-
"""
智能客服demo 线上服务
"""

import sys
sys.path.append("/data/jupyter/lihuicong/customer_server")
sys.path.append("/data/jupyter/lihuicong/customer_server/qq_match")

from sentence_spell_correct import SentenceSpellCorrect
# from text_detect_language_lemma_process import TextDetectLanguageLemmaProcess
from inference import Inference
from TextProcess import TextProcess

import faiss
#import re
#import pandas as pd
import numpy as np
import logging
from collections import OrderedDict

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
#from nltk.corpus import stopwords

#import stanza
#from stanza.models.common.doc import Document
#from stanza.pipeline.core import Pipeline
#from stanza.pipeline.multilingual import MultilingualPipeline

from flask import Flask,request,render_template
from flask.templating import render_template_string

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s - %(message)s')


app = Flask(__name__)

sc = SentenceSpellCorrect()     # 拼写纠错类
tp = TextProcess()      # 文本预处理类
TextMatchModel = Inference()     # 文本匹配模型预测类


min_threshold = 0.5    # 最小阈值，小于该阈值则认为模型匹配错误
believable_threshold = 0.75    # 可置信阈值，大于等于该阈值则认为模型匹配正确，可直接输出答案


lang_dict = {"en":"英语",
             "ar":"阿拉伯语",
             "tr":"土语",
             "id":"印尼语",
             "pt":"葡萄牙语",
             "hi":"印地语",
             "ur":"乌尔都语"
             }


RawQuestion_Reply_map_file = "/data/jupyter/lihuicong/customer_server/qq_match/data/npy/RawQuestion_Reply_map.npy"
StandardQuestion_RawQuestion_map_file = "/data/jupyter/lihuicong/customer_server/qq_match/data/npy/StandardQuestion_RawQuestion_map.npy"
StandardQuestion_StandardQuestion_map_file = "/data/jupyter/lihuicong/customer_server/qq_match/data/npy/StandardQuestion_StandardQuestion_map.npy"

StandardQuestion_RawQuestion_map = np.load(StandardQuestion_RawQuestion_map_file,allow_pickle=True).item()    # [标准问题-原始问题] 映射表  
RawQuestion_Reply_map = np.load(RawQuestion_Reply_map_file,allow_pickle=True).item()               # [原始问题-客服回复] 映射表
StandardQuestion_StandardQuestion_map = np.load(StandardQuestion_StandardQuestion_map_file,allow_pickle=True).item()    # [标准问题-标准问题] 映射表 N->1

# 知识库中的所有question（已经经过数据预处理）
question_warehouse = np.array([i for i in StandardQuestion_StandardQuestion_map])

# 知识库中的所有question对应的embedding向量
embeddings = np.array([TextMatchModel._vector(i)[0] for i in question_warehouse])
    
# 建立向量检索的索引
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


# 对list元素去重，并保持原有的顺序
def duplicate_keep_order(list1):
    _list = sorted(set(list1), key=list1.index)    # 先采用set去重，再按照key关键词定义的元素进行排序
    # _list = list(OrderedDict.fromkeys(list1))    # OrderedDict进行去重，再转换为list
    return _list
        
        
# 用户query与知识库中的标准question做匹配，返回去重后的top标准问题
def qq_match(sentence, topk=3):
    sentence = tp.process(sentence)        # 用户输入文本的预处理，包括拼写纠错、词形还原等  
    vector = TextMatchModel._vector(sentence)       # 获得输入文本的向量表示（文本匹配模型）
    top_distance, top_index = index.search(vector, k=200)      # faiss检索，取top200
    similar_questions = question_warehouse[top_index[0]]       # 检索到的所有的相似问题
    similar_standard_question = [StandardQuestion_StandardQuestion_map[i] for i in similar_questions]   # 获取所有问题对应的标准问题
    similar_standard_question_dup = duplicate_keep_order(similar_standard_question)[:topk]         # 去重并保持原有顺序，取top
    raw_question = [StandardQuestion_RawQuestion_map[i] for i in similar_standard_question_dup]    # 获取对用户友好的原始问题(未经过文本处理的问题)
    
    idx = np.array([similar_standard_question.index(i) for i in similar_standard_question_dup])    # 获取第一次出现的索引位置
    score = top_distance[0][idx]    # 获取相似距离

    # 大于等于最小阈值的数量
    available_num = len([i for i in score if i >= min_threshold])

    if available_num == 0:
        return [sentence, [], []]
    else:
        return [sentence, raw_question[:available_num], score[:available_num]]



#首页默认渲染entry.html中的内容，请求首页时显示
@app.route('/') 
def homepage() -> 'html':
    return render_template('home.html')


# 用户检索，同时支持post和get请求 
#@app.route('/get', methods=['POST', 'GET']) 
#def do_search() -> 'html':
#    sentence = request.args.get("msg")       # 获取用户post请求的输入
#    if sentence == '':
#        ans = """
#            <p class="botText"><img src="../static/customer_service.png" align="top"><span>
#                <b>Please enter your question</b>
#            </span></p>
#            """  
#        return render_template_string(ans)
#    else:
#        res = search(sentence, bm25Model, q_lang, original_q)     # bm25模型检索
#        lang = lang_dict.get(res.get("test_lang"))      # 测试语句的语言
#        sim_q = res.get("sim_q", "")       # 返回库中相似文本
#
#        # url_for进行链接跳转，调用answer函数 跳转到search/<query>页面
#        querys = [[url_for("answer", query=i), i] for i in sim_q]     
#
#        ans = """<p class="botText"><img src="../static/customer_service.png" align="top">
#                 {% if sim_q == [] %}
#                     <span><b>Sorry, I can't answer this question, please try another question o(╥﹏╥)o</b></span>
#                 {% else %}
#                     <span>Guess you want to ask the following questions</span>
#                     <ul>
#                     {% for query_info in querys %}
#                         <li>
#                             <a id=question href="{{ query_info[0] }}" data-name="{{ query_info[1] }}">
#                             <font size="5">{{ query_info[1] }}</font></a>
#                         </li>
#                         <br>
#                     {% endfor %}
#                     </ul>
#
#                 {% endif %}
#               </p>
#              """
#        return render_template_string(ans, querys=querys, sim_q=sim_q)
    


@app.route('/get', methods=['POST', 'GET']) 
def do_search() -> 'html':
    sentence = request.args.get("msg", "")       # 获取用户post请求的输入
    if sentence.strip() == '':
        ans = """
            <div class="botText">
              <img src="../static/customer_service.png" align="top">
              <span>
              <font size="3">
                <b>Please enter your question</b>
              </font>
              </span>
            </div>
            """  
        return render_template_string(ans)
    
    else:
        sentence, sim_q, score = qq_match(sentence)  
        logging.info("\nsentence:{}\nsim_q:{}\nscore:{}\n".format(sentence, sim_q, score))
#         sim_q = ["{} ({:.4f})".format(i[0],i[1]) for i in zip(sim_q,score)]
        
        if sim_q == [] or score[0] < min_threshold:
            ans = """<div class="botText">
                     <img src="../static/customer_service.png" align="top">
                     <span>
                       <b>
                       <font size="3">
                         Sorry, I can't answer this question, please give more information or try another question o(╥﹏╥)o
                       </font>
                       </b>
                     </span>
                   </div>
                """
            return render_template_string(ans)

        elif score[0] >= believable_threshold:
            _ans = RawQuestion_Reply_map.get(sim_q[0])
            return _ans

        else:
            ans = """<div class="botText">
                      <img src="../static/customer_service.png" align="top">
                      <span>
                        <font size="3">
                          Guess you want to ask the following questions
                        </font>
                      </span>
                      <ul class="ques-ans">
                      {% for query_info in querys %}
                        <li>
                          <a id=question href="javascript:void(0);" onclick ='getBotQuestionResponse("{{query_info}}")'>
                            <font size="3">{{ query_info }}</font>
                          </a>
                        </li>
                         {% endfor %}
                      </ul>
                    </div>
                  """        
            return render_template_string(ans, querys=sim_q)


# @app.route("/search/<query>", methods=['POST', 'GET'])
# def answer(query):
#     _ans = RawQuestion_Reply_map.get(query)
#     return render_template_string(_ans)    # 渲染字符串


@app.route("/question", methods=['POST', 'GET'])
def answers():
    query = request.args.get("query")
    _ans = RawQuestion_Reply_map.get(query)
    return _ans    



if __name__ == '__main__':
    app.run(host='172.21.35.7', port=8080, debug=True)   # debug=True开启debug模式，当代码文本有修改时，会自动加载，不用手动重启
