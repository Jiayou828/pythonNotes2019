'''03_豆瓣电影抓取-Ajax.py'''
import requests
import json
import csv

# url要写抓到的GET ：URL
url = "https://movie.douban.com/j/chart/top_list?"
headers = {"User-Agent":"Mozilla/5.0"}
L = ["剧情","喜剧","动作"]
tp_list = [{"剧情":"11"},{"喜剧":"24"},{"动作":"5"}]
tp = input("请输入电影类型:")

if tp in L:
    num = input("请输入要爬取的数量：")
    for film_dict in tp_list:
        for key,value in film_dict.items():
            if tp == key:
                params = {
                        "type":value,
                        "interval_id":"100:90",
                        "action":"",	
                        "start":"0",
                        "limit":num
                    }
                
                res = requests.get(url,params=params,headers=headers)
                res.encoding = "utf-8"
                # html为json格式的数组[{电影1信息},{},{}]
                html = res.text
                # 数组 -> 列表
                html = json.loads(html)
                # 用for循环遍历每一个电影信息{}
                for film in html:
                    name = film['title']
                    score = film["rating"][0]
                    #{"rating":["9.6","50"],...}
                    with open("豆瓣电影.csv","a",newline="") as f:
                        writer = csv.writer(f)
                        L = [name,score]
                        writer.writerow(L)
else:
    print("您输入的类型不存在!")


















