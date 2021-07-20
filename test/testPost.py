import requests
"""
    使用requests库来模拟发送各种HTTP请求，查看服务器运行正常与否
"""
post_data = {
    "90D": 0,
    "RevolvingRatio": 0.766127,
    "30-59D": 2,
    "60-89D": 0,
    "Age": 45
}
if __name__ == "__main__":
    try:
        url = "http://localhost:5000/api/score"
        res = requests.post(url=url, json=post_data)
        print(res.json())
    except Exception as e:
        print(e)
