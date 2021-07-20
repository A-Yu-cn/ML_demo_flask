from flask import Flask
from flask_restful import Api

from src.Model import Model

app = Flask(__name__)
api = Api(app)

s_model = Model()


@app.route('/')
def hello_world():
    return "Hello World!"


# 这个如果放到最前边好像会递归包括就报错了
from src.scoreModel import scoreModel

# 注册路径，预测模型
api.add_resource(scoreModel, '/api/score')

if __name__ == '__main__':
    app.run(port=2333, host='0.0.0.0')
