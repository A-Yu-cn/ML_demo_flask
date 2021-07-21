from flask import request, make_response, jsonify
from flask_restful import Resource
import json
import logging

# 引入全局变量，防止多次重复训练模型
from app import s_model


def make_new_response(data):
    res = make_response(jsonify({'code': 0, 'data': data}))
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Method'] = '*'
    res.headers['Access-Control-Allow-Headers'] = '*'

    return res


class scoreModel(Resource):

    @staticmethod
    def post():
        try:
            data = json.loads(request.data)
            # print(data)
            data_list = list(data.values())
            d_list = []
            for i in data_list:
                d_list.append(float(i))
            print(d_list)
            logging.info("get a new post request data:" + str(data))
            return make_new_response({"result": str(s_model.predict(d_list))})
        except Exception as e:
            logging.error(str(e))

    @staticmethod
    def get():
        return "这是一个错误的访问0.0"
