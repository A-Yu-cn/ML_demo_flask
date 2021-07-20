from flask import request
from flask_restful import Resource
import json
import logging

# 引入全局变量，防止多次重复训练模型
from app import s_model


class scoreModel(Resource):

    @staticmethod
    def post():
        try:
            data = json.loads(request.data)
            # print(data)
            data_list = list(data.values())
            # print(data_list)
            logging.info("get a new post request data:" + str(data))

            return {
                "result": s_model.predict(data_list)
            }
        except Exception as e:
            logging.error("get information error of unknown reasons:" + str(e))

    @staticmethod
    def get():
        return "这是一个错误的访问0.0"
