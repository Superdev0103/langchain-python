from flask import request
from flasgger import LazyString

swagger_template = dict(
info = {
    'title': LazyString(lambda: 'BookBlaster ChatBot API document'),
    'version': LazyString(lambda: '1.0'),
    'description': LazyString(lambda: 'Swagger API Documentation. API description the upload file and summarize text so that question and answer '),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'api_docs',
            "route": '/apidocs.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}