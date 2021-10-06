import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return {'Welcome to': 'Ankan'}

@app.get('/yolo')
def get_yolo():
    return {'Welcome to YOLO version': '3'}



if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=60000)
