from fastapi import FastAPI,File,UploadFile,Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from requests import request
from Utils.validation import Switch
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse,StreamingResponse
from PIL import Image as IMG
from Utils.validation import Message
from Utils.validation import Switch
import cv2
import numpy as np
import main as m
import json
import base64
from io import BytesIO


class Routes():

    def __init__(self):
        self.app = FastAPI()
        self.app.mount("/static", StaticFiles(directory="./static"), name="static")
        self.templates = Jinja2Templates(directory="templates")
        self.webcam = None
        self.webcam2 = None
        self.webcam3 = None
        self.webcam4 = None
        self.webcam5 = None
        self.image = None
        self.requests = None
        self.stream = None
        self.pages = None
        self.switch = True
        self.filename = None
        self.prediction = None
        self.time_prediction = None
        self.source_filename = None
        self.text = ""

    def create(self):
        print("Create main routes")
        self.start()
        return self.app

    def restart_stream(self):
        print("Restart Web Application Delhaize")
        self.stop_stream()
        self.start()

    def get_switch(self):
        return self.switch

    def set_switch(self,switch):
        self.switch = switch

    def detect_date(self):
        #print("PRED:",m.main(self.image))
        self.prediction,self.time_prediction = m.main(self.image)
        print("PREDICTION:",self.prediction)
        print("TIME PREDICTION:",self.time_prediction)

    def start(self):
        print("Start Web Application Delhaize")
        app = self.app
        templates = self.templates

        @app.post("/reboot/")
        def reboot(switch:Switch):
            print("REBOOT WebApplication")
            #my_dict = switch.dict()
            #response = my_dict["state"]
            print("restart stream")
            print("start cameras")


        print("Web Application Delhaize started")


        @app.get("/",response_class=HTMLResponse)
        async def index(request:Request):
            print("loading index")
            context = {"request":request}
            return templates.TemplateResponse("index.html",context)


        @app.get("/my_api/",response_class=HTMLResponse)
        async def my_api(request:Request):
            print("loading my_api web page")
            context = {"request":request}
            return templates.TemplateResponse("my_api.html",context)


        @app.get("/picture/",response_class=HTMLResponse)
        async def picture(request:Request):
            print("loading picture")
            self.source_filename = None
            context = {"request":request}
            return templates.TemplateResponse("picture.html",context)


        @app.get("/options/",response_class=HTMLResponse)
        async def options(request:Request):
            print("loading options")
            context = {"request":request}
            return templates.TemplateResponse("options.html",context)


        print("Pages loaded")


        @app.post("/file_image/")
        def file(filename):
            cv2.imread(filename)

            return {"file_name":filename}

        @app.post("/submitform",response_class=HTMLResponse)
        async def handle_form(request:Request, my_picture_file:UploadFile = File(...)):
            print("type of file : ",my_picture_file.content_type)
            print("name of the file : ",my_picture_file.filename)
            #filename = my_picture_file.filename

            file = await my_picture_file.read()
            print("type of file : ",type(file))
            print("my_picture_file: ",type(my_picture_file))
            #print("my_picture_file.file: ",type(my_picture_file.file))
            self.source_filename = "picture2.jpg"
            image = cv2.imdecode(np.frombuffer(file, np.uint8),cv2.IMREAD_COLOR)
            cv2.imwrite("./static/Images/" + self.source_filename,image)
            print("Picture saved in 'picture2.jpg'")
            self.image = image
            self.detect_date()
            context = {"request":request}
            context["filename"] = my_picture_file.filename
            context["prediction"] = self.prediction
            context["time_prediction"] = round(float(self.time_prediction),2)
            context["source_filename"] = self.source_filename
            return templates.TemplateResponse("detected.html",context)

        @app.post("/show_picture",response_class=HTMLResponse)
        async def show_picture(request:Request):
            filename = "picture1.jpg"
            path = "../static/Images/"+filename
            image = cv2.imread(path)
            print("Picture 'picture1.jpg' read")
            self.image = image #detect_date() need self.image
            cv2.imwrite("./static/Images/" + filename,image)
            self.detect_date()
            context = {"request":request}
            context["filename"] = filename
            context["prediction"] = self.prediction
            context["time_prediction"] = round(float(self.time_prediction),2)
            context["source_filename"] = self.source_filename
            return templates.TemplateResponse("detected.html",context)



        @app.post("/API")
        async def api(file:bytes=File()):
            print("type of file : ",type(file))
            image = cv2.imdecode(np.frombuffer(file, np.uint8),cv2.IMREAD_COLOR)
            self.image = image
            self.detect_date()
            context = {}
            context["prediction"] = self.prediction
            context["time_prediction"] = round(float(self.time_prediction),2)
            print(context)
            return context
