from fastapi import FastAPI,File,UploadFile,Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from requests import request
from Utils.validation import Switch
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse,StreamingResponse
from PIL import Image as IMG
from Utils.ocr import detect2
from Utils.validation import Message
from Utils.camera import Webcam
from Utils.camera import Stream
from Utils.validation import Switch
import cv2
import numpy as np
import main as m
import json
import base64
import os
from Utils import results
from Utils.results import Result
from Utils.database import SessionLocal,engine,sessionmaker
import Utils.database



class Routes():

    def __init__(self):
        self.app = FastAPI()
        self.app.mount("/static", StaticFiles(directory="./static"), name="static") #that is the only way in fastapi to serve static files
        self.templates = Jinja2Templates(directory="templates") # Load templates using Jinja2 for the frontend
        results.Base.metadata.create_all(bind=engine)
        self.webcam = None # Webcam for the stream using the class Webcam in the Utils folder
        self.webcam2 = None # Webcam2 for the stream using the class Webcam in the Utils folder
        self.webcam3 = None # Webcam3 for the stream using the class Webcam in the Utils folder
        self.webcam4 = None # Webcam4 for the stream using the class Webcam in the Utils folder
        self.webcam5 = None # Webcam5 for the stream using the class Webcam in the Utils folder
        self.image = None # this content the image from the picture and will be used for the prediction
        self.requests = None # Request from the frontend
        self.stream = None # Stream of the webcam using the class Stream in the Utils folder
        self.switch = True # Switch for the stream
        self.filename = None # Filename of the image that is uploaded
        self.prediction = None # Prediction of the image showing the expire date of the image
        self.time_prediction = None # Time prediction of the image showing the expire date of the image
        self.source_filename = None # filename of the image that is uploaded via picture link
    
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def create(self):
        print("Create main routes")
        self.start()
        return self.app
        
    def stop_stream(self):
        print("Stop Web Application Delhaize")
        self.stream.stop()
        

    def restart_stream(self):
        """
        This function stop the stream and then start it again.
        It is called during the reboot
        """
        print("Restart Web Application Delhaize")
        self.stop_stream()
        self.start()

    def start_webcam(self):
        """
        This function initialize all the webcams.
        (the 4 that we can see in camera link)
        """
        print("function start_webcam")
        self.webcam = Webcam()
        self.webcam2 = Webcam()
        self.webcam3 = Webcam()
        self.webcam4 = Webcam()
        self.webcam5 = Webcam()
        print("Webcam initialized")


    def get_stream(self):
        """
        Getter stream.
        """
        return self.stream

    def get_switch(self):
        """
        Getter switch.
        """
        return self.switch

    def set_switch(self,switch):
        """
        Setter switch to True or False.
        """
        self.switch = switch

    def start_stream(self):
        """
        Start the stream of the webcam. Using the class stream in the Utils folder.
        And then switch the switch to True using the set_switch function.
        """
        print("function start_stream")
        self.stream = Stream()
        self.set_switch(True)
        

    def detect_date(self):
        """
        Call the function that detect the date of the image.
        """
        self.prediction,self.time_prediction = m.main(self.image)
        print("PREDICTION:",self.prediction)
        print("TIME PREDICTION:",self.time_prediction)

    def start(self):
        """
        Start the web application. 
        Run the app
        Run the template 
        Start the stream
        """
        print("Start Web Application Delhaize")
        self.start_webcam()
        app = self.app
        templates = self.templates
        self.start_stream()


        @app.post("/reboot/")
        def reboot(switch:Switch):
            """
            This root is used to reboot the web application.
            """
            print("REBOOT WebApplication")
            #my_dict = switch.dict()
            #response = my_dict["state"]
            self.restart_stream()
            print("restart stream")
            print("start cameras")

        
        print("Web Application Delhaize started")
            

        @app.get("/",response_class=HTMLResponse)
        async def index(request:Request):
            """
            This function is the main route of this web application.
            It will lead the index.html webpage
            """
            print("loading index")
            context = {"request":request}
            return templates.TemplateResponse("index.html",context)


        @app.get("/camera/",response_class=HTMLResponse)
        async def my_camera(request:Request):
            """
            This function is the root of the camera and lead to camera.html webpage
            """
            print("loading camera web page")
            context = {"request":request}
            return templates.TemplateResponse("camera.html",context)


        @app.get("/my_api/",response_class=HTMLResponse)
        async def my_api(request:Request):
            """
            This function is the root of the api and lead to my_api.html webpage
            """
            print("loading my_api web page")
            context = {"request":request}
            return templates.TemplateResponse("my_api.html",context)


        @app.get("/picture/",response_class=HTMLResponse)
        async def picture(request:Request):
            """
            This function is the root of the picture and lead to picture.html webpage
            """
            print("loading picture")
            self.source_filename = None
            context = {"request":request}
            return templates.TemplateResponse("picture.html",context)


        @app.get("/database/",response_class=HTMLResponse)
        async def picture(request:Request):
            """
            This function is the root of the picture and lead to database.html webpage
            """
            print("loading picture")
            self.source_filename = None
            # my_worker.name = Result
            # print("my_worker.name : ",my_worker.name)
            # print("worker_request.name: ",worker_request.name)
            # db.add(my_worker)
            # db.commit()
            
            context = {"request":request}
            return templates.TemplateResponse("database.html",context)


        @app.get("/options/",response_class=HTMLResponse)
        async def options(request:Request):
            """
            This function is the root of the picture and lead to picture.html webpage
            Deactivated for now
            """
            print("loading options")
            context = {"request":request}
            return templates.TemplateResponse("options.html",context)

        
        print("Pages loaded")

        
        @app.post("/take_picture/")
        async def take_picture():  
            """
            This route is called when we send the picture that we select  
            in the picture web page. It will load the picture and detect
            the expired date 
            This function is called from brython (/static/Brython/base.py)
            """       
            self.stream.save_picture()
            picture = "./picture2.jpg"
            self.image,jpg = self.stream.load_picture(picture)
            text = detect2(self.image)
            print("Request take a picture done")
            return text

        @app.post("/take_picture_camera/")
        async def take_picture_camera():    
            """
            This route is called when we press the button take a picture 
            in the camera web page. It will load the picture and detect
            the expired date.This function is called from brython 
            (/static/Brython/base.py)
            """       
            self.stream.save_picture_camera()
            picture = "./picture1.jpg"
            self.image,jpg = self.stream.load_picture(picture)
            text = detect2(self.image)
            print("Request take a picture done")
            return text

        @app.post("/load_picture/")
        async def load_picture(message:Message):
            """
            This function is called from brython (/static/Brython/base.py)
            to load a picture with the name of the file
            """
            file_name = message
            self.image = self.stream.load_picture(file_name)
            print("Request loaded image")


        @app.get("/video_original/",response_class=HTMLResponse)
        def video_original(request:Request):
            """
            This route is the video stream of the first video. The original one on
            the camera web page
            """
            print("webcam : ",self.webcam.get_switch_webcam())
            return StreamingResponse(self.webcam.generate(self.stream,"original"),
            media_type="multipart/x-mixed-replace;boundary=frame"
            )


        # @app.get("/video_gray/",response_class=HTMLResponse)
        # def video_gray(request:Request):
        #     """
        #     This route is the video stream of the second video. The gray one on
        #     the camera web page
        #     """
        #     print("webcam2 : ",self.webcam2.get_switch_webcam())
        #     return StreamingResponse(self.webcam2.generate(self.stream,"gray"),
        #     media_type="multipart/x-mixed-replace;boundary=frame"
        #     )


        # @app.get("/video_blurr/",response_class=HTMLResponse)
        # def video_blurr(request:Request):
        #     """
        #     This route is the video stream of the third video. The blurr one on
        #     the camera web page
        #     """
        #     print("webcam3 : ",self.webcam3.get_switch_webcam())
        #     return StreamingResponse(self.webcam3.generate(self.stream,"blurr"),
        #     media_type="multipart/x-mixed-replace;boundary=frame"
        #     )


        @app.get("/video_thresold/",response_class=HTMLResponse)
        def video_thresold(request:Request):
            """
            This route is the video stream of the fourth video. The thresold one on
            the camera web page
            """
            print("webcam4 : ",self.webcam4.get_switch_webcam())
            return StreamingResponse(self.webcam4.generate(self.stream,"thresold"),
            media_type="multipart/x-mixed-replace;boundary=frame"
            )


        @app.get("/video_frame/",response_class=HTMLResponse)
        def video_frame(request:Request):
            """
            This route is the video stream of the fifth video. Not in 
            the camera web page right now
            """
            print("webcam5 : ",self.webcam5.get_switch_webcam())
            self.text = self.stream.get_text()
            return StreamingResponse(self.webcam5.generate(self.stream,"video_frame"),
            media_type="multipart/x-mixed-replace;boundary=frame"
            )

        @app.post("/switch/")
        def switch(switch:Switch):
            
            print("get switch : ",self.get_switch())
            switch = switch.dict()
            print("Changing switch state")
            self.set_switch(switch["state"])
            print("switch state changed to : ",self.get_switch())
            self.webcam.set_switch_webcam(switch["state"])
            self.webcam2.set_switch_webcam(switch["state"])
            self.webcam3.set_switch_webcam(switch["state"])
            self.webcam4.set_switch_webcam(switch["state"])
            self.webcam5.set_switch_webcam(switch["state"])
            print("self.webcam5 : ",self.webcam.get_switch_webcam())
            print("self.webcam5 : ",self.webcam2.get_switch_webcam())
            print("self.webcam5 : ",self.webcam3.get_switch_webcam())
            print("self.webcam5 : ",self.webcam4.get_switch_webcam())
            print("self.webcam5 : ",self.webcam5.get_switch_webcam())
            if switch["state"] == True:
                print("switch state is True")
            else:
                print("switch state is False")

            #return {"Switch State":switch["state"]}

        print("Stream loaded")


        @app.post("/file_image/")
        def file(filename):
            cv2.imread(filename)
            
            return {"file_name":filename}

        @app.post("/submitform",response_class=HTMLResponse)
        async def handle_form(request:Request, my_picture_file:UploadFile = File(...)):
            print("type of file : ",my_picture_file.content_type)
            print("name of the file : ",my_picture_file.filename)

            file = await my_picture_file.read()
            print("type of file : ",type(file))
            print("my_picture_file: ",type(my_picture_file))
            self.source_filename = "picture2.jpg"
            image = cv2.imdecode(np.frombuffer(file, np.uint8),cv2.IMREAD_COLOR)
            cv2.imwrite("./static/Images/" + self.source_filename,image)
            print("Picture saved in 'picture2.jpg'")
            print("TYPE OF IMAGE : ",type(image))
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
            #self.stream.save_picture_camera()
            print("ENTER IN THE SHOW PICTURE")
            print("self.stream : ",type(self.stream))
            print("stream.get_frame : ",type(self.stream.get_frame()))
            ret,image2 = self.stream.read_stream()
            print("image2 : ",type(image2))
            ret,jpeg = cv2.imencode(".jpg",image2)
            print("jpeg : ",type(jpeg))
            tobytes = jpeg.tobytes()
            print("tobytes : ",type(tobytes))
            filename = "picture1.jpg"
            print("Image size : ",image2.size)
            self.source_filename = filename
            self.image = image2 #detect_date() need self.image
            root = "/static/Images/"
            path = root + filename
            print("path : ",path)
            image2 = self.stream.resize_image(image2)
            cv2.imwrite("./static/Images/" + self.source_filename,image2)
            print("Picture saved in 'picture1.jpg'")
            self.detect_date()
            context = {"request":request}
            context["filename"] = filename
            context["prediction"] = self.prediction
            context["time_prediction"] = round(float(self.time_prediction),2)
            context["image"] = image2
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
      
