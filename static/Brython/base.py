
from browser import document,html,ajax,bind
import json
from typing import Dict

print("Brython activated!!!!!")
print("Javascript = NO THANKS")


class Picture():
    def __init__(self):
        print("Initializing Picture")
        self.request1 = ajax.ajax()


    def load(self):
        print("function load -- request Picture")
        request1 = self.request1
        request1.bind('complete',self.on_complete)
        request1.open('POST',"/load_picture/",True)
        text = document["formFileLg"].value
        print("TEXT VALUE = ",text)
        filename = json.dumps({"text":text})
        request1.send(filename)
        print("Picture loaded")
    
    def get(self):
        print("function get -- request Picture")
        request1 = self.request1
        request1.bind('complete',self.on_complete)
        request1.open('POST',"/file_image/",True)

        request1.send()
        print("Picture received")
    
    def on_complete(request1):
        print("On complete loading picture")
        if request1.status == 200 or request1.status == 0:
            print("LOADING IS COMPLETE")
        elif request1.status == 422:
            pass
        elif request1.status == 405:
            print("METHOD NOT ALLOWED - code 405")

class Switch_boutton():
    def __init__(self):
        print("Initializing switch_boutton") 
        print("start ajax switch")
        self.request3 = ajax.Ajax()
        self.state = False


    def change(self,state:bool):
        self.state = state
        if self.state == True:
            print("SWITCH IS ON")
        else:
            print("SWITCH IS OFF")

        url3 = "/switch/"
        self.request3.bind("complete",self.on_complete)
        self.request3.open("POST",url3,True)
        self.request3.set_header("content-type","application/json")
        self.request3.send(json.dumps({"state":state}))
        print("End ajax switch")
        print("!!SEND!!")


    def on_complete(request3):
        print("On complete switch button")
        if request3.status == 200 or request3.status == 0:
            print("SWITCH IS COMPLETE")         
        elif request3.status == 422:
            pass



class Camera():
    def __init__(self):
        print("Camera initialized")
        print("start ajax camera")
        self.request = ajax.Ajax()

    def shoot(self):
        print("start function shoot")
        url2 = "/take_picture/"
        self.request.open('POST', url2, True)
        self.request.set_header('content-type', 'application/json')
        self.request.send()
        print("!!!!!!!SEND!!!!!!!")

    def shoot_camera(self):
        print("start function shoot")
        url6 = "/take_picture_camera/"
        self.request.open('POST', url6, True)

        self.request.set_header('content-type', 'application/json')
        self.request.send()
        print("!!!!!!!SEND!!!!!!!")

    def show(self):
        print("start function show")
        url = "/show_picture"
        self.request.open('POST', url, True)
        self.request.send()
        print("!!!!!!!SEND!!!!!!!")
            


def change_button():
    print("Change the button")
    document["button-load-image"].text = "processing..."

   


if "my_button_picture" in document:
    print("my_button_picture found")
    @bind('#my_button_picture',"click")             
    def clicked(ev):
        print("clicked on #my_button_picture !")
        print(ev.currentTarget.text)
        if ev.currentTarget.text == "Take a picture":
            cam = Camera()
            cam.shoot_camera()
            cam.show()
            document["my_button_picture"].text = "Loading..."
            print("Change the text 'Take a picture' to 'Loading...'")
            


if "my-loaded-image" in document:
    print("my_loaded_image found")
    @bind('#button-load-image',"click")             
    def image_loaded(ev):
        print("loaded on #my-loaded-image !")
        picture = Picture()
        picture.get()

else:
    print("!!!!!!!!!!my_loaded_image not found")
            

if "button-load-image" in document:
    print("button-load-image found")
    @bind('#button-load-image',"click")             
    def image_load(ev):
        print("load on #button-load-image !")
        change_button()

else:
    print("!!!!!!!!!!button-load-image not found")
            

