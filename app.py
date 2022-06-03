import uvicorn
from Utils.routes import Routes
import os

routes = Routes()
app = routes.create()


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=os.environ.get("PORT", 5000))
