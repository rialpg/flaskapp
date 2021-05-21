from flask import Flask

app = Flask(__name__)
app.secret_key = "hhfsdfhs00390dsafjsdafkh30940"
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

from app import views
