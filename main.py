from fastapi import FastAPI
from app import define_routes

app = FastAPI()
define_routes(app)
