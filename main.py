from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Accueil
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Schéma des données
class UserProfile(BaseModel):
    age: int
    sexe: str
    poids: int
    taille: int
    objectif: str
    historique_sportif: str

@app.post("/predict_program")
def predict_program(data: UserProfile):
    df = pd.DataFrame([data.dict()])

    encodage = {
        "sexe": {"M": 0, "F": 1},
        "objectif": {"perte de poids": 0, "prise de muscle": 1, "endurance": 2},
        "historique_sportif": {"débutant": 0, "intermédiaire": 1, "avancé": 2}
    }

    for col, mapping in encodage.items():
        df[col] = df[col].map(mapping)

    model = joblib.load("model_reco.pkl")
    pred = model.predict(df)[0]

    programme_labels = {
        0: "cardio",
        1: "HIIT",
        2: "musculation"
    }
    return {"programme_recommandé": programme_labels.get(int(pred), "Inconnu")}
