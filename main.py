from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Modèle pour la reco programme
class UserProfile(BaseModel):
    age: int
    sexe: str
    poids: int
    taille: int
    objectif: str
    historique_sportif: str

# Modèle pour la performance
class PerfProfile(BaseModel):
    nb_squats: float
    nb_bench_press: float
    heures_sommeil: float
    qualité_nutrition: float

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
    pred_code = int(model.predict(df)[0])

    programme_labels = {
        0: ("cardio", ["Tapis de course", "Vélo", "Burpees", "Jumping jacks", "Montées de genoux"]),
        1: ("HIIT", ["Sprints", "Squats sautés", "Pompes explosives", "Fentes sautées"]),
        2: ("musculation", ["Squats", "Développé couché", "Soulevé de terre", "Tractions"])
    }

    label, details = programme_labels.get(pred_code, ("Inconnu", []))
    return {"programme_recommandé": label, "exercices": details}

@app.post("/predict_performance")
def predict_performance(data: PerfProfile):
    df = pd.DataFrame([data.dict()])
    model_perf = joblib.load("model_perf.pkl")
    prediction = model_perf.predict(df)[0]
    return {"progression_future": round(prediction, 2)}
