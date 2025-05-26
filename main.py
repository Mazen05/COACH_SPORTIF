from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers templates et static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Page d'accueil
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Données reçues du formulaire
class UserProfile(BaseModel):
    age: int
    sexe: str
    poids: int
    taille: int
    objectif: str
    historique_sportif: str


# Mapping des recommandations
programme_details = {
    "cardio": ["Tapis de course", "Vélo", "Burpees", "Jumping jacks", "Montées de genoux"],
    "HIIT": ["Squat jumps", "Pompes", "Gainage", "Fentes sautées", "Mountain climbers"],
    "musculation": ["Développé couché", "Soulevé de terre", "Squat barre", "Rowing haltères", "Tractions"]
}

# Endpoint de prédiction
@app.post("/predict_program")
def predict_program(data: UserProfile):
    df = pd.DataFrame([data.dict()])

    # Encodage
    encodage = {
        "sexe": {"M": 0, "F": 1},
        "objectif": {"perte de poids": 0, "prise de muscle": 1, "endurance": 2},
        "historique_sportif": {"débutant": 0, "intermédiaire": 1, "avancé": 2}
    }

    for col, mapping in encodage.items():
        df[col] = df[col].map(mapping)

    model = joblib.load("model_reco.pkl")
    pred_code = int(model.predict(df)[0])

    code_to_label = {
        0: "cardio",
        1: "HIIT",
        2: "musculation"
    }
    label = code_to_label.get(pred_code, "inconnu")
    details = programme_details.get(label, [])

    return {
        "programme": label,
        "exercices": details
    }
