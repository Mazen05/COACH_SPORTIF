from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Charger les modèles
model_perf = joblib.load("models/model_perf.pkl")
model_reco = joblib.load("models/model_reco.pkl")

@app.get("/predict_program", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "programme": None,
        "exercices": [],
        "prediction": None
    })

@app.post("/predict_program", response_class=HTMLResponse)
async def predict_program(
    request: Request,
    age: int = Form(...),
    sexe: str = Form(...),
    poids: float = Form(...),
    taille: float = Form(...),
    objectif: str = Form(...),
    historique_sportif: str = Form(...)
):
    # Encodage manuel
    sexe_encoded = 0 if sexe.upper() == "M" else 1
    objectif_map = {"Perte de poids": 0, "Prise de muscle": 1, "Endurance": 2}
    niveau_map = {"Débutant": 0, "Intermédiaire": 1, "Avancé": 2}

    input_data = {
        "age": age,
        "sexe": sexe_encoded,
        "poids": poids,
        "taille": taille,
        "objectif": objectif_map.get(objectif, 0),
        "historique_sportif": niveau_map.get(historique_sportif, 0)
    }

    df = pd.DataFrame([input_data])
    prediction = int(model_reco.predict(df)[0])  # s'assurer que c'est bien un int

    # Mapping des codes vers le type de programme
    programme_mapping = {
        0: "prise de muscle",
        1: "perte de poids",
        2: "endurance"
    }

    recommandations = {
        'prise de muscle': ["Développé couché", "Squat barre", "Soulevé de terre", "Rowing", "Tractions"],
        'perte de poids': ["HIIT", "Mountain climbers", "Burpees", "Sauts", "Gainage"],
        'endurance': ["Course", "Rameur", "Burpees", "Corde à sauter"]
    }

    programme_label = programme_mapping.get(prediction, "Programme inconnu")
    exercices = recommandations.get(programme_label, ["Programme non reconnu"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "programme": [f"[ {prediction} ]"],
        "exercices": exercices,
        "prediction": None
    })

@app.post("/predict_performance", response_class=HTMLResponse)
async def predict_performance(
    request: Request,
    nb_squats: float = Form(...),
    nb_bench_press: float = Form(...),
    heures_sommeil: float = Form(...),
    qualité_nutrition: float = Form(...)
):
    data = {
        "nb_squats": nb_squats,
        "nb_bench_press": nb_bench_press,
        "heures_sommeil": heures_sommeil,
        "qualité_nutrition": qualité_nutrition
    }
    df = pd.DataFrame([data])
    prediction = model_perf.predict(df)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": round(prediction, 2),
        "programme": None,
        "exercices": []
    })
