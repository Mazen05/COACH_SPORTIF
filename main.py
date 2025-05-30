
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = joblib.load("model.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_program", response_class=HTMLResponse)
async def predict_program(
    request: Request,
    age: int = Form(...),
    sexe: str = Form(...),
    objectif: str = Form(...),
    historique_sportif: str = Form(...)
):
    try:
        df = pd.DataFrame([{
            "age": age,
            "sexe": sexe,
            "objectif": objectif,
            "historique_sportif": historique_sportif
        }])

        # Mise en forme pour éviter les NaN dus à des erreurs de casse
        df["objectif"] = df["objectif"].str.lower()
        df["historique_sportif"] = df["historique_sportif"].str.lower()
        df["sexe"] = df["sexe"].str.upper()

        # Encodage
        encodage = {
            "sexe": {"M": 0, "F": 1},
            "objectif": {"perte de poids": 0, "prise de muscle": 1, "endurance": 2},
            "historique_sportif": {"débutant": 0, "intermédiaire": 1, "avancé": 2}
        }
        for col, mapping in encodage.items():
            df[col] = df[col].map(mapping)

        # Vérifie l'absence de NaN après encodage
        if df.isnull().values.any():
            raise ValueError("Les données contiennent des valeurs non reconnues.")

        pred_code = int(model.predict(df)[0])

        programmes = {
            0: "Programme Perte de poids :\n- 30 min de cardio\n- Circuit training\n- 15 min de HIIT",
            1: "Programme Prise de muscle :\n- Squat, Bench press, Deadlift\n- Séries 4x12\n- Repos 1 min entre les séries",
            2: "Programme Endurance :\n- Course 45 min\n- Vélo 30 min\n- Étirements"
        }

        result = programmes.get(pred_code, "Programme inconnu")
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Erreur : {str(e)}"})
