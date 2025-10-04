**Current Format of demo app**
1. Run the xgboost model in **fdmv2_feedback.ipynb** -> Model is here(Path):  Model Trainings and Accuracy Testing -> Model Selction -> **Selecting model to dumbs**
2. Run app.py ->
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
4. Run frontend.py streamlit ->
   streamlit run frontend.py
