from calendar import c
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from timetable import solve_optimal,generate_html_timetable
from io import BytesIO
import pandas as pd

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
templates = Jinja2Templates(directory="templates")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploads")
async def uploads(request: Request,courses: UploadFile = File(...),classRoom: UploadFile = File(...)):
    courses_byte = await courses.read()
    classRoom_byte = await classRoom.read()
    try:
        courses_df = pd.read_csv(BytesIO(courses_byte))
    except Exception as e:
        courses_df = pd.read_excel(BytesIO(courses_byte))
    try:
        classRoom_df = pd.read_csv(BytesIO(classRoom_byte))
    except Exception as e:
        classRoom_df = pd.read_excel(BytesIO(classRoom_byte))
    result=solve_optimal(courses_df,classRoom_df)
    return templates.TemplateResponse("uploads.html", {"request": request,"table":generate_html_timetable(result)})

@app.exception_handler(404)
def Error404(request: Request, exc: HTTPException):
    path = request.url.path
    return templates.TemplateResponse("404.html", {"request": request, "path": path}, status_code=404)