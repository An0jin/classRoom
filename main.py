from calendar import c
from fastapi import FastAPI, Form, HTTPException, Request,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from timetable import solve_optimal, LLM,generate_html_timetable
from fastapi.responses import FileResponse,StreamingResponse
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup
import urllib.parse

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
templates = Jinja2Templates(directory="templates")
app.mount("/Assets", StaticFiles(directory="Assets"), name="Assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get("/")
def Main(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
async def UploadFile_to_DataFrame(file:UploadFile):
    file_byte=await file.read()
    return pd.read_excel(file_byte)

@app.post("/upload")
async def Upload(request: Request,semester:int=Form(...),subject:UploadFile = File(...),classroom:UploadFile = File(...),professor_room:UploadFile = File(None),professor_day:UploadFile = File(None)):
    

    subject_df=await UploadFile_to_DataFrame(subject)
    if semester==2:
        subject_df.loc[subject_df['개설학년']==3,'교과목학점']*=2
    classroom_df=await UploadFile_to_DataFrame(classroom)
    if professor_room is not None:
        professor_room_df=await UploadFile_to_DataFrame(professor_room)
    if professor_day is not None:
        professor_day_df=await UploadFile_to_DataFrame(professor_day)
    result=solve_optimal(subject_df,classroom_df,professor_room_df,professor_day_df)
    html=generate_html_timetable(result)
    return templates.TemplateResponse("uploads.html", {"request": request,"table":html})

@app.post("/llm")
async def Llm(request: Request,question:str=Form(...),table:str=Form(...)):
    llm=LLM()
    result=llm.invok(question,table)
    return result

@app.exception_handler(404)
def Error404(request: Request, exc: HTTPException):
    path = request.url.path
    return templates.TemplateResponse("404.html", {"request": request, "path": path}, status_code=404)

@app.get("/download/{file}")
def Download(request: Request,file: str):
    filename=f"{file}.xlsx"
    return FileResponse(
        path=f"assets/{filename}",
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        filename=filename
    )
# @app.post("/result")
# def Result(request: Request,html:str=Form(...)):
#     soup = BeautifulSoup(html, 'html.parser')
#     titles = soup.select('h2')
#     df=pd.read_html(html)
#     size=len(titles)
#     output_stream=BytesIO()
#     with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
#         for i in range(size):
#             print(titles[i].getText().split(": ")[1])
#             print(df[i])
#             df[i].set_index("교시",inplace=True)
#             df[i].to_excel(writer, sheet_name=titles[i].getText().split(": ")[1])
#     output_stream.seek(0)
#     filename = "최종시간표.xlsx"
#     encoded_filename = urllib.parse.quote(filename)
#     headers = {
#             'Content-Disposition': f'attachment; filename*=UTF-8\'\'{encoded_filename}'
#     }
#     return StreamingResponse(
#         output_stream,
#         media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
#         headers=headers
#     )