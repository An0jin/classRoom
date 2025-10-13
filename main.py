from calendar import c
from fastapi import FastAPI, Form, HTTPException, Request,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from DataControll import solve_optimal, LLM,generate_html_timetable,UploadFile_to_DataFrame
from fastapi.responses import FileResponse

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
templates = Jinja2Templates(directory="templates")
app.mount("/asset", StaticFiles(directory="asset"), name="asset")
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


@app.post("/upload")
async def Upload(request: Request,semester:int=Form(...),subject:UploadFile = File(...),classroom:UploadFile = File(...),professor_room:UploadFile = File(None),professor_day:UploadFile = File(None)):
    subject_df=await UploadFile_to_DataFrame(subject)
    subject_df['그룹순번'] = subject_df.groupby(['개설학과', '교과목명', '개설학년']).cumcount()
    subject_df['반'] = subject_df['그룹순번'].apply(lambda x : chr(ord('A') + x))
    subject_df = subject_df.drop(columns=['그룹순번'])
    if semester==2:
        subject_df.loc[subject_df['개설학년']==3,'교과목학점']*=2
    classroom_df=await UploadFile_to_DataFrame(classroom)
    professor_room_df = None
    professor_day_df = None
    if professor_room is not None and professor_room.filename:
        try:
            professor_room_df=await UploadFile_to_DataFrame(professor_room)
        except ValueError:
             print("경고: 교수 선호 강의실 파일의 형식을 인식할 수 없습니다. None으로 처리합니다.")
             professor_room_df = None
    
    if professor_day is not None and professor_day.filename:
        try:
            professor_day_df=await UploadFile_to_DataFrame(professor_day)
        except ValueError:
            print("경고: 교수 선호 요일 파일의 형식을 인식할 수 없습니다. None으로 처리합니다.")
            professor_day_df = None
    result=solve_optimal(subject_df,classroom_df,professor_room_df,professor_day_df)
    if(type(result)==str):
        return templates.TemplateResponse("uploads.html", {"request": request,"table":result})
    else:
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
        path=f"asset/{filename}",
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