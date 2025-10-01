from calendar import c
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from timetable import demonstrate_solution, LLM
from starlette.concurrency import run_in_threadpool

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
        return templates.TemplateResponse("uploads.html", {"request": request,"table":demonstrate_solution()})
@app.post("/")
async def main(request: Request, question: str = Form(...), table: str = Form(...)):
    lm = LLM()
    result = await run_in_threadpool(lm.invok, question, table)  # offload sync
    print(f"결과 : {result}")
    return templates.TemplateResponse(
        "uploads.html", {"request": request, "table": table, "result": result,"question":question}
    )
@app.exception_handler(404)
def Error404(request: Request, exc: HTTPException):
    path = request.url.path
    return templates.TemplateResponse("404.html", {"request": request, "path": path}, status_code=404)