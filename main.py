from fastapi import FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app=FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.exception_handler(404)
def Error404(request: Request, exc: HTTPException):
    path = request.url.path
    html = f"""
    <!doctype html>
    <html lang=\"ko\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>404 Not Found</title>
      <link rel="stylesheet" href="/assets/style.css">

    </head>
    <body>
      <main class=\"card\">
        <span class=\"badge\">404</span>
        <h1>요청하신 페이지를 찾을 수 없습니다.</h1>
        <p>경로 <code>{path}</code> 에 해당하는 리소스가 존재하지 않습니다.</p>
        <div class=\"actions\">
          <a class=\"btn\" href=\"/\">홈으로 이동</a>
          <a class=\"btn\" href=\"#\" onclick=\"history.back();return false;\">이전 페이지</a>
        </div>
      </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=exc.status_code)
