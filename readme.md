## 최적화 문제 vs. 실행 가능성 문제 (Near VIBE Coding)
- 최적화 문제(Optimization Problem): 목적식을 최소화하거나 최대화하는 해를 Find
- 실행 가능성 문제(Feasibility Problem): 목적식은 중요하지 않고, 모든 제약 조건을 만족하는 해가 존재하는지 여부만 확인. 즉, "조건에 맞는 해가 하나라도 있는가?"가 핵심 : 즉 Z = 0
- 목적식 수정 가능
> - 강의실 활용률 최대화: 모든 시간 슬롯과 강의실의 조합 중 사용된 수를 최대화합니다.
> - 교수 선호 시간대 배정: 교수들이 선호하는 시간대에 강의를 배정했을 때 얻는 점수를 최대화합니다.
> - 빈 강의실 최소화: 사용되지 않는 강의실 시간대를 최소화합니다.
>
## 기능 요구사항
- FastAPI, Database(postrgres or sqlites), HTML or TS : Input data(교과목 시수)와 조건식을 받아 Output data(요일별 교과목 강의실 배정)
- Gemini LLM (Python SDK) : 배정되 ㄴ결과에 대한 조건식 수정 및 조회, Dashboard 등 prompt

## 사용기술
- FastAPI, Ubuntu, Postgres, Docker, Gemini LLM, HTML, CSS