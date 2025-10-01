import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import time # 지연 시간 테스트용으로 사용될 수 있으으나 여기서는 생략
import psycopg2
from psycopg2 import connect
from google import genai
import markdown
# PuLP Solver는 실행 환경에 따라 설치 필요: pip install pulp
# from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value

# 1. 환경 변수 로드
load_dotenv()

# 2. 환경 변수에서 데이터베이스 정보 추출
host = os.getenv('host')
port = os.getenv('port')
user = os.getenv('user')
password = os.getenv('password')
dbname = "postgres"  # 또는 실제 사용하는 DB 이름

# 3. PostgreSQL 연결 문자열 생성 (SQLAlchemy Engine 생성용)
# 포트가 int 타입이어야 하므로 os.getenv('port')를 int로 변환
try:
    port_int = int(port)
except (TypeError, ValueError):
    print("Error: Port number is not defined or is not an integer.")
    exit()

# PostgreSQL용 SQLAlchemy Engine 생성
# 형식: 'postgresql+psycopg2://<user>:<password>@<host>:<port>/<dbname>'
DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port_int}/{dbname}"
engine = create_engine(DATABASE_URL)

# 4. 데이터 로드 및 SQL 삽입 (기존 DB 로더 코드를 통합)
def load_data_to_db():
    """CSV 파일을 읽어 데이터베이스에 삽입합니다."""
    # 이전에 사용된 DB 연결 로직(psycopg2) 대신, SQLAlchemy engine을 사용합니다.
    try:
        print("Loading data from courses_data_feature.csv...")
        # 'courses_data_feature.csv' 파일이 현재 경로에 있다고 가정합니다.
        courses = pd.read_csv('courses_data_feature.csv')
        
        # [수정] 데이터베이스 삽입 전에 컬럼명 오타 수정 (peofessor -> professor)
        if 'peofessor' in courses.columns:
            courses = courses.rename(columns={'peofessor': 'professor'})
        
        # [가정] 수강 인원 컬럼 추가 (데이터에 없으므로 임의값 30으로 설정)
        if 'enrollment' not in courses.columns:
             courses['enrollment'] = 30 # 임의값 설정
        
        table_name = 'courses'
        
        print(f"Attempting to insert data into PostgreSQL table: '{table_name}'")
        
        # if_exists='replace': 이미 테이블이 존재하면 삭제 후 재생성 (개발 단계에서 편리함)
        courses.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"Data successfully loaded into PostgreSQL table '{table_name}'.")

    except Exception as e:
        print(f"An unexpected error occurred during database operation: {e}")
        # PostgreSQL 연결 오류, 인증 오류 등은 여기서 잡힘
    finally:
        # SQLAlchemy Engine은 사용 후 반드시 Dispose
        if 'engine' in globals():
            engine.dispose()
            print("SQLAlchemy Engine disposed.")


# --- 기존 코드에 추가된 DB 인터페이스 클래스 ---
class DB:
    """PostgreSQL 데이터베이스 연결 및 데이터 접근을 담당하는 클래스"""
    def __init__(self):
        self._host=os.getenv('host')
        self._port=int(os.getenv('port'))
        self._user=os.getenv('user')
        self._password=os.getenv('password')
        # psycopg2를 사용하여 직접 연결 (pandas.read_sql을 위해)
        self._conn = connect(f"host={self._host} port={self._port} user={self._user} password={self._password} dbname=postgres")
    
    @property
    def conn(self):
        return self._conn
    
    @property
    def courses(self):
        # 'courses' 테이블에서 데이터 로드
        courses_df = pd.read_sql("SELECT * FROM courses", self.conn)
        
        # [DB 로드 후 수정] 로드 시 오타가 수정되지 않았을 경우를 대비 (DB에 peofessor로 저장되었을 가능성)
        if 'peofessor' in courses_df.columns:
            courses_df = courses_df.rename(columns={'peofessor': 'professor'})
        
        return courses_df
    
    @property
    def room(self):
        # 'room' 테이블에서 데이터 로드 (room.csv로 가정)
        return pd.read_sql("SELECT * FROM room", self.conn)

# [새로운 제약 조건 상수] 교수 최대 연속 강의 허용 시간 (예: 3교시)
# 이 상수는 이제 사용되지 않지만, HTML 보고서 출력을 위해 그대로 유지합니다.
MAX_CONTINUOUS_LECTURE_HOURS = 3 

def solve_optimal(courses_df: pd.DataFrame, classRoom_df: pd.DataFrame) -> pd.DataFrame:
    """
    정수 선형 계획법(ILP) 모델을 사용하여 과목-강의실 최적 배정 해를 도출하는 함수.
    (월-금, 1-9교시 주간 시간표 기반)
    
    Args:
        courses_df (pd.DataFrame): 과목 정보 (Course_ID, 교과목 시간, 수강인원, 강좌대표교수 포함).
        classRoom_df (pd.DataFrame): 강의실 정보 (호실번호 포함, Capacity는 함수 내에서 가정).
    
    Returns:
        pd.DataFrame: 배정 결과 (요일, 교과목명, 강의실 ID, 시작 시간, 종료 시간)를 담은 DataFrame.
    """
    
    # PuLP 객체는 실행 환경에 따라 직접 로드해야 함
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value, LpBinary
    except ImportError:
        print("오류: PuLP 라이브러리가 설치되어 있지 않아 실행할 수 없습니다.")
        return pd.DataFrame()

    # --- 0. 데이터 준비 및 주간 시간표 설정 ---
    
    DAYS = ['월', '화', '수', '목', '금']
    T_MAX = 9  # 가정: 총 시간 슬롯 수 (1교시~9교시)
    TIME_SLOTS = range(1, T_MAX + 1)
    
    
    # 강의실 수용 인원 가정 (입력 데이터 불완전성 보완)
    classRoom_df = classRoom_df.copy()
    if 'Capacity' not in classRoom_df.columns:
        # 6개 강의실에 임의의 수용 인원 부여
        capacities = [45, 45, 45, 45, 60, 60]
        if len(classRoom_df) == len(capacities):
            classRoom_df['Capacity'] = capacities
        else:
            classRoom_df['Capacity'] = 50 

    # [수정] Course_ID가 이미 외부 (demonstrate_solution)에서 생성됨
    courses_df = courses_df.copy()

    classRoom_df['Room_ID'] = classRoom_df['number'].astype(str)
    
    COURSES = list(courses_df['Course_ID'])
    CLASSROOMS = list(classRoom_df['Room_ID'])
    PROFESSORS = courses_df['professor'].unique().tolist()

    # Parameters (Dictionaries)
    course_duration = courses_df.set_index('Course_ID')['time'].to_dict()
    course_professor = courses_df.set_index('Course_ID')['professor'].to_dict()
    room_capacity = classRoom_df.set_index('Room_ID')['Capacity'].to_dict()
    
    # [핵심 수정] 누락된 'course_enrollment' 딕셔너리 정의 (NameError 해결)
    if 'enrollment' in courses_df.columns:
        course_enrollment = courses_df.set_index('Course_ID')['enrollment'].to_dict()
    else:
        # 'enrollment' 컬럼이 없으면, 임의의 기본값 30으로 설정하여 시뮬레이션
        print("경고: courses_df에 'enrollment' 컬럼이 없습니다. 모든 과목의 수강 인원을 30명으로 가정합니다.")
        course_enrollment = {c: 30 for c in COURSES}


    # --- 1. ILP 모델 및 결정 변수 정의 ---

    prob = LpProblem("Weekly_Course_Classroom_Assignment_Optimal", LpMaximize)

    # 결정 변수: X_crdt (1이면 과목 c가 강의실 r, 요일 d, 시간 t에 시작하여 배정)
    X = LpVariable.dicts("Assignment",
                          ((c, r, d, t)
                           for c in COURSES
                           for r in CLASSROOMS
                           for d in DAYS
                           for t in TIME_SLOTS
                           # 수업 종료 시간이 T_MAX를 넘지 않도록 처리
                           if t + course_duration.get(c, 0) - 1 <= T_MAX),
                          0, 1, LpBinary)

    # --- 2. 목적 함수 (Objective Function) ---
    # 배정된 과목의 총 개수 최대화 (모든 과목을 배정할 필요는 없음)
    prob += lpSum(X[c, r, d, t] for c, r, d, t in X if (c, r, d, t) in X), "Total_Courses_Assigned"

    # --- 3. 제약 조건 (Constraints) ---

    # 3.1. 과목당 최대 1회 배정 제약 (주간 전체에서 1회)
    for c in COURSES:
        prob += lpSum(X[c, r, d, t] 
                      for r in CLASSROOMS 
                      for d in DAYS 
                      for t in TIME_SLOTS 
                      if (c, r, d, t) in X) <= 1, f"Max_One_Assignment_for_Course_{c}"

    # 3.2. 수용 인원 제약: 수강 인원 > 강의실 Capacity이면 배정 불가 (X=0 강제)
    for c in COURSES:
        for r in CLASSROOMS:
            # [핵심 수정] 정의된 course_enrollment 사용
            if course_enrollment.get(c, 0) > room_capacity.get(r, float('inf')):
                for d in DAYS:
                    for t in TIME_SLOTS:
                        if (c, r, d, t) in X:
                            prob += X[c, r, d, t] == 0, f"Capacity_Check_{c}_{r}_{d}_{t}"

    # 3.3. 강의실 시간 충돌 제약: 한 강의실은 특정 요일 d, 시간 슬롯 k에 하나의 과목만 배정 가능 (연속 배정 포함)
    for r in CLASSROOMS:
        for d in DAYS:
            for k in TIME_SLOTS:
                prob += lpSum(X[c, r_check, d_check, t]
                              for c, r_check, d_check, t in X
                              if r_check == r and d_check == d and t <= k <= t + course_duration.get(c, 0) - 1) <= 1, f"Room_{r}_Day_{d}_Time_{k}_Conflict_Check"

    # 3.4. 교수 시간표 충돌 제약: 한 교수는 특정 요일 d, 시간 슬롯 k에 하나의 과목만 배정 가능
    # (동일 시간에 겹치는 수업만 방지. 연속 강의 허용)
    for p in PROFESSORS:
        for d in DAYS:
            for k in TIME_SLOTS:
                prob += lpSum(X[c, r, d_check, t]
                              for c, r, d_check, t in X
                              if course_professor.get(c) == p and d_check == d and t <= k <= t + course_duration.get(c, 0) - 1
                             ) <= 1, f"Professor_{p}_Day_{d}_Time_{k}_Conflict_Check"

    # 3.5. [제거됨] 교수 최대 연속 강의 시간 제한

    # --- 4. 문제 해결 및 결과 추출 ---
    
    # CBC Solver로 문제 해결 (msg=0은 로그 출력 방지)
    prob.solve(PULP_CBC_CMD(msg=0))
    
    if LpStatus[prob.status] == "Optimal" or LpStatus[prob.status] == "Feasible":
        assigned_courses = []
        for c, r, d, t in X:
            # 결정 변수 값이 1인 경우 (배정 성공)
            if X[c, r, d, t].varValue == 1.0:
                # [수정] original_row를 가져올 때 Course_ID를 사용합니다.
                original_row = courses_df[courses_df['Course_ID'] == c].iloc[0]
                assigned_courses.append({
                    'Day': d,
                    '교과목명': original_row['subject'],
                    '수강분반': original_row['class'],
                    '교과목 시간': original_row['time'],
                    '강좌대표교수': original_row['professor'],
                    'Classroom_ID': r,
                    'Start_Time': t,
                    'End_Time': t + original_row['time'] - 1,
                    # ILP 결과를 merge에 사용하기 위해 Course_ID를 포함
                    'Course_ID': c 
                })

        assignment_df = pd.DataFrame(assigned_courses)
        # 학과/학년 기준으로 그룹화하기 위해 Course_ID를 유지합니다.
        return assignment_df.sort_values(by=['Classroom_ID', 'Day', 'Start_Time']).reset_index(drop=True)
    else:
        print(f"경고: 최적화에 실패했거나 실행 불가능한 상태입니다. Status: {LpStatus[prob.status]}")
        # 실패 시 빈 스케줄 DataFrame 반환
        return pd.DataFrame(columns=['Day', '교과목명', '수강분반', '교과목 시간', '강좌대표교수', 'Classroom_ID', 'Start_Time', 'End_Time', 'Course_ID'])

def generate_html_timetable(schedule_df: pd.DataFrame, courses_df: pd.DataFrame, classRoom_df: pd.DataFrame, unassigned_courses_df: pd.DataFrame) -> str:
    """
    최적화된 배정 결과를 학과, 학년, 분반 기준으로 주간 시간표 그리드 형태로 HTML을 생성하고,
    미배정 과목 보고서를 포함합니다.
    """
    
    DAYS = ['월', '화', '수', '목', '금']
    TIME_SLOTS = range(1, 10) # 1교시부터 9교시
    
    # 1. 스케줄 DataFrame에 학과/학년 정보 통합
    # courses_df['Course_ID']는 이미 외부에서 생성되었음을 가정합니다.
    
    courses_info = courses_df[['Course_ID', 'dept', 'grade']].drop_duplicates(subset=['Course_ID'])
    schedule_df = pd.merge(schedule_df, courses_info, on='Course_ID', how='left')
    
    # 컬럼 이름이 '개설학과', '개설학년'이라고 가정하고 사용합니다.
    schedule_df = schedule_df.rename(columns={'dept': '개설학과', 'grade': '개설학년'})

    # 2. 그룹 ID 정의: 학과, 학년, 분반
    schedule_df['Group_ID'] = (schedule_df['개설학과'] + ' (' + 
                                schedule_df['개설학년'].astype(str) + '학년) ' + 
                                schedule_df['수강분반'].astype(str) + '반')

    grouped_schedule = schedule_df.groupby('Group_ID')
    
    timetable_data = {group: {day: {} for day in DAYS} for group in grouped_schedule.groups.keys()}
    
    for group_id, group_df in grouped_schedule:
        for _, row in group_df.iterrows():
            day = row['Day']
            start = row['Start_Time']
            duration = row['교과목 시간']
            
            cell_division = row.get('수강분반', '분반 정보 없음') 
            cell_professor = row.get('강좌대표교수', '교수 정보 없음')
            
            timetable_data[group_id][day][start] = {
                'name': row['교과목명'],
                'division': cell_division,
                'professor': cell_professor,
                'room_id': row['Classroom_ID'], 
                'duration': duration,
                'end': row['End_Time']
            }
            
            for t in range(start + 1, start + duration):
                timetable_data[group_id][day][t] = 'occupied'

    html = [
        '<div class="container mx-auto p-4 bg-gray-100 rounded-xl shadow-2xl">',
        '    <h1 class="text-3xl font-extrabold text-indigo-800 text-center mb-6">최적 학과/학년별 주간 배정표 (월-금, 1-9교시)</h1>',
        f'    <p class="text-center text-gray-600 mb-6">배정 성공 과목 수: {len(schedule_df)} / 총 과목 수: {len(courses_df)} (미배정: {len(unassigned_courses_df)}개)</p>'
    ]

    # --- 시간표 테이블 생성 ---
    if schedule_df.empty and unassigned_courses_df.empty:
         html.append('<div class="p-6 text-center text-gray-500">배정 가능한 과목이 없습니다.</div>')
    elif schedule_df.empty and not unassigned_courses_df.empty:
         # 모든 과목이 미배정되었을 경우
         html.append('<div class="p-6 text-center text-red-600">제약 조건이 너무 타이트하여 배정된 과목이 없습니다.</div>')
    else:
        for group_id, group_timetable in timetable_data.items():
            html.append(f'<div class="mb-10 p-5 bg-white rounded-xl shadow-lg">')
            html.append(f'    <h3 class="text-xl font-bold text-indigo-700 mb-4 border-b pb-2">배정 그룹: {group_id}</h3>')
            
            html.append('    <div class="overflow-x-auto">')
            html.append('        <table class="min-w-full divide-y divide-gray-200 border border-gray-300">')
            html.append('            <thead class="bg-indigo-500 text-white">')
            html.append('                <tr><th class="px-4 py-2 border-r border-indigo-400 w-1/12">교시</th>')
            for day in DAYS:
                html.append(f'<th class="px-4 py-2 border-r border-indigo-400">{day}</th>')
            html.append('                </tr>')
            html.append('            </thead>')
            html.append('            <tbody class="divide-y divide-gray-200">')

            for t in TIME_SLOTS:
                html.append('<tr>')
                html.append(f'    <td class="px-3 py-2 text-center text-xs font-semibold bg-gray-100 border-r border-gray-300">{t}교시</td>')
                
                for day in DAYS:
                    slot_content = group_timetable[day].get(t)
                    
                    if slot_content == 'occupied':
                        continue
                    
                    elif isinstance(slot_content, dict):
                        duration = slot_content['duration']
                        
                        cell_content = f"""
                        <div class="font-bold text-sm text-indigo-800">{slot_content['name']} ({slot_content['division']})</div>
                        <div class="text-xs text-gray-600">({slot_content['professor']})</div>
                        <div class="text-xs text-red-500 font-bold">({slot_content['room_id']}호 배정)</div>
                        """
                        
                        html.append(f'<td class="px-3 py-2 text-center border bg-yellow-100/70" rowspan="{duration}">')
                        html.append(cell_content)
                        html.append('</td>')
                        
                    else:
                        html.append('<td class="px-3 py-2 text-center text-gray-400 border bg-white hover:bg-gray-50/50"></td>')

                html.append('</tr>')
                
            html.append('            </tbody>')
            html.append('        </table>')
            html.append('    </div>')
            html.append('</div>')

    # --- 미배정 보고서 추가 (진단 목적) ---
    if not unassigned_courses_df.empty:
        html.append(f'<div class="p-5 bg-red-50 border border-red-200 rounded-xl shadow-lg mt-10">')
        html.append(f'    <h3 class="text-xl font-bold text-red-700 mb-4 border-b pb-2">🚨 미배정 과목 보고서 ({len(unassigned_courses_df)}개)</h3>')
        # MAX_CONTINUOUS_LECTURE_HOURS 상수는 이제 연속 강의 제한 해제되었음을 알립니다.
        html.append(f'    <p class="text-sm text-gray-600 mb-3">미배정된 과목들은 강의실 및 시간 자원의 부족 때문에 배정될 수 없었습니다. (교수 연속 강의 제한은 해제됨)</p>')
        
        unassigned_by_grade = unassigned_courses_df.groupby('grade')['subject'].apply(list).to_dict()

        if unassigned_by_grade:
            for grade, subjects in unassigned_by_grade.items():
                html.append(f'    <p class="font-semibold text-red-600 mt-2">{grade}학년 미배정 과목 ({len(subjects)}개):</p>')
                html.append(f'    <ul class="list-disc list-inside ml-4 text-sm text-gray-700">')
                for subject in subjects:
                    html.append(f'        <li>{subject}</li>')
                html.append(f'    </ul>')
        
        html.append('</div>')
        
    html.append('</div>')
    return "".join(html)

def demonstrate_solution():
    """
    솔루션 함수를 시연하기 위한 DB에서 데이터 로드 및 실행
    """
    # 데이터베이스에 파일 기반 데이터를 먼저 로드하는 단계 (필수 실행)
    load_data_to_db() 
    
    try:
        # 데이터베이스 연결 및 데이터 로드 시도
        db_instance = DB()
        courses_df = db_instance.courses
        classRoom_df = db_instance.room
        db_instance.conn.close()
        
    except psycopg2.Error as e:
        print(f"데이터베이스 연결 또는 데이터 로드 중 오류 발생: {e}")
        return ""
    except Exception as e:
        # 파일 로드가 아닌 DB 로드이므로 FileNotFoundError 대신 다른 예외를 처리할 수 있음
        print(f"데이터 로드 실패: {e}")
        return ""
    
    # 데이터 유효성 검사
    if courses_df.empty or classRoom_df.empty:
        print("경고: 데이터베이스에서 과목 또는 강의실 데이터를 로드하지 못했습니다.")
        return ""
    
    # [핵심 수정] Course_ID 생성 (solve_optimal 호출 전에 생성해야 함)
    # KeyError 'Course_ID' 문제를 해결합니다.
    courses_df['Course_ID'] = courses_df['subject'] + '_' + courses_df['class'].astype(str)
        
    # 최적화 솔루션 실행
    schedule_df = solve_optimal(courses_df, classRoom_df)

    # --- 진단 로직 추가: 미배정 과목 추출 ---
    assigned_course_ids = set(schedule_df['Course_ID'])
    all_course_ids = set(courses_df['Course_ID'])
    unassigned_course_ids = list(all_course_ids - assigned_course_ids)
    
    # 미배정된 과목들의 원본 정보를 추출하여 HTML 생성 함수에 전달
    unassigned_courses_df = courses_df[courses_df['Course_ID'].isin(unassigned_course_ids)]
    
    # HTML 보고서 생성 시 unassigned_courses_df 전달
    html_output = generate_html_timetable(schedule_df, courses_df, classRoom_df, unassigned_courses_df)
    
    return html_output
class LLM:
    def __init__(self) -> None:
        self.client = genai.Client(
    api_key=os.getenv('gemini'))
    def invok(self,msg,table)->str:
        response = self.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{table}이 HTML코드를 보고 {msg}에 대한 답을 해줘라"
)
        return markdown.markdown(response.text)