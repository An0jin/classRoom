import os
import pandas as pd
from google import genai
import markdown
from pulp import *

def solve_optimal(courses_df: pd.DataFrame, rooms_df: pd.DataFrame, 
                  prof_rooms_df: pd.DataFrame=None, prof_days_df: pd.DataFrame=None) -> pd.DataFrame:
    
    # PULP 및 Pandas는 외부에서 import 되었다고 가정합니다.
    # from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus
    # import pandas as pd 

    # 1. 데이터 전처리 및 컬럼 통일
    
    # 1.1. 사용자 정의 컬럼 매핑 (파일 양식 기반)
    COL_MAP = {
        'course_prof_id': '강좌대표교수', 'pref_prof_id': '교수명', 'capacity': '수강인원',         
        'hours': '교과목학점', 'room_id': '호실번호', 'preferred_room_id': '호실번호', 
        'preferred_day': '요일', 
    }
    
    # 1.2. courses_df 전처리
    try:
        courses_df['group_id'] = courses_df['개설학과'] + '_' + courses_df['개설학년'].astype(str) + '학년_' + courses_df['반'].astype(str) + '반'
        courses_df['course_id'] = courses_df['교과목명'] + '_' + courses_df['개설학년'].astype(str) + '_' + courses_df['반'].astype(str)
        courses_df = courses_df.rename(columns={
            COL_MAP['course_prof_id']: 'prof_id', COL_MAP['capacity']: 'capacity', COL_MAP['hours']: 'hours'
        }, errors='raise')
    except KeyError as e:
        # 이 에러는 COL_MAP이 잘못되었거나 DF에 필수 컬럼이 누락되었을 때 발생합니다.
        raise ValueError(f"강의목록 데이터에서 필수 컬럼 '{e.args[0]}'을 찾을 수 없습니다. COL_MAP을 확인하세요.")
    
    # 1.3. rooms_df 전처리 (수용 인원 임시 삽입)
    try:
        rooms_df = rooms_df.rename(columns={
            COL_MAP['room_id']: 'room_id'
        }, errors='raise')
        
        # 🚨 강의실 DF에 '수용인원' 컬럼이 없으므로, 모델 실행을 위해 50명 임시 설정 🚨
        if 'size' not in rooms_df.columns:
            rooms_df['size'] = 50 
            
    except KeyError as e:
        raise ValueError(f"강의실정보 데이터에서 필수 컬럼 '{e.args[0]}'을 찾을 수 없습니다.")

    # 1.4. 시간 블록 정의 (T)
    DAYS = ['월', '화', '수', '목', '금']
    TIMES = [str(i) for i in range(1, 10)] # 9교시는 8교시 다음에 오지 않으므로 9교시까지 순회합니다.
    T = [d + t for d in DAYS for t in TIMES]
    T.append('금10') # 금9교시는 마지막 시간블록으로 추가

    # 1.5. 선택적 데이터 선호도 처리
    
    # A. 교수 선호 강의실
    prof_pref_room = {}
    if prof_rooms_df is not None and not prof_rooms_df.empty:
        prof_rooms_df = prof_rooms_df.rename(columns={
            COL_MAP['pref_prof_id']: 'prof_id', COL_MAP['preferred_room_id']: 'preferred_room_id'
        }, errors='raise')
        prof_pref_room = prof_rooms_df.groupby('prof_id')['preferred_room_id'].apply(set).to_dict()

    # B. 교수 선호 요일
    prof_preferred_days = {} 
    if prof_days_df is not None and not prof_days_df.empty:
        prof_days_df = prof_days_df.rename(columns={
            COL_MAP['pref_prof_id']: 'prof_id', COL_MAP['preferred_day']: 'preferred_day'
        }, errors='raise')
        preferred_map = {}
        for _, row in prof_days_df.iterrows():
            prof = row['prof_id']
            day = row['preferred_day']
            preferred_slots = {day + t for t in TIMES + ['9']}
            if prof not in preferred_map: preferred_map[prof] = set()
            preferred_map[prof].update(preferred_slots)
        prof_preferred_days = preferred_map
    
    # 2. 집합(Sets) 및 사전 생성 
    C = courses_df['course_id'].tolist() 
    R = rooms_df['room_id'].tolist() 
    P = courses_df['prof_id'].unique().tolist() 
    G = courses_df['group_id'].unique().tolist() 
    course_data = courses_df.set_index('course_id').to_dict('index')
    room_data = rooms_df.set_index('room_id').to_dict('index')
    
    # 3. 문제 및 결정 변수 정의
    prob = LpProblem("Timetable_Optimization", LpMinimize)
    X = LpVariable.dicts("X", (C, R, T), 0, 1, LpBinary) # X[c, r, t]: 배정 여부
    Y = LpVariable.dicts("Y", (C, R, T), 0, 1, LpBinary) # Y[c, r, t]: 시작 시간 여부
    Z = LpVariable.dicts("Z", (C, R), 0, 1, LpBinary)    # Z[c, r]: 강의실 고정 여부 (c가 r을 사용하면 1)
    
    M = 1000 # Big M 상수

    # 4. 목적 함수 정의 (패널티 최소화 / 보상 최대화)
    W_SIZE = 50; W_ROOM_PREF = 100; W_DAY_PREF = 25      
    objective_elements = []

    for c in C:
        prof_id = course_data[c]['prof_id']; required_capacity = course_data[c]['capacity']
        for r in R:
            room_size = room_data[r]['size']; penalty_capacity = 0
            
            # (A) 정원 불일치 패널티
            if required_capacity > room_size:
                 penalty_capacity = (required_capacity - room_size) * W_SIZE * 5 
            elif room_size > required_capacity * 1.5:
                 penalty_capacity = (room_size - required_capacity) * W_SIZE * 0.1
            
            # (B) 선호 강의실 보상
            reward_room = 0
            if prof_id in prof_pref_room and r in prof_pref_room[prof_id]:
                reward_room = -W_ROOM_PREF 

            for t in T:
                # (C) 선호 요일 보상
                reward_day = 0
                if prof_id in prof_preferred_days and t in prof_preferred_days[prof_id]:
                    reward_day = -W_DAY_PREF 
                
                total_penalty = penalty_capacity + reward_room + reward_day
                objective_elements.append(total_penalty * X[c][r][t])

    prob += lpSum(objective_elements), "Total_Penalty"

    # 5. 필수 제약 조건 (Hard Constraints)
    
    # 5.1. 강의 시간 완수
    for c in C: prob += lpSum([X[c][r][t] for r in R for t in T]) == course_data[c]['hours'], f"C{c}_Hours"

    # 5.2. 강의실 충돌 방지
    for r in R:
        for t in T: prob += lpSum([X[c][r][t] for c in C]) <= 1, f"R{r}_Conflict_{t}"

    # 5.3. 교수 충돌 방지
    for p in P:
        prof_courses = courses_df[courses_df['prof_id'] == p]['course_id'].tolist()
        for t in T: prob += lpSum([X[c][r][t] for c in prof_courses for r in R]) <= 1, f"P{p}_Conflict_{t}"
    
    # 5.4. 학생 그룹 충돌 방지
    for g in G:
        group_courses = courses_df[courses_df['group_id'] == g]['course_id'].tolist()
        for t in T: prob += lpSum([X[c][r][t] for c in group_courses for r in R]) <= 1, f"G{g}_Conflict_{t}"

    # 5.5. 강의실 정원 제약
    for c in C:
        required_capacity = course_data[c]['capacity']
        for r in R:
            room_size = room_data[r]['size']
            if required_capacity > room_size:
                for t in T: prob += X[c][r][t] == 0, f"C{c}_R{r}_TooSmall"
            
    # ==========================================================
    # 🚨 5.6. 강의 연속성 및 강의실 고정 제약 (수정 및 강화된 로직) 🚨
    # ==========================================================
    
    # 5.6.1. 강의실 고정 제약 (Single Room Constraint)
    for c in C:
        # X[c,r,t]가 1이면, Z[c,r]도 1이어야 함. (Big M 대신 X의 총합 사용)
        for r in R:
            prob += lpSum(X[c][r][t] for t in T) <= M * Z[c][r], f"C{c}_R{r}_Z_Link" 
        
        # 모든 강의실 중 단 하나만 Z[c,r]이 1이어야 함. (단일 강의실 사용 강제)
        prob += lpSum(Z[c][r] for r in R) == 1, f"C{c}_SingleRoom" 

    # 5.6.2. 연속성 강제 제약 (Contiguity & Fixed Room Enforced)
    for c in C:
        hours_c = course_data[c]['hours']
        
        # A. 전체 시작 횟수 제한: 강의는 배정된 총 시간에 대해 한 번만 시작해야 함 (연속 블록으로 나뉘지 않도록 강제)
        prob += lpSum(Y[c][r][t] for r in R for t in T) == 1, f"C{c}_ExactlyOneStart"

        for r in R:
            for t_start_index, t_start in enumerate(T):
                t_end_index = t_start_index + hours_c
                
                # B. 시작-배정 연결: Y[c,r,t_start]=1 이면, 다음 hours_c 블록은 반드시 X[c,r,t_k]=1 이어야 함.
                # 같은 요일 내에서만 연속 허용 확인 (t_start[:1]은 요일)
                is_contiguous_in_day = t_end_index <= len(T) and all(T[t_start_index + k][:1] == t_start[:1] for k in range(hours_c))
                
                if is_contiguous_in_day:
                    for k in range(hours_c):
                        t_k = T[t_start_index + k]
                        prob += Y[c][r][t_start] <= X[c][r][t_k], f"C{c}_R{r}_T{t_start}_Cont_{k}"
                else:
                    # 유효하지 않은 시작 시간은 0으로 강제
                    prob += Y[c][r][t_start] == 0, f"C{c}_R{r}_T{t_start}_InvalidStart"

        # C. X와 Y의 역방향 연결: X[c,r,t]=1 이면, t와 그 이전 hours_c-1 블록 중 하나는 시작 시간(Y=1)이어야 함.
        for r in R:
            for t_index, t in enumerate(T):
                 # 가능한 시작 시간 범위 계산
                 possible_starts = [T[t_start_index] 
                                    for t_start_index in range(t_index - hours_c + 1, t_index + 1) 
                                    if t_start_index >= 0 and T[t_start_index][:1] == t[:1]]
                 
                 # X[c,r,t] <= Y[c,r,t] + Y[c,r,t-1] + ...
                 prob += X[c][r][t] <= lpSum(Y[c][r][t_start] for t_start in possible_starts), f"C{c}_R{r}_T{t}_X_Link"

    # 6. 문제 풀이 및 결과 정리
    prob.solve()

    if LpStatus[prob.status] in ["Optimal", "Feasible"]:
        results = []
        for c in C:
            for r in R:
                for t in T:
                    if X[c][r][t].varValue == 1.0:
                        # 🚨 결과 DataFrame 가독성 개선 🚨
                        results.append({
                            '강의 고유 ID': c, 
                            '강의실': r, 
                            '요일': t[:1],           # 시간대에서 요일 추출
                            '교시': t[1:],           # 시간대에서 교시 추출
                            '담당 교수': course_data[c]['prof_id'], 
                            '학생 그룹 ID': course_data[c]['group_id'] 
                        })
        
        return pd.DataFrame(results)
    else:
        print(f"**경고: 최적 해를 찾지 못했습니다. 상태: {LpStatus[prob.status]}**")
        print("-> 연속성 제약 때문에 해를 찾지 못했을 수 있습니다. 요구시간(학점) 데이터를 확인하세요.")
        return pd.DataFrame()





def generate_html_timetable(schedule_df: pd.DataFrame) -> str:
    """
    Pandas DataFrame 형태의 시간표 데이터를 받아서
    그룹별 HTML 테이블을 생성합니다.
    이 함수는 HTML의 <table> 부분을 포함하는 문자열만 반환합니다.

    :param schedule_df: 시간표 데이터프레임.
                        필수 컬럼: '학생 그룹 ID', '강의 고유 ID', '요일', '교시', '담당 교수', '강의실'
    :return: 생성된 HTML 테이블 문자열
    """

    # '교시' 컬럼이 숫자형이 아닐 경우를 대비하여 숫자형으로 변환합니다.
    # 이렇게 하면 함수 외부에서 넘어오는 데이터의 타입에 관계없이 안정적으로 작동합니다.
    try:
        schedule_df['교시'] = pd.to_numeric(schedule_df['교시'])
    except (ValueError, TypeError) as e:
        return f"<p><strong>Error:</strong> '교시' 컬럼에 숫자로 변환할 수 없는 값이 포함되어 있습니다. 원본 데이터를 확인해주세요. (오류: {e})</p>"


    # 요일 순서 정의
    days_order = ['월', '화', '수', '목', '금']
    
    all_tables_html = ""
    
    # 학생 그룹 ID로 그룹화
    grouped = schedule_df.groupby('학생 그룹 ID')
    
    for group_name, group_df in grouped:
        all_tables_html += f"<h2>배정 그룹: {group_name}</h2>"
        
        # 비어있는 그룹은 건너뜁니다.
        if group_df.empty:
            continue
        
        # 시간표 그리드 생성 (최대 9교시, 5일)
        max_period = schedule_df['교시'].max()
        grid = [['' for _ in range(len(days_order))] for _ in range(max_period)]
        
        # 강의 기간 계산 및 그리드에 배치
        for day_idx, day in enumerate(days_order):
            day_classes = group_df[group_df['요일'] == day].sort_values('교시')
            
            processed_periods = set()
            
            for _, row in day_classes.iterrows():
                period = row['교시']
                if period in processed_periods:
                    continue

                class_id = row['강의 고유 ID']
                
                # 동일 강의 시간 찾기
                same_class_periods = day_classes[day_classes['강의 고유 ID'] == class_id]['교시'].tolist()
                start_period = min(same_class_periods)
                end_period = max(same_class_periods)
                duration = len(same_class_periods)

                # 강의 정보 HTML 생성
                subject_name = class_id.split('_')[0]
                professor = row['담당 교수']
                classroom = row['강의실']
                
                class_html = f"""
                <div class="subject">{subject_name}</div>
                <div class="details">({professor} / {classroom}호)</div>
                """
                
                # 그리드에 강의 정보와 rowspan 정보 저장
                grid[start_period-1][day_idx] = {'html': class_html, 'rowspan': duration}
                
                # 처리된 교시 기록
                for p in same_class_periods:
                    processed_periods.add(p)
                    if p != start_period:
                        grid[p-1][day_idx] = 'occupied' # 병합될 셀 표시

        # HTML 테이블 생성
        table_html = "<table><thead><tr><th class='period-col'>교시</th>"
        for day in days_order:
            table_html += f"<th>{day}</th>"
        table_html += "</tr></thead><tbody>"
        
        for period_idx in range(max_period):
            table_html += f"<tr><td class='period-col'>{period_idx+1}교시</td>"
            for day_idx in range(len(days_order)):
                cell_content = grid[period_idx][day_idx]
                if cell_content == 'occupied':
                    continue # 이미 병합된 셀이므로 건너뜀
                elif isinstance(cell_content, dict):
                    table_html += f"<td class='class-cell' rowspan='{cell_content['rowspan']}'>{cell_content['html']}</td>"
                else:
                    table_html += "<td></td>" # 빈 셀
            table_html += "</tr>"
            
        table_html += "</tbody></table>"
        all_tables_html += table_html

    return all_tables_html


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