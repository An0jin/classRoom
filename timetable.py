import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value
import numpy as np

def solve_optimal(courses_df: pd.DataFrame, classRoom_df: pd.DataFrame) -> pd.DataFrame:
    """
    강의 데이터와 강의실 데이터를 기반으로 최적의 시간표를 배정하는 함수.
    
    ✅ 적용된 제약 조건:
    1. 강의 시간 충족 및 연속성 보장 (C5)
    2. 강의실 중복 사용 방지 (C2)
    3. 교수 중복 스케줄링 방지 (C3)
    4. 강의실 수용 인원 충족 (C4 - 임시 가정)
    5. 학생 그룹(학과/학년/분반) 충돌 방지 (C6 - 핵심)

    Args:
        courses_df (pd.DataFrame): 강좌 정보 (개설학과, 개설학년, 수강분반 포함).
        classRoom_df (pd.DataFrame): 강의실 정보.

    Returns:
        pd.DataFrame: 배정된 시간표. 해를 찾지 못할 경우 빈 DataFrame을 반환합니다.
    """
    
    course_df = courses_df.copy()
    course_df['강좌ID'] = course_df.index
    Courses = course_df['강좌ID'].tolist()

    Days = ['월', '화', '수', '목', '금']
    Hours = list(range(9, 18)) # 9시부터 17시 시작 슬롯
    Rooms = classRoom_df['호실번호'].astype(str).tolist()

    # 제약 조건에 필요한 데이터 매핑
    Course_Req_Hours = course_df.set_index('강좌ID')['교과목 시간'].to_dict()
    Course_Enrollment = course_df.set_index('강좌ID')['수강인원'].to_dict()
    Course_Professor = course_df.set_index('강좌ID')['강좌대표교수'].to_dict()
    Professors = course_df['강좌대표교수'].unique().tolist()
    
    # 🚨 강의실 수용 인원 (임시 가정: 실제 데이터로 대체 필수)
    ROOM_CAPACITY = 50 
    Room_Capacity = {room: ROOM_CAPACITY for room in Rooms}
    
    # 🌟 C6을 위한 학생 그룹 정의: (학과, 학년, 분반)
    course_df['학생그룹'] = course_df['개설학과'].astype(str) + '_' + \
                             course_df['개설학년'].astype(str) + '_' + \
                             course_df['수강분반'].astype(str)
                             
    StudentGroups = course_df['학생그룹'].unique().tolist()
    Course_StudentGroup = course_df.set_index('강좌ID')['학생그룹'].to_dict()


    # 2. PuLP 최적화 모델 정의
    model = LpProblem("TimeTable Optimization with Student Groups and Contiguity", LpMinimize)
    model += 0 

    # 3. 결정 변수
    x = LpVariable.dicts("Schedule", (Courses, Rooms, Days, Hours), 0, 1, cat='Binary')
    y = LpVariable.dicts("Start", (Courses, Rooms, Days, Hours), 0, 1, cat='Binary')


    # 4. 제약 조건 설정

    # C5a: 연속성 제약 - 강의 종료 시간 제한 (18시 초과 금지)
    for c in Courses:
        L_c = Course_Req_Hours[c]
        for r in Rooms:
            for d in Days:
                for h in Hours:
                    if h + L_c > 18:
                        model += y[c][r][d][h] == 0, f"C5a_End_Time_Invalid_{c}_{d}_{h}"

    # C5b: 연속성 제약 - 각 강의(강좌ID)는 단 한 번만 시작해야 함 (단일 연속 블록)
    for c in Courses:
        model += lpSum(y[c][r][d][h] for r in Rooms for d in Days for h in Hours) == 1, f"C5b_Single_Continuous_Block_{c}"

    # C5c: 연속성 및 시간 충족 제약 - x와 y 연결 (x[h]는 h를 포함하는 시작 블록의 합과 같음)
    for c in Courses:
        L_c = Course_Req_Hours[c]
        for r in Rooms:
            for d in Days:
                for h in Hours:
                    start_min = max(9, h - L_c + 1)
                    start_max = min(17, h)
                    valid_starts = [h_start for h_start in range(start_min, start_max + 1) 
                                    if h_start + L_c <= 18]

                    if valid_starts:
                         model += x[c][r][d][h] == lpSum(y[c][r][d][h_start] for h_start in valid_starts), f"C5c_Link_X_Y_{c}_{r}_{d}_{h}"
                    else:
                        model += x[c][r][d][h] == 0, f"C5c_Link_X_Y_Zero_{c}_{r}_{d}_{h}"

    # C2: 한 강의실에는 한 시간에 하나의 강의만 배정 가능
    for r in Rooms:
        for d in Days:
            for h in Hours:
                model += lpSum(x[c][r][d][h] for c in Courses) <= 1, f"C2_Room_Conflict_{r}_{d}_{h}"

    # C3: 한 교수는 한 시간에 하나의 강의만 담당 가능
    for p in Professors:
        prof_courses = [c for c in Courses if Course_Professor[c] == p]
        for d in Days:
            for h in Hours:
                model += lpSum(x[c][r][d][h] for c in prof_courses for r in Rooms) <= 1, f"C3_Professor_Conflict_{p}_{d}_{h}"
    
    # 🌟 C6: 학생 그룹 충돌 방지 (가장 중요한 제약)
    for g in StudentGroups:
        group_courses = [c for c in Courses if Course_StudentGroup[c] == g]
        for d in Days:
            for h in Hours:
                # 해당 학생 그룹의 모든 강의는 해당 시간에 1개 이하로 배정되어야 함
                model += lpSum(x[c][r][d][h] for c in group_courses for r in Rooms) <= 1, f"C6_Student_Group_Conflict_{g}_{d}_{h}"

    # C4: 강의실 수용 인원 제약 (강의실 수용 인원이 부족하면 시작 불가)
    for c in Courses:
        enrollment = Course_Enrollment[c]
        for r in Rooms:
            if Room_Capacity[r] < enrollment: 
                model += lpSum(y[c][r][d][h] for d in Days for h in Hours) == 0, f"C4_Capacity_Fail_Start_{c}_{r}"


    # 5. 모델 풀이 및 결과 정리
    try:
        model.solve()
    except Exception as e:
        print(f"PuLP Solver 오류 발생: {e}")
        return pd.DataFrame()

    
    if value(model.status) == 1: 
        
        timetable = []
        
        # 배정된 결과만 추출 (x 변수 기준)
        for c in Courses:
            for r in Rooms:
                for d in Days:
                    # y 변수를 사용하여 연속된 블록의 시작점 찾기
                    for h in Hours:
                        if value(y[c][r][d][h]) == 1:
                            course_info = course_df[course_df['강좌ID'] == c].iloc[0]
                            L_c = Course_Req_Hours[c]
                            
                            timetable.append({
                                '교과목명': course_info['교과목명'],
                                '강좌대표교수': course_info['강좌대표교수'],
                                '요일': d,
                                '시간': f"{h}:00-{h + L_c}:00 ({L_c}시간 연속)",
                                '호실번호': r
                            })

        schedule_df = pd.DataFrame(timetable)

        # 정렬 후 반환
        day_order = {day: i for i, day in enumerate(Days)}
        schedule_df['요일순서'] = schedule_df['요일'].map(day_order)
        schedule_df['시작시간'] = schedule_df['시간'].apply(lambda x: int(x.split(':')[0]))
        schedule_df = schedule_df.sort_values(by=['요일순서', '시작시간', '호실번호'])
        
        return schedule_df[['교과목명', '강좌대표교수', '요일', '시간', '호실번호']]
    
    else:
        print(f"❌ 모델 해결 실패 (Status: {value(model.status)}). 현재 제약 조건을 만족하는 해를 찾을 수 없습니다.")
        return pd.DataFrame()

def generate_html_timetable(schedule_df: pd.DataFrame) -> str:
    """
    최적화된 시간표 DataFrame을 HTML 테이블 문자열로 변환합니다.
    (rowspan을 사용하여 연속된 수업을 병합)
    """
    
    # 9시부터 18시까지의 시간대 정의
    TIME_SLOTS = list(range(9, 18))
    DAYS = ['월', '화', '수', '목', '금']
    
    # 1. HTML 렌더링을 위한 매트릭스 변환 (이전 단계의 함수 로직 사용)
    timetable_matrix = {day: {} for day in DAYS}
    occupied_slots = {day: set() for day in DAYS}
    
    for _, row in schedule_df.iterrows():
        day = row['요일']
        time_str = row['시간'] # 예: "9:00-14:00 (5시간 연속)"
        
        start_hour = int(time_str.split(':')[0])
        try:
            duration = int(time_str.split('(')[1].split('시간')[0])
        except (IndexError, ValueError):
            duration = 1 
        
        # 이미 처리된 슬롯은 건너뜁니다.
        if start_hour in occupied_slots[day]:
            continue

        course_info = {
            'name': row['교과목명'],
            'professor': row['강좌대표교수'],
            'room': row['호실번호'],
            'duration': duration,
            'is_start': True
        }
        
        timetable_matrix[day][start_hour] = course_info
        
        # 연속된 시간을 occupied로 표시
        for h in range(start_hour, start_hour + duration):
            occupied_slots[day].add(h)
    
    # 2. HTML 테이블 구조 생성
    html = ['<table class="timetable">',
            '<thead><tr><th class="time-header">시간</th><th>월</th><th>화</th><th>수</th><th>목</th><th>금</th></tr></thead>',
            '<tbody>']
    
    for hour in TIME_SLOTS:
        html.append(f'<tr id="hour-{hour}">')
        # 시간 컬럼
        html.append(f'<td class="time-slot">{hour}:00 - {hour + 1}:00</td>')
        
        for day in DAYS:
            course_info = timetable_matrix[day].get(hour)
            
            if course_info and course_info['is_start']:
                # 강의가 시작될 때 셀 병합 적용
                duration = course_info['duration']
                
                cell_content = f"""
                <div class="course-name">{course_info['name']}</div>
                <div class="course-professor">({course_info['professor']})</div>
                <div class="course-room">{course_info['room']}호</div>
                """
                
                html.append(f'<td class="course-cell" rowspan="{duration}" data-room="{course_info["room"]}">')
                html.append(cell_content)
                html.append('</td>')
                
            elif hour in occupied_slots[day] and timetable_matrix[day].get(hour) is None:
                # 이미 rowspan에 의해 병합된 슬롯이면 셀을 생략 (Jinja2의 'elif not course_info or not course_info.is_running' 역할)
                continue
            else:
                # 빈 슬롯일 때
                html.append('<td class="empty-cell"></td>')

        html.append('</tr>')
        
    html.append('</tbody></table>')
    
    return "".join(html)