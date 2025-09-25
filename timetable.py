import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value
import numpy as np

def solve_optimal(courses_df: pd.DataFrame, classRoom_df: pd.DataFrame) -> pd.DataFrame:
    """
    강의 데이터와 강의실 데이터를 기반으로 최적의 시간표를 배정하는 함수.
    
    주의: 현재 강의실 수용 인원(capacity) 데이터가 없어, 
    모든 강의실의 수용 인원을 50명으로 임시 가정하여 제약 조건을 설정합니다.
    실제 사용 시 classRoom_df에 '수용인원' 컬럼을 추가하고 코드를 수정해야 합니다.
    
    Args:
        courses_df (pd.DataFrame): 강좌 정보 (교과목명, 수강인원, 교과목 시간, 강좌대표교수 등).
        classRoom_df (pd.DataFrame): 강의실 정보 (호실번호, 강의실명).

    Returns:
        pd.DataFrame: 배정된 시간표 (교과목명, 교수명, 요일, 시간, 호실번호).
                      해를 찾지 못할 경우 빈 DataFrame을 반환합니다.
    """
    
    # 1. 데이터 전처리 및 정의
    course_df = courses_df.copy()
    course_df['강좌ID'] = course_df.index
    Courses = course_df['강좌ID'].tolist()

    Days = ['월', '화', '수', '목', '금']
    Hours = list(range(9, 18)) # 9시부터 17시 시작 슬롯 (9시간)
    
    Rooms = classRoom_df['호실번호'].astype(str).tolist()

    Course_Req_Hours = course_df.set_index('강좌ID')['교과목 시간'].to_dict()
    Course_Enrollment = course_df.set_index('강좌ID')['수강인원'].to_dict()
    Course_Professor = course_df.set_index('강좌ID')['강좌대표교수'].to_dict()
    Professors = course_df['강좌대표교수'].unique().tolist()

    # 🚨 강의실 수용 인원 (임시 가정)
    ROOM_CAPACITY = 50 
    Room_Capacity = {room: ROOM_CAPACITY for room in Rooms}


    # 2. PuLP 최적화 모델 정의
    model = LpProblem("TimeTable Optimization", LpMinimize)
    model += 0 # 목적 함수 (제약 조건 만족이 주 목표)

    # 3. 결정 변수
    x = LpVariable.dicts("Schedule", (Courses, Rooms, Days, Hours), 0, 1, cat='Binary')


    # 4. 제약 조건 설정
    
    # C1: 모든 강의는 요구된 시간을 정확히 충족해야 함
    for c in Courses:
        model += lpSum(x[c][r][d][h] for r in Rooms for d in Days for h in Hours) == Course_Req_Hours[c], f"C1_Req_Hours_{c}"

    # C2: 한 강의실에는 한 시간에 하나의 강의만 배정 가능 (강의실 중복 사용 금지)
    for r in Rooms:
        for d in Days:
            for h in Hours:
                model += lpSum(x[c][r][d][h] for c in Courses) <= 1, f"C2_Room_Conflict_{r}_{d}_{h}"

    # C3: 한 교수는 한 시간에 하나의 강의만 담당 가능 (교수 중복 스케줄링 금지)
    for p in Professors:
        prof_courses = [c for c in Courses if Course_Professor[c] == p]
        for d in Days:
            for h in Hours:
                model += lpSum(x[c][r][d][h] for c in prof_courses for r in Rooms) <= 1, f"C3_Professor_Conflict_{p}_{d}_{h}"

    # C4: 강의실 수용 인원 제약 (임시 가정된 50명 사용)
    for c in Courses:
        enrollment = Course_Enrollment[c]
        for r in Rooms:
            # 🚨 강의실 수용 인원 데이터가 추가되면, 아래 조건문이 Room_Capacity[r] < enrollment 로 변경되어야 함
            if Room_Capacity[r] < enrollment: 
                model += lpSum(x[c][r][d][h] for d in Days for h in Hours) == 0, f"C4_Capacity_Fail_{c}_{r}"


    # 5. 모델 풀이 및 결과 정리
    try:
        model.solve()
    except Exception as e:
        # Solver 호출 오류 발생 시
        print(f"PuLP Solver 오류 발생: {e}")
        return pd.DataFrame()


    
    timetable = []
    
    # 배정된 결과만 추출
    for c in Courses:
        for r in Rooms:
            for d in Days:
                for h in Hours:
                    if value(x[c][r][d][h]) == 1:
                        course_info = course_df[course_df['강좌ID'] == c].iloc[0]
                        
                        timetable.append({
                            '교과목명': course_info['교과목명'],
                            '강좌대표교수': course_info['강좌대표교수'],
                            '요일': d,
                            '시간': f"{h}:00-{h+1}:00",
                            '호실번호': r
                        })

    schedule_df = pd.DataFrame(timetable)
    
    # 시간표 보기 좋게 정렬 후 반환
    day_order = {day: i for i, day in enumerate(Days)}
    schedule_df['요일순서'] = schedule_df['요일'].map(day_order)
    schedule_df['시간순서'] = schedule_df['시간'].apply(lambda x: int(x.split(':')[0]))
    schedule_df = schedule_df.sort_values(by=['요일순서', '시간순서', '호실번호'])
    
    return schedule_df[['교과목명', '강좌대표교수', '요일', '시간', '호실번호']]

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