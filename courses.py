#%%
import requests

def get_course_availability():
    url = "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/searchResults/searchResults"
    
    # Query parameters
    params = {
        "txt_subject": "CS",
        "txt_courseNumber": "8803", 
        "txt_term": "202502",
        "startDatepicker": "",
        "endDatepicker": "",
        "uniqueSessionId": "71pta1736520119296",
        "pageOffset": "0",
        "pageMaxSize": "10",
        "sortColumn": "subjectDescription",
        "sortDirection": "asc"
    }
    
    # Headers from curl command
    headers = {
        "Host": "registration.banner.gatech.edu",
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "X-Synchronizer-Token": "40c73a76-d957-46d2-b27e-86aabe7b8528",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
        "Referer": "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/classRegistration/classRegistration",
        # "Cookie": "JSESSIONID=4BE834E2BC514F5525E46539A3923BA1; _ga_L3KDESR6P8=GS1.1.1733005378.1.1.1733006203.0.0.0; _ga=GA1.1.911336017.1733005379; _clck=18be4b9%7C2%7Cfrs%7C0%7C1795; _hjSessionUser_3867652=eyJpZCI6ImJlNjYzOGE0LTUyOWUtNWIyOC1iMjJmLTBjZmRmMjVlZTA2ZSIsImNyZWF0ZWQiOjE3MzMwMDUzNzk1MDMsImV4aXN0aW5nIjp0cnVlfQ==; _ga_0QEC0F4BTL=GS1.1.1734435868.5.0.1734435868.0.0.0; _ga_DBY1MDQHS1=GS1.1.1736519033.21.1.1736519978.0.0.0; _ga_NQ72LFXR2E=GS1.1.1733347934.2.0.1733347934.0.0.0; _ga_FLTL39B1PG=GS1.1.1733418208.1.0.1733418208.60.0.0; _ga_DBF4MB426N=GS1.1.1734472312.2.0.1734472312.60.0.0; _ga_93GEGKW8BW=GS1.1.1736519043.15.0.1736519043.0.0.0; _ga_JZJKHTL2BT=GS1.1.1736519366.18.1.1736520121.0.0.0; _ga_EF4W4FWYSY=GS1.1.1736393222.2.1.1736393521.0.0.0; _gid=GA1.2.585536405.1736174125; _ga_4TW76PBJ0Q=GS1.1.1736519502.5.0.1736519502.0.0.0; _ga_30TT54P0XE=GS1.1.1736519792.2.1.1736519822.0.0.0; _ga_8HB70LE2J4=GS1.1.1736175224.1.0.1736175225.0.0.0; __utmzz=utmcsr=google|utmcmd=organic|utmccn=(not set)|utmctr=(not provided); _gcl_au=1.1.1976068631.1736175260; _ga_8XJDVR2ZKP=GS1.1.1736175259.1.0.1736175263.56.0.1171685795; _uetvid=76b8c71072e411ef988233a2ef8f2a67; _ga_BH0P2FNXXQ=GS1.1.1736358522.4.0.1736358525.0.0.0; _ga_GP3B01FPCR=GS1.1.1736258859.1.1.1736258880.0.0.0; BIGipServer~BANNER~registration.coda=1688391554.64288.0000; _ga_7SESDMW4NW=GS1.1.1736520169.1.1.1736520220.0.0.0",
        # "Cookie": "JSESSIONID=02C10F1C624C2247ADF75C50A7DBFBC3; _ga_L3KDESR6P8=GS1.1.1733005378.1.1.1733006203.0.0.0; _ga=GA1.2.911336017.1733005379; _clck=18be4b9%7C2%7Cfrs%7C0%7C1795; _hjSessionUser_3867652=eyJpZCI6ImJlNjYzOGE0LTUyOWUtNWIyOC1iMjJmLTBjZmRmMjVlZTA2ZSIsImNyZWF0ZWQiOjE3MzMwMDUzNzk1MDMsImV4aXN0aW5nIjp0cnVlfQ==; _ga_0QEC0F4BTL=GS1.1.1734435868.5.0.1734435868.0.0.0; _ga_DBY1MDQHS1=GS1.1.1736527115.22.1.1736527122.0.0.0; _ga_NQ72LFXR2E=GS1.1.1733347934.2.0.1733347934.0.0.0; _ga_FLTL39B1PG=GS1.1.1733418208.1.0.1733418208.60.0.0; _ga_DBF4MB426N=GS1.1.1734472312.2.0.1734472312.60.0.0; _ga_93GEGKW8BW=GS1.1.1736527122.17.0.1736527122.0.0.0; _ga_JZJKHTL2BT=GS1.1.1736527032.20.1.1736527130.0.0.0; _ga_EF4W4FWYSY=GS1.1.1736393222.2.1.1736393521.0.0.0; _gid=GA1.2.585536405.1736174125; _ga_4TW76PBJ0Q=GS1.1.1736519502.5.0.1736519502.0.0.0; _ga_30TT54P0XE=GS1.1.1736519792.2.1.1736519822.0.0.0; _ga_8HB70LE2J4=GS1.1.1736175224.1.0.1736175225.0.0.0; __utmzz=utmcsr=google|utmcmd=organic|utmccn=(not set)|utmctr=(not provided); _gcl_au=1.1.1976068631.1736175260; _ga_8XJDVR2ZKP=GS1.1.1736175259.1.0.1736175263.56.0.1171685795; _uetvid=76b8c71072e411ef988233a2ef8f2a67; _ga_BH0P2FNXXQ=GS1.1.1736358522.4.0.1736358525.0.0.0; _ga_GP3B01FPCR=GS1.1.1736258859.1.1.1736258880.0.0.0; BIGipServer~BANNER~registration.coda=1688391554.64288.0000; _ga_7SESDMW4NW=GS1.1.1736520169.1.1.1736520220.0.0.0; _ga_CZ8LWWH02E=GS1.1.1736524370.1.0.1736524584.0.0.0; _gat_gtag_UA_217558002_6=1; _gat_gtag_UA_146262441_1=1",
        "Cookie": "JSESSIONID=0BA18F0E1B8CF211ADA117C051A23432; _ga_L3KDESR6P8=GS1.1.1733005378.1.1.1733006203.0.0.0; _ga=GA1.2.911336017.1733005379; _clck=18be4b9%7C2%7Cfrs%7C0%7C1795; _hjSessionUser_3867652=eyJpZCI6ImJlNjYzOGE0LTUyOWUtNWIyOC1iMjJmLTBjZmRmMjVlZTA2ZSIsImNyZWF0ZWQiOjE3MzMwMDUzNzk1MDMsImV4aXN0aW5nIjp0cnVlfQ==; _ga_0QEC0F4BTL=GS1.1.1734435868.5.0.1734435868.0.0.0; _ga_DBY1MDQHS1=GS1.1.1736527115.22.1.1736527122.0.0.0; _ga_NQ72LFXR2E=GS1.1.1733347934.2.0.1733347934.0.0.0; _ga_FLTL39B1PG=GS1.1.1733418208.1.0.1733418208.60.0.0; _ga_DBF4MB426N=GS1.1.1734472312.2.0.1734472312.60.0.0; _ga_93GEGKW8BW=GS1.1.1736527122.17.0.1736527122.0.0.0; _ga_JZJKHTL2BT=GS1.1.1736527032.20.1.1736528657.0.0.0; _ga_EF4W4FWYSY=GS1.1.1736393222.2.1.1736393521.0.0.0; _gid=GA1.2.585536405.1736174125; _ga_4TW76PBJ0Q=GS1.1.1736519502.5.0.1736519502.0.0.0; _ga_30TT54P0XE=GS1.1.1736519792.2.1.1736519822.0.0.0; _ga_8HB70LE2J4=GS1.1.1736175224.1.0.1736175225.0.0.0; __utmzz=utmcsr=google|utmcmd=organic|utmccn=(not set)|utmctr=(not provided); _gcl_au=1.1.1976068631.1736175260; _ga_8XJDVR2ZKP=GS1.1.1736175259.1.0.1736175263.56.0.1171685795; _uetvid=76b8c71072e411ef988233a2ef8f2a67; _ga_BH0P2FNXXQ=GS1.1.1736358522.4.0.1736358525.0.0.0; _ga_GP3B01FPCR=GS1.1.1736258859.1.1.1736258880.0.0.0; BIGipServer~BANNER~registration.coda=1688391554.64288.0000; _ga_7SESDMW4NW=GS1.1.1736520169.1.1.1736520220.0.0.0; _ga_CZ8LWWH02E=GS1.1.1736524370.1.0.1736524584.0.0.0",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors", 
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        content = response.json()
        data = content.get('data', [])
        print(f"Number of courses: {len(data)}")
        if data is None:
            print("No data found in response")
            return
        
        # Find course with 'inforcement' in the title
        for course in data:
            if 'inforcement' in course.get('courseTitle', '').lower():
                seats_available = course.get('seatsAvailable', 'N/A')
                return seats_available
                
        return 'N/A'
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return 'N/A'
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return 'N/A'

from playsound import playsound
iterations = 0
import time
while True:
    try:
        iterations += 1
        seats = get_course_availability()
        if seats != 'N/A':
            print(f"Seats Available: {seats}")
            from datetime import datetime
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if seats != 'N/A' and int(seats) > 0:
                print("SEATS AVAILABLE!")
                playsound('mixkit-bell-notification-933.wav')
            if (iterations % 1000 == 10):
                print("JUST KIDDING")
                playsound('mixkit-bell-notification-933.wav')
        elif seats == 'N/A':
            print('Problem with request')
            playsound('mixkit-bell-notification-933.wav')
    except Exception as e:
        print(f"Error occurred: {e}")
        playsound('mixkit-bell-notification-933.wav')
    time.sleep(1)

#%%
# !pip install playsound --break-system-packages
from playsound import playsound
playsound('mixkit-bell-notification-933.wav')  # Requires an audio file
