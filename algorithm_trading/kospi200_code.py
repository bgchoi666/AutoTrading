import requests
import pandas as pd
from io import BytesIO

get_req_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
date_ = '20010228'

query_str_params={
'name': 'fileDown',
'filetype': 'xls',
'url': 'MKD/03/0304/03040101/mkd03040101T3_01',
'ind_tp_cd': '1',
'idx_ind_cd': '028',
'lang': 'ko',
'compst_isu_tp': '1',
'schdate': date_,
'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp'
}

r = requests.get(get_req_url, query_str_params)

gen_req_url='http://file.krx.co.kr/download.jspx'
headers = {
    'Referer': 'http://marketdata.krx.co.kr/mdi'
}

form_data = {
        'code' : r.content
}

r = requests.post(gen_req_url, form_data, headers=headers)

df = pd.read_excel(BytesIO(r.content))

file_name = "KOSPI200.xlsx"

df.to_excel(file_name)

