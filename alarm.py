# -*- coding:utf-8 -*-

import smtplib
from email.mime.text import MIMEText

smtp = smtplib.SMTP('smtp.gmail.com', 587)
smtp.ehlo()  # say Hello
smtp.starttls()  # TLS 사용시 필요
smtp.login('bgchoi666@gmail.com', 'zxwzswqgfdvthsgu')

msg = MIMEText('프로그램 다운')
msg['Subject'] = '자동거래'
msg['To'] = 'bgchoi666@gmail.com'
smtp.sendmail('bgchoi666@gmail.com', 'bgchoi666@gmail.com', msg.as_string())

smtp.quit()