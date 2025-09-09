SETLOCAL EnableExtensions
 :: 아래의 notepad.exe에 프로세스 이름을 넣습니다. 
set EXE=python.exe
FOR /F %%x IN ('tasklist /NH /FI "IMAGENAME eq %EXE%"') DO IF %%x == %EXE% goto FOUND
echo Not running
C:\Users\user\Anaconda3\envs\test\python.exe H:\알고리즘트레이딩2_bat\alarm.py
goto FIN
:FOUND 
echo Running
:FIN
