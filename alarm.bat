SETLOCAL EnableExtensions
 :: �Ʒ��� notepad.exe�� ���μ��� �̸��� �ֽ��ϴ�. 
set EXE=python.exe
FOR /F %%x IN ('tasklist /NH /FI "IMAGENAME eq %EXE%"') DO IF %%x == %EXE% goto FOUND
echo Not running
C:\Users\user\Anaconda3\envs\test\python.exe H:\�˰���Ʈ���̵�2_bat\alarm.py
goto FIN
:FOUND 
echo Running
:FIN
