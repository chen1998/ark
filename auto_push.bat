@echo off
REM === �۰� Git push �}�� ===
REM �`�N�G�ݦb�w��l�� git ����Ƨ�������I

:: �]�w����C��]�i��^
color 0A

echo ���b�ˬd git ���A...
git status

echo.
echo �[�J�Ҧ��ܧ��ɮ�...
git add .

echo.
echo �п�J���满���G
set /p commit_msg="Commit message: "
git commit -m "%commit_msg%"

echo.
echo ���e�컷�ݭܮw...
git push

echo.
echo ���e�����A�нT�{ GitHub�I
pause