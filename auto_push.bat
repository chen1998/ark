@echo off
REM === 自動 Git push 腳本 ===
REM 注意：需在已初始化 git 的資料夾中執行！

:: 設定顯示顏色（可選）
color 0A

echo 正在檢查 git 狀態...
git status

echo.
echo 加入所有變更檔案...
git add .

echo.
echo 請輸入提交說明：
set /p commit_msg="Commit message: "
git commit -m "%commit_msg%"

echo.
echo 推送到遠端倉庫...
git push

echo.
echo 推送完成，請確認 GitHub！
pause