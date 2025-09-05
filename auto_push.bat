@echo off
REM === 智能 Git push 腳本 ===
REM 放在任何已初始化的 git repo 目錄中執行

setlocal EnableExtensions EnableDelayedExpansion
color 0A

REM 1) 確認在 git 版本庫內
git rev-parse --is-inside-work-tree >NUL 2>&1
if errorlevel 1 (
  echo [錯誤] 這不是 Git 版本庫資料夾。
  pause
  exit /b 1
)

REM 2) 取得目前分支名稱
for /f "usebackq tokens=*" %%i in (`git rev-parse --abbrev-ref HEAD`) do set BRANCH=%%i
echo 目前分支：%BRANCH%

REM 3) 先同步遠端資訊
echo.
echo [步驟] 取得遠端更新...
git fetch --prune
if errorlevel 1 (
  echo [錯誤] git fetch 失敗。
  pause
  exit /b 1
)

REM 4) 檢查與遠端的差異（ ahead/behind ）
for /f "usebackq tokens=1,2" %%a in (`
  git rev-list --left-right --count origin/%BRANCH%...%BRANCH% 2^>NUL
`) do (
  set BEHIND=%%a
  set AHEAD=%%b
)

echo 與遠端差異：behind=%BEHIND%  ahead=%AHEAD%

REM 5) 若 behind>0 (遠端有新提交)，先 pull --rebase
if defined BEHIND if not "!BEHIND!"=="0" (
  echo.
  echo [步驟] 遠端有新提交，執行 pull --rebase...
  git pull --rebase origin %BRANCH%
  if errorlevel 1 (
    echo.
    echo [衝突處理提示]
    echo - 解衝突後：git add <檔案>
    echo - 繼續 rebase：git rebase --continue
    echo - 或放棄 rebase：git rebase --abort
    pause
    exit /b 1
  )
)

REM 6) 加入變更（工作區或暫存區有變更才 commit）
echo.
echo [步驟] 檢查是否有變更需要提交...
git diff --quiet && git diff --cached --quiet
if errorlevel 1 (
  echo 發現變更，加入所有檔案...
  git add -A

  echo.
  set /p commit_msg="請輸入提交說明（留空則使用自動訊息）： "
  if "%commit_msg%"=="" (
    for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"`) do set NOW=%%i
    set commit_msg=Auto commit at %NOW%
  )

  git commit -m "%commit_msg%"
  if errorlevel 1 (
    echo [警告] commit 失敗或沒有可提交的變更。
  )
) else (
  echo 沒有任何變更可提交（working tree clean）。
)

REM 7) 推送（若未設 upstream 則自動設定）
echo.
echo [步驟] 推送到遠端...
git rev-parse --abbrev-ref --symbolic-full-name @{u} >NUL 2>&1
if errorlevel 1 (
  echo 未設定 upstream，將以 --set-upstream 推送。
  git push --set-upstream origin %BRANCH%
) else (
  git push
)

if errorlevel 1 (
  echo.
  echo [提示] 若再次被拒，通常是推送期間遠端又有新提交。
  echo        請再執行：git fetch ^&^& git pull --rebase origin %BRANCH% ^&^& git push
  pause
  exit /b 1
)

echo.
echo 推送完成，請到遠端（如 GitHub）確認！
pause