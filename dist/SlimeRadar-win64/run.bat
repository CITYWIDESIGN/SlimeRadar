@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "EXE=%~dp0slimerander.exe"

:menu
cls
echo ============================================
echo    SlimeRadar - 史莱姆区块搜索器
echo ============================================
echo.
set "seed="
set "range="
set "thr="
set "fmt="

set /p "seed=请输入世界种子 : "
if "!seed!"=="" (
    echo [错误] 种子不能为空。
    echo.
    pause
    goto menu
)

set /p "range=请输入搜索区块半径 : "
if "!range!"=="" set "range=10000"

set /p "thr=请输入数量阈值 : "
if "!thr!"=="" set "thr=45"

set /p "fmt=保存为 CSV 文件吗? [ y,N ]: "

echo.
echo --------------------------------------------
echo  种子=!seed!   范围=+-!range!   阈值=!thr!
echo --------------------------------------------
echo.

set "T0=%time%"
if /i "!fmt!"=="y" (
    set "outfile=result_!seed!_!range!.csv"
    "!EXE!" -f csv !seed! !range! !thr! > "!outfile!"
    echo 已保存到: !outfile!
) else (
    "!EXE!" !seed! !range! !thr!
)
set "T1=%time%"

set "T0=!T0: =0!"
set "T1=!T1: =0!"
for /f "tokens=1-4 delims=:.," %%a in ("!T0!") do set /a "C0=(((1%%a-100)*60+(1%%b-100))*60+(1%%c-100))*100+(1%%d-100)"
for /f "tokens=1-4 delims=:.," %%a in ("!T1!") do set /a "C1=(((1%%a-100)*60+(1%%b-100))*60+(1%%c-100))*100+(1%%d-100)"
set /a "D=C1-C0"
if !D! LSS 0 set /a "D=D+8640000"
set /a "SEC=D/100"
set /a "CS=D%%100"
if !CS! LSS 10 set "CS=0!CS!"

echo.
echo --------------------------------------------
echo  搜索完成，用时 !SEC!.!CS! 秒
echo --------------------------------------------
echo 按任意键返回菜单，或关闭窗口退出。
pause >nul
goto menu
