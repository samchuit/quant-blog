@echo off
chcp 65001 >nul
title 量化交易监控系统
cls

echo ================================================
echo       量化交易监控系统 - 启动面板
echo ================================================
echo.
echo 当前时间: %date% %time%
echo.
echo 选择运行模式:
echo   1. 运行一次 (立即执行)
echo   2. 定时运行 (每60分钟)
echo   3. 定时运行 (每30分钟)
echo   4. 查看交易日志
echo   5. 查看今日报告
echo   6. 退出
echo.
set /p choice=请输入选项 (1-6): 

if "%choice%"=="1" goto once
if "%choice%"=="2" goto schedule60
if "%choice%"=="3" goto schedule30
if "%choice%"=="4" goto log
if "%choice%"=="5" goto report
if "%choice%"=="6" goto end

:once
echo.
echo [运行一次]
python trading_monitor.py --mode once
echo.
pause
goto menu

:schedule60
echo.
echo [定时运行 - 每60分钟]
echo 按 Ctrl+C 可停止
echo.
python trading_monitor.py --mode schedule --interval 60
goto menu

:schedule30
echo.
echo [定时运行 - 每30分钟]
echo 按 Ctrl+C 可停止
echo.
python trading_monitor.py --mode schedule --interval 30
goto menu

:log
echo.
echo [交易日志]
echo.
type trading_log.json
echo.
pause
goto menu

:report
echo.
echo [今日报告]
echo.
type daily_report.txt
echo.
pause
goto menu

:menu
cls
goto start

:end
echo.
echo 再见！
timeout /t 2 >nul
exit

:start
