@echo off
chcp 65001 >nul
echo ============================================
echo     量化交易策略 - 管理面板
echo ============================================
echo.
echo 选择操作:
echo   1. 运行模拟盘交易
echo   2. 执行月度重训练
echo   3. 生成监控报告
echo   4. 启动所有服务
echo   5. 退出
echo.
set /p choice=请输入选项 (1-5): 

if "%choice%"=="1" goto trading
if "%choice%"=="2" goto retrain
if "%choice%"=="3" goto report
if "%choice%"=="4" goto all
if "%choice%"=="5" goto end

:trading
echo.
echo 启动模拟盘交易...
python live_trading.py
pause
goto menu

:retrain
echo.
echo 执行模型重训练...
python auto_retrain.py --mode now
pause
goto menu

:report
echo.
echo 生成监控报告...
python monitor_report.py
type monitor_report.txt
pause
goto menu

:all
echo.
echo 启动所有服务...
start "监控服务" cmd /c "python monitor_report.py --mode daily"
echo 监控服务已启动 (每日9:00和21:00)
echo.
echo 按任意键运行一次模拟交易...
pause >nul
python live_trading.py
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
