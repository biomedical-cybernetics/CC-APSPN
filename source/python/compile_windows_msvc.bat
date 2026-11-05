@echo off
REM Compatibility entry point. Pass --eigen PATH or set EIGEN3_INCLUDE_DIR.
python "%~dp0build_native.py" %*
exit /b %ERRORLEVEL%
