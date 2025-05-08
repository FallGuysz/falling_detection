@echo off
echo Starting Human Fall Detection Streaming Server...

rem Set environment variables to hide warnings
set PYTHONWARNINGS=ignore::FutureWarning
set PYTHONWARNINGS=ignore::UserWarning

python stream_server.py -C 0 --device cuda 2>nul

pause 