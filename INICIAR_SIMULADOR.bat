@echo off
chcp 65001 > nul
title Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
cd /d "%~dp0"
echo ============================================================================
echo   STEEL BALL RUN - NEURAL NETWORK SIMULATOR (MySQL Workbench Edition)
echo ============================================================================
echo Iniciando simulador com 250 cavalos simultaneos...
python main.py 250
if errorlevel 1 (
    echo.
    echo Ocorreu um erro ao executar o simulador.
    pause
)

