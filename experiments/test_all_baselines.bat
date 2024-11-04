@echo off
REM Script for evaluation some test cases of load-dependent Chinese postman problem
REM Author: Dr. Truong Son Hy
REM Copyright 2023

REM Remove previous executables if they exist
if exist test_greedy.exe del test_greedy.exe
if exist test_ils_multithreads.exe del test_ils_multithreads.exe
if exist test_vns.exe del test_vns.exe
if exist test_de_multithreads.exe del test_de_multithreads.exe
if exist test_ea_multithreads.exe del test_ea_multithreads.exe
if exist test_aco_multithreads.exe del test_aco_multithreads.exe
if exist test_brute_force_multithreads.exe del test_brute_force_multithreads.exe

REM Compile the C++ test files
g++ test_greedy.cpp -o test_greedy.exe
g++ test_ils_multithreads.cpp -o test_ils_multithreads.exe -lpthread
g++ test_vns.cpp -o test_vns.exe
g++ test_de_multithreads.cpp -o test_de_multithreads.exe -lpthread
g++ test_ea_multithreads.cpp -o test_ea_multithreads.exe -lpthread
g++ test_aco_multithreads.cpp -o test_aco_multithreads.exe -lpthread
g++ test_brute_force_multithreads.cpp -o test_brute_force_multithreads.exe -lpthread

set data_dir=..\data\

REM Run tests on sample input files
for %%f in (%data_dir%sample_input_1.txt %data_dir%sample_input_2.txt) do (
    .\test_greedy.exe %%f
    .\test_ils_multithreads.exe %%f
    .\test_vns.exe %%f
    .\test_ea_multithreads.exe %%f
    .\test_aco_multithreads.exe %%f
    echo -------------------------------------------------------
)

REM Run tests on small input files
for /L %%i in (1,1,18) do (
    .\test_greedy.exe %data_dir%small_%%i.txt
    .\test_ils_multithreads.exe %data_dir%small_%%i.txt
    .\test_vns.exe %data_dir%small_%%i.txt
    .\test_ea_multithreads.exe %data_dir%small_%%i.txt
    .\test_aco_multithreads.exe %data_dir%small_%%i.txt
    echo -------------------------------------------------------
)

echo Done all baselines

REM Clean up executables
if exist test_greedy.exe del test_greedy.exe
if exist test_ils_multithreads.exe del test_ils_multithreads.exe
if exist test_vns.exe del test_vns.exe
if exist test_de_multithreads.exe del test_de_multithreads.exe
if exist test_ea_multithreads.exe del test_ea_multithreads.exe
if exist test_aco_multithreads.exe del test_aco_multithreads.exe
if exist test_brute_force_multithreads.exe del test_brute_force_multithreads.exe
