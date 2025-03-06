@echo off
setlocal EnableDelayedExpansion

set main_batch_folder="B:\SourceControl\_BATCH"
set threads=12

CALL %main_batch_folder%\compile.bat  --clean=true --compile_path="%cd%" --threads=!threads! --program=true --templates=true --export_path=B:\GodotWindows\Godot4.5\