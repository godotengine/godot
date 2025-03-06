@echo off
setlocal EnableDelayedExpansion

set main_batch_folder="B:\SourceControl\_BATCH"
set threads=16
set threads_max=32

if not [%1] == [] (
	SET "var="&for /f "delims=0123456789" %%i in ("%1") do set var=%%i
	if not defined var (
		if [%1] GEQ [1] (
			if [%1] LEQ [%threads_max%] (
				set threads=%1
			) else (
			set threads=%threads_max%
			)
		) else (
			set threads=1
		)
	)
)

CALL %main_batch_folder%\compile.bat --compile_path="%cd%" --threads=!threads! --program=true --templates=true --export_path=B:\GodotWindows\Godot4.5\