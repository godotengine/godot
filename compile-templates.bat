@echo off
setlocal EnableDelayedExpansion

set main_batch_folder=B:\SourceControl\_BATCH
set threads=16

set main_args="%main_batch_folder%\compile.bat" --compile_path="%~dp0" --threads=!threads! --program=false --templates=true --export_path=B:\GodotWindows\Godot4.5\

set disabled_features=
if not [%1]==[] (
	
	set filename=%~n1
	
	for /f delims^=^ eol^= %%a in (%1) do (
		set first_char=%%~a
		rem echo !first_char:~0,1!
		rem pause
		if "!first_char:~0,1!" neq "#" (
			if not defined disabled_features (
				set "disabled_features=--disable=%%~a"
			) else (
				set disabled_features=!disabled_features! --disable=%%~a
			)
		)
	)
	set main_args=%main_args%  !disabled_features! --build_name=!filename!
)

rem echo !main_args!
rem pause

CALL !main_args!