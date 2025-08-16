@rem Script to build LuaJIT with the Xbox One SDK.
@rem Donated to the public domain.
@rem
@rem Open a "Visual Studio .NET Command Prompt" (64 bit host compiler)
@rem Then cd to this directory and run this script.

@if not defined INCLUDE goto :FAIL
@if not defined DurangoXDK goto :FAIL

@setlocal
@echo ---- Host compiler ----
@set LJCOMPILE=cl /nologo /c /MD /O2 /W3 /D_CRT_SECURE_NO_DEPRECATE
@set LJLINK=link /nologo
@set LJMT=mt /nologo
@set DASMDIR=..\dynasm
@set DASM=%DASMDIR%\dynasm.lua
@set ALL_LIB=lib_base.c lib_math.c lib_bit.c lib_string.c lib_table.c lib_io.c lib_os.c lib_package.c lib_debug.c lib_jit.c lib_ffi.c lib_buffer.c

%LJCOMPILE% host\minilua.c
@if errorlevel 1 goto :BAD
%LJLINK% /out:minilua.exe minilua.obj
@if errorlevel 1 goto :BAD
if exist minilua.exe.manifest^
  %LJMT% -manifest minilua.exe.manifest -outputresource:minilua.exe

@rem Error out for 64 bit host compiler
@minilua
@if not errorlevel 8 goto :FAIL

@set DASMFLAGS=-D WIN -D FFI -D P64
minilua %DASM% -LN %DASMFLAGS% -o host\buildvm_arch.h vm_x64.dasc
@if errorlevel 1 goto :BAD

if exist ..\.git ( git show -s --format=%%ct >luajit_relver.txt ) else ( type ..\.relver >luajit_relver.txt )
minilua host\genversion.lua

%LJCOMPILE% /I "." /I %DASMDIR% /D_DURANGO host\buildvm*.c
@if errorlevel 1 goto :BAD
%LJLINK% /out:buildvm.exe buildvm*.obj
@if errorlevel 1 goto :BAD
if exist buildvm.exe.manifest^
  %LJMT% -manifest buildvm.exe.manifest -outputresource:buildvm.exe

buildvm -m peobj -o lj_vm.obj
@if errorlevel 1 goto :BAD
buildvm -m bcdef -o lj_bcdef.h %ALL_LIB%
@if errorlevel 1 goto :BAD
buildvm -m ffdef -o lj_ffdef.h %ALL_LIB%
@if errorlevel 1 goto :BAD
buildvm -m libdef -o lj_libdef.h %ALL_LIB%
@if errorlevel 1 goto :BAD
buildvm -m recdef -o lj_recdef.h %ALL_LIB%
@if errorlevel 1 goto :BAD
buildvm -m vmdef -o jit\vmdef.lua %ALL_LIB%
@if errorlevel 1 goto :BAD
buildvm -m folddef -o lj_folddef.h lj_opt_fold.c
@if errorlevel 1 goto :BAD

@echo ---- Cross compiler ----

@set CWD=%cd%
@call "%DurangoXDK%\xdk\DurangoVars.cmd" XDK
@cd /D "%CWD%"
@shift

@set LJCOMPILE="cl" /nologo /c /W3 /GF /Gm- /GR- /GS- /Gy /openmp- /D_CRT_SECURE_NO_DEPRECATE /D_LIB /D_UNICODE /D_DURANGO
@set LJLIB="lib" /nologo

@if "%1"=="debug" (
  @shift
  @set LJCOMPILE=%LJCOMPILE% /Zi /MDd /Od
  @set LJLINK=%LJLINK% /debug 
) else (
  @set LJCOMPILE=%LJCOMPILE% /MD /O2 /DNDEBUG
)

@if "%1"=="amalg" goto :AMALG
%LJCOMPILE% /DLUA_BUILD_AS_DLL lj_*.c lib_*.c
@if errorlevel 1 goto :BAD
%LJLIB% /OUT:luajit.lib lj_*.obj lib_*.obj
@if errorlevel 1 goto :BAD
@goto :NOAMALG
:AMALG
%LJCOMPILE% /DLUA_BUILD_AS_DLL ljamalg.c
@if errorlevel 1 goto :BAD
%LJLIB% /OUT:luajit.lib ljamalg.obj lj_vm.obj
@if errorlevel 1 goto :BAD
:NOAMALG

@del *.obj *.manifest minilua.exe buildvm.exe
@echo.
@echo === Successfully built LuaJIT for Xbox One ===

@goto :END
:BAD
@echo.
@echo *******************************************************
@echo *** Build FAILED -- Please check the error messages ***
@echo *******************************************************
@goto :END
:FAIL
@echo To run this script you must open a "Visual Studio .NET Command Prompt"
@echo (64 bit host compiler). The Xbox One SDK must be installed, too.
:END
