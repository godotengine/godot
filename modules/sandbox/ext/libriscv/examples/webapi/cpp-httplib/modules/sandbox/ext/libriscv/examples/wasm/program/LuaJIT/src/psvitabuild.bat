@rem Script to build LuaJIT with the PS Vita SDK.
@rem Donated to the public domain.
@rem
@rem Open a "Visual Studio .NET Command Prompt" (32 bit host compiler)
@rem Then cd to this directory and run this script.

@if not defined INCLUDE goto :FAIL
@if not defined SCE_PSP2_SDK_DIR goto :FAIL

@setlocal
@rem ---- Host compiler ----
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

@rem Check for 32 bit host compiler.
@minilua
@if errorlevel 8 goto :FAIL

@set DASMFLAGS=-D FPU -D HFABI
minilua %DASM% -LN %DASMFLAGS% -o host\buildvm_arch.h vm_arm.dasc
@if errorlevel 1 goto :BAD

if exist ..\.git ( git show -s --format=%%ct >luajit_relver.txt ) else ( type ..\.relver >luajit_relver.txt )
minilua host\genversion.lua

%LJCOMPILE% /I "." /I %DASMDIR% -DLUAJIT_TARGET=LUAJIT_ARCH_ARM -DLUAJIT_OS=LUAJIT_OS_OTHER -DLUAJIT_DISABLE_JIT -DLUAJIT_DISABLE_FFI -DLJ_TARGET_PSVITA=1 host\buildvm*.c
@if errorlevel 1 goto :BAD
%LJLINK% /out:buildvm.exe buildvm*.obj
@if errorlevel 1 goto :BAD
if exist buildvm.exe.manifest^
  %LJMT% -manifest buildvm.exe.manifest -outputresource:buildvm.exe

buildvm -m elfasm -o lj_vm.s
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

@rem ---- Cross compiler ----
@set LJCOMPILE="%SCE_PSP2_SDK_DIR%\host_tools\build\bin\psp2snc" -c -w -DLUAJIT_DISABLE_FFI -DLUAJIT_USE_SYSMALLOC
@set LJLIB="%SCE_PSP2_SDK_DIR%\host_tools\build\bin\psp2ld32" -r --output=
@set INCLUDE=""

"%SCE_PSP2_SDK_DIR%\host_tools\build\bin\psp2as" -o lj_vm.o lj_vm.s

@if "%1" neq "debug" goto :NODEBUG
@shift
@set LJCOMPILE=%LJCOMPILE% -g -O0
@set TARGETLIB=libluajitD.a
goto :BUILD
:NODEBUG
@set LJCOMPILE=%LJCOMPILE% -O2
@set TARGETLIB=libluajit.a
:BUILD
del %TARGETLIB%

%LJCOMPILE% ljamalg.c
@if errorlevel 1 goto :BAD
%LJLIB%%TARGETLIB% ljamalg.o lj_vm.o
@if errorlevel 1 goto :BAD

@del *.o *.obj *.manifest minilua.exe buildvm.exe
@echo.
@echo === Successfully built LuaJIT for PS Vita ===

@goto :END
:BAD
@echo.
@echo *******************************************************
@echo *** Build FAILED -- Please check the error messages ***
@echo *******************************************************
@goto :END
:FAIL
@echo To run this script you must open a "Visual Studio .NET Command Prompt"
@echo (32 bit host compiler). The PS Vita SDK must be installed, too.
:END
