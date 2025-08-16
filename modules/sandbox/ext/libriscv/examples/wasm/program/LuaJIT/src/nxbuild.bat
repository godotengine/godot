@rem Script to build LuaJIT with NintendoSDK + NX Addon.
@rem Donated to the public domain by Swyter.
@rem
@rem To run this script you must open a "Native Tools Command Prompt for VS".
@rem
@rem Either the x86 version for NX32, or x64 for the NX64 target.
@rem This is because the pointer size of the LuaJIT host tools (buildvm.exe)
@rem must match the cross-compiled target (32 or 64 bits).
@rem
@rem Then cd to this directory and run this script.
@rem
@rem Recommended invocation:
@rem
@rem nxbuild            # release build, amalgamated
@rem nxbuild debug      # debug build, amalgamated
@rem
@rem Additional command-line options (not generally recommended):
@rem
@rem noamalg            # (after debug) non-amalgamated build

@if not defined INCLUDE goto :FAIL
@if not defined NINTENDO_SDK_ROOT goto :FAIL
@if not defined PLATFORM goto :FAIL

@if "%platform%" == "x86" goto :DO_NX32
@if "%platform%" == "x64" goto :DO_NX64

@echo Error: Current host platform is %platform%!
@echo.
@goto :FAIL

@setlocal

:DO_NX32
@set DASC=vm_arm.dasc
@set DASMFLAGS= -D HFABI -D FPU
@set DASMTARGET= -D LUAJIT_TARGET=LUAJIT_ARCH_ARM
@set HOST_PTR_SIZE=4
goto :BEGIN

:DO_NX64
@set DASC=vm_arm64.dasc
@set DASMFLAGS= -D ENDIAN_LE
@set DASMTARGET= -D LUAJIT_TARGET=LUAJIT_ARCH_ARM64
@set HOST_PTR_SIZE=8

:BEGIN
@rem ---- Host compiler ----
@set LJCOMPILE=cl /nologo /c /MD /O2 /W3 /wo4146 /wo4244 /D_CRT_SECURE_NO_DEPRECATE
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

@rem Check that we have the right 32/64 bit host compiler to generate the right virtual machine files.
@minilua
@if "%ERRORLEVEL%" == "%HOST_PTR_SIZE%" goto :PASSED_PTR_CHECK

@echo The pointer size of the host in bytes (%HOST_PTR_SIZE%) does not match the expected value (%errorlevel%).
@echo Check that the script is being ran under the correct x86/x64 VS prompt.
@goto :BAD

:PASSED_PTR_CHECK
@set DASMFLAGS=%DASMFLAGS% %DASMTARGET% -D LJ_TARGET_NX -D LUAJIT_OS=LUAJIT_OS_OTHER -D LUAJIT_DISABLE_JIT -D LUAJIT_DISABLE_FFI
minilua %DASM% -LN %DASMFLAGS% -o host\buildvm_arch.h %DASC%
@if errorlevel 1 goto :BAD

if exist ..\.git ( git show -s --format=%%ct >luajit_relver.txt ) else ( type ..\.relver >luajit_relver.txt )
minilua host\genversion.lua

%LJCOMPILE% /I "." /I %DASMDIR% %DASMTARGET% -D LJ_TARGET_NX -DLUAJIT_OS=LUAJIT_OS_OTHER -DLUAJIT_DISABLE_JIT -DLUAJIT_DISABLE_FFI host\buildvm*.c
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
@set NXCOMPILER_ROOT="%NINTENDO_SDK_ROOT%\Compilers\NintendoClang"
@if "%platform%" neq "x64" goto :NX32_CROSSBUILD
@set LJCOMPILE="%NXCOMPILER_ROOT%\bin\clang" --target=aarch64-nintendo-nx-elf -Wall -I%NINTENDO_SDK_ROOT%\Include %DASMTARGET% -DLUAJIT_OS=LUAJIT_OS_OTHER -DLUAJIT_DISABLE_JIT -DLUAJIT_DISABLE_FFI -DLUAJIT_USE_SYSMALLOC -c
@set LJLIB="%NXCOMPILER_ROOT%\bin\llvm-ar" rc
@set TARGETLIB_SUFFIX=nx64

%NXCOMPILER_ROOT%\bin\clang --target=aarch64-nintendo-nx-elf -o lj_vm.o -c lj_vm.s
goto :DEBUGCHECK

:NX32_CROSSBUILD
@set LJCOMPILE="%NXCOMPILER_ROOT%\bin\clang" --target=armv7l-nintendo-nx-eabihf -Wall -I%NINTENDO_SDK_ROOT%\Include %DASMTARGET% -DLUAJIT_OS=LUAJIT_OS_OTHER -DLUAJIT_DISABLE_JIT -DLUAJIT_DISABLE_FFI -DLUAJIT_USE_SYSMALLOC -c
@set LJLIB="%NXCOMPILER_ROOT%\bin\llvm-ar" rc
@set TARGETLIB_SUFFIX=nx32

%NXCOMPILER_ROOT%\bin\clang --target=armv7l-nintendo-nx-eabihf -o lj_vm.o -c lj_vm.s
:DEBUGCHECK

@if "%1" neq "debug" goto :NODEBUG
@shift
@set LJCOMPILE=%LJCOMPILE% -DNN_SDK_BUILD_DEBUG -g -O0
@set TARGETLIB=libluajitD_%TARGETLIB_SUFFIX%.a
goto :BUILD
:NODEBUG
@set LJCOMPILE=%LJCOMPILE% -DNN_SDK_BUILD_RELEASE -O3
@set TARGETLIB=libluajit_%TARGETLIB_SUFFIX%.a
:BUILD
del %TARGETLIB%
@set LJCOMPILE=%LJCOMPILE% -fPIC
@if "%1" neq "noamalg" goto :AMALG
for %%f in (lj_*.c lib_*.c) do (
  %LJCOMPILE% %%f
  @if errorlevel 1 goto :BAD
)

%LJLIB% %TARGETLIB% lj_*.o lib_*.o
@if errorlevel 1 goto :BAD
@goto :NOAMALG
:AMALG
%LJCOMPILE% ljamalg.c
@if errorlevel 1 goto :BAD
%LJLIB% %TARGETLIB% ljamalg.o lj_vm.o
@if errorlevel 1 goto :BAD
:NOAMALG

@del *.o *.obj *.manifest minilua.exe buildvm.exe
@echo.
@echo === Successfully built LuaJIT for Nintendo Switch (%TARGETLIB_SUFFIX%) ===

@goto :END
:BAD
@echo.
@echo *******************************************************
@echo *** Build FAILED -- Please check the error messages ***
@echo *******************************************************
@goto :END
:FAIL
@echo To run this script you must open a "Native Tools Command Prompt for VS".
@echo.
@echo Either the x86 version for NX32, or x64 for the NX64 target.
@echo This is because the pointer size of the LuaJIT host tools (buildvm.exe)
@echo must match the cross-compiled target (32 or 64 bits).
@echo.
@echo Keep in mind that NintendoSDK + NX Addon must be installed, too.
:END
