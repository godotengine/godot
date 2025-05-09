if %2.==one. goto setxboxone
rem Xbox Series compile
set DXC="%GameDKLatest%\GXDK\bin\Scarlett\DXC.exe"
set SUFFIX=_Series.h
goto startbuild

:setxboxone
set DXC="%GameDKLatest%\GXDK\bin\XboxOne\DXC.exe"
set SUFFIX=_One.h

:startbuild

call "%~dp0\compile_shaders.bat"
