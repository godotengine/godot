# Microsoft Developer Studio Project File - Name="LzmaUtil" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=LzmaUtil - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "LzmaUtil.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "LzmaUtil.mak" CFG="LzmaUtil - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "LzmaUtil - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "LzmaUtil - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "LzmaUtil - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MT /W4 /WX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /FD /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x419 /d "NDEBUG"
# ADD RSC /l 0x419 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386 /out:"c:\util\7lzma.exe"

!ELSEIF  "$(CFG)" == "LzmaUtil - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W4 /WX /Gm /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x419 /d "_DEBUG"
# ADD RSC /l 0x419 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /out:"c:\util\7lzma.exe" /pdbtype:sept

!ENDIF 

# Begin Target

# Name "LzmaUtil - Win32 Release"
# Name "LzmaUtil - Win32 Debug"
# Begin Source File

SOURCE=..\..\7zFile.c
# End Source File
# Begin Source File

SOURCE=..\..\7zFile.h
# End Source File
# Begin Source File

SOURCE=..\..\7zStream.c
# End Source File
# Begin Source File

SOURCE=..\..\7zTypes.h
# End Source File
# Begin Source File

SOURCE=..\..\7zVersion.h
# End Source File
# Begin Source File

SOURCE=..\..\7zWindows.h
# End Source File
# Begin Source File

SOURCE=..\..\Alloc.c
# End Source File
# Begin Source File

SOURCE=..\..\Alloc.h
# End Source File
# Begin Source File

SOURCE=..\..\Compiler.h
# End Source File
# Begin Source File

SOURCE=..\..\CpuArch.c
# End Source File
# Begin Source File

SOURCE=..\..\CpuArch.h
# End Source File
# Begin Source File

SOURCE=..\..\LzFind.c
# End Source File
# Begin Source File

SOURCE=..\..\LzFind.h
# End Source File
# Begin Source File

SOURCE=..\..\LzFindMt.c
# End Source File
# Begin Source File

SOURCE=..\..\LzFindMt.h
# End Source File
# Begin Source File

SOURCE=..\..\LzFindOpt.c
# End Source File
# Begin Source File

SOURCE=..\..\LzHash.h
# End Source File
# Begin Source File

SOURCE=..\..\LzmaDec.c
# End Source File
# Begin Source File

SOURCE=..\..\LzmaDec.h
# End Source File
# Begin Source File

SOURCE=..\..\LzmaEnc.c
# End Source File
# Begin Source File

SOURCE=..\..\LzmaEnc.h
# End Source File
# Begin Source File

SOURCE=.\LzmaUtil.c
# End Source File
# Begin Source File

SOURCE=..\..\Precomp.h
# End Source File
# Begin Source File

SOURCE=.\Precomp.h
# End Source File
# Begin Source File

SOURCE=..\..\Threads.c
# End Source File
# Begin Source File

SOURCE=..\..\Threads.h
# End Source File
# End Target
# End Project
