# Microsoft Developer Studio Project File - Name="LzmaLib" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=LzmaLib - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "LzmaLib.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "LzmaLib.mak" CFG="LzmaLib - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "LzmaLib - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "LzmaLib - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "LzmaLib - Win32 Release"

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
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LZMALIB_EXPORTS" /YX /FD /c
# ADD CPP /nologo /Gr /MT /W4 /WX /O2 /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LZMALIB_EXPORTS" /FD /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x419 /d "NDEBUG"
# ADD RSC /l 0x419 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386 /out:"C:\Util\LZMA.dll" /opt:NOWIN98
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "LzmaLib - Win32 Debug"

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
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LZMALIB_EXPORTS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W4 /WX /Gm /ZI /Od /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "LZMALIB_EXPORTS" /D "COMPRESS_MF_MT" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x419 /d "_DEBUG"
# ADD RSC /l 0x419 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /out:"C:\Util\LZMA.dll" /pdbtype:sept

!ENDIF 

# Begin Target

# Name "LzmaLib - Win32 Release"
# Name "LzmaLib - Win32 Debug"
# Begin Group "Spec"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\LzmaLib.def
# End Source File
# Begin Source File

SOURCE=.\LzmaLibExports.c
# End Source File
# Begin Source File

SOURCE=.\Precomp.h
# End Source File
# End Group
# Begin Source File

SOURCE=..\..\7zTypes.h
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

SOURCE=..\..\IStream.h
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

SOURCE=..\..\LzmaLib.c
# End Source File
# Begin Source File

SOURCE=..\..\LzmaLib.h
# End Source File
# Begin Source File

SOURCE=..\..\Precomp.h
# End Source File
# Begin Source File

SOURCE=.\resource.rc
# End Source File
# Begin Source File

SOURCE=..\..\Threads.c
# End Source File
# Begin Source File

SOURCE=..\..\Threads.h
# End Source File
# End Target
# End Project
