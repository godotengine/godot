7-Zip for installers 9.38
-------------------------

7-Zip is a file archiver for Windows NT/2000/2003/2008/XP/Vista/7/8/10. 

7-Zip for installers is part of LZMA SDK.
LZMA SDK is written and placed in the public domain by Igor Pavlov.

It's allowed to join 7-Zip SFX module with another software.
It's allowed to change resources of 7-Zip's SFX modules.


HOW to use
-----------

7zr.exe is reduced version of 7za.exe of 7-Zip.
7zr.exe supports only format with these codecs: LZMA, LZMA2, BCJ, BCJ2, ARM, Copy.

Example of compressing command for installation packages:

7zr a archive.7z files

7zSD.sfx is SFX module for installers. 7zSD.sfx uses msvcrt.dll.

SFX modules for installers allow to create installation program. 
Such module extracts archive to temp folder and then runs specified program and removes 
temp files after program finishing. Self-extract archive for installers must be created 
as joining 3 files: SFX_Module, Installer_Config, 7z_Archive. 
Installer_Config is optional file. You can use the following command to create installer 
self-extract archive:

copy /b 7zSD.sfx + config.txt + archive.7z archive.exe

The smallest installation package size can be achieved, if installation files was 
uncompressed before including to 7z archive.

-y switch for installer module (at runtime) specifies quiet mode for extracting.

Installer Config file format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Config file contains commands for Installer. File begins from string 
;!@Install@!UTF-8! and ends with ;!@InstallEnd@!. File must be written 
in UTF-8 encoding. File contains string pairs: 

ID_String="Value"

ID_String          Description 

Title              Title for messages 
BeginPrompt        Begin Prompt message 
Progress           Value can be "yes" or "no". Default value is "yes". 
RunProgram         Command for executing. Default value is "setup.exe". 
                   Substring %%T will be replaced with path to temporary 
                   folder, where files were extracted 
Directory          Directory prefix for "RunProgram". Default value is ".\\" 
ExecuteFile        Name of file for executing 
ExecuteParameters  Parameters for "ExecuteFile" 


You can omit any string pair.

There are two ways to run program: RunProgram and ExecuteFile. 
Use RunProgram, if you want to run some program from .7z archive. 
Use ExecuteFile, if you want to open some document from .7z archive or 
if you want to execute some command from Windows.

If you use RunProgram and if you specify empty directory prefix: Directory="", 
the system searches for the executable file in the following sequence:

1. The directory from which the application (installer) loaded. 
2. The temporary folder, where files were extracted. 
3. The Windows system directory. 


Config file Examples
~~~~~~~~~~~~~~~~~~~~

;!@Install@!UTF-8!
Title="7-Zip 4.00"
BeginPrompt="Do you want to install the 7-Zip 4.00?"
RunProgram="setup.exe"
;!@InstallEnd@!



;!@Install@!UTF-8!
Title="7-Zip 4.00"
BeginPrompt="Do you want to install the 7-Zip 4.00?"
ExecuteFile="7zip.msi"
;!@InstallEnd@!



;!@Install@!UTF-8!
Title="7-Zip 4.01 Update"
BeginPrompt="Do you want to install the 7-Zip 4.01 Update?"
ExecuteFile="msiexec.exe"
ExecuteParameters="/i 7zip.msi REINSTALL=ALL REINSTALLMODE=vomus"
;!@InstallEnd@!



Small SFX modules for installers
--------------------------------

7zS2.sfx     - small SFX module (GUI version)
7zS2con.sfx  - small SFX module (Console version)

Small SFX modules support this codecs: LZMA, LZMA2, BCJ, BCJ2, ARM, COPY

Small SFX module is similar to common SFX module for installers.
The difference (what's new in small version):
 - Smaller size (30 KB vs 100 KB)
 - C source code instead of Ñ++
 - No installer Configuration file
 - No extracting progress window
 - It decompresses solid 7z blocks (it can be whole 7z archive) to RAM.
   So user that calls SFX installer must have free RAM of size of largest 
   solid 7z block (size of 7z archive at simplest case).

How to use
----------

copy /b 7zS2.sfx + archive.7z sfx.exe

When you run installer sfx module (sfx.exe)
1) It creates "7zNNNNNNNN" temp folder in system temp folder.
2) It extracts .7z archive to that folder
3) It executes one file from "7zNNNNNNNN" temp folder. 
4) It removes "7zNNNNNNNN" temp folder

You can send parameters to installer, and installer will transfer them to extracted .exe file.

Small SFX uses 3 levels of priorities to select file to execute:

  1) Files in root folder have higher priority than files in subfolders.
  2) File extension priorities (from high to low priority order): 
       bat, cmd, exe, inf, msi, cab (under Windows CE), html, htm
  3) File name priorities (from high to low priority order): 
       setup, install, run, start

Windows CE (ARM) version of 7zS2.sfx is included to 7-Zip for Windows Mobile package.


Examples
--------

1) To create compressed console 7-Zip:

7zr a c.7z 7z.exe 7z.dll -mx
copy /b 7zS2con.sfx + c.7z 7zCompr.exe
7zCompr.exe b -md22


2) To create compressed GUI 7-Zip:

7zr a g.7z 7zg.exe 7z.dll -mx
copy /b 7zS2.sfx + g.7z 7zgCompr.exe
7zgCompr.exe b -md22


3) To open some file:

7zr a h.7z readme.txt -mx
copy /b 7zS2.sfx + h.7z 7zTxt.exe 
7zTxt.exe
