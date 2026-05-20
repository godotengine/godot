PROG = lzma.exe
MY_CONSOLE = 1

CFLAGS = $(CFLAGS) -DZ7_NO_LONG_PATH
# CFLAGS = $(CFLAGS) -DZ7_ST


CURRENT_OBJS = \
  $O\LzmaAlone.obj \

COMPRESS_OBJS = \
  $O\LzmaDecoder.obj \
  $O\LzmaEncoder.obj \
  $O\LzmaRegister.obj \

COMMON_OBJS = \
  $O\CommandLineParser.obj \
  $O\CRC.obj \
  $O\CrcReg.obj \
  $O\IntToString.obj \
  $O\LzFindPrepare.obj \
  $O\MyString.obj \
  $O\MyVector.obj \
  $O\NewHandler.obj \
  $O\StringConvert.obj \
  $O\StringToInt.obj \
  $O\Wildcard.obj \

WIN_OBJS = \
  $O\FileIO.obj \
  $O\PropVariant.obj \
  $O\Registry.obj \
  $O\System.obj \
  $O\SystemInfo.obj \

7ZIP_COMMON_OBJS = \
  $O\CreateCoder.obj \
  $O\CWrappers.obj \
  $O\FileStreams.obj \
  $O\FilterCoder.obj \
  $O\MethodProps.obj \
  $O\StreamObjects.obj \
  $O\StreamUtils.obj \

UI_COMMON_OBJS = \
  $O\Bench.obj \

CONSOLE_OBJS = \
  $O\ConsoleClose.obj \
  $O\BenchCon.obj \

C_OBJS = \
  $O\Alloc.obj \
  $O\Bra86.obj \
  $O\CpuArch.obj \
  $O\LzFind.obj \
  $O\LzFindMt.obj \
  $O\Lzma86Dec.obj \
  $O\Lzma86Enc.obj \
  $O\LzmaDec.obj \
  $O\LzmaEnc.obj \
  $O\Threads.obj \

!include "../../Crc.mak"
!include "../../LzFindOpt.mak"
!include "../../LzmaDec.mak"

!include "../../7zip.mak"
