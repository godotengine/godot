PROG = 7zS.sfx
MY_FIXED = 1

CFLAGS = $(CFLAGS) \
  -DZ7_NO_REGISTRY \
  -DZ7_EXTRACT_ONLY \
  -DZ7_NO_READ_FROM_CODER \
  -DZ7_SFX \
  -DZ7_NO_CRYPTO \
  -DZ7_NO_LONG_PATH \
  -DZ7_NO_LARGE_PAGES \

CURRENT_OBJS = \
  $O\SfxSetup.obj \
  $O\ExtractCallbackSfx.obj \
  $O\ExtractEngine.obj \

COMMON_OBJS = \
  $O\CommandLineParser.obj \
  $O\CRC.obj \
  $O\IntToString.obj \
  $O\NewHandler.obj \
  $O\MyString.obj \
  $O\StringConvert.obj \
  $O\TextConfig.obj \
  $O\UTFConvert.obj \
  $O\MyVector.obj \
  $O\Wildcard.obj \

WIN_OBJS = \
  $O\DLL.obj \
  $O\ErrorMsg.obj \
  $O\FileDir.obj \
  $O\FileFind.obj \
  $O\FileIO.obj \
  $O\FileName.obj \
  $O\PropVariant.obj \
  $O\ResourceString.obj \
  $O\Synchronization.obj \
  $O\System.obj \
  $O\TimeUtils.obj \
  $O\Window.obj \

WIN_CTRL_OBJS = \
  $O\Dialog.obj \

7ZIP_COMMON_OBJS = \
  $O\CreateCoder.obj \
  $O\CWrappers.obj \
  $O\FileStreams.obj \
  $O\InBuffer.obj \
  $O\FilterCoder.obj \
  $O\LimitedStreams.obj \
  $O\OutBuffer.obj \
  $O\ProgressUtils.obj \
  $O\PropId.obj \
  $O\StreamBinder.obj \
  $O\StreamObjects.obj \
  $O\StreamUtils.obj \
  $O\VirtThread.obj \

UI_COMMON_OBJS = \
  $O\ArchiveOpenCallback.obj \
  $O\DefaultName.obj \
  $O\LoadCodecs.obj \
  $O\OpenArchive.obj \

EXPLORER_OBJS = \
  $O\MyMessages.obj \

FM_OBJS = \
  $O\FormatUtils.obj \
  $O\ProgressDialog.obj \

AR_COMMON_OBJS = \
  $O\CoderMixer2.obj \
  $O\ItemNameUtils.obj \
  $O\OutStreamWithCRC.obj \

7Z_OBJS = \
  $O\7zDecode.obj \
  $O\7zExtract.obj \
  $O\7zHandler.obj \
  $O\7zIn.obj \
  $O\7zRegister.obj \

COMPRESS_OBJS = \
  $O\Bcj2Coder.obj \
  $O\Bcj2Register.obj \
  $O\BcjCoder.obj \
  $O\BcjRegister.obj \
  $O\BranchMisc.obj \
  $O\BranchRegister.obj \
  $O\CopyCoder.obj \
  $O\CopyRegister.obj \
  $O\DeltaFilter.obj \
  $O\Lzma2Decoder.obj \
  $O\Lzma2Register.obj \
  $O\LzmaDecoder.obj \
  $O\LzmaRegister.obj \

C_OBJS = \
  $O\7zStream.obj \
  $O\Alloc.obj \
  $O\Bcj2.obj \
  $O\Bra.obj \
  $O\Bra86.obj \
  $O\BraIA64.obj \
  $O\CpuArch.obj \
  $O\Delta.obj \
  $O\DllSecur.obj \
  $O\Lzma2Dec.obj \
  $O\Lzma2DecMt.obj \
  $O\LzmaDec.obj \
  $O\MtDec.obj \
  $O\Threads.obj \

!include "../../Crc.mak"
!include "../../LzmaDec.mak"

!include "../../7zip.mak"
