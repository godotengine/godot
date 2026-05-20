PROG = 7zS2.sfx
MY_FIXED = 1

CFLAGS = $(CFLAGS) \
  -DZ7_EXTRACT_ONLY \

C_OBJS = \
  $O\7zAlloc.obj \
  $O\7zArcIn.obj \
  $O\7zBuf.obj \
  $O\7zBuf2.obj \
  $O\7zFile.obj \
  $O\7zDec.obj \
  $O\7zStream.obj \
  $O\Bcj2.obj \
  $O\Bra.obj \
  $O\Bra86.obj \
  $O\BraIA64.obj \
  $O\CpuArch.obj \
  $O\Delta.obj \
  $O\DllSecur.obj \
  $O\Lzma2Dec.obj \
  $O\LzmaDec.obj \

7Z_OBJS = \
  $O\SfxSetup.obj \

!include "../../../CPP/7zip/Crc.mak"
# !include "../../../CPP/7zip/LzmaDec.mak"

OBJS = \
  $(7Z_OBJS) \
  $(C_OBJS) \
  $(ASM_OBJS) \
  $O\resource.res

!include "../../../CPP/Build.mak"

$(7Z_OBJS): $(*B).c
	$(COMPL_O1)
$(C_OBJS): ../../$(*B).c
	$(COMPL_O1)

!include "../../Asm_c.mak"
