MY_STATIC_LINK=1
SLIB = sLZMA.lib
PROG = LZMA.dll
SLIBPATH = $O\$(SLIB)

DEF_FILE = LzmaLib.def
CFLAGS = $(CFLAGS) \

LIB_OBJS = \
  $O\LzmaLibExports.obj \

C_OBJS = \
  $O\Alloc.obj \
  $O\CpuArch.obj \
  $O\LzFind.obj \
  $O\LzFindMt.obj \
  $O\LzmaDec.obj \
  $O\LzmaEnc.obj \
  $O\LzmaLib.obj \
  $O\Threads.obj \

!include "../../../CPP/7zip/LzFindOpt.mak"
!include "../../../CPP/7zip/LzmaDec.mak"

OBJS = \
  $O\Precomp.obj \
  $(LIB_OBJS) \
  $(C_OBJS) \
  $(ASM_OBJS) \
  $O\resource.res

!include "../../../CPP/Build.mak"

$(SLIBPATH): $O $(OBJS)
	lib -out:$(SLIBPATH) $(OBJS) $(LIBS)


MAK_SINGLE_FILE = 1

$O\Precomp.obj: Precomp.c
	$(CCOMPL_PCH)

!IFDEF MAK_SINGLE_FILE

$(LIB_OBJS): $(*B).c
	$(CCOMPL_USE)
$(C_OBJS): ../../$(*B).c
	$(CCOMPL_USE)

!ELSE

{.}.c{$O}.obj::
	$(CCOMPLB_USE)
{../../../C}.c{$O}.obj::
	$(CCOMPLB_USE)

!ENDIF

!include "../../Asm_c.mak"
