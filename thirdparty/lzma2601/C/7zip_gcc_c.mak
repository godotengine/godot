
MY_ARCH_2 = $(MY_ARCH)

MY_ASM = jwasm
MY_ASM = asmc

ifndef RC
#RC=windres.exe --target=pe-x86-64
#RC=windres.exe -F pe-i386
RC=windres.exe
endif

PROGPATH = $(O)/$(PROG)
PROGPATH_STATIC = $(O)/$(PROG)s

ifneq ($(CC), xlc)
CFLAGS_WARN_WALL = -Wall -Werror -Wextra
endif

# for object file
CFLAGS_BASE_LIST = -c
# for ASM file
# CFLAGS_BASE_LIST = -S

FLAGS_FLTO = -flto
FLAGS_FLTO =

CFLAGS_BASE = $(MY_ARCH_2) -O2 $(CFLAGS_BASE_LIST) $(CFLAGS_WARN_WALL) $(CFLAGS_WARN) \
 -DNDEBUG -D_REENTRANT -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE


ifdef SystemDrive
IS_MINGW = 1
else
ifdef SYSTEMDRIVE
# ifdef OS
IS_MINGW = 1
endif
endif

ifdef IS_MINGW
LDFLAGS_STATIC_2 = -static
else
ifndef DEF_FILE
ifndef IS_NOT_STANDALONE
ifndef MY_DYNAMIC_LINK
ifneq ($(CC), clang)
LDFLAGS_STATIC_2 =
# -static
# -static-libstdc++ -static-libgcc
endif
endif
endif
endif
endif

LDFLAGS_STATIC = -DNDEBUG $(LDFLAGS_STATIC_2)

ifdef DEF_FILE


ifdef IS_MINGW
SHARED_EXT=.dll
LDFLAGS = -shared -DEF $(DEF_FILE) $(LDFLAGS_STATIC)
else
SHARED_EXT=.so
LDFLAGS = -shared -fPIC  $(LDFLAGS_STATIC)
CC_SHARED=-fPIC
endif


else

LDFLAGS = $(LDFLAGS_STATIC)
# -s is not required for clang, do we need it for GGC ???
# -s

#-static -static-libgcc -static-libstdc++

ifdef IS_MINGW
SHARED_EXT=.exe
else
SHARED_EXT=
endif

endif


PROGPATH = $(O)/$(PROG)$(SHARED_EXT)
PROGPATH_STATIC = $(O)/$(PROG)s$(SHARED_EXT)
	
ifndef O
O=_o
endif

ifdef IS_MINGW

ifdef MSYSTEM
RM = rm -f
MY_MKDIR=mkdir -p
DEL_OBJ_EXE = -$(RM) $(PROGPATH) $(PROGPATH_STATIC) $(OBJS)
else
RM = del
MY_MKDIR=mkdir
DEL_OBJ_EXE = -$(RM) $(O)\*.o $(O)\$(PROG).exe $(O)\$(PROG).dll
endif


LIB2 = -lole32 -loleaut32 -luuid -ladvapi32 -luser32 -lshell32

CFLAGS_EXTRA = -DUNICODE -D_UNICODE
# -Wno-delete-non-virtual-dtor

 
else

RM = rm -f
MY_MKDIR=mkdir -p
# CFLAGS_BASE := $(CFLAGS_BASE) -DZ7_ST
# CFLAGS_EXTRA = -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE

# LOCAL_LIBS=-lpthread
# LOCAL_LIBS_DLL=$(LOCAL_LIBS) -ldl
LIB2 = -lpthread -ldl

DEL_OBJ_EXE = -$(RM) $(PROGPATH) $(PROGPATH_STATIC) $(OBJS)

endif


ifdef IS_X64
AFLAGS_ABI = -elf64 -DABI_LINUX
else
AFLAGS_ABI = -elf -DABI_LINUX -DABI_CDECL
# -DABI_CDECL
# -DABI_LINUX
# -DABI_CDECL
endif
AFLAGS = $(AFLAGS_ABI) -Fo$(O)/

C_WARN_FLAGS =

CFLAGS = $(LOCAL_FLAGS) $(CFLAGS_BASE2) $(CFLAGS_BASE) $(CFLAGS_EXTRA) $(C_WARN_FLAGS) $(FLAGS_FLTO) $(CC_SHARED) -o $@

STATIC_TARGET=
ifdef COMPL_STATIC
STATIC_TARGET=$(PROGPATH_STATIC)
endif


all: $(O) $(PROGPATH) $(STATIC_TARGET)

$(O):
	$(MY_MKDIR) $(O)

ifneq ($(CC), $(CROSS_COMPILE)clang)
LFLAGS_STRIP = -s
endif

LFLAGS_ALL = $(LFLAGS_STRIP) $(MY_ARCH_2) $(LDFLAGS) $(FLAGS_FLTO) $(LD_arch) $(OBJS) $(MY_LIBS) $(LIB2)
$(PROGPATH): $(OBJS)
	$(CC) -o $(PROGPATH) $(LFLAGS_ALL)

$(PROGPATH_STATIC): $(OBJS)
	$(CC) -static -o $(PROGPATH_STATIC) $(LFLAGS_ALL)


ifndef NO_DEFAULT_RES
# old mingw without -FO
# windres.exe $(RFLAGS) resource.rc $O/resource.o
$O/resource.o: resource.rc
	$(RC) $(RFLAGS) resource.rc $(O)/resource.o
endif
# windres.exe $(RFLAGS) resource.rc $(O)\resource.o
# windres.exe $(RFLAGS) resource.rc -FO $(O)/resource.o
# $(RC) $(RFLAGS) resource.rc -FO $(O)/resource.o



$O/7zAlloc.o: ../../../C/7zAlloc.c
	$(CC) $(CFLAGS) $<
$O/7zArcIn.o: ../../../C/7zArcIn.c
	$(CC) $(CFLAGS) $<
$O/7zBuf.o: ../../../C/7zBuf.c
	$(CC) $(CFLAGS) $<
$O/7zBuf2.o: ../../../C/7zBuf2.c
	$(CC) $(CFLAGS) $<
$O/7zCrc.o: ../../../C/7zCrc.c
	$(CC) $(CFLAGS) $<
$O/7zDec.o: ../../../C/7zDec.c
	$(CC) $(CFLAGS) $<
$O/7zFile.o: ../../../C/7zFile.c
	$(CC) $(CFLAGS) $<
$O/7zStream.o: ../../../C/7zStream.c
	$(CC) $(CFLAGS) $<
$O/Aes.o: ../../../C/Aes.c
	$(CC) $(CFLAGS) $<
$O/Alloc.o: ../../../C/Alloc.c
	$(CC) $(CFLAGS) $<
$O/Bcj2.o: ../../../C/Bcj2.c
	$(CC) $(CFLAGS) $<
$O/Bcj2Enc.o: ../../../C/Bcj2Enc.c
	$(CC) $(CFLAGS) $<
$O/Blake2s.o: ../../../C/Blake2s.c
	$(CC) $(CFLAGS) $<
$O/Bra.o: ../../../C/Bra.c
	$(CC) $(CFLAGS) $<
$O/Bra86.o: ../../../C/Bra86.c
	$(CC) $(CFLAGS) $<
$O/BraIA64.o: ../../../C/BraIA64.c
	$(CC) $(CFLAGS) $<
$O/BwtSort.o: ../../../C/BwtSort.c
	$(CC) $(CFLAGS) $<

$O/CpuArch.o: ../../../C/CpuArch.c
	$(CC) $(CFLAGS) $<
$O/Delta.o: ../../../C/Delta.c
	$(CC) $(CFLAGS) $<
$O/DllSecur.o: ../../../C/DllSecur.c
	$(CC) $(CFLAGS) $<
$O/HuffEnc.o: ../../../C/HuffEnc.c
	$(CC) $(CFLAGS) $<
$O/LzFind.o: ../../../C/LzFind.c
	$(CC) $(CFLAGS) $<

# ifdef MT_FILES
$O/LzFindMt.o: ../../../C/LzFindMt.c
	$(CC) $(CFLAGS) $<
$O/LzFindOpt.o: ../../../C/LzFindOpt.c
	$(CC) $(CFLAGS) $<

$O/Threads.o: ../../../C/Threads.c
	$(CC) $(CFLAGS) $<
# endif

$O/LzmaEnc.o: ../../../C/LzmaEnc.c
	$(CC) $(CFLAGS) $<
$O/Lzma86Dec.o: ../../../C/Lzma86Dec.c
	$(CC) $(CFLAGS) $<
$O/Lzma86Enc.o: ../../../C/Lzma86Enc.c
	$(CC) $(CFLAGS) $<
$O/Lzma2Dec.o: ../../../C/Lzma2Dec.c
	$(CC) $(CFLAGS) $<
$O/Lzma2DecMt.o: ../../../C/Lzma2DecMt.c
	$(CC) $(CFLAGS) $<
$O/Lzma2Enc.o: ../../../C/Lzma2Enc.c
	$(CC) $(CFLAGS) $<
$O/LzmaLib.o: ../../../C/LzmaLib.c
	$(CC) $(CFLAGS) $<
$O/MtCoder.o: ../../../C/MtCoder.c
	$(CC) $(CFLAGS) $<
$O/MtDec.o: ../../../C/MtDec.c
	$(CC) $(CFLAGS) $<
$O/Ppmd7.o: ../../../C/Ppmd7.c
	$(CC) $(CFLAGS) $<
$O/Ppmd7aDec.o: ../../../C/Ppmd7aDec.c
	$(CC) $(CFLAGS) $<
$O/Ppmd7Dec.o: ../../../C/Ppmd7Dec.c
	$(CC) $(CFLAGS) $<
$O/Ppmd7Enc.o: ../../../C/Ppmd7Enc.c
	$(CC) $(CFLAGS) $<
$O/Ppmd8.o: ../../../C/Ppmd8.c
	$(CC) $(CFLAGS) $<
$O/Ppmd8Dec.o: ../../../C/Ppmd8Dec.c
	$(CC) $(CFLAGS) $<
$O/Ppmd8Enc.o: ../../../C/Ppmd8Enc.c
	$(CC) $(CFLAGS) $<
$O/Sha1.o: ../../../C/Sha1.c
	$(CC) $(CFLAGS) $<
$O/Sha256.o: ../../../C/Sha256.c
	$(CC) $(CFLAGS) $<
$O/Sort.o: ../../../C/Sort.c
	$(CC) $(CFLAGS) $<
$O/SwapBytes.o: ../../../C/SwapBytes.c
	$(CC) $(CFLAGS) $<
$O/Xz.o: ../../../C/Xz.c
	$(CC) $(CFLAGS) $<
$O/XzCrc64.o: ../../../C/XzCrc64.c
	$(CC) $(CFLAGS) $<
$O/XzDec.o: ../../../C/XzDec.c
	$(CC) $(CFLAGS) $<
$O/XzEnc.o: ../../../C/XzEnc.c
	$(CC) $(CFLAGS) $<
$O/XzIn.o: ../../../C/XzIn.c
	$(CC) $(CFLAGS) $<


ifdef USE_ASM
ifdef IS_X64
USE_X86_ASM=1
else
ifdef IS_X86
USE_X86_ASM=1
endif
endif
endif

ifdef USE_X86_ASM
$O/7zCrcOpt.o: ../../../Asm/x86/7zCrcOpt.asm
	$(MY_ASM) $(AFLAGS) $<
$O/XzCrc64Opt.o: ../../../Asm/x86/XzCrc64Opt.asm
	$(MY_ASM) $(AFLAGS) $<
$O/AesOpt.o: ../../../Asm/x86/AesOpt.asm
	$(MY_ASM) $(AFLAGS) $<
$O/Sha1Opt.o: ../../../Asm/x86/Sha1Opt.asm
	$(MY_ASM) $(AFLAGS) $<
$O/Sha256Opt.o: ../../../Asm/x86/Sha256Opt.asm
	$(MY_ASM) $(AFLAGS) $<
else
$O/7zCrcOpt.o: ../../7zCrcOpt.c
	$(CC) $(CFLAGS) $<
$O/XzCrc64Opt.o: ../../XzCrc64Opt.c
	$(CC) $(CFLAGS) $<
$O/Sha1Opt.o: ../../Sha1Opt.c
	$(CC) $(CFLAGS) $<
$O/Sha256Opt.o: ../../Sha256Opt.c
	$(CC) $(CFLAGS) $<
$O/AesOpt.o: ../../AesOpt.c
	$(CC) $(CFLAGS) $<
endif


ifdef USE_LZMA_DEC_ASM

ifdef IS_X64
$O/LzmaDecOpt.o: ../../../Asm/x86/LzmaDecOpt.asm
	$(MY_ASM) $(AFLAGS) $<
endif

ifdef IS_ARM64
$O/LzmaDecOpt.o: ../../../Asm/arm64/LzmaDecOpt.S ../../../Asm/arm64/7zAsm.S
	$(CC) $(CFLAGS) $(ASM_FLAGS) $<
endif

$O/LzmaDec.o: ../../LzmaDec.c
	$(CC) $(CFLAGS) -DZ7_LZMA_DEC_OPT $<

else

$O/LzmaDec.o: ../../LzmaDec.c
	$(CC) $(CFLAGS) $<

endif



$O/7zMain.o: ../../../C/Util/7z/7zMain.c
	$(CC) $(CFLAGS) $<
$O/7zipInstall.o: ../../../C/Util/7zipInstall/7zipInstall.c
	$(CC) $(CFLAGS) $<
$O/7zipUninstall.o: ../../../C/Util/7zipUninstall/7zipUninstall.c
	$(CC) $(CFLAGS) $<
$O/LzmaUtil.o: ../../../C/Util/Lzma/LzmaUtil.c
	$(CC) $(CFLAGS) $<
$O/XzUtil.o: ../../../C/Util/Xz/XzUtil.c
	$(CC) $(CFLAGS) $<


clean:
	-$(DEL_OBJ_EXE)
