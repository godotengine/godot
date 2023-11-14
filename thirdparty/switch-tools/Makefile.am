# Makefile.am -- Process this file with automake to produce Makefile.in
bin_PROGRAMS = elf2nso elf2nro elf2kip build_pfs0 build_romfs nacptool npdmtool nxlink

build_pfs0_SOURCES	=	src/build_pfs0.c src/types.h

build_romfs_SOURCES = src/build_romfs.c src/romfs.c src/romfs.h src/filepath.c src/filepath.h src/types.h

elf2nro_SOURCES	=	src/elf2nro.c src/elf64.h src/romfs.c src/filepath.c src/filepath.h src/romfs.h src/elf_common.h

elf2nso_SOURCES	=	src/elf2nso.c src/sha256.c src/sha256.h src/elf64.h src/elf_common.h

elf2kip_SOURCES	=	src/elf2kip.c src/cJSON.c src/cJSON.h src/blz.c src/blz.h src/elf64.h src/elf_common.h

nacptool_SOURCES	=	src/nacptool.c

npdmtool_SOURCES	=	src/npdmtool.c src/cJSON.c src/cJSON.h

nxlink_SOURCES = src/nxlink.c

nxlink_CPPFLAGS = @ZLIB_CFLAGS@
nxlink_LDADD = @ZLIB_LIBS@ @NET_LIBS@

elf2nso_CPPFLAGS = @LZ4_CFLAGS@
elf2nso_LDADD = @LZ4_LIBS@

elf2nro_CPPFLAGS = @LZ4_CFLAGS@
elf2nro_LDADD = @LZ4_LIBS@


EXTRA_DIST = autogen.sh
