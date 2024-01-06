##############################################################################
# LuaJIT top level Makefile for installation. Requires GNU Make.
#
# Please read doc/install.html before changing any variables!
#
# Suitable for POSIX platforms (Linux, *BSD, OSX etc.).
# Note: src/Makefile has many more configurable options.
#
# ##### This Makefile is NOT useful for Windows! #####
# For MSVC, please follow the instructions given in src/msvcbuild.bat.
# For MinGW and Cygwin, cd to src and run make with the Makefile there.
#
# Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
##############################################################################

MAJVER=  2
MINVER=  1
ABIVER=  5.1

# LuaJIT uses rolling releases. The release version is based on the time of
# the latest git commit. The 'git' command must be available during the build.
RELVER= $(shell cat src/luajit_relver.txt 2>/dev/null || : )
# Note: setting it with := doesn't work, since it will change during the build.

MMVERSION= $(MAJVER).$(MINVER)
VERSION= $(MMVERSION).$(RELVER)

##############################################################################
#
# Change the installation path as needed. This automatically adjusts
# the paths in src/luaconf.h, too. Note: PREFIX must be an absolute path!
#
export PREFIX= /usr/local
export MULTILIB= lib
##############################################################################

DPREFIX= $(DESTDIR)$(PREFIX)
INSTALL_BIN=   $(DPREFIX)/bin
INSTALL_LIB=   $(DPREFIX)/$(MULTILIB)
INSTALL_SHARE= $(DPREFIX)/share
INSTALL_DEFINC= $(DPREFIX)/include/luajit-$(MMVERSION)
INSTALL_INC=   $(INSTALL_DEFINC)

INSTALL_LJLIBD= $(INSTALL_SHARE)/luajit-$(MMVERSION)
INSTALL_JITLIB= $(INSTALL_LJLIBD)/jit
INSTALL_LMODD= $(INSTALL_SHARE)/lua
INSTALL_LMOD= $(INSTALL_LMODD)/$(ABIVER)
INSTALL_CMODD= $(INSTALL_LIB)/lua
INSTALL_CMOD= $(INSTALL_CMODD)/$(ABIVER)
INSTALL_MAN= $(INSTALL_SHARE)/man/man1
INSTALL_PKGCONFIG= $(INSTALL_LIB)/pkgconfig

INSTALL_TNAME= luajit-$(VERSION)
INSTALL_TSYMNAME= luajit
INSTALL_ANAME= libluajit-$(ABIVER).a
INSTALL_SOSHORT1= libluajit-$(ABIVER).so
INSTALL_SOSHORT2= libluajit-$(ABIVER).so.$(MAJVER)
INSTALL_SONAME= libluajit-$(ABIVER).so.$(VERSION)
INSTALL_DYLIBSHORT1= libluajit-$(ABIVER).dylib
INSTALL_DYLIBSHORT2= libluajit-$(ABIVER).$(MAJVER).dylib
INSTALL_DYLIBNAME= libluajit-$(ABIVER).$(VERSION).dylib
INSTALL_PCNAME= luajit.pc

INSTALL_STATIC= $(INSTALL_LIB)/$(INSTALL_ANAME)
INSTALL_DYN= $(INSTALL_LIB)/$(INSTALL_SONAME)
INSTALL_SHORT1= $(INSTALL_LIB)/$(INSTALL_SOSHORT1)
INSTALL_SHORT2= $(INSTALL_LIB)/$(INSTALL_SOSHORT2)
INSTALL_T= $(INSTALL_BIN)/$(INSTALL_TNAME)
INSTALL_TSYM= $(INSTALL_BIN)/$(INSTALL_TSYMNAME)
INSTALL_PC= $(INSTALL_PKGCONFIG)/$(INSTALL_PCNAME)

INSTALL_DIRS= $(INSTALL_BIN) $(INSTALL_LIB) $(INSTALL_INC) $(INSTALL_MAN) \
  $(INSTALL_PKGCONFIG) $(INSTALL_JITLIB) $(INSTALL_LMOD) $(INSTALL_CMOD)
UNINSTALL_DIRS= $(INSTALL_JITLIB) $(INSTALL_LJLIBD) $(INSTALL_INC) \
  $(INSTALL_LMOD) $(INSTALL_LMODD) $(INSTALL_CMOD) $(INSTALL_CMODD)

RM= rm -f
MKDIR= mkdir -p
RMDIR= rmdir 2>/dev/null
SYMLINK= ln -sf
INSTALL_X= install -m 0755
INSTALL_F= install -m 0644
UNINSTALL= $(RM)
LDCONFIG= ldconfig -n 2>/dev/null
SED_PC= sed -e "s|^prefix=.*|prefix=$(PREFIX)|" \
	    -e "s|^multilib=.*|multilib=$(MULTILIB)|" \
	    -e "s|^relver=.*|relver=$(RELVER)|"
ifneq ($(INSTALL_DEFINC),$(INSTALL_INC))
  SED_PC+= -e "s|^includedir=.*|includedir=$(INSTALL_INC)|"
endif

FILE_T= luajit
FILE_A= libluajit.a
FILE_SO= libluajit.so
FILE_MAN= luajit.1
FILE_PC= luajit.pc
FILES_INC= lua.h lualib.h lauxlib.h luaconf.h lua.hpp luajit.h
FILES_JITLIB= bc.lua bcsave.lua dump.lua p.lua v.lua zone.lua \
	      dis_x86.lua dis_x64.lua dis_arm.lua dis_arm64.lua \
	      dis_arm64be.lua dis_ppc.lua dis_mips.lua dis_mipsel.lua \
	      dis_mips64.lua dis_mips64el.lua \
	      dis_mips64r6.lua dis_mips64r6el.lua \
		  dis_riscv.lua dis_riscv64.lua \
	      vmdef.lua

ifeq (,$(findstring Windows,$(OS)))
  HOST_SYS:= $(shell uname -s)
else
  HOST_SYS= Windows
endif
TARGET_SYS?= $(HOST_SYS)

ifeq (Darwin,$(TARGET_SYS))
  INSTALL_SONAME= $(INSTALL_DYLIBNAME)
  INSTALL_SOSHORT1= $(INSTALL_DYLIBSHORT1)
  INSTALL_SOSHORT2= $(INSTALL_DYLIBSHORT2)
  LDCONFIG= :
endif

##############################################################################

INSTALL_DEP= src/luajit

default all $(INSTALL_DEP):
	@echo "==== Building LuaJIT $(MMVERSION) ===="
	$(MAKE) -C src
	@echo "==== Successfully built LuaJIT $(MMVERSION) ===="

install: $(INSTALL_DEP)
	@echo "==== Installing LuaJIT $(VERSION) to $(PREFIX) ===="
	$(MKDIR) $(INSTALL_DIRS)
	cd src && $(INSTALL_X) $(FILE_T) $(INSTALL_T)
	cd src && test -f $(FILE_A) && $(INSTALL_F) $(FILE_A) $(INSTALL_STATIC) || :
	$(RM) $(INSTALL_DYN) $(INSTALL_SHORT1) $(INSTALL_SHORT2)
	cd src && test -f $(FILE_SO) && \
	  $(INSTALL_X) $(FILE_SO) $(INSTALL_DYN) && \
	  ( $(LDCONFIG) $(INSTALL_LIB) || : ) && \
	  $(SYMLINK) $(INSTALL_SONAME) $(INSTALL_SHORT1) && \
	  $(SYMLINK) $(INSTALL_SONAME) $(INSTALL_SHORT2) || :
	cd etc && $(INSTALL_F) $(FILE_MAN) $(INSTALL_MAN)
	cd etc && $(SED_PC) $(FILE_PC) > $(FILE_PC).tmp && \
	  $(INSTALL_F) $(FILE_PC).tmp $(INSTALL_PC) && \
	  $(RM) $(FILE_PC).tmp
	cd src && $(INSTALL_F) $(FILES_INC) $(INSTALL_INC)
	cd src/jit && $(INSTALL_F) $(FILES_JITLIB) $(INSTALL_JITLIB)
	$(SYMLINK) $(INSTALL_TNAME) $(INSTALL_TSYM)
	@echo "==== Successfully installed LuaJIT $(VERSION) to $(PREFIX) ===="

uninstall:
	@echo "==== Uninstalling LuaJIT $(VERSION) from $(PREFIX) ===="
	$(UNINSTALL) $(INSTALL_TSYM) $(INSTALL_T) $(INSTALL_STATIC) $(INSTALL_DYN) $(INSTALL_SHORT1) $(INSTALL_SHORT2) $(INSTALL_MAN)/$(FILE_MAN) $(INSTALL_PC)
	for file in $(FILES_JITLIB); do \
	  $(UNINSTALL) $(INSTALL_JITLIB)/$$file; \
	  done
	for file in $(FILES_INC); do \
	  $(UNINSTALL) $(INSTALL_INC)/$$file; \
	  done
	$(LDCONFIG) $(INSTALL_LIB)
	$(RMDIR) $(UNINSTALL_DIRS) || :
	@echo "==== Successfully uninstalled LuaJIT $(VERSION) from $(PREFIX) ===="

##############################################################################

amalg:
	@echo "==== Building LuaJIT $(MMVERSION) (amalgamation) ===="
	$(MAKE) -C src amalg
	@echo "==== Successfully built LuaJIT $(MMVERSION) (amalgamation) ===="

clean:
	$(MAKE) -C src clean

.PHONY: all install amalg clean

##############################################################################
