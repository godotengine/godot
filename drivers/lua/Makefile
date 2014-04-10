# makefile for installing Lua
# see INSTALL for installation instructions
# see src/Makefile and src/luaconf.h for further customization

# == CHANGE THE SETTINGS BELOW TO SUIT YOUR ENVIRONMENT =======================

# Your platform. See PLATS for possible values.
PLAT= none

# Where to install. The installation starts in the src and doc directories,
# so take care if INSTALL_TOP is not an absolute path.
INSTALL_TOP= /usr/local
INSTALL_BIN= $(INSTALL_TOP)/bin
INSTALL_INC= $(INSTALL_TOP)/include
INSTALL_LIB= $(INSTALL_TOP)/lib
INSTALL_MAN= $(INSTALL_TOP)/man/man1
#
# You probably want to make INSTALL_LMOD and INSTALL_CMOD consistent with
# LUA_ROOT, LUA_LDIR, and LUA_CDIR in luaconf.h (and also with etc/lua.pc).
INSTALL_LMOD= $(INSTALL_TOP)/share/lua/$V
INSTALL_CMOD= $(INSTALL_TOP)/lib/lua/$V

# How to install. If your install program does not support "-p", then you
# may have to run ranlib on the installed liblua.a (do "make ranlib").
INSTALL= install -p
INSTALL_EXEC= $(INSTALL) -m 0755
INSTALL_DATA= $(INSTALL) -m 0644
#
# If you don't have install you can use cp instead.
# INSTALL= cp -p
# INSTALL_EXEC= $(INSTALL)
# INSTALL_DATA= $(INSTALL)

# Utilities.
MKDIR= mkdir -p
RANLIB= ranlib

# == END OF USER SETTINGS. NO NEED TO CHANGE ANYTHING BELOW THIS LINE =========

# Convenience platforms targets.
PLATS= aix ansi bsd freebsd generic linux macosx mingw posix solaris

# What to install.
TO_BIN= lua luac
TO_INC= lua.h luaconf.h lualib.h lauxlib.h ../etc/lua.hpp
TO_LIB= liblua.a
TO_MAN= lua.1 luac.1

# Lua version and release.
V= 5.1
R= 5.1.5

all:	$(PLAT)

$(PLATS) clean:
	cd src && $(MAKE) $@

test:	dummy
	src/lua test/hello.lua

install: dummy
	cd src && $(MKDIR) $(INSTALL_BIN) $(INSTALL_INC) $(INSTALL_LIB) $(INSTALL_MAN) $(INSTALL_LMOD) $(INSTALL_CMOD)
	cd src && $(INSTALL_EXEC) $(TO_BIN) $(INSTALL_BIN)
	cd src && $(INSTALL_DATA) $(TO_INC) $(INSTALL_INC)
	cd src && $(INSTALL_DATA) $(TO_LIB) $(INSTALL_LIB)
	cd doc && $(INSTALL_DATA) $(TO_MAN) $(INSTALL_MAN)

ranlib:
	cd src && cd $(INSTALL_LIB) && $(RANLIB) $(TO_LIB)

local:
	$(MAKE) install INSTALL_TOP=..

none:
	@echo "Please do"
	@echo "   make PLATFORM"
	@echo "where PLATFORM is one of these:"
	@echo "   $(PLATS)"
	@echo "See INSTALL for complete instructions."

# make may get confused with test/ and INSTALL in a case-insensitive OS
dummy:

# echo config parameters
echo:
	@echo ""
	@echo "These are the parameters currently set in src/Makefile to build Lua $R:"
	@echo ""
	@cd src && $(MAKE) -s echo
	@echo ""
	@echo "These are the parameters currently set in Makefile to install Lua $R:"
	@echo ""
	@echo "PLAT = $(PLAT)"
	@echo "INSTALL_TOP = $(INSTALL_TOP)"
	@echo "INSTALL_BIN = $(INSTALL_BIN)"
	@echo "INSTALL_INC = $(INSTALL_INC)"
	@echo "INSTALL_LIB = $(INSTALL_LIB)"
	@echo "INSTALL_MAN = $(INSTALL_MAN)"
	@echo "INSTALL_LMOD = $(INSTALL_LMOD)"
	@echo "INSTALL_CMOD = $(INSTALL_CMOD)"
	@echo "INSTALL_EXEC = $(INSTALL_EXEC)"
	@echo "INSTALL_DATA = $(INSTALL_DATA)"
	@echo ""
	@echo "See also src/luaconf.h ."
	@echo ""

# echo private config parameters
pecho:
	@echo "V = $(V)"
	@echo "R = $(R)"
	@echo "TO_BIN = $(TO_BIN)"
	@echo "TO_INC = $(TO_INC)"
	@echo "TO_LIB = $(TO_LIB)"
	@echo "TO_MAN = $(TO_MAN)"

# echo config parameters as Lua code
# uncomment the last sed expression if you want nil instead of empty strings
lecho:
	@echo "-- installation parameters for Lua $R"
	@echo "VERSION = '$V'"
	@echo "RELEASE = '$R'"
	@$(MAKE) echo | grep = | sed -e 's/= /= "/' -e 's/$$/"/' #-e 's/""/nil/'
	@echo "-- EOF"

# list targets that do not create files (but not all makes understand .PHONY)
.PHONY: all $(PLATS) clean test install local none dummy echo pecho lecho

# (end of Makefile)
