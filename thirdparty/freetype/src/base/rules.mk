#
# FreeType 2 base layer configuration rules
#


# Copyright 1996-2018 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# It sets the following variables which are used by the master Makefile
# after the call:
#
#   BASE_OBJ_S:   The single-object base layer.
#   BASE_OBJ_M:   A list of all objects for a multiple-objects build.
#   BASE_EXT_OBJ: A list of base layer extensions, i.e., components found
#                 in `src/base' which are not compiled within the base
#                 layer proper.


BASE_COMPILE := $(CC) $(ANSIFLAGS)                             \
                      $I$(subst /,$(COMPILER_SEP),$(BASE_DIR)) \
                      $(INCLUDE_FLAGS)                         \
                      $(FT_CFLAGS)


# Base layer sources
#
#   ftsystem, ftinit, and ftdebug are handled by freetype.mk
#
# All files listed here should be included in `ftbase.c' (for a `single'
# build).
#
BASE_SRC := $(BASE_DIR)/basepic.c  \
            $(BASE_DIR)/ftadvanc.c \
            $(BASE_DIR)/ftcalc.c   \
            $(BASE_DIR)/ftdbgmem.c \
            $(BASE_DIR)/ftfntfmt.c \
            $(BASE_DIR)/ftgloadr.c \
            $(BASE_DIR)/fthash.c   \
            $(BASE_DIR)/ftlcdfil.c \
            $(BASE_DIR)/ftobjs.c   \
            $(BASE_DIR)/ftoutln.c  \
            $(BASE_DIR)/ftpic.c    \
            $(BASE_DIR)/ftpsprop.c \
            $(BASE_DIR)/ftrfork.c  \
            $(BASE_DIR)/ftsnames.c \
            $(BASE_DIR)/ftstream.c \
            $(BASE_DIR)/fttrigon.c \
            $(BASE_DIR)/ftutil.c


ifneq ($(ftmac_c),)
  BASE_SRC += $(BASE_DIR)/$(ftmac_c)
endif

# for simplicity, we also handle `md5.c' (which gets included by `ftobjs.h')
BASE_H := $(BASE_DIR)/basepic.h \
          $(BASE_DIR)/ftbase.h  \
          $(BASE_DIR)/md5.c     \
          $(BASE_DIR)/md5.h

# Base layer `extensions' sources
#
# An extension is added to the library file as a separate object.  It is
# then linked to the final executable only if one of its symbols is used by
# the application.
#
BASE_EXT_SRC := $(patsubst %,$(BASE_DIR)/%,$(BASE_EXTENSIONS))

# Default extensions objects
#
BASE_EXT_OBJ := $(BASE_EXT_SRC:$(BASE_DIR)/%.c=$(OBJ_DIR)/%.$O)


# Base layer object(s)
#
#   BASE_OBJ_M is used during `multi' builds (each base source file compiles
#   to a single object file).
#
#   BASE_OBJ_S is used during `single' builds (the whole base layer is
#   compiled as a single object file using ftbase.c).
#
BASE_OBJ_M := $(BASE_SRC:$(BASE_DIR)/%.c=$(OBJ_DIR)/%.$O)
BASE_OBJ_S := $(OBJ_DIR)/ftbase.$O

# Base layer root source file for single build
#
BASE_SRC_S := $(BASE_DIR)/ftbase.c


# Base layer - single object build
#
$(BASE_OBJ_S): $(BASE_SRC_S) $(BASE_SRC) $(FREETYPE_H) $(BASE_H)
	$(BASE_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(BASE_SRC_S))


# Multiple objects build + extensions
#
$(OBJ_DIR)/%.$O: $(BASE_DIR)/%.c $(FREETYPE_H) $(BASE_H)
	$(BASE_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# EOF
