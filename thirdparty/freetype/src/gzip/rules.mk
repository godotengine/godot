#
# FreeType 2 GZip support configuration rules
#


# Copyright (C) 2002-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# gzip driver directory
#
GZIP_DIR := $(SRC_DIR)/gzip


# compilation flags for the driver
#
ifeq ($(SYSTEM_ZLIB),)
  GZIP_COMPILE := $(CC) $(ANSIFLAGS)                             \
                        $I$(subst /,$(COMPILER_SEP),$(GZIP_DIR)) \
                        $(INCLUDE_FLAGS)                         \
                        $(FT_CFLAGS)
else
  GZIP_COMPILE := $(CC) $(ANSIFLAGS)     \
                        $(INCLUDE_FLAGS) \
                        $(FT_CFLAGS)
endif


# gzip support sources
#
# All source and header files get loaded by `ftgzip.c' only if SYSTEM_ZLIB
# is not defined (regardless whether we have a `single' or a `multi' build).
# However, it doesn't harm if we add everything as a dependency
# unconditionally.
#
GZIP_DRV_SRCS := $(GZIP_DIR)/adler32.c  \
                 $(GZIP_DIR)/ftzconf.h  \
                 $(GZIP_DIR)/infblock.c \
                 $(GZIP_DIR)/infblock.h \
                 $(GZIP_DIR)/infcodes.c \
                 $(GZIP_DIR)/infcodes.h \
                 $(GZIP_DIR)/inffixed.h \
                 $(GZIP_DIR)/inflate.c  \
                 $(GZIP_DIR)/inftrees.c \
                 $(GZIP_DIR)/inftrees.h \
                 $(GZIP_DIR)/infutil.c  \
                 $(GZIP_DIR)/infutil.h  \
                 $(GZIP_DIR)/zlib.h     \
                 $(GZIP_DIR)/zutil.c    \
                 $(GZIP_DIR)/zutil.h


# gzip driver object(s)
#
#   GZIP_DRV_OBJ is used during both `single' and `multi' builds
#
GZIP_DRV_OBJ := $(OBJ_DIR)/ftgzip.$O


# gzip main source file
#
GZIP_DRV_SRC := $(GZIP_DIR)/ftgzip.c


# gzip support - object
#
$(GZIP_DRV_OBJ): $(GZIP_DRV_SRC) $(GZIP_DRV_SRCS) $(FREETYPE_H)
	$(GZIP_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(GZIP_DRV_SRC))


# update main driver object lists
#
DRV_OBJS_S += $(GZIP_DRV_OBJ)
DRV_OBJS_M += $(GZIP_DRV_OBJ)


# EOF
