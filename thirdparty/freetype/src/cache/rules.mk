#
# FreeType 2 Cache configuration rules
#


# Copyright (C) 2000-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# Cache driver directory
#
CACHE_DIR   := $(SRC_DIR)/cache


# compilation flags for the driver
#
CACHE_COMPILE := $(CC) $(ANSIFLAGS)                              \
                       $I$(subst /,$(COMPILER_SEP),$(CACHE_DIR)) \
                       $(INCLUDE_FLAGS)                          \
                       $(FT_CFLAGS)


# Cache driver sources (i.e., C files)
#
CACHE_DRV_SRC := $(CACHE_DIR)/ftcbasic.c \
                 $(CACHE_DIR)/ftccache.c \
                 $(CACHE_DIR)/ftccmap.c  \
                 $(CACHE_DIR)/ftcglyph.c \
                 $(CACHE_DIR)/ftcimage.c \
                 $(CACHE_DIR)/ftcmanag.c \
                 $(CACHE_DIR)/ftcmru.c   \
                 $(CACHE_DIR)/ftcsbits.c


# Cache driver headers
#
CACHE_DRV_H := $(CACHE_DIR)/ftccache.h \
               $(CACHE_DIR)/ftccback.h \
               $(CACHE_DIR)/ftcerror.h \
               $(CACHE_DIR)/ftcglyph.h \
               $(CACHE_DIR)/ftcimage.h \
               $(CACHE_DIR)/ftcmanag.h \
               $(CACHE_DIR)/ftcmru.h   \
               $(CACHE_DIR)/ftcsbits.h


# Cache driver object(s)
#
#   CACHE_DRV_OBJ_M is used during `multi' builds.
#   CACHE_DRV_OBJ_S is used during `single' builds.
#
CACHE_DRV_OBJ_M := $(CACHE_DRV_SRC:$(CACHE_DIR)/%.c=$(OBJ_DIR)/%.$O)
CACHE_DRV_OBJ_S := $(OBJ_DIR)/ftcache.$O

# Cache driver source file for single build
#
CACHE_DRV_SRC_S := $(CACHE_DIR)/ftcache.c


# Cache driver - single object
#
$(CACHE_DRV_OBJ_S): $(CACHE_DRV_SRC_S) $(CACHE_DRV_SRC) \
                   $(FREETYPE_H) $(CACHE_DRV_H)
	$(CACHE_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(CACHE_DRV_SRC_S))


# Cache driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(CACHE_DIR)/%.c $(FREETYPE_H) $(CACHE_DRV_H)
	$(CACHE_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(CACHE_DRV_OBJ_S)
DRV_OBJS_M += $(CACHE_DRV_OBJ_M)


# EOF
