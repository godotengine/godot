#
# FreeType 2 TrueType driver configuration rules
#


# Copyright (C) 1996-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# TrueType driver directory
#
TT_DIR := $(SRC_DIR)/truetype


# compilation flags for the driver
#
TT_COMPILE := $(CC) $(ANSIFLAGS)                           \
                    $I$(subst /,$(COMPILER_SEP),$(TT_DIR)) \
                    $(INCLUDE_FLAGS)                       \
                    $(FT_CFLAGS)


# TrueType driver sources (i.e., C files)
#
TT_DRV_SRC := $(TT_DIR)/ttdriver.c \
              $(TT_DIR)/ttgload.c  \
              $(TT_DIR)/ttgxvar.c  \
              $(TT_DIR)/ttinterp.c \
              $(TT_DIR)/ttobjs.c   \
              $(TT_DIR)/ttpload.c  \
              $(TT_DIR)/ttsubpix.c

# TrueType driver headers
#
TT_DRV_H := $(TT_DRV_SRC:%.c=%.h) \
            $(TT_DIR)/tterrors.h


# TrueType driver object(s)
#
#   TT_DRV_OBJ_M is used during `multi' builds
#   TT_DRV_OBJ_S is used during `single' builds
#
TT_DRV_OBJ_M := $(TT_DRV_SRC:$(TT_DIR)/%.c=$(OBJ_DIR)/%.$O)
TT_DRV_OBJ_S := $(OBJ_DIR)/truetype.$O

# TrueType driver source file for single build
#
TT_DRV_SRC_S := $(TT_DIR)/truetype.c


# TrueType driver - single object
#
$(TT_DRV_OBJ_S): $(TT_DRV_SRC_S) $(TT_DRV_SRC) $(FREETYPE_H) $(TT_DRV_H)
	$(TT_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(TT_DRV_SRC_S))


# driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(TT_DIR)/%.c $(FREETYPE_H) $(TT_DRV_H)
	$(TT_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(TT_DRV_OBJ_S)
DRV_OBJS_M += $(TT_DRV_OBJ_M)


# EOF
