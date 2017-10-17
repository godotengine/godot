#
# FreeType 2 smooth renderer module build rules
#


# Copyright 1996-2017 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# smooth driver directory
#
SMOOTH_DIR := $(SRC_DIR)/smooth


# compilation flags for the driver
#
SMOOTH_COMPILE := $(CC) $(ANSIFLAGS)                               \
                        $I$(subst /,$(COMPILER_SEP),$(SMOOTH_DIR)) \
                        $(INCLUDE_FLAGS)                           \
                        $(FT_CFLAGS)


# smooth driver sources (i.e., C files)
#
SMOOTH_DRV_SRC := $(SMOOTH_DIR)/ftgrays.c  \
                  $(SMOOTH_DIR)/ftsmooth.c \
                  $(SMOOTH_DIR)/ftspic.c


# smooth driver headers
#
SMOOTH_DRV_H := $(SMOOTH_DRV_SRC:%c=%h)  \
                $(SMOOTH_DIR)/ftsmerrs.h


# smooth driver object(s)
#
#   SMOOTH_DRV_OBJ_M is used during `multi' builds.
#   SMOOTH_DRV_OBJ_S is used during `single' builds.
#
SMOOTH_DRV_OBJ_M := $(SMOOTH_DRV_SRC:$(SMOOTH_DIR)/%.c=$(OBJ_DIR)/%.$O)
SMOOTH_DRV_OBJ_S := $(OBJ_DIR)/smooth.$O

# smooth driver source file for single build
#
SMOOTH_DRV_SRC_S := $(SMOOTH_DIR)/smooth.c


# smooth driver - single object
#
$(SMOOTH_DRV_OBJ_S): $(SMOOTH_DRV_SRC_S) $(SMOOTH_DRV_SRC) \
                     $(FREETYPE_H) $(SMOOTH_DRV_H)
	$(SMOOTH_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(SMOOTH_DRV_SRC_S))


# smooth driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(SMOOTH_DIR)/%.c $(FREETYPE_H) $(SMOOTH_DRV_H)
	$(SMOOTH_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(SMOOTH_DRV_OBJ_S)
DRV_OBJS_M += $(SMOOTH_DRV_OBJ_M)


# EOF
