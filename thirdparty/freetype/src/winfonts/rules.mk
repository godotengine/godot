#
# FreeType 2 Windows FNT/FON driver configuration rules
#


# Copyright 1996-2016 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# Windows driver directory
#
FNT_DIR := $(SRC_DIR)/winfonts


FNT_COMPILE := $(CC) $(ANSIFLAGS)                            \
                     $I$(subst /,$(COMPILER_SEP),$(FNT_DIR)) \
                     $(INCLUDE_FLAGS)                        \
                     $(FT_CFLAGS)


# Windows driver sources (i.e., C files)
#
FNT_DRV_SRC := $(FNT_DIR)/winfnt.c

# Windows driver headers
#
FNT_DRV_H := $(FNT_DRV_SRC:%.c=%.h) \
             $(FNT_DIR)/fnterrs.h


# Windows driver object(s)
#
#   FNT_DRV_OBJ_M is used during `multi' builds
#   FNT_DRV_OBJ_S is used during `single' builds
#
FNT_DRV_OBJ_M := $(FNT_DRV_SRC:$(FNT_DIR)/%.c=$(OBJ_DIR)/%.$O)
FNT_DRV_OBJ_S := $(OBJ_DIR)/winfnt.$O

# Windows driver source file for single build
#
FNT_DRV_SRC_S := $(FNT_DIR)/winfnt.c


# Windows driver - single object
#
$(FNT_DRV_OBJ_S): $(FNT_DRV_SRC_S) $(FNT_DRV_SRC) $(FREETYPE_H) $(FNT_DRV_H)
	$(FNT_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(FNT_DRV_SRC_S))


# Windows driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(FNT_DIR)/%.c $(FREETYPE_H) $(FNT_DRV_H)
	$(FNT_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(FNT_DRV_OBJ_S)
DRV_OBJS_M += $(FNT_DRV_OBJ_M)


# EOF
