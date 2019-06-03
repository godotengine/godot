#
# FreeType 2 BZIP2 support configuration rules
#

# Copyright (C) 2010-2019 by
# Joel Klinghed.
#
# based on `src/lzw/rules.mk'
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# BZIP2 driver directory
#
BZIP2_DIR := $(SRC_DIR)/bzip2


# compilation flags for the driver
#
BZIP2_COMPILE := $(CC) $(ANSIFLAGS)     \
                       $(INCLUDE_FLAGS) \
                       $(FT_CFLAGS)


# BZIP2 support sources (i.e., C files)
#
BZIP2_DRV_SRC := $(BZIP2_DIR)/ftbzip2.c

# BZIP2 driver object(s)
#
#   BZIP2_DRV_OBJ_M is used during `multi' builds
#   BZIP2_DRV_OBJ_S is used during `single' builds
#
BZIP2_DRV_OBJ_M := $(OBJ_DIR)/ftbzip2.$O
BZIP2_DRV_OBJ_S := $(OBJ_DIR)/ftbzip2.$O

# BZIP2 support source file for single build
#
BZIP2_DRV_SRC_S := $(BZIP2_DIR)/ftbzip2.c


# BZIP2 support - single object
#
$(BZIP2_DRV_OBJ_S): $(BZIP2_DRV_SRC_S) $(BZIP2_DRV_SRC) $(FREETYPE_H) $(BZIP2_DRV_H)
	$(BZIP2_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(BZIP2_DRV_SRC_S))


# BZIP2 support - multiple objects
#
$(OBJ_DIR)/%.$O: $(BZIP2_DIR)/%.c $(FREETYPE_H) $(BZIP2_DRV_H)
	$(BZIP2_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(BZIP2_DRV_OBJ_S)
DRV_OBJS_M += $(BZIP2_DRV_OBJ_M)


# EOF
