#
# FreeType 2 Type1 driver configuration rules
#


# Copyright 1996-2016 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# Type1 driver directory
#
T1_DIR := $(SRC_DIR)/type1


# compilation flags for the driver
#
T1_COMPILE := $(CC) $(ANSIFLAGS)                           \
                    $I$(subst /,$(COMPILER_SEP),$(T1_DIR)) \
                    $(INCLUDE_FLAGS)                       \
                    $(FT_CFLAGS)


# Type1 driver sources (i.e., C files)
#
T1_DRV_SRC := $(T1_DIR)/t1parse.c  \
              $(T1_DIR)/t1load.c   \
              $(T1_DIR)/t1driver.c \
              $(T1_DIR)/t1afm.c    \
              $(T1_DIR)/t1gload.c  \
              $(T1_DIR)/t1objs.c

# Type1 driver headers
#
T1_DRV_H := $(T1_DRV_SRC:%.c=%.h) \
            $(T1_DIR)/t1tokens.h  \
            $(T1_DIR)/t1errors.h


# Type1 driver object(s)
#
#   T1_DRV_OBJ_M is used during `multi' builds
#   T1_DRV_OBJ_S is used during `single' builds
#
T1_DRV_OBJ_M := $(T1_DRV_SRC:$(T1_DIR)/%.c=$(OBJ_DIR)/%.$O)
T1_DRV_OBJ_S := $(OBJ_DIR)/type1.$O

# Type1 driver source file for single build
#
T1_DRV_SRC_S := $(T1_DIR)/type1.c


# Type1 driver - single object
#
$(T1_DRV_OBJ_S): $(T1_DRV_SRC_S) $(T1_DRV_SRC) $(FREETYPE_H) $(T1_DRV_H)
	$(T1_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(T1_DRV_SRC_S))


# Type1 driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(T1_DIR)/%.c $(FREETYPE_H) $(T1_DRV_H)
	$(T1_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(T1_DRV_OBJ_S)
DRV_OBJS_M += $(T1_DRV_OBJ_M)


# EOF
