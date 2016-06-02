#
# FreeType 2 Type42 driver configuration rules
#


# Copyright 2002, 2003, 2008 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# Type42 driver directory
#
T42_DIR := $(SRC_DIR)/type42


# compilation flags for the driver
#
T42_COMPILE := $(FT_COMPILE) $I$(subst /,$(COMPILER_SEP),$(T42_DIR))


# Type42 driver source
#
T42_DRV_SRC := $(T42_DIR)/t42objs.c  \
               $(T42_DIR)/t42parse.c \
               $(T42_DIR)/t42drivr.c

# Type42 driver headers
#
T42_DRV_H := $(T42_DRV_SRC:%.c=%.h) \
             $(T42_DIR)/t42error.h  \
             $(T42_DIR)/t42types.h


# Type42 driver object(s)
#
#   T42_DRV_OBJ_M is used during `multi' builds
#   T42_DRV_OBJ_S is used during `single' builds
#
T42_DRV_OBJ_M := $(T42_DRV_SRC:$(T42_DIR)/%.c=$(OBJ_DIR)/%.$O)
T42_DRV_OBJ_S := $(OBJ_DIR)/type42.$O

# Type42 driver source file for single build
#
T42_DRV_SRC_S := $(T42_DIR)/type42.c


# Type42 driver - single object
#
$(T42_DRV_OBJ_S): $(T42_DRV_SRC_S) $(T42_DRV_SRC) $(FREETYPE_H) $(T42_DRV_H)
	$(T42_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(T42_DRV_SRC_S))


# Type42 driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(T42_DIR)/%.c $(FREETYPE_H) $(T42_DRV_H)
	$(T42_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(T42_DRV_OBJ_S)
DRV_OBJS_M += $(T42_DRV_OBJ_M)


# EOF
