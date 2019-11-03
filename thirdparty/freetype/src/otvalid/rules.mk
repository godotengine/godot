#
# FreeType 2 OpenType validation driver configuration rules
#


# Copyright (C) 2004-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# OTV driver directory
#
OTV_DIR := $(SRC_DIR)/otvalid


# compilation flags for the driver
#
OTV_COMPILE := $(CC) $(ANSIFLAGS)                            \
                     $I$(subst /,$(COMPILER_SEP),$(OTV_DIR)) \
                     $(INCLUDE_FLAGS)                        \
                     $(FT_CFLAGS)


# OTV driver sources (i.e., C files)
#
OTV_DRV_SRC := $(OTV_DIR)/otvbase.c  \
               $(OTV_DIR)/otvcommn.c \
               $(OTV_DIR)/otvgdef.c  \
               $(OTV_DIR)/otvgpos.c  \
               $(OTV_DIR)/otvgsub.c  \
               $(OTV_DIR)/otvjstf.c  \
               $(OTV_DIR)/otvmath.c  \
               $(OTV_DIR)/otvmod.c

# OTV driver headers
#
OTV_DRV_H := $(OTV_DIR)/otvalid.h  \
             $(OTV_DIR)/otvcommn.h \
             $(OTV_DIR)/otverror.h \
             $(OTV_DIR)/otvgpos.h  \
             $(OTV_DIR)/otvmod.h


# OTV driver object(s)
#
#   OTV_DRV_OBJ_M is used during `multi' builds.
#   OTV_DRV_OBJ_S is used during `single' builds.
#
OTV_DRV_OBJ_M := $(OTV_DRV_SRC:$(OTV_DIR)/%.c=$(OBJ_DIR)/%.$O)
OTV_DRV_OBJ_S := $(OBJ_DIR)/otvalid.$O

# OTV driver source file for single build
#
OTV_DRV_SRC_S := $(OTV_DIR)/otvalid.c


# OTV driver - single object
#
$(OTV_DRV_OBJ_S): $(OTV_DRV_SRC_S) $(OTV_DRV_SRC) \
                   $(FREETYPE_H) $(OTV_DRV_H)
	$(OTV_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(OTV_DRV_SRC_S))


# OTV driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(OTV_DIR)/%.c $(FREETYPE_H) $(OTV_DRV_H)
	$(OTV_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(OTV_DRV_OBJ_S)
DRV_OBJS_M += $(OTV_DRV_OBJ_M)


# EOF
