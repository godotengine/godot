#
# FreeType 2 PFR driver configuration rules
#


# Copyright (C) 2002-2020 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# pfr driver directory
#
PFR_DIR := $(SRC_DIR)/pfr


# compilation flags for the driver
#
PFR_COMPILE := $(CC) $(ANSIFLAGS)                            \
                     $I$(subst /,$(COMPILER_SEP),$(PFR_DIR)) \
                     $(INCLUDE_FLAGS)                        \
                     $(FT_CFLAGS)


# pfr driver sources (i.e., C files)
#
PFR_DRV_SRC := $(PFR_DIR)/pfrload.c  \
               $(PFR_DIR)/pfrgload.c \
               $(PFR_DIR)/pfrcmap.c  \
               $(PFR_DIR)/pfrdrivr.c \
               $(PFR_DIR)/pfrsbit.c  \
               $(PFR_DIR)/pfrobjs.c

# pfr driver headers
#
PFR_DRV_H := $(PFR_DRV_SRC:%.c=%.h) \
             $(PFR_DIR)/pfrerror.h  \
             $(PFR_DIR)/pfrtypes.h


# Pfr driver object(s)
#
#   PFR_DRV_OBJ_M is used during `multi' builds
#   PFR_DRV_OBJ_S is used during `single' builds
#
PFR_DRV_OBJ_M := $(PFR_DRV_SRC:$(PFR_DIR)/%.c=$(OBJ_DIR)/%.$O)
PFR_DRV_OBJ_S := $(OBJ_DIR)/pfr.$O

# pfr driver source file for single build
#
PFR_DRV_SRC_S := $(PFR_DIR)/pfr.c


# pfr driver - single object
#
$(PFR_DRV_OBJ_S): $(PFR_DRV_SRC_S) $(PFR_DRV_SRC) $(FREETYPE_H) $(PFR_DRV_H)
	$(PFR_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(PFR_DRV_SRC_S))


# pfr driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(PFR_DIR)/%.c $(FREETYPE_H) $(PFR_DRV_H)
	$(PFR_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(PFR_DRV_OBJ_S)
DRV_OBJS_M += $(PFR_DRV_OBJ_M)


# EOF
