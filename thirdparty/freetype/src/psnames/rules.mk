#
# FreeType 2 PSNames driver configuration rules
#


# Copyright 1996-2017 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# PSNames driver directory
#
PSNAMES_DIR := $(SRC_DIR)/psnames


# compilation flags for the driver
#
PSNAMES_COMPILE := $(CC) $(ANSIFLAGS)                                \
                         $I$(subst /,$(COMPILER_SEP),$(PSNAMES_DIR)) \
                         $(INCLUDE_FLAGS)                            \
                         $(FT_CFLAGS)


# PSNames driver sources (i.e., C files)
#
PSNAMES_DRV_SRC := $(PSNAMES_DIR)/psmodule.c \
                   $(PSNAMES_DIR)/pspic.c


# PSNames driver headers
#
PSNAMES_DRV_H := $(PSNAMES_DRV_SRC:%.c=%.h) \
                 $(PSNAMES_DIR)/psnamerr.h  \
                 $(PSNAMES_DIR)/pstables.h


# PSNames driver object(s)
#
#   PSNAMES_DRV_OBJ_M is used during `multi' builds
#   PSNAMES_DRV_OBJ_S is used during `single' builds
#
PSNAMES_DRV_OBJ_M := $(PSNAMES_DRV_SRC:$(PSNAMES_DIR)/%.c=$(OBJ_DIR)/%.$O)
PSNAMES_DRV_OBJ_S := $(OBJ_DIR)/psnames.$O

# PSNames driver source file for single build
#
PSNAMES_DRV_SRC_S := $(PSNAMES_DIR)/psnames.c


# PSNames driver - single object
#
$(PSNAMES_DRV_OBJ_S): $(PSNAMES_DRV_SRC_S) $(PSNAMES_DRV_SRC) \
                      $(FREETYPE_H) $(PSNAMES_DRV_H)
	$(PSNAMES_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(PSNAMES_DRV_SRC_S))


# PSNames driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(PSNAMES_DIR)/%.c $(FREETYPE_H) $(PSNAMES_DRV_H)
	$(PSNAMES_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(PSNAMES_DRV_OBJ_S)
DRV_OBJS_M += $(PSNAMES_DRV_OBJ_M)


# EOF
