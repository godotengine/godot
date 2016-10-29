#
# FreeType 2 PSHinter driver configuration rules
#


# Copyright 2001-2016 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# PSHINTER driver directory
#
PSHINTER_DIR := $(SRC_DIR)/pshinter


# compilation flags for the driver
#
PSHINTER_COMPILE := $(CC) $(ANSIFLAGS)                                 \
                          $I$(subst /,$(COMPILER_SEP),$(PSHINTER_DIR)) \
                          $(INCLUDE_FLAGS)                             \
                          $(FT_CFLAGS)


# PSHINTER driver sources (i.e., C files)
#
PSHINTER_DRV_SRC := $(PSHINTER_DIR)/pshalgo.c \
                    $(PSHINTER_DIR)/pshglob.c \
                    $(PSHINTER_DIR)/pshmod.c  \
                    $(PSHINTER_DIR)/pshpic.c  \
                    $(PSHINTER_DIR)/pshrec.c


# PSHINTER driver headers
#
PSHINTER_DRV_H := $(PSHINTER_DRV_SRC:%c=%h) \
                  $(PSHINTER_DIR)/pshnterr.h


# PSHINTER driver object(s)
#
#   PSHINTER_DRV_OBJ_M is used during `multi' builds.
#   PSHINTER_DRV_OBJ_S is used during `single' builds.
#
PSHINTER_DRV_OBJ_M := $(PSHINTER_DRV_SRC:$(PSHINTER_DIR)/%.c=$(OBJ_DIR)/%.$O)
PSHINTER_DRV_OBJ_S := $(OBJ_DIR)/pshinter.$O

# PSHINTER driver source file for single build
#
PSHINTER_DRV_SRC_S := $(PSHINTER_DIR)/pshinter.c


# PSHINTER driver - single object
#
$(PSHINTER_DRV_OBJ_S): $(PSHINTER_DRV_SRC_S) $(PSHINTER_DRV_SRC) \
                       $(FREETYPE_H) $(PSHINTER_DRV_H)
	$(PSHINTER_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(PSHINTER_DRV_SRC_S))


# PSHINTER driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(PSHINTER_DIR)/%.c $(FREETYPE_H) $(PSHINTER_DRV_H)
	$(PSHINTER_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(PSHINTER_DRV_OBJ_S)
DRV_OBJS_M += $(PSHINTER_DRV_OBJ_M)


# EOF
