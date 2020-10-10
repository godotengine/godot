#
# FreeType 2 auto-fitter module configuration rules
#


# Copyright (C) 2003-2020 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# AUTOF driver directory
#
AUTOF_DIR := $(SRC_DIR)/autofit


# compilation flags for the driver
#
AUTOF_COMPILE := $(CC) $(ANSIFLAGS)                              \
                       $I$(subst /,$(COMPILER_SEP),$(AUTOF_DIR)) \
                       $(INCLUDE_FLAGS)                          \
                       $(FT_CFLAGS)


# AUTOF driver sources (i.e., C files)
#
AUTOF_DRV_SRC := $(AUTOF_DIR)/afangles.c \
                 $(AUTOF_DIR)/afblue.c   \
                 $(AUTOF_DIR)/afcjk.c    \
                 $(AUTOF_DIR)/afdummy.c  \
                 $(AUTOF_DIR)/afglobal.c \
                 $(AUTOF_DIR)/afhints.c  \
                 $(AUTOF_DIR)/afindic.c  \
                 $(AUTOF_DIR)/aflatin.c  \
                 $(AUTOF_DIR)/afloader.c \
                 $(AUTOF_DIR)/afmodule.c \
                 $(AUTOF_DIR)/afranges.c \
                 $(AUTOF_DIR)/afshaper.c \
                 $(AUTOF_DIR)/afwarp.c

# AUTOF driver headers
#
AUTOF_DRV_H := $(AUTOF_DRV_SRC:%c=%h)  \
               $(AUTOF_DIR)/afcover.h  \
               $(AUTOF_DIR)/aferrors.h \
               $(AUTOF_DIR)/afscript.h \
               $(AUTOF_DIR)/afstyles.h \
               $(AUTOF_DIR)/aftypes.h  \
               $(AUTOF_DIR)/afwrtsys.h


# AUTOF driver object(s)
#
#   AUTOF_DRV_OBJ_M is used during `multi' builds.
#   AUTOF_DRV_OBJ_S is used during `single' builds.
#
AUTOF_DRV_OBJ_M := $(AUTOF_DRV_SRC:$(AUTOF_DIR)/%.c=$(OBJ_DIR)/%.$O)
AUTOF_DRV_OBJ_S := $(OBJ_DIR)/autofit.$O

# AUTOF driver source file for single build
#
AUTOF_DRV_SRC_S := $(AUTOF_DIR)/autofit.c


# AUTOF driver - single object
#
$(AUTOF_DRV_OBJ_S): $(AUTOF_DRV_SRC_S) $(AUTOF_DRV_SRC) \
                   $(FREETYPE_H) $(AUTOF_DRV_H)
	$(AUTOF_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(AUTOF_DRV_SRC_S))


# AUTOF driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(AUTOF_DIR)/%.c $(FREETYPE_H) $(AUTOF_DRV_H)
	$(AUTOF_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(AUTOF_DRV_OBJ_S)
DRV_OBJS_M += $(AUTOF_DRV_OBJ_M)


# EOF
