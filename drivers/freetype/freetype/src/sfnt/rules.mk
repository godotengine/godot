#
# FreeType 2 SFNT driver configuration rules
#


# Copyright 1996-2000, 2002-2007, 2009, 2011, 2013 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# SFNT driver directory
#
SFNT_DIR := $(SRC_DIR)/sfnt


# compilation flags for the driver
#
SFNT_COMPILE := $(FT_COMPILE) $I$(subst /,$(COMPILER_SEP),$(SFNT_DIR))


# SFNT driver sources (i.e., C files)
#
SFNT_DRV_SRC := $(SFNT_DIR)/ttload.c   \
                $(SFNT_DIR)/ttmtx.c    \
                $(SFNT_DIR)/ttcmap.c   \
                $(SFNT_DIR)/ttsbit.c   \
                $(SFNT_DIR)/ttpost.c   \
                $(SFNT_DIR)/ttkern.c   \
                $(SFNT_DIR)/ttbdf.c    \
                $(SFNT_DIR)/sfobjs.c   \
                $(SFNT_DIR)/sfdriver.c \
                $(SFNT_DIR)/sfntpic.c  \
                $(SFNT_DIR)/pngshim.c

# SFNT driver headers
#
SFNT_DRV_H := $(SFNT_DRV_SRC:%c=%h)  \
              $(SFNT_DIR)/sferrors.h


# SFNT driver object(s)
#
#   SFNT_DRV_OBJ_M is used during `multi' builds.
#   SFNT_DRV_OBJ_S is used during `single' builds.
#
SFNT_DRV_OBJ_M := $(SFNT_DRV_SRC:$(SFNT_DIR)/%.c=$(OBJ_DIR)/%.$O)
SFNT_DRV_OBJ_S := $(OBJ_DIR)/sfnt.$O

# SFNT driver source file for single build
#
SFNT_DRV_SRC_S := $(SFNT_DIR)/sfnt.c


# SFNT driver - single object
#
$(SFNT_DRV_OBJ_S): $(SFNT_DRV_SRC_S) $(SFNT_DRV_SRC) \
                   $(FREETYPE_H) $(SFNT_DRV_H)
	$(SFNT_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(SFNT_DRV_SRC_S))


# SFNT driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(SFNT_DIR)/%.c $(FREETYPE_H) $(SFNT_DRV_H)
	$(SFNT_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(SFNT_DRV_OBJ_S)
DRV_OBJS_M += $(SFNT_DRV_OBJ_M)


# EOF
