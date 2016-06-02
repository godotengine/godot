#
# FreeType 2 renderer module build rules
#


# Copyright 1996-2000, 2001, 2003, 2008, 2009, 2011 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# raster driver directory
#
RASTER_DIR := $(SRC_DIR)/raster

# compilation flags for the driver
#
RASTER_COMPILE := $(FT_COMPILE) $I$(subst /,$(COMPILER_SEP),$(RASTER_DIR))


# raster driver sources (i.e., C files)
#
RASTER_DRV_SRC := $(RASTER_DIR)/ftraster.c \
                  $(RASTER_DIR)/ftrend1.c  \
                  $(RASTER_DIR)/rastpic.c


# raster driver headers
#
RASTER_DRV_H := $(RASTER_DRV_SRC:%.c=%.h) \
                $(RASTER_DIR)/rasterrs.h


# raster driver object(s)
#
#   RASTER_DRV_OBJ_M is used during `multi' builds.
#   RASTER_DRV_OBJ_S is used during `single' builds.
#
RASTER_DRV_OBJ_M := $(RASTER_DRV_SRC:$(RASTER_DIR)/%.c=$(OBJ_DIR)/%.$O)
RASTER_DRV_OBJ_S := $(OBJ_DIR)/raster.$O

# raster driver source file for single build
#
RASTER_DRV_SRC_S := $(RASTER_DIR)/raster.c


# raster driver - single object
#
$(RASTER_DRV_OBJ_S): $(RASTER_DRV_SRC_S) $(RASTER_DRV_SRC) \
                     $(FREETYPE_H) $(RASTER_DRV_H)
	$(RASTER_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(RASTER_DRV_SRC_S))


# raster driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(RASTER_DIR)/%.c $(FREETYPE_H) $(RASTER_DRV_H)
	$(RASTER_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(RASTER_DRV_OBJ_S)
DRV_OBJS_M += $(RASTER_DRV_OBJ_M)


# EOF
