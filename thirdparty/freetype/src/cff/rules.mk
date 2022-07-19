#
# FreeType 2 OpenType/CFF driver configuration rules
#


# Copyright (C) 1996-2020 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# OpenType driver directory
#
CFF_DIR := $(SRC_DIR)/cff


CFF_COMPILE := $(CC) $(ANSIFLAGS)                            \
                     $I$(subst /,$(COMPILER_SEP),$(CFF_DIR)) \
                     $(INCLUDE_FLAGS)                        \
                     $(FT_CFLAGS)


# CFF driver sources (i.e., C files)
#
CFF_DRV_SRC := $(CFF_DIR)/cffcmap.c  \
               $(CFF_DIR)/cffdrivr.c \
               $(CFF_DIR)/cffgload.c \
               $(CFF_DIR)/cffload.c  \
               $(CFF_DIR)/cffobjs.c  \
               $(CFF_DIR)/cffparse.c


# CFF driver headers
#
CFF_DRV_H := $(CFF_DRV_SRC:%.c=%.h) \
             $(CFF_DIR)/cfferrs.h   \
             $(CFF_DIR)/cfftoken.h


# CFF driver object(s)
#
#   CFF_DRV_OBJ_M is used during `multi' builds
#   CFF_DRV_OBJ_S is used during `single' builds
#
CFF_DRV_OBJ_M := $(CFF_DRV_SRC:$(CFF_DIR)/%.c=$(OBJ_DIR)/%.$O)
CFF_DRV_OBJ_S := $(OBJ_DIR)/cff.$O

# CFF driver source file for single build
#
CFF_DRV_SRC_S := $(CFF_DIR)/cff.c


# CFF driver - single object
#
$(CFF_DRV_OBJ_S): $(CFF_DRV_SRC_S) $(CFF_DRV_SRC) $(FREETYPE_H) $(CFF_DRV_H)
	$(CFF_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(CFF_DRV_SRC_S))


# CFF driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(CFF_DIR)/%.c $(FREETYPE_H) $(CFF_DRV_H)
	$(CFF_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(CFF_DRV_OBJ_S)
DRV_OBJS_M += $(CFF_DRV_OBJ_M)


# EOF
