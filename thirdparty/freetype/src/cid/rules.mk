#
# FreeType 2 CID driver configuration rules
#


# Copyright 1996-2016 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# CID driver directory
#
CID_DIR := $(SRC_DIR)/cid


CID_COMPILE := $(CC) $(ANSIFLAGS)                            \
                     $I$(subst /,$(COMPILER_SEP),$(CID_DIR)) \
                     $(INCLUDE_FLAGS)                        \
                     $(FT_CFLAGS)


# CID driver sources (i.e., C files)
#
CID_DRV_SRC := $(CID_DIR)/cidparse.c \
               $(CID_DIR)/cidload.c  \
               $(CID_DIR)/cidriver.c \
               $(CID_DIR)/cidgload.c \
               $(CID_DIR)/cidobjs.c

# CID driver headers
#
CID_DRV_H := $(CID_DRV_SRC:%.c=%.h) \
             $(CID_DIR)/cidtoken.h  \
             $(CID_DIR)/ciderrs.h


# CID driver object(s)
#
#   CID_DRV_OBJ_M is used during `multi' builds
#   CID_DRV_OBJ_S is used during `single' builds
#
CID_DRV_OBJ_M := $(CID_DRV_SRC:$(CID_DIR)/%.c=$(OBJ_DIR)/%.$O)
CID_DRV_OBJ_S := $(OBJ_DIR)/type1cid.$O

# CID driver source file for single build
#
CID_DRV_SRC_S := $(CID_DIR)/type1cid.c


# CID driver - single object
#
$(CID_DRV_OBJ_S): $(CID_DRV_SRC_S) $(CID_DRV_SRC) $(FREETYPE_H) $(CID_DRV_H)
	$(CID_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(CID_DRV_SRC_S))


# CID driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(CID_DIR)/%.c $(FREETYPE_H) $(CID_DRV_H)
	$(CID_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(CID_DRV_OBJ_S)
DRV_OBJS_M += $(CID_DRV_OBJ_M)


# EOF
