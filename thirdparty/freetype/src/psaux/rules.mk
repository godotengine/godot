#
# FreeType 2 PSaux driver configuration rules
#


# Copyright 1996-2016 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# PSAUX driver directory
#
PSAUX_DIR := $(SRC_DIR)/psaux


# compilation flags for the driver
#
PSAUX_COMPILE := $(CC) $(ANSIFLAGS)                              \
                       $I$(subst /,$(COMPILER_SEP),$(PSAUX_DIR)) \
                       $(INCLUDE_FLAGS)                          \
                       $(FT_CFLAGS)


# PSAUX driver sources (i.e., C files)
#
PSAUX_DRV_SRC := $(PSAUX_DIR)/psobjs.c   \
                 $(PSAUX_DIR)/t1decode.c \
                 $(PSAUX_DIR)/t1cmap.c   \
                 $(PSAUX_DIR)/afmparse.c \
                 $(PSAUX_DIR)/psconv.c   \
                 $(PSAUX_DIR)/psauxmod.c

# PSAUX driver headers
#
PSAUX_DRV_H := $(PSAUX_DRV_SRC:%c=%h)  \
               $(PSAUX_DIR)/psauxerr.h


# PSAUX driver object(s)
#
#   PSAUX_DRV_OBJ_M is used during `multi' builds.
#   PSAUX_DRV_OBJ_S is used during `single' builds.
#
PSAUX_DRV_OBJ_M := $(PSAUX_DRV_SRC:$(PSAUX_DIR)/%.c=$(OBJ_DIR)/%.$O)
PSAUX_DRV_OBJ_S := $(OBJ_DIR)/psaux.$O

# PSAUX driver source file for single build
#
PSAUX_DRV_SRC_S := $(PSAUX_DIR)/psaux.c


# PSAUX driver - single object
#
$(PSAUX_DRV_OBJ_S): $(PSAUX_DRV_SRC_S) $(PSAUX_DRV_SRC) \
                   $(FREETYPE_H) $(PSAUX_DRV_H)
	$(PSAUX_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(PSAUX_DRV_SRC_S))


# PSAUX driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(PSAUX_DIR)/%.c $(FREETYPE_H) $(PSAUX_DRV_H)
	$(PSAUX_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(PSAUX_DRV_OBJ_S)
DRV_OBJS_M += $(PSAUX_DRV_OBJ_M)


# EOF
