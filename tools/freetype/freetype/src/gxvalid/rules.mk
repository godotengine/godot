#
# FreeType 2 TrueTypeGX/AAT validation driver configuration rules
#


# Copyright 2004, 2005 by suzuki toshiya, Masatake YAMATO, Red Hat K.K.,
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


# GXV driver directory
#
GXV_DIR := $(SRC_DIR)/gxvalid


# compilation flags for the driver
#
GXV_COMPILE := $(FT_COMPILE) $I$(subst /,$(COMPILER_SEP),$(GXV_DIR))


# GXV driver sources (i.e., C files)
#
GXV_DRV_SRC := $(GXV_DIR)/gxvcommn.c \
               $(GXV_DIR)/gxvfeat.c  \
               $(GXV_DIR)/gxvbsln.c  \
               $(GXV_DIR)/gxvtrak.c  \
               $(GXV_DIR)/gxvopbd.c  \
               $(GXV_DIR)/gxvprop.c  \
               $(GXV_DIR)/gxvjust.c  \
               $(GXV_DIR)/gxvmort.c  \
               $(GXV_DIR)/gxvmort0.c \
               $(GXV_DIR)/gxvmort1.c \
               $(GXV_DIR)/gxvmort2.c \
               $(GXV_DIR)/gxvmort4.c \
               $(GXV_DIR)/gxvmort5.c \
               $(GXV_DIR)/gxvmorx.c  \
               $(GXV_DIR)/gxvmorx0.c \
               $(GXV_DIR)/gxvmorx1.c \
               $(GXV_DIR)/gxvmorx2.c \
               $(GXV_DIR)/gxvmorx4.c \
               $(GXV_DIR)/gxvmorx5.c \
               $(GXV_DIR)/gxvlcar.c  \
               $(GXV_DIR)/gxvkern.c  \
               $(GXV_DIR)/gxvmod.c

# GXV driver headers
#
GXV_DRV_H := $(GXV_DIR)/gxvalid.h  \
             $(GXV_DIR)/gxverror.h \
             $(GXV_DIR)/gxvcommn.h \
             $(GXV_DIR)/gxvfeat.h  \
             $(GXV_DIR)/gxvmod.h   \
             $(GXV_DIR)/gxvmort.h  \
             $(GXV_DIR)/gxvmorx.h


# GXV driver object(s)
#
#   GXV_DRV_OBJ_M is used during `multi' builds.
#   GXV_DRV_OBJ_S is used during `single' builds.
#
GXV_DRV_OBJ_M := $(GXV_DRV_SRC:$(GXV_DIR)/%.c=$(OBJ_DIR)/%.$O)
GXV_DRV_OBJ_S := $(OBJ_DIR)/gxvalid.$O

# GXV driver source file for single build
#
GXV_DRV_SRC_S := $(GXV_DIR)/gxvalid.c


# GXV driver - single object
#
$(GXV_DRV_OBJ_S): $(GXV_DRV_SRC_S) $(GXV_DRV_SRC) \
                   $(FREETYPE_H) $(GXV_DRV_H)
	$(GXV_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(GXV_DRV_SRC_S))


# GXV driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(GXV_DIR)/%.c $(FREETYPE_H) $(GXV_DRV_H)
	$(GXV_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(GXV_DRV_OBJ_S)
DRV_OBJS_M += $(GXV_DRV_OBJ_M)


# EOF
