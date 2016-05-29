#
# FreeType 2 pcf driver configuration rules
#


# Copyright (C) 2000, 2001, 2003, 2008 by
# Francesco Zappa Nardelli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# pcf driver directory
#
PCF_DIR := $(SRC_DIR)/pcf


PCF_COMPILE := $(FT_COMPILE) $I$(subst /,$(COMPILER_SEP),$(PCF_DIR))


# pcf driver sources (i.e., C files)
#
PCF_DRV_SRC := $(PCF_DIR)/pcfdrivr.c \
               $(PCF_DIR)/pcfread.c  \
               $(PCF_DIR)/pcfutil.c

# pcf driver headers
#
PCF_DRV_H := $(PCF_DRV_SRC:%.c=%.h) \
             $(PCF_DIR)/pcf.h       \
             $(PCF_DIR)/pcferror.h

# pcf driver object(s)
#
#   PCF_DRV_OBJ_M is used during `multi' builds
#   PCF_DRV_OBJ_S is used during `single' builds
#
PCF_DRV_OBJ_M := $(PCF_DRV_SRC:$(PCF_DIR)/%.c=$(OBJ_DIR)/%.$O)
PCF_DRV_OBJ_S := $(OBJ_DIR)/pcf.$O

# pcf driver source file for single build
#
PCF_DRV_SRC_S := $(PCF_DIR)/pcf.c


# pcf driver - single object
#
$(PCF_DRV_OBJ_S): $(PCF_DRV_SRC_S) $(PCF_DRV_SRC) $(FREETYPE_H) $(PCF_DRV_H)
	$(PCF_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $(PCF_DRV_SRC_S))


# pcf driver - multiple objects
#
$(OBJ_DIR)/%.$O: $(PCF_DIR)/%.c $(FREETYPE_H) $(PCF_DRV_H)
	$(PCF_COMPILE) $T$(subst /,$(COMPILER_SEP),$@ $<)


# update main driver object lists
#
DRV_OBJS_S += $(PCF_DRV_OBJ_S)
DRV_OBJS_M += $(PCF_DRV_OBJ_M)


# EOF
