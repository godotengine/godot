##
##  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

# List of tools to build.
TOOLS-yes            += tiny_ssim.c
tiny_ssim.SRCS       += vpx/vpx_integer.h y4minput.c y4minput.h \
                        vpx/vpx_codec.h vpx/src/vpx_image.c
tiny_ssim.SRCS       += vpx_mem/vpx_mem.c vpx_mem/vpx_mem.h
tiny_ssim.SRCS       += vpx_dsp/ssim.h vpx_scale/yv12config.h
tiny_ssim.SRCS       += vpx_ports/mem.h vpx_ports/mem.h
tiny_ssim.SRCS       += vpx_mem/include/vpx_mem_intrnl.h
tiny_ssim.GUID        = 3afa9b05-940b-4d68-b5aa-55157d8ed7b4
tiny_ssim.DESCRIPTION = Generate SSIM/PSNR from raw .yuv files

#
# End of specified files. The rest of the build rules should happen
# automagically from here.
#


# Expand list of selected tools to build (as specified above)
TOOLS           = $(addprefix tools/,$(call enabled,TOOLS))
ALL_SRCS        = $(foreach ex,$(TOOLS),$($(notdir $(ex:.c=)).SRCS))
CFLAGS += -I../include

ifneq ($(CONFIG_CODEC_SRCS), yes)
  CFLAGS += -I../include/vpx
endif

# Expand all tools sources into a variable containing all sources
# for that tools (not just them main one specified in TOOLS)
# and add this file to the list (for MSVS workspace generation)
$(foreach ex,$(TOOLS),$(eval $(notdir $(ex:.c=)).SRCS += $(ex) tools.mk))


# Create build/install dependencies for all tools. The common case
# is handled here. The MSVS case is handled below.
NOT_MSVS = $(if $(CONFIG_MSVS),,yes)
DIST-BINS-$(NOT_MSVS)      += $(addprefix bin/,$(TOOLS:.c=$(EXE_SFX)))
DIST-SRCS-yes              += $(ALL_SRCS)
OBJS-$(NOT_MSVS)           += $(call objs,$(ALL_SRCS))
BINS-$(NOT_MSVS)           += $(addprefix $(BUILD_PFX),$(TOOLS:.c=$(EXE_SFX)))

# Instantiate linker template for all tools.
$(foreach bin,$(BINS-yes),\
    $(eval $(bin):)\
    $(eval $(call linker_template,$(bin),\
        $(call objs,$($(notdir $(bin:$(EXE_SFX)=)).SRCS)) -lm)))

# The following pairs define a mapping of locations in the distribution
# tree to locations in the source/build trees.
INSTALL_MAPS += src/%.c   %.c
INSTALL_MAPS += src/%     $(SRC_PATH_BARE)/%
INSTALL_MAPS += bin/%     %
INSTALL_MAPS += %         %


# Build Visual Studio Projects. We use a template here to instantiate
# explicit rules rather than using an implicit rule because we want to
# leverage make's VPATH searching rather than specifying the paths on
# each file in TOOLS. This has the unfortunate side effect that
# touching the source files trigger a rebuild of the project files
# even though there is no real dependency there (the dependency is on
# the makefiles). We may want to revisit this.
define vcproj_template
$(1): $($(1:.$(VCPROJ_SFX)=).SRCS) vpx.$(VCPROJ_SFX)
	$(if $(quiet),@echo "    [vcproj] $$@")
	$(qexec)$$(GEN_VCPROJ)\
            --exe\
            --target=$$(TOOLCHAIN)\
            --name=$$(@:.$(VCPROJ_SFX)=)\
            --ver=$$(CONFIG_VS_VERSION)\
            --proj-guid=$$($$(@:.$(VCPROJ_SFX)=).GUID)\
            --src-path-bare="$(SRC_PATH_BARE)" \
            --as=$$(AS) \
            $$(if $$(CONFIG_STATIC_MSVCRT),--static-crt) \
            --out=$$@ $$(INTERNAL_CFLAGS) $$(CFLAGS) \
            $$(INTERNAL_LDFLAGS) $$(LDFLAGS) $$^
endef
TOOLS_BASENAME := $(notdir $(TOOLS))
PROJECTS-$(CONFIG_MSVS) += $(TOOLS_BASENAME:.c=.$(VCPROJ_SFX))
INSTALL-BINS-$(CONFIG_MSVS) += $(foreach p,$(VS_PLATFORMS),\
                               $(addprefix bin/$(p)/,$(TOOLS_BASENAME:.c=.exe)))
$(foreach proj,$(call enabled,PROJECTS),\
    $(eval $(call vcproj_template,$(proj))))

# Generate a list of all enabled sources, in particular for exporting to gyp
# based build systems.
tiny_ssim_srcs.txt:
	@echo "    [CREATE] $@"
	@echo $(tiny_ssim.SRCS) | xargs -n1 echo | LC_ALL=C sort -u > $@
CLEAN-OBJS += tiny_ssim_srcs.txt

#
# Documentation Rules
#
%.dox: %.c
	@echo "    [DOXY] $@"
	@mkdir -p $(dir $@)
	@echo "/*!\page tools_$(@F:.dox=) $(@F:.dox=)" > $@
	@echo "   \includelineno $(<F)" >> $@
	@echo "*/" >> $@

tools.dox: tools.mk
	@echo "    [DOXY] $@"
	@echo "/*!\page tools Tools" > $@
	@echo "    This SDK includes a number of tools/utilities."\
	      "The following tools are included: ">>$@
	@$(foreach ex,$(sort $(notdir $(TOOLS:.c=))),\
	   echo "     - \subpage tools_$(ex) $($(ex).DESCRIPTION)" >> $@;)
	@echo "*/" >> $@

CLEAN-OBJS += tools.doxy tools.dox $(TOOLS:.c=.dox)
DOCS-yes += tools.doxy tools.dox
tools.doxy: tools.dox $(TOOLS:.c=.dox)
	@echo "INPUT += $^" > $@
