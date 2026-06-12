##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


# ARM assembly files are written in RVCT-style. We use some make magic to
# filter those files to allow GCC compilation
ifeq ($(VPX_ARCH_ARM),yes)
  ASM:=$(if $(filter yes,$(CONFIG_GCC)$(CONFIG_MSVS)),.asm.S,.asm)
else
  ASM:=.asm
endif

#
# Rule to generate runtime cpu detection files
#
define rtcd_h_template
$$(BUILD_PFX)$(1).h: $$(SRC_PATH_BARE)/$(2)
	@echo "    [CREATE] $$@"
	$$(qexec)$$(SRC_PATH_BARE)/build/make/rtcd.pl --arch=$$(TGT_ISA) \
          --sym=$(1) \
          --config=$$(CONFIG_DIR)$$(target)-$$(TOOLCHAIN).mk \
          $$(RTCD_OPTIONS) $$^ > $$@
CLEAN-OBJS += $$(BUILD_PFX)$(1).h
RTCD += $$(BUILD_PFX)$(1).h
endef

CODEC_SRCS-yes += CHANGELOG
CODEC_SRCS-yes += libs.mk

include $(SRC_PATH_BARE)/vpx/vpx_codec.mk
CODEC_SRCS-yes += $(addprefix vpx/,$(call enabled,API_SRCS))
CODEC_DOC_SRCS += $(addprefix vpx/,$(call enabled,API_DOC_SRCS))

include $(SRC_PATH_BARE)/vpx_mem/vpx_mem.mk
CODEC_SRCS-yes += $(addprefix vpx_mem/,$(call enabled,MEM_SRCS))

include $(SRC_PATH_BARE)/vpx_scale/vpx_scale.mk
CODEC_SRCS-yes += $(addprefix vpx_scale/,$(call enabled,SCALE_SRCS))

include $(SRC_PATH_BARE)/vpx_ports/vpx_ports.mk
CODEC_SRCS-yes += $(addprefix vpx_ports/,$(call enabled,PORTS_SRCS))

include $(SRC_PATH_BARE)/vpx_dsp/vpx_dsp.mk
CODEC_SRCS-yes += $(addprefix vpx_dsp/,$(call enabled,DSP_SRCS))

include $(SRC_PATH_BARE)/vpx_util/vpx_util.mk
CODEC_SRCS-yes += $(addprefix vpx_util/,$(call enabled,UTIL_SRCS))

ifeq ($(CONFIG_VP8),yes)
  VP8_PREFIX=vp8/
  include $(SRC_PATH_BARE)/$(VP8_PREFIX)vp8_common.mk
endif

ifeq ($(CONFIG_VP8_ENCODER),yes)
  include $(SRC_PATH_BARE)/$(VP8_PREFIX)vp8cx.mk
  CODEC_SRCS-yes += $(addprefix $(VP8_PREFIX),$(call enabled,VP8_CX_SRCS))
  CODEC_EXPORTS-yes += $(addprefix $(VP8_PREFIX),$(VP8_CX_EXPORTS))
  INSTALL-LIBS-yes += include/vpx/vp8.h include/vpx/vp8cx.h
  INSTALL-LIBS-yes += include/vpx/vpx_ext_ratectrl.h
  INSTALL_MAPS += include/vpx/% $(SRC_PATH_BARE)/$(VP8_PREFIX)/%
  CODEC_DOC_SECTIONS += vp8 vp8_encoder
endif

ifeq ($(CONFIG_VP8_DECODER),yes)
  include $(SRC_PATH_BARE)/$(VP8_PREFIX)vp8dx.mk
  CODEC_SRCS-yes += $(addprefix $(VP8_PREFIX),$(call enabled,VP8_DX_SRCS))
  CODEC_EXPORTS-yes += $(addprefix $(VP8_PREFIX),$(VP8_DX_EXPORTS))
  INSTALL-LIBS-yes += include/vpx/vp8.h include/vpx/vp8dx.h
  INSTALL_MAPS += include/vpx/% $(SRC_PATH_BARE)/$(VP8_PREFIX)/%
  CODEC_DOC_SECTIONS += vp8 vp8_decoder
endif

ifeq ($(CONFIG_VP9),yes)
  VP9_PREFIX=vp9/
  include $(SRC_PATH_BARE)/$(VP9_PREFIX)vp9_common.mk
endif

ifeq ($(CONFIG_VP9_ENCODER),yes)
  VP9_PREFIX=vp9/
  include $(SRC_PATH_BARE)/$(VP9_PREFIX)vp9cx.mk
  CODEC_SRCS-yes += $(addprefix $(VP9_PREFIX),$(call enabled,VP9_CX_SRCS))
  CODEC_EXPORTS-yes += $(addprefix $(VP9_PREFIX),$(VP9_CX_EXPORTS))
  CODEC_SRCS-yes += $(VP9_PREFIX)vp9cx.mk vpx/vp8.h vpx/vp8cx.h
  CODEC_SRCS-yes += vpx/vpx_ext_ratectrl.h
  INSTALL-LIBS-yes += include/vpx/vp8.h include/vpx/vp8cx.h
  INSTALL-LIBS-yes += include/vpx/vpx_ext_ratectrl.h
  INSTALL_MAPS += include/vpx/% $(SRC_PATH_BARE)/$(VP9_PREFIX)/%
  CODEC_DOC_SRCS += vpx/vp8.h vpx/vp8cx.h vpx/vpx_ext_ratectrl.h
  CODEC_DOC_SECTIONS += vp9 vp9_encoder
endif

RC_RTC_SRCS := vpx/vp8.h vpx/vp8cx.h
RC_RTC_SRCS += vpx/vpx_ext_ratectrl.h
RC_RTC_SRCS += vpx/internal/vpx_ratectrl_rtc.h
ifeq ($(CONFIG_VP9_ENCODER),yes)
  VP9_PREFIX=vp9/
  RC_RTC_SRCS += $(addprefix $(VP9_PREFIX),$(call enabled,VP9_CX_SRCS))
  RC_RTC_SRCS += $(VP9_PREFIX)vp9cx.mk
  RC_RTC_SRCS += $(VP9_PREFIX)ratectrl_rtc.cc
  RC_RTC_SRCS += $(VP9_PREFIX)ratectrl_rtc.h
  INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(VP9_PREFIX)ratectrl_rtc.cc
  INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(VP9_PREFIX)ratectrl_rtc.h
endif
ifeq ($(CONFIG_VP8_ENCODER),yes)
  VP8_PREFIX=vp8/
  RC_RTC_SRCS += $(addprefix $(VP8_PREFIX),$(call enabled,VP8_CX_SRCS))
  RC_RTC_SRCS += $(VP8_PREFIX)vp8_ratectrl_rtc.cc
  RC_RTC_SRCS += $(VP8_PREFIX)vp8_ratectrl_rtc.h
  INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(VP8_PREFIX)vp8_ratectrl_rtc.cc
  INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(VP8_PREFIX)vp8_ratectrl_rtc.h
endif

ifeq ($(CONFIG_VP9_DECODER),yes)
  VP9_PREFIX=vp9/
  include $(SRC_PATH_BARE)/$(VP9_PREFIX)vp9dx.mk
  CODEC_SRCS-yes += $(addprefix $(VP9_PREFIX),$(call enabled,VP9_DX_SRCS))
  CODEC_EXPORTS-yes += $(addprefix $(VP9_PREFIX),$(VP9_DX_EXPORTS))
  CODEC_SRCS-yes += $(VP9_PREFIX)vp9dx.mk vpx/vp8.h vpx/vp8dx.h
  INSTALL-LIBS-yes += include/vpx/vp8.h include/vpx/vp8dx.h
  INSTALL_MAPS += include/vpx/% $(SRC_PATH_BARE)/$(VP9_PREFIX)/%
  CODEC_DOC_SRCS += vpx/vp8.h vpx/vp8dx.h
  CODEC_DOC_SECTIONS += vp9 vp9_decoder
endif

ifeq ($(CONFIG_ENCODERS),yes)
  CODEC_DOC_SECTIONS += encoder
endif
ifeq ($(CONFIG_DECODERS),yes)
  CODEC_DOC_SECTIONS += decoder
endif

ifeq ($(CONFIG_MSVS),yes)
CODEC_LIB=$(if $(CONFIG_STATIC_MSVCRT),vpxmt,vpxmd)
GTEST_LIB=$(if $(CONFIG_STATIC_MSVCRT),gtestmt,gtestmd)
RC_RTC_LIB=$(if $(CONFIG_STATIC_MSVCRT),vpxrcmt,vpxrcmd)
# This variable uses deferred expansion intentionally, since the results of
# $(wildcard) may change during the course of the Make.
VS_PLATFORMS = $(foreach d,$(wildcard */Release/$(CODEC_LIB).lib),$(word 1,$(subst /, ,$(d))))
endif

# The following pairs define a mapping of locations in the distribution
# tree to locations in the source/build trees.
INSTALL_MAPS += include/vpx/% $(SRC_PATH_BARE)/vpx/%
INSTALL_MAPS += include/vpx/% $(SRC_PATH_BARE)/vpx_ports/%
INSTALL_MAPS += $(LIBSUBDIR)/%     %
INSTALL_MAPS += src/%     $(SRC_PATH_BARE)/%
ifeq ($(CONFIG_MSVS),yes)
INSTALL_MAPS += $(foreach p,$(VS_PLATFORMS),$(LIBSUBDIR)/$(p)/%  $(p)/Release/%)
INSTALL_MAPS += $(foreach p,$(VS_PLATFORMS),$(LIBSUBDIR)/$(p)/%  $(p)/Debug/%)
endif

CODEC_SRCS-yes += build/make/version.sh
CODEC_SRCS-yes += build/make/rtcd.pl
CODEC_SRCS-yes += vpx_ports/emmintrin_compat.h
CODEC_SRCS-yes += vpx_ports/mem_ops.h
CODEC_SRCS-yes += vpx_ports/mem_ops_aligned.h
CODEC_SRCS-yes += vpx_ports/vpx_once.h
CODEC_SRCS-yes += $(BUILD_PFX)vpx_config.c
INSTALL-SRCS-no += $(BUILD_PFX)vpx_config.c
ifeq ($(VPX_ARCH_X86)$(VPX_ARCH_X86_64),yes)
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += third_party/x86inc/x86inc.asm
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += vpx_dsp/x86/bitdepth_conversion_sse2.asm
endif
CODEC_EXPORTS-yes += vpx/exports_com
CODEC_EXPORTS-$(CONFIG_ENCODERS) += vpx/exports_enc
CODEC_EXPORTS-$(CONFIG_DECODERS) += vpx/exports_dec

INSTALL-LIBS-yes += include/vpx/vpx_codec.h
INSTALL-LIBS-yes += include/vpx/vpx_frame_buffer.h
INSTALL-LIBS-yes += include/vpx/vpx_image.h
INSTALL-LIBS-yes += include/vpx/vpx_integer.h
INSTALL-LIBS-$(CONFIG_DECODERS) += include/vpx/vpx_decoder.h
INSTALL-LIBS-$(CONFIG_ENCODERS) += include/vpx/vpx_encoder.h
INSTALL-LIBS-$(CONFIG_ENCODERS) += include/vpx/vpx_tpl.h
ifeq ($(CONFIG_EXTERNAL_BUILD),yes)
ifeq ($(CONFIG_MSVS),yes)
INSTALL-LIBS-yes                  += $(foreach p,$(VS_PLATFORMS),$(LIBSUBDIR)/$(p)/$(CODEC_LIB).lib)
ifeq ($(CONFIG_STATIC),yes)
INSTALL-LIBS-$(CONFIG_DEBUG_LIBS) += $(foreach p,$(VS_PLATFORMS),$(LIBSUBDIR)/$(p)/$(CODEC_LIB)d.lib)
endif
INSTALL-LIBS-$(CONFIG_SHARED) += $(foreach p,$(VS_PLATFORMS),$(LIBSUBDIR)/$(p)/vpx.dll)
INSTALL-LIBS-$(CONFIG_SHARED) += $(foreach p,$(VS_PLATFORMS),$(LIBSUBDIR)/$(p)/vpx.exp)
endif
else
INSTALL-LIBS-$(CONFIG_STATIC) += $(LIBSUBDIR)/libvpx.a
ifeq ($(CONFIG_STATIC),yes)
INSTALL-LIBS-$(CONFIG_DEBUG_LIBS) += $(LIBSUBDIR)/libvpx_g.a
endif
endif

CODEC_SRCS=$(call enabled,CODEC_SRCS)

INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(CODEC_SRCS)
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(call enabled,CODEC_EXPORTS)


# Generate a list of all enabled sources, in particular for exporting to gyp
# based build systems.
libvpx_srcs.txt:
	@echo "    [CREATE] $@"
	@echo $(CODEC_SRCS) | xargs -n1 echo | LC_ALL=C sort -u > $@
CLEAN-OBJS += libvpx_srcs.txt

libvpxrc_srcs.txt:
	@echo "    [CREATE] $@"
	@echo $(RC_RTC_SRCS) | xargs -n1 echo | LC_ALL=C sort -u > $@
CLEAN-OBJS += libvpxrc_srcs.txt

# Assembly files that are included, but don't define symbols themselves.
# Filtered out to avoid Windows build warnings.
ASM_INCLUDES := \
    third_party/x86inc/x86inc.asm \
    vpx_config.asm \
    vpx_ports/x86_abi_support.asm \
    vpx_dsp/x86/bitdepth_conversion_sse2.asm \

ifeq ($(CONFIG_EXTERNAL_BUILD),yes)
ifeq ($(CONFIG_MSVS),yes)

vpx.def: $(call enabled,CODEC_EXPORTS)
	@echo "    [CREATE] $@"
	$(qexec)$(SRC_PATH_BARE)/build/make/gen_msvs_def.sh\
            --name=vpx\
            --out=$@ $^
CLEAN-OBJS += vpx.def

vpx.$(VCPROJ_SFX): VCPROJ_SRCS=$(filter-out $(addprefix %, $(ASM_INCLUDES)), $^)

vpx.$(VCPROJ_SFX): $(CODEC_SRCS) vpx.def
	@echo "    [CREATE] $@"
	$(qexec)$(GEN_VCPROJ) \
            $(if $(CONFIG_SHARED),--dll,--lib) \
            --target=$(TOOLCHAIN) \
            $(if $(CONFIG_STATIC_MSVCRT),--static-crt) \
            --name=vpx \
            --proj-guid=DCE19DAF-69AC-46DB-B14A-39F0FAA5DB74 \
            --module-def=vpx.def \
            --ver=$(CONFIG_VS_VERSION) \
            --src-path-bare="$(SRC_PATH_BARE)" \
            --out=$@ $(CFLAGS) \
            --as=$(AS) \
            $(filter $(SRC_PATH_BARE)/vp8/%.c, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vp8/%.h, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vp9/%.c, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vp9/%.h, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vpx/%, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vpx_dsp/%, $(VCPROJ_SRCS)) \
            $(filter-out $(addprefix $(SRC_PATH_BARE)/, \
                           vp8/%.c vp8/%.h vp9/%.c vp9/%.h vpx/% vpx_dsp/%), \
              $(VCPROJ_SRCS)) \
            --src-path-bare="$(SRC_PATH_BARE)" \

PROJECTS-yes += vpx.$(VCPROJ_SFX)

vpx.$(VCPROJ_SFX): vpx_config.asm
vpx.$(VCPROJ_SFX): $(RTCD)

vpxrc.$(VCPROJ_SFX): \
    VCPROJ_SRCS=$(filter-out $(addprefix %, $(ASM_INCLUDES)), $^)

vpxrc.$(VCPROJ_SFX): $(RC_RTC_SRCS)
	@echo "    [CREATE] $@"
	$(qexec)$(GEN_VCPROJ) \
            $(if $(CONFIG_SHARED),--dll,--lib) \
            --target=$(TOOLCHAIN) \
            $(if $(CONFIG_STATIC_MSVCRT),--static-crt) \
            --name=vpxrc \
            --proj-guid=C26FF952-9494-4838-9A3F-7F3D4F613385 \
            --ver=$(CONFIG_VS_VERSION) \
            --src-path-bare="$(SRC_PATH_BARE)" \
            --out=$@ $(CFLAGS) \
            --as=$(AS) \
            $(filter $(SRC_PATH_BARE)/vp9/%.c, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vp9/%.cc, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vp9/%.h, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vpx/%, $(VCPROJ_SRCS)) \
            $(filter $(SRC_PATH_BARE)/vpx_dsp/%, $(VCPROJ_SRCS)) \
            $(filter-out $(addprefix $(SRC_PATH_BARE)/, \
                           vp8/%.c vp8/%.h vp9/%.c vp9/%.cc vp9/%.h vpx/% \
                           vpx_dsp/%), \
              $(VCPROJ_SRCS)) \
            --src-path-bare="$(SRC_PATH_BARE)" \

PROJECTS-yes += vpxrc.$(VCPROJ_SFX)

vpxrc.$(VCPROJ_SFX): vpx_config.asm
vpxrc.$(VCPROJ_SFX): $(RTCD)

endif # ifeq ($(CONFIG_MSVS),yes)
else # ifeq ($(CONFIG_EXTERNAL_BUILD),yes)
LIBVPX_OBJS=$(call objs, $(filter-out $(ASM_INCLUDES), $(CODEC_SRCS)))
OBJS-yes += $(LIBVPX_OBJS)
LIBS-$(if yes,$(CONFIG_STATIC)) += $(BUILD_PFX)libvpx.a $(BUILD_PFX)libvpx_g.a
$(BUILD_PFX)libvpx_g.a: $(LIBVPX_OBJS)

# Updating version info.
# https://www.gnu.org/software/libtool/manual/libtool.html#Updating-version-info
# For libtool: c=<current>, a=<age>, r=<revision>
# libtool generates .so file as .so.[c-a].a.r, while -version-info c:r:a is
# passed to libtool.
#
# libvpx library file is generated as libvpx.so.<MAJOR>.<MINOR>.<PATCH>
# MAJOR = c-a, MINOR = a, PATCH = r
#
# To determine SO_VERSION_{MAJOR,MINOR,PATCH}, calculate c,a,r with current
# SO_VERSION_* then follow the rules in the link to detemine the new version
# (c1, a1, r1) and set MAJOR to [c1-a1], MINOR to a1 and PATCH to r1
SO_VERSION_MAJOR := 12
SO_VERSION_MINOR := 0
SO_VERSION_PATCH := 0
ifeq ($(filter darwin%,$(TGT_OS)),$(TGT_OS))
LIBVPX_SO               := libvpx.$(SO_VERSION_MAJOR).dylib
SHARED_LIB_SUF          := .dylib
EXPORT_FILE             := libvpx.syms
LIBVPX_SO_SYMLINKS      := $(addprefix $(LIBSUBDIR)/, \
                             libvpx.dylib  )
else
ifeq ($(filter iphonesimulator%,$(TGT_OS)),$(TGT_OS))
LIBVPX_SO               := libvpx.$(SO_VERSION_MAJOR).dylib
SHARED_LIB_SUF          := .dylib
EXPORT_FILE             := libvpx.syms
LIBVPX_SO_SYMLINKS      := $(addprefix $(LIBSUBDIR)/, libvpx.dylib)
else
ifeq ($(filter os2%,$(TGT_OS)),$(TGT_OS))
LIBVPX_SO               := libvpx$(SO_VERSION_MAJOR).dll
SHARED_LIB_SUF          := _dll.a
EXPORT_FILE             := libvpx.def
LIBVPX_SO_SYMLINKS      :=
LIBVPX_SO_IMPLIB        := libvpx_dll.a
else
LIBVPX_SO               := libvpx.so.$(SO_VERSION_MAJOR).$(SO_VERSION_MINOR).$(SO_VERSION_PATCH)
SHARED_LIB_SUF          := .so
EXPORT_FILE             := libvpx.ver
LIBVPX_SO_SYMLINKS      := $(addprefix $(LIBSUBDIR)/, \
                             libvpx.so libvpx.so.$(SO_VERSION_MAJOR) \
                             libvpx.so.$(SO_VERSION_MAJOR).$(SO_VERSION_MINOR))
endif
endif
endif

LIBS-$(CONFIG_SHARED) += $(BUILD_PFX)$(LIBVPX_SO)\
                           $(notdir $(LIBVPX_SO_SYMLINKS)) \
                           $(if $(LIBVPX_SO_IMPLIB), $(BUILD_PFX)$(LIBVPX_SO_IMPLIB))
$(BUILD_PFX)$(LIBVPX_SO): $(LIBVPX_OBJS) $(EXPORT_FILE)
$(BUILD_PFX)$(LIBVPX_SO): extralibs += -lm
$(BUILD_PFX)$(LIBVPX_SO): SONAME = libvpx.so.$(SO_VERSION_MAJOR)
$(BUILD_PFX)$(LIBVPX_SO): EXPORTS_FILE = $(EXPORT_FILE)

libvpx.def: $(call enabled,CODEC_EXPORTS)
	@echo "    [CREATE] $@"
	$(qexec)echo LIBRARY $(LIBVPX_SO:.dll=) INITINSTANCE TERMINSTANCE > $@
	$(qexec)echo "DATA MULTIPLE NONSHARED" >> $@
	$(qexec)echo "EXPORTS" >> $@
	$(qexec)awk '!/vpx_svc_*/ {print "_"$$2}' $^ >>$@
CLEAN-OBJS += libvpx.def

libvpx_dll.a: $(LIBVPX_SO)
	@echo "    [IMPLIB] $@"
	$(qexec)emximp -o $@ $<
CLEAN-OBJS += libvpx_dll.a

define libvpx_symlink_template
$(1): $(2)
	@echo "    [LN]     $(2) $$@"
	$(qexec)mkdir -p $$(dir $$@)
	$(qexec)ln -sf $(2) $$@
endef

$(eval $(call libvpx_symlink_template,\
    $(addprefix $(BUILD_PFX),$(notdir $(LIBVPX_SO_SYMLINKS))),\
    $(BUILD_PFX)$(LIBVPX_SO)))
$(eval $(call libvpx_symlink_template,\
    $(addprefix $(DIST_DIR)/,$(LIBVPX_SO_SYMLINKS)),\
    $(LIBVPX_SO)))


INSTALL-LIBS-$(CONFIG_SHARED) += $(LIBVPX_SO_SYMLINKS)
INSTALL-LIBS-$(CONFIG_SHARED) += $(LIBSUBDIR)/$(LIBVPX_SO)
INSTALL-LIBS-$(CONFIG_SHARED) += $(if $(LIBVPX_SO_IMPLIB),$(LIBSUBDIR)/$(LIBVPX_SO_IMPLIB))


LIBS-yes += vpx.pc
vpx.pc: config.mk libs.mk
	@echo "    [CREATE] $@"
	$(qexec)echo '# pkg-config file from libvpx $(VERSION_STRING)' > $@
	$(qexec)echo 'prefix=$(PREFIX)' >> $@
	$(qexec)echo 'exec_prefix=$${prefix}' >> $@
	$(qexec)echo 'libdir=$${prefix}/$(LIBSUBDIR)' >> $@
	$(qexec)echo 'includedir=$${prefix}/include' >> $@
	$(qexec)echo '' >> $@
	$(qexec)echo 'Name: vpx' >> $@
	$(qexec)echo 'Description: WebM Project VPx codec implementation' >> $@
	$(qexec)echo 'Version: $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)' >> $@
	$(qexec)echo 'Requires:' >> $@
	$(qexec)echo 'Conflicts:' >> $@
	$(qexec)echo 'Libs: -L$${libdir} -lvpx -lm' >> $@
ifeq ($(HAVE_PTHREAD_H),yes)
	$(qexec)echo 'Libs.private: -lm -lpthread' >> $@
else
	$(qexec)echo 'Libs.private: -lm' >> $@
endif
	$(qexec)echo 'Cflags: -I$${includedir}' >> $@
INSTALL-LIBS-yes += $(LIBSUBDIR)/pkgconfig/vpx.pc
INSTALL_MAPS += $(LIBSUBDIR)/pkgconfig/%.pc %.pc
CLEAN-OBJS += vpx.pc

ifeq ($(CONFIG_ENCODERS),yes)
  RC_RTC_OBJS=$(call objs,$(RC_RTC_SRCS))
  OBJS-yes += $(RC_RTC_OBJS)
  LIBS-yes += $(BUILD_PFX)libvpxrc.a $(BUILD_PFX)libvpxrc_g.a
  $(BUILD_PFX)libvpxrc_g.a: $(RC_RTC_OBJS)
endif

endif # ifeq ($(CONFIG_EXTERNAL_BUILD),yes)

libvpx.ver: $(call enabled,CODEC_EXPORTS)
	@echo "    [CREATE] $@"
	$(qexec)echo "{ global:" > $@
	$(qexec)for f in $?; do awk '{print $$2";"}' < $$f >>$@; done
	$(qexec)echo "local: *; };" >> $@
CLEAN-OBJS += libvpx.ver

libvpx.syms: $(call enabled,CODEC_EXPORTS)
	@echo "    [CREATE] $@"
	$(qexec)awk '{print "_"$$2}' $^ >$@
CLEAN-OBJS += libvpx.syms

#
# Rule to make assembler configuration file from C configuration file
#
ifeq ($(VPX_ARCH_X86)$(VPX_ARCH_X86_64),yes)
# YASM
$(BUILD_PFX)vpx_config.asm: $(BUILD_PFX)vpx_config.h
	@echo "    [CREATE] $@"
	@LC_ALL=C grep -E "#define [A-Z0-9_]+ [01]" $< \
	    | awk '{print $$2 " equ " $$3}' > $@
else
ADS2GAS=$(if $(filter yes,$(CONFIG_GCC)),| $(ASM_CONVERSION))
$(BUILD_PFX)vpx_config.asm: $(BUILD_PFX)vpx_config.h
	@echo "    [CREATE] $@"
	@LC_ALL=C grep -E "#define [A-Z0-9_]+ [01]" $< \
	    | awk '{print $$2 " EQU " $$3}' $(ADS2GAS) > $@
	@echo "        END" $(ADS2GAS) >> $@
CLEAN-OBJS += $(BUILD_PFX)vpx_config.asm
endif

#
# Add assembler dependencies for configuration.
#
$(filter %.S.o,$(OBJS-yes)):     $(BUILD_PFX)vpx_config.asm
$(filter %$(ASM).o,$(OBJS-yes)): $(BUILD_PFX)vpx_config.asm


$(shell $(SRC_PATH_BARE)/build/make/version.sh "$(SRC_PATH_BARE)" $(BUILD_PFX)vpx_version.h)
CLEAN-OBJS += $(BUILD_PFX)vpx_version.h

#
# Add include path for libwebm sources.
#
ifeq ($(CONFIG_WEBM_IO),yes)
  CXXFLAGS += -I$(SRC_PATH_BARE)/third_party/libwebm
endif

##
## libvpx test directives
##
ifeq ($(CONFIG_UNIT_TESTS),yes)
LIBVPX_TEST_DATA_PATH ?= .

include $(SRC_PATH_BARE)/test/test.mk

# addprefix_clean behaves like addprefix if the target doesn't start with "../"
# However, if the target starts with "../", instead of adding prefix,
# it will remove "../".
# Using addprefix_clean, we can avoid two different targets building the
# same file, i.e.
# test/../ivfenc.c.d: ivfenc.o
# ivfenc.c.d: ivfenc.o
# Note that the other way to solve this problem is using "realpath".
# The "realpath" is supported by make 3.81 or later.
addprefix_clean=$(patsubst $(1)../%,%,$(addprefix $(1), $(2)))
LIBVPX_TEST_SRCS=$(call addprefix_clean,test/,$(call enabled,LIBVPX_TEST_SRCS))

LIBVPX_TEST_BIN=./test_libvpx$(EXE_SFX)
LIBVPX_TEST_DATA=$(addprefix $(LIBVPX_TEST_DATA_PATH)/,\
                     $(call enabled,LIBVPX_TEST_DATA))
libvpx_test_data_url=https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/$(1)

TEST_INTRA_PRED_SPEED_BIN=./test_intra_pred_speed$(EXE_SFX)
TEST_INTRA_PRED_SPEED_SRCS=$(call addprefix_clean,test/,\
                           $(call enabled,TEST_INTRA_PRED_SPEED_SRCS))
TEST_INTRA_PRED_SPEED_OBJS := $(sort $(call objs,$(TEST_INTRA_PRED_SPEED_SRCS)))

ifeq ($(CONFIG_ENCODERS),yes)
RC_INTERFACE_TEST_BIN=./test_rc_interface$(EXE_SFX)
RC_INTERFACE_TEST_SRCS=$(call addprefix_clean,test/,\
                       $(call enabled,RC_INTERFACE_TEST_SRCS))
RC_INTERFACE_TEST_OBJS := $(sort $(call objs,$(RC_INTERFACE_TEST_SRCS)))
endif

libvpx_test_srcs.txt:
	@echo "    [CREATE] $@"
	@echo $(LIBVPX_TEST_SRCS) | xargs -n1 echo | LC_ALL=C sort -u > $@
CLEAN-OBJS += libvpx_test_srcs.txt

# Attempt to download the file using curl, retrying once if it fails for a
# partial file (18).
$(LIBVPX_TEST_DATA): $(SRC_PATH_BARE)/test/test-data.sha1
	@echo "    [DOWNLOAD] $@"
	$(qexec)( \
	  trap 'rm -f $@' INT TERM; \
	  curl="curl -S -s --retry 1 -L -o $@ $(call libvpx_test_data_url,$(@F))"; \
	  $$curl; ret=$$?; \
	  case "$$ret" in \
	    18) $$curl -C - ;; \
	    *) exit $$ret ;; \
	  esac \
	)

testdata: $(LIBVPX_TEST_DATA)
	$(qexec)[ -x "$$(which sha1sum)" ] && sha1sum=sha1sum;\
          [ -x "$$(which shasum)" ] && sha1sum=shasum;\
          [ -x "$$(which sha1)" ] && sha1sum=sha1;\
          if [ -n "$${sha1sum}" ]; then\
            set -e;\
            echo "Checking test data:";\
            for f in $(call enabled,LIBVPX_TEST_DATA); do\
                grep $$f $(SRC_PATH_BARE)/test/test-data.sha1 |\
                    (cd "$(LIBVPX_TEST_DATA_PATH)"; $${sha1sum} -c);\
            done; \
        else\
            echo "Skipping test data integrity check, sha1sum not found.";\
        fi

ifeq ($(CONFIG_EXTERNAL_BUILD),yes)
ifeq ($(CONFIG_MSVS),yes)

gtest.$(VCPROJ_SFX): $(SRC_PATH_BARE)/third_party/googletest/src/src/gtest-all.cc
	@echo "    [CREATE] $@"
	$(qexec)$(GEN_VCPROJ) \
            --lib \
            --target=$(TOOLCHAIN) \
            $(if $(CONFIG_STATIC_MSVCRT),--static-crt) \
            --name=gtest \
            --proj-guid=EC00E1EC-AF68-4D92-A255-181690D1C9B1 \
            --ver=$(CONFIG_VS_VERSION) \
            --src-path-bare="$(SRC_PATH_BARE)" \
            --as=$(AS) \
            -D_VARIADIC_MAX=10 \
            --out=gtest.$(VCPROJ_SFX) $(SRC_PATH_BARE)/third_party/googletest/src/src/gtest-all.cc \
            -I. -I"$(SRC_PATH_BARE)/third_party/googletest/src/include" -I"$(SRC_PATH_BARE)/third_party/googletest/src"

PROJECTS-$(CONFIG_MSVS) += gtest.$(VCPROJ_SFX)

test_libvpx.$(VCPROJ_SFX): $(LIBVPX_TEST_SRCS) vpx.$(VCPROJ_SFX) gtest.$(VCPROJ_SFX)
	@echo "    [CREATE] $@"
	$(qexec)$(GEN_VCPROJ) \
            --exe \
            --target=$(TOOLCHAIN) \
            --name=test_libvpx \
            -D_VARIADIC_MAX=10 \
            --proj-guid=CD837F5F-52D8-4314-A370-895D614166A7 \
            --ver=$(CONFIG_VS_VERSION) \
            --src-path-bare="$(SRC_PATH_BARE)" \
            --as=$(AS) \
            $(if $(CONFIG_STATIC_MSVCRT),--static-crt) \
            --out=$@ $(INTERNAL_CFLAGS) $(CFLAGS) \
            -I. -I"$(SRC_PATH_BARE)/third_party/googletest/src/include" \
            $(if $(CONFIG_WEBM_IO),-I"$(SRC_PATH_BARE)/third_party/libwebm") \
            -L. -l$(CODEC_LIB) -l$(GTEST_LIB) $^

PROJECTS-$(CONFIG_MSVS) += test_libvpx.$(VCPROJ_SFX)

LIBVPX_TEST_BIN := $(addprefix $(TGT_OS:win64=x64)/Release/,$(notdir $(LIBVPX_TEST_BIN)))

ifneq ($(strip $(TEST_INTRA_PRED_SPEED_OBJS)),)
PROJECTS-$(CONFIG_MSVS) += test_intra_pred_speed.$(VCPROJ_SFX)
test_intra_pred_speed.$(VCPROJ_SFX): $(TEST_INTRA_PRED_SPEED_SRCS) vpx.$(VCPROJ_SFX) gtest.$(VCPROJ_SFX)
	@echo "    [CREATE] $@"
	$(qexec)$(GEN_VCPROJ) \
            --exe \
            --target=$(TOOLCHAIN) \
            --name=test_intra_pred_speed \
            -D_VARIADIC_MAX=10 \
            --proj-guid=CD837F5F-52D8-4314-A370-895D614166A7 \
            --ver=$(CONFIG_VS_VERSION) \
            --src-path-bare="$(SRC_PATH_BARE)" \
            --as=$(AS) \
            $(if $(CONFIG_STATIC_MSVCRT),--static-crt) \
            --out=$@ $(INTERNAL_CFLAGS) $(CFLAGS) \
            -I. -I"$(SRC_PATH_BARE)/third_party/googletest/src/include" \
            -L. -l$(CODEC_LIB) -l$(GTEST_LIB) $^
endif  # TEST_INTRA_PRED_SPEED

ifeq ($(CONFIG_ENCODERS),yes)
ifneq ($(strip $(RC_INTERFACE_TEST_OBJS)),)
PROJECTS-$(CONFIG_MSVS) += test_rc_interface.$(VCPROJ_SFX)
test_rc_interface.$(VCPROJ_SFX): $(RC_INTERFACE_TEST_SRCS) vpx.$(VCPROJ_SFX) \
	vpxrc.$(VCPROJ_SFX) gtest.$(VCPROJ_SFX)
	@echo "    [CREATE] $@"
	$(qexec)$(GEN_VCPROJ) \
            --exe \
            --target=$(TOOLCHAIN) \
            --name=test_rc_interface \
            -D_VARIADIC_MAX=10 \
            --proj-guid=30458F88-1BC6-4689-B41C-50F3737AAB27 \
            --ver=$(CONFIG_VS_VERSION) \
            --as=$(AS) \
            --src-path-bare="$(SRC_PATH_BARE)" \
            $(if $(CONFIG_STATIC_MSVCRT),--static-crt) \
            --out=$@ $(INTERNAL_CFLAGS) $(CFLAGS) \
            -I. -I"$(SRC_PATH_BARE)/third_party/googletest/src/include" \
            -L. -l$(CODEC_LIB) -l$(RC_RTC_LIB) -l$(GTEST_LIB) $^
endif  # RC_INTERFACE_TEST
endif  # CONFIG_ENCODERS
endif  # CONFIG_MSVS
else

include $(SRC_PATH_BARE)/third_party/googletest/gtest.mk
GTEST_SRCS := $(addprefix third_party/googletest/src/,$(call enabled,GTEST_SRCS))
GTEST_OBJS=$(call objs,$(GTEST_SRCS))
ifeq ($(filter win%,$(TGT_OS)),$(TGT_OS))
# Disabling pthreads globally will cause issues on darwin and possibly elsewhere
$(GTEST_OBJS) $(GTEST_OBJS:.o=.d): CXXFLAGS += -DGTEST_HAS_PTHREAD=0
endif
GTEST_INCLUDES := -I$(SRC_PATH_BARE)/third_party/googletest/src
GTEST_INCLUDES += -I$(SRC_PATH_BARE)/third_party/googletest/src/include
$(GTEST_OBJS) $(GTEST_OBJS:.o=.d): CXXFLAGS += $(GTEST_INCLUDES)
OBJS-yes += $(GTEST_OBJS)
LIBS-yes += $(BUILD_PFX)libgtest.a $(BUILD_PFX)libgtest_g.a
$(BUILD_PFX)libgtest_g.a: $(GTEST_OBJS)

LIBVPX_TEST_OBJS=$(sort $(call objs,$(LIBVPX_TEST_SRCS)))
$(LIBVPX_TEST_OBJS) $(LIBVPX_TEST_OBJS:.o=.d): CXXFLAGS += $(GTEST_INCLUDES)
OBJS-yes += $(LIBVPX_TEST_OBJS)
BINS-yes += $(LIBVPX_TEST_BIN)

CODEC_LIB=$(if $(CONFIG_DEBUG_LIBS),vpx_g,vpx)
CODEC_LIB_SUF=$(if $(CONFIG_SHARED),$(SHARED_LIB_SUF),.a)
TEST_LIBS := lib$(CODEC_LIB)$(CODEC_LIB_SUF) libgtest.a
$(LIBVPX_TEST_BIN): $(TEST_LIBS)
$(eval $(call linkerxx_template,$(LIBVPX_TEST_BIN), \
              $(LIBVPX_TEST_OBJS) \
              -L. -lvpx -lgtest $(extralibs) -lm))

ifneq ($(strip $(TEST_INTRA_PRED_SPEED_OBJS)),)
$(TEST_INTRA_PRED_SPEED_OBJS) $(TEST_INTRA_PRED_SPEED_OBJS:.o=.d): CXXFLAGS += $(GTEST_INCLUDES)
OBJS-yes += $(TEST_INTRA_PRED_SPEED_OBJS)
BINS-yes += $(TEST_INTRA_PRED_SPEED_BIN)

$(TEST_INTRA_PRED_SPEED_BIN): $(TEST_LIBS)
$(eval $(call linkerxx_template,$(TEST_INTRA_PRED_SPEED_BIN), \
              $(TEST_INTRA_PRED_SPEED_OBJS) \
              -L. -lvpx -lgtest $(extralibs) -lm))
endif  # TEST_INTRA_PRED_SPEED

ifeq ($(CONFIG_ENCODERS),yes)
ifneq ($(strip $(RC_INTERFACE_TEST_OBJS)),)
$(RC_INTERFACE_TEST_OBJS) $(RC_INTERFACE_TEST_OBJS:.o=.d): \
  CXXFLAGS += $(GTEST_INCLUDES)
OBJS-yes += $(RC_INTERFACE_TEST_OBJS)
BINS-yes += $(RC_INTERFACE_TEST_BIN)

$(RC_INTERFACE_TEST_BIN): $(TEST_LIBS) libvpxrc.a
$(eval $(call linkerxx_template,$(RC_INTERFACE_TEST_BIN), \
              $(RC_INTERFACE_TEST_OBJS) \
              -L. -lvpx -lgtest -lvpxrc $(extralibs) -lm))
endif  # RC_INTERFACE_TEST
endif  # CONFIG_ENCODERS

endif  # CONFIG_EXTERNAL_BUILD

# Install test sources only if codec source is included
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(patsubst $(SRC_PATH_BARE)/%,%,\
    $(shell find $(SRC_PATH_BARE)/third_party/googletest -type f))
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(LIBVPX_TEST_SRCS)
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(TEST_INTRA_PRED_SPEED_SRCS)
INSTALL-SRCS-$(CONFIG_CODEC_SRCS) += $(RC_INTERFACE_TEST_SRCS)

define test_shard_template
test: test_shard.$(1)
test-no-data-check: test_shard_ndc.$(1)
test_shard.$(1) test_shard_ndc.$(1): $(LIBVPX_TEST_BIN)
	@set -e; \
	 export GTEST_SHARD_INDEX=$(1); \
	 export GTEST_TOTAL_SHARDS=$(2); \
	 $(LIBVPX_TEST_BIN)
test_shard.$(1): testdata
.PHONY: test_shard.$(1) test_shard_ndc.$(1)
endef

NUM_SHARDS := 10
SHARDS := 0 1 2 3 4 5 6 7 8 9
$(foreach s,$(SHARDS),$(eval $(call test_shard_template,$(s),$(NUM_SHARDS))))

endif  # CONFIG_UNIT_TESTS

##
## documentation directives
##
CLEAN-OBJS += libs.doxy
DOCS-yes += libs.doxy
libs.doxy: $(CODEC_DOC_SRCS)
	@echo "    [CREATE] $@"
	@rm -f $@
	@echo "INPUT += $^" >> $@
	@echo "INCLUDE_PATH += ." >> $@;
	@echo "ENABLED_SECTIONS += $(sort $(CODEC_DOC_SECTIONS))" >> $@

## Generate rtcd.h for all objects
ifeq ($(CONFIG_DEPENDENCY_TRACKING),yes)
$(OBJS-yes:.o=.d): $(RTCD)
else
$(OBJS-yes): $(RTCD)
endif

## Update the global src list
SRCS += $(CODEC_SRCS) $(LIBVPX_TEST_SRCS) $(GTEST_SRCS)
SRCS += $(RC_INTERFACE_TEST_SRCS)

##
## vpxdec/vpxenc tests.
##
ifeq ($(CONFIG_UNIT_TESTS),yes)
TEST_BIN_PATH = .
ifeq ($(CONFIG_MSVS),yes)
# MSVC will build both Debug and Release configurations of tools in a
# sub directory named for the current target. Assume the user wants to
# run the Release tools, and assign TEST_BIN_PATH accordingly.
# TODO(tomfinegan): Is this adequate for ARM?
# TODO(tomfinegan): Support running the debug versions of tools?
TEST_BIN_PATH := $(addsuffix /$(TGT_OS:win64=x64)/Release, $(TEST_BIN_PATH))
endif
utiltest utiltest-no-data-check:
	$(qexec)$(SRC_PATH_BARE)/test/vpxdec.sh \
		--test-data-path "$(LIBVPX_TEST_DATA_PATH)" \
		--bin-path $(TEST_BIN_PATH)
	$(qexec)$(SRC_PATH_BARE)/test/vpxenc.sh \
		--test-data-path "$(LIBVPX_TEST_DATA_PATH)" \
		--bin-path $(TEST_BIN_PATH)
utiltest: testdata
else
utiltest utiltest-no-data-check:
	@echo Unit tests must be enabled to make the utiltest target.
endif

##
## Example tests.
##
ifeq ($(CONFIG_UNIT_TESTS),yes)
# All non-MSVC targets output example targets in a sub dir named examples.
EXAMPLES_BIN_PATH = examples
ifeq ($(CONFIG_MSVS),yes)
# MSVC will build both Debug and Release configurations of the examples in a
# sub directory named for the current target. Assume the user wants to
# run the Release tools, and assign EXAMPLES_BIN_PATH accordingly.
# TODO(tomfinegan): Is this adequate for ARM?
# TODO(tomfinegan): Support running the debug versions of tools?
EXAMPLES_BIN_PATH := $(TGT_OS:win64=x64)/Release
endif
exampletest exampletest-no-data-check: examples
	$(qexec)$(SRC_PATH_BARE)/test/examples.sh \
		--test-data-path "$(LIBVPX_TEST_DATA_PATH)" \
		--bin-path $(EXAMPLES_BIN_PATH)
exampletest: testdata
else
exampletest exampletest-no-data-check:
	@echo Unit tests must be enabled to make the exampletest target.
endif
