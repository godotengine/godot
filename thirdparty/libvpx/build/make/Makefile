##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


include config.mk
quiet?=true
ifeq ($(target),)
# If a target wasn't specified, invoke for all enabled targets.
.DEFAULT:
	@for t in $(ALL_TARGETS); do \
	     $(MAKE) --no-print-directory target=$$t $(MAKECMDGOALS) || exit $$?;\
        done
all: .DEFAULT
clean:: .DEFAULT
exampletest: .DEFAULT
install:: .DEFAULT
test: .DEFAULT
test-no-data-check: .DEFAULT
testdata: .DEFAULT
utiltest: .DEFAULT
exampletest-no-data-check utiltest-no-data-check: .DEFAULT
test_%: .DEFAULT ;

# Note: md5sum is not installed on OS X, but openssl is. Openssl may not be
# installed on cygwin, so we need to autodetect here.
md5sum := $(firstword $(wildcard \
          $(foreach e,md5sum openssl,\
          $(foreach p,$(subst :, ,$(PATH)),$(p)/$(e)*))\
          ))
md5sum := $(if $(filter %openssl,$(md5sum)),$(md5sum) dgst -md5,$(md5sum))

TGT_CC:=$(word 3, $(subst -, ,$(TOOLCHAIN)))
dist:
	@for t in $(ALL_TARGETS); do \
	     $(MAKE) --no-print-directory target=$$t $(MAKECMDGOALS) || exit $$?;\
        done
        # Run configure for the user with the current toolchain.
	@if [ -d "$(DIST_DIR)/src" ]; then \
            mkdir -p "$(DIST_DIR)/build"; \
            cd "$(DIST_DIR)/build"; \
            echo "Rerunning configure $(CONFIGURE_ARGS)"; \
            ../src/configure $(CONFIGURE_ARGS); \
            $(if $(filter vs%,$(TGT_CC)),make NO_LAUNCH_DEVENV=1;) \
        fi
	@if [ -d "$(DIST_DIR)" ]; then \
            echo "    [MD5SUM] $(DIST_DIR)"; \
	    cd $(DIST_DIR) && \
	    $(md5sum) `find . -name md5sums.txt -prune -o -type f -print` \
                | sed -e 's/MD5(\(.*\))= \([0-9a-f]\{32\}\)/\2  \1/' \
                > md5sums.txt;\
        fi
endif

# Since we invoke make recursively for multiple targets we need to include the
# .mk file for the correct target, but only when $(target) is non-empty.
ifneq ($(target),)
include $(target)-$(TOOLCHAIN).mk
endif
BUILD_ROOT?=.
VPATH=$(SRC_PATH_BARE)
CFLAGS+=-I$(BUILD_PFX)$(BUILD_ROOT) -I$(SRC_PATH)
CXXFLAGS+=-I$(BUILD_PFX)$(BUILD_ROOT) -I$(SRC_PATH)
ASFLAGS+=-I$(BUILD_PFX)$(BUILD_ROOT)/ -I$(SRC_PATH)/
DIST_DIR?=dist
HOSTCC?=gcc
TGT_ISA:=$(word 1, $(subst -, ,$(TOOLCHAIN)))
TGT_OS:=$(word 2, $(subst -, ,$(TOOLCHAIN)))
TGT_CC:=$(word 3, $(subst -, ,$(TOOLCHAIN)))
quiet:=$(if $(or $(verbose), $(V)),, yes)
qexec=$(if $(quiet),@)

# Cancel built-in implicit rules
%: %.o
%.asm:
%.a:
%: %.cc

#
# Common rules"
#
.PHONY: all
all:

.PHONY: clean
clean::
	rm -f $(OBJS-yes) $(OBJS-yes:.o=.d) $(OBJS-yes:.asm.S.o=.asm.S)
	rm -f $(CLEAN-OBJS)

.PHONY: clean
distclean: clean
	if [ -z "$(target)" ]; then \
      rm -f Makefile; \
      rm -f config.log config.mk; \
      rm -f vpx_config.[hc] vpx_config.asm; \
      rm -f arm_neon.h; \
    else \
      rm -f $(target)-$(TOOLCHAIN).mk; \
    fi

.PHONY: dist
dist:
.PHONY: exampletest
exampletest:
.PHONY: install
install::
.PHONY: test
test:
.PHONY: testdata
testdata:
.PHONY: utiltest
utiltest:
.PHONY: test-no-data-check exampletest-no-data-check utiltest-no-data-check
test-no-data-check:
exampletest-no-data-check utiltest-no-data-check:

# Force to realign stack always on OS/2
ifeq ($(TOOLCHAIN), x86-os2-gcc)
CFLAGS += -mstackrealign
endif

# x86[_64]
$(BUILD_PFX)%_mmx.c.d: CFLAGS += -mmmx
$(BUILD_PFX)%_mmx.c.o: CFLAGS += -mmmx
$(BUILD_PFX)%_sse2.c.d: CFLAGS += -msse2
$(BUILD_PFX)%_sse2.c.o: CFLAGS += -msse2
$(BUILD_PFX)%_sse3.c.d: CFLAGS += -msse3
$(BUILD_PFX)%_sse3.c.o: CFLAGS += -msse3
$(BUILD_PFX)%_ssse3.c.d: CFLAGS += -mssse3
$(BUILD_PFX)%_ssse3.c.o: CFLAGS += -mssse3
$(BUILD_PFX)%_sse4.c.d: CFLAGS += -msse4.1
$(BUILD_PFX)%_sse4.c.o: CFLAGS += -msse4.1
$(BUILD_PFX)%_avx.c.d: CFLAGS += -mavx
$(BUILD_PFX)%_avx.c.o: CFLAGS += -mavx
$(BUILD_PFX)%_avx2.c.d: CFLAGS += -mavx2
$(BUILD_PFX)%_avx2.c.o: CFLAGS += -mavx2
$(BUILD_PFX)%_avx512.c.d: CFLAGS += -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl
$(BUILD_PFX)%_avx512.c.o: CFLAGS += -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl

# AARCH64
$(BUILD_PFX)%_neon_dotprod.c.d: CFLAGS += -march=armv8.2-a+dotprod
$(BUILD_PFX)%_neon_dotprod.c.o: CFLAGS += -march=armv8.2-a+dotprod
$(BUILD_PFX)%_neon_i8mm.c.d: CFLAGS += -march=armv8.2-a+dotprod+i8mm
$(BUILD_PFX)%_neon_i8mm.c.o: CFLAGS += -march=armv8.2-a+dotprod+i8mm
$(BUILD_PFX)%_sve.c.d: CFLAGS += -march=armv8.2-a+dotprod+i8mm+sve
$(BUILD_PFX)%_sve.c.o: CFLAGS += -march=armv8.2-a+dotprod+i8mm+sve
$(BUILD_PFX)%_sve2.c.d: CFLAGS += -march=armv9-a+i8mm+sve2
$(BUILD_PFX)%_sve2.c.o: CFLAGS += -march=armv9-a+i8mm+sve2

# POWER
$(BUILD_PFX)%_vsx.c.d: CFLAGS += -maltivec -mvsx
$(BUILD_PFX)%_vsx.c.o: CFLAGS += -maltivec -mvsx

# MIPS
$(BUILD_PFX)%_msa.c.d: CFLAGS += -mmsa
$(BUILD_PFX)%_msa.c.o: CFLAGS += -mmsa

# LOONGARCH
$(BUILD_PFX)%_lsx.c.d:  CFLAGS += -mlsx
$(BUILD_PFX)%_lsx.c.o:  CFLAGS += -mlsx
$(BUILD_PFX)%_lasx.c.d: CFLAGS += -mlasx
$(BUILD_PFX)%_lasx.c.o: CFLAGS += -mlasx

$(BUILD_PFX)%.c.d: %.c
	$(if $(quiet),@echo "    [DEP] $@")
	$(qexec)mkdir -p $(dir $@)
	$(qexec)$(CC) $(INTERNAL_CFLAGS) $(CFLAGS) -M $< | $(fmt_deps) > $@

$(BUILD_PFX)%.c.o: %.c
	$(if $(quiet),@echo "    [CC] $@")
	$(qexec)$(if $(CONFIG_DEPENDENCY_TRACKING),,mkdir -p $(dir $@))
	$(qexec)$(CC) $(INTERNAL_CFLAGS) $(CFLAGS) -c -o $@ $<

$(BUILD_PFX)%.cc.d: %.cc
	$(if $(quiet),@echo "    [DEP] $@")
	$(qexec)mkdir -p $(dir $@)
	$(qexec)$(CXX) $(INTERNAL_CFLAGS) $(CXXFLAGS) -M $< | $(fmt_deps) > $@

$(BUILD_PFX)%.cc.o: %.cc
	$(if $(quiet),@echo "    [CXX] $@")
	$(qexec)$(if $(CONFIG_DEPENDENCY_TRACKING),,mkdir -p $(dir $@))
	$(qexec)$(CXX) $(INTERNAL_CFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_PFX)%.cpp.d: %.cpp
	$(if $(quiet),@echo "    [DEP] $@")
	$(qexec)mkdir -p $(dir $@)
	$(qexec)$(CXX) $(INTERNAL_CFLAGS) $(CXXFLAGS) -M $< | $(fmt_deps) > $@

$(BUILD_PFX)%.cpp.o: %.cpp
	$(if $(quiet),@echo "    [CXX] $@")
	$(qexec)$(if $(CONFIG_DEPENDENCY_TRACKING),,mkdir -p $(dir $@))
	$(qexec)$(CXX) $(INTERNAL_CFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_PFX)%.asm.d: %.asm
	$(if $(quiet),@echo "    [DEP] $@")
	$(qexec)mkdir -p $(dir $@)
	$(qexec)$(SRC_PATH_BARE)/build/make/gen_asm_deps.sh \
            --build-pfx=$(BUILD_PFX) --depfile=$@ $(ASFLAGS) $< > $@

$(BUILD_PFX)%.asm.o: %.asm
	$(if $(quiet),@echo "    [AS] $@")
	$(qexec)$(if $(CONFIG_DEPENDENCY_TRACKING),,mkdir -p $(dir $@))
	$(qexec)$(AS) $(ASFLAGS) -o $@ $<

$(BUILD_PFX)%.S.d: %.S
	$(if $(quiet),@echo "    [DEP] $@")
	$(qexec)mkdir -p $(dir $@)
	$(qexec)$(SRC_PATH_BARE)/build/make/gen_asm_deps.sh \
            --build-pfx=$(BUILD_PFX) --depfile=$@ $(ASFLAGS) $< > $@

$(BUILD_PFX)%.S.o: %.S
	$(if $(quiet),@echo "    [AS] $@")
	$(qexec)$(if $(CONFIG_DEPENDENCY_TRACKING),,mkdir -p $(dir $@))
	$(qexec)$(AS) $(ASFLAGS) -o $@ $<

.PRECIOUS: %.c.S
%.c.S: CFLAGS += -DINLINE_ASM
$(BUILD_PFX)%.c.S: %.c
	$(if $(quiet),@echo "    [GEN] $@")
	$(qexec)$(if $(CONFIG_DEPENDENCY_TRACKING),,mkdir -p $(dir $@))
	$(qexec)$(CC) -S $(CFLAGS) -o $@ $<

.PRECIOUS: %.asm.S
$(BUILD_PFX)%.asm.S: %.asm
	$(if $(quiet),@echo "    [ASM CONVERSION] $@")
	$(qexec)mkdir -p $(dir $@)
	$(qexec)$(ASM_CONVERSION) <$< >$@

# If we're in debug mode, pretend we don't have GNU strip, to fall back to
# the copy implementation
HAVE_GNU_STRIP := $(if $(CONFIG_DEBUG),,$(HAVE_GNU_STRIP))
ifeq ($(HAVE_GNU_STRIP),yes)
# Older binutils strip global symbols not needed for relocation processing
# when given --strip-unneeded. Using nm and awk to identify globals and
# keep them caused command line length issues under mingw and segfaults in
# test_libvpx were observed under OS/2: simply use --strip-debug.
%.a: %_g.a
	$(if $(quiet),@echo "    [STRIP] $@ < $<")
	$(qexec)$(STRIP) --strip-debug \
          -o $@ $<
else
%.a: %_g.a
	$(if $(quiet),@echo "    [CP] $@ < $<")
	$(qexec)cp $< $@
endif

#
# Utility functions
#
pairmap=$(if $(strip $(2)),\
    $(call $(1),$(word 1,$(2)),$(word 2,$(2)))\
    $(call pairmap,$(1),$(wordlist 3,$(words $(2)),$(2)))\
)

enabled=$(filter-out $($(1)-no),$($(1)-yes))
cond_enabled=$(if $(filter yes,$($(1))), $(call enabled,$(2)))

find_file1=$(word 1,$(wildcard $(subst //,/,$(addsuffix /$(1),$(2)))))
find_file=$(foreach f,$(1),$(call find_file1,$(strip $(f)),$(strip $(2))) )
obj_pats=.c=.c.o $(AS_SFX)=$(AS_SFX).o .cc=.cc.o .cpp=.cpp.o
objs=$(addprefix $(BUILD_PFX),$(foreach p,$(obj_pats),$(filter %.o,$(1:$(p))) ))

install_map_templates=$(eval $(call install_map_template,$(1),$(2)))

not=$(subst yes,no,$(1))

ifeq ($(CONFIG_MSVS),yes)
lib_file_name=$(1).lib
else
lib_file_name=lib$(1).a
endif
#
# Rule Templates
#
define linker_template
$(1): $(filter-out -%,$(2))
$(1):
	$(if $(quiet),@echo    "    [LD] $$@")
	$(qexec)$$(LD) $$(strip $$(INTERNAL_LDFLAGS) $$(LDFLAGS) -o $$@ $(2) $(3) $$(extralibs))
endef
define linkerxx_template
$(1): $(filter-out -%,$(2))
$(1):
	$(if $(quiet),@echo    "    [LD] $$@")
	$(qexec)$$(CXX) $$(strip $$(INTERNAL_LDFLAGS) $$(LDFLAGS) -o $$@ $(2) $(3) $$(extralibs))
endef
# make-3.80 has a bug with expanding large input strings to the eval function,
# which was triggered in some cases by the following component of
# linker_template:
#   $(1): $$(call find_file, $(patsubst -l%,lib%.a,$(filter -l%,$(2))),\
#                           $$(patsubst -L%,%,$$(filter -L%,$$(LDFLAGS) $(2))))
# This may be useful to revisit in the future (it tries to locate libraries
# in a search path and add them as prerequisites

define install_map_template
$(DIST_DIR)/$(1): $(2)
	$(if $(quiet),@echo "    [INSTALL] $$@")
	$(qexec)mkdir -p $$(dir $$@)
	$(qexec)cp -p $$< $$@
endef

define archive_template
# Not using a pattern rule here because we don't want to generate empty
# archives when they are listed as a dependency in files not responsible
# for creating them.
$(1):
	$(if $(quiet),@echo "    [AR] $$@")
	$(qexec)$$(AR) $$(ARFLAGS) $$@ $$^
endef

# Don't use -Wl,-z,defs with Clang's sanitizers.
#
# Clang's AddressSanitizer documentation says "When linking shared libraries,
# the AddressSanitizer run-time is not linked, so -Wl,-z,defs may cause link
# errors (don't use it with AddressSanitizer)." See
# https://clang.llvm.org/docs/AddressSanitizer.html#usage.
NO_UNDEFINED := -Wl,-z,defs
ifeq ($(findstring clang,$(CC)),clang)
    ifneq ($(filter -fsanitize=%,$(LDFLAGS)),)
        NO_UNDEFINED :=
    endif
endif

define so_template
# Not using a pattern rule here because we don't want to generate empty
# archives when they are listed as a dependency in files not responsible
# for creating them.
#
# This needs further abstraction for dealing with non-GNU linkers.
$(1):
	$(if $(quiet),@echo "    [LD] $$@")
	$(qexec)$$(LD) -shared $$(LDFLAGS) \
            $(NO_UNDEFINED) \
            -Wl,-soname,$$(SONAME) \
            -Wl,--version-script,$$(EXPORTS_FILE) -o $$@ \
            $$(filter %.o,$$^) $$(extralibs)
endef

define dl_template
# Not using a pattern rule here because we don't want to generate empty
# archives when they are listed as a dependency in files not responsible
# for creating them.
$(1):
	$(if $(quiet),@echo "    [LD] $$@")
	$(qexec)$$(LD) -dynamiclib $$(LDFLAGS) \
	    -exported_symbols_list $$(EXPORTS_FILE) \
        -Wl,-headerpad_max_install_names,-compatibility_version,1.0,-current_version,$$(VERSION_MAJOR) \
        -o $$@ \
        $$(filter %.o,$$^) $$(extralibs)
endef

define dll_template
# Not using a pattern rule here because we don't want to generate empty
# archives when they are listed as a dependency in files not responsible
# for creating them.
$(1):
	$(if $(quiet),@echo "    [LD] $$@")
	$(qexec)$$(LD) -Zdll $$(LDFLAGS) \
        -o $$@ \
        $$(filter %.o,$$^) $$(extralibs) $$(EXPORTS_FILE)
endef


#
# Get current configuration
#
ifneq ($(target),)
include $(SRC_PATH_BARE)/$(target:-$(TOOLCHAIN)=).mk
endif

skip_deps := $(filter %clean,$(MAKECMDGOALS))
skip_deps += $(findstring testdata,$(MAKECMDGOALS))
ifeq ($(strip $(skip_deps)),)
  ifeq ($(CONFIG_DEPENDENCY_TRACKING),yes)
    # Older versions of make don't like -include directives with no arguments
    ifneq ($(filter %.d,$(OBJS-yes:.o=.d)),)
      -include $(filter %.d,$(OBJS-yes:.o=.d))
    endif
  endif
endif

#
# Configuration dependent rules
#
$(call pairmap,install_map_templates,$(INSTALL_MAPS))

DOCS=$(call cond_enabled,CONFIG_INSTALL_DOCS,DOCS)
.docs: $(DOCS)
	@touch $@

INSTALL-DOCS=$(call cond_enabled,CONFIG_INSTALL_DOCS,INSTALL-DOCS)
ifeq ($(MAKECMDGOALS),dist)
INSTALL-DOCS+=$(call cond_enabled,CONFIG_INSTALL_DOCS,DIST-DOCS)
endif
.install-docs: .docs $(addprefix $(DIST_DIR)/,$(INSTALL-DOCS))
	@touch $@

clean::
	rm -f .docs .install-docs $(DOCS)

BINS=$(call enabled,BINS)
.bins: $(BINS)
	@touch $@

INSTALL-BINS=$(call cond_enabled,CONFIG_INSTALL_BINS,INSTALL-BINS)
ifeq ($(MAKECMDGOALS),dist)
INSTALL-BINS+=$(call cond_enabled,CONFIG_INSTALL_BINS,DIST-BINS)
endif
.install-bins: .bins $(addprefix $(DIST_DIR)/,$(INSTALL-BINS))
	@touch $@

clean::
	rm -f .bins .install-bins $(BINS)

LIBS=$(call enabled,LIBS)
.libs: $(LIBS)
	@touch $@
$(foreach lib,$(filter %_g.a,$(LIBS)),$(eval $(call archive_template,$(lib))))
$(foreach lib,$(filter %so.$(SO_VERSION_MAJOR).$(SO_VERSION_MINOR).$(SO_VERSION_PATCH),$(LIBS)),$(eval $(call so_template,$(lib))))
$(foreach lib,$(filter %$(SO_VERSION_MAJOR).dylib,$(LIBS)),$(eval $(call dl_template,$(lib))))
$(foreach lib,$(filter %$(SO_VERSION_MAJOR).dll,$(LIBS)),$(eval $(call dll_template,$(lib))))

INSTALL-LIBS=$(call cond_enabled,CONFIG_INSTALL_LIBS,INSTALL-LIBS)
ifeq ($(MAKECMDGOALS),dist)
INSTALL-LIBS+=$(call cond_enabled,CONFIG_INSTALL_LIBS,DIST-LIBS)
endif
.install-libs: .libs $(addprefix $(DIST_DIR)/,$(INSTALL-LIBS))
	@touch $@

clean::
	rm -f .libs .install-libs $(LIBS)

ifeq ($(CONFIG_EXTERNAL_BUILD),yes)
PROJECTS=$(call enabled,PROJECTS)
.projects: $(PROJECTS)
	@touch $@

INSTALL-PROJECTS=$(call cond_enabled,CONFIG_INSTALL_PROJECTS,INSTALL-PROJECTS)
ifeq ($(MAKECMDGOALS),dist)
INSTALL-PROJECTS+=$(call cond_enabled,CONFIG_INSTALL_PROJECTS,DIST-PROJECTS)
endif
.install-projects: .projects $(addprefix $(DIST_DIR)/,$(INSTALL-PROJECTS))
	@touch $@

clean::
	rm -f .projects .install-projects $(PROJECTS)
endif

# If there are any source files to be distributed, then include the build
# system too.
ifneq ($(call enabled,DIST-SRCS),)
    DIST-SRCS-yes            += configure
    DIST-SRCS-yes            += build/make/configure.sh
    DIST-SRCS-yes            += build/make/gen_asm_deps.sh
    DIST-SRCS-yes            += build/make/Makefile
    DIST-SRCS-$(CONFIG_MSVS)  += build/make/gen_msvs_def.sh
    DIST-SRCS-$(CONFIG_MSVS)  += build/make/gen_msvs_sln.sh
    DIST-SRCS-$(CONFIG_MSVS)  += build/make/gen_msvs_vcxproj.sh
    DIST-SRCS-$(CONFIG_MSVS)  += build/make/msvs_common.sh
    DIST-SRCS-$(CONFIG_RVCT) += build/make/armlink_adapter.sh
    DIST-SRCS-$(VPX_ARCH_ARM) += build/make/ads2gas.pl
    DIST-SRCS-$(VPX_ARCH_ARM) += build/make/ads2gas_apple.pl
    DIST-SRCS-$(VPX_ARCH_ARM) += build/make/ads2armasm_ms.pl
    DIST-SRCS-$(VPX_ARCH_ARM) += build/make/thumb.pm
    DIST-SRCS-yes            += $(target:-$(TOOLCHAIN)=).mk
endif
INSTALL-SRCS := $(call cond_enabled,CONFIG_INSTALL_SRCS,INSTALL-SRCS)
ifeq ($(MAKECMDGOALS),dist)
INSTALL-SRCS += $(call cond_enabled,CONFIG_INSTALL_SRCS,DIST-SRCS)
endif
.install-srcs: $(addprefix $(DIST_DIR)/src/,$(INSTALL-SRCS))
	@touch $@

clean::
	rm -f .install-srcs

ifeq ($(CONFIG_EXTERNAL_BUILD),yes)
    BUILD_TARGETS += .projects
    INSTALL_TARGETS += .install-projects
endif
BUILD_TARGETS += .docs .libs .bins
INSTALL_TARGETS += .install-docs .install-srcs .install-libs .install-bins
all: $(BUILD_TARGETS)
install:: $(INSTALL_TARGETS)
dist: $(INSTALL_TARGETS)
test:

.SUFFIXES:  # Delete default suffix rules
