##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

# libvpx reverse dependencies (targets that depend on libvpx)
VPX_NONDEPS=$(addsuffix .$(VCPROJ_SFX),vpx gtest)
VPX_RDEPS=$(foreach vcp,\
              $(filter-out $(VPX_NONDEPS),$^), --dep=$(vcp:.$(VCPROJ_SFX)=):vpx)

vpx.sln: $(wildcard *.$(VCPROJ_SFX))
	@echo "    [CREATE] $@"
	$(SRC_PATH_BARE)/build/make/gen_msvs_sln.sh \
            $(if $(filter vpx.$(VCPROJ_SFX),$^),$(VPX_RDEPS)) \
            --dep=test_libvpx:gtest \
            --ver=$(CONFIG_VS_VERSION)\
            --out=$@ $^
vpx.sln.mk: vpx.sln
	@true

PROJECTS-yes += vpx.sln vpx.sln.mk
-include vpx.sln.mk

# Always install this file, as it is an unconditional post-build rule.
INSTALL_MAPS += src/%     $(SRC_PATH_BARE)/%
INSTALL-SRCS-yes            += $(target).mk
