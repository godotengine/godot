##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


INSTALL_MAPS += docs/%    docs/%
INSTALL_MAPS += src/%     %
INSTALL_MAPS += %         %

# Static documentation authored in doxygen
CODEC_DOX :=    mainpage.dox \
		keywords.dox \
		usage.dox \
		usage_cx.dox \
		usage_dx.dox \

# Other doxy files sourced in Markdown
TXT_DOX = $(call enabled,TXT_DOX)

EXAMPLE_PATH += $(SRC_PATH_BARE) #for CHANGELOG, README, etc
EXAMPLE_PATH += $(SRC_PATH_BARE)/examples

doxyfile: $(if $(findstring examples, $(ALL_TARGETS)),examples.doxy)
doxyfile: libs.doxy_template libs.doxy
	@echo "    [CREATE] $@"
	@cat $^ > $@
	@echo "STRIP_FROM_PATH += $(SRC_PATH_BARE) $(BUILD_ROOT)" >> $@
	@echo "INPUT += $(addprefix $(SRC_PATH_BARE)/,$(CODEC_DOX))" >> $@;
	@echo "INPUT += $(TXT_DOX)" >> $@;
	@echo "EXAMPLE_PATH += $(EXAMPLE_PATH)" >> $@

CLEAN-OBJS += doxyfile $(wildcard docs/html/*)
docs/html/index.html: doxyfile $(CODEC_DOX) $(TXT_DOX)
	@echo "    [DOXYGEN] $<"
	@doxygen $<
DOCS-yes += docs/html/index.html

DIST-DOCS-yes = $(wildcard docs/html/*)
DIST-DOCS-$(CONFIG_CODEC_SRCS) += $(addprefix src/,$(CODEC_DOX))
DIST-DOCS-$(CONFIG_CODEC_SRCS) += src/libs.doxy_template
DIST-DOCS-yes                  += CHANGELOG
DIST-DOCS-yes                  += README
