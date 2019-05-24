ROOT_SHADERC_PATH := $(call my-dir)

include $(ROOT_SHADERC_PATH)/third_party/Android.mk
include $(ROOT_SHADERC_PATH)/libshaderc_util/Android.mk
include $(ROOT_SHADERC_PATH)/libshaderc/Android.mk

ALL_LIBS:=libglslang.a \
	libOGLCompiler.a \
	libOSDependent.a \
	libshaderc.a \
	libshaderc_util.a \
	libSPIRV.a \
	libHLSL.a \
	libSPIRV-Tools.a \
	libSPIRV-Tools-opt.a

define gen_libshaderc
$(1)/combine.ar: $(addprefix $(1)/, $(ALL_LIBS))
	@echo "create libshaderc_combined.a" > $(1)/combine.ar
	$(foreach lib,$(ALL_LIBS),
		@echo "addlib $(lib)" >> $(1)/combine.ar
	)
	@echo "save" >> $(1)/combine.ar
	@echo "end" >> $(1)/combine.ar

$(1)/libshaderc_combined.a: $(addprefix $(1)/, $(ALL_LIBS)) $(1)/combine.ar
	@echo "[$(TARGET_ARCH_ABI)] Combine: libshaderc_combined.a <= $(ALL_LIBS)"
	@cd $(1) && $(2)ar -M < combine.ar && cd $(ROOT_SHADERC_PATH)
	@$(2)objcopy --strip-debug $(1)/libshaderc_combined.a

$(NDK_APP_LIBS_OUT)/$(APP_STL)/$(TARGET_ARCH_ABI)/libshaderc.a: \
		$(1)/libshaderc_combined.a
	$(call host-mkdir,$(NDK_APP_LIBS_OUT)/$(APP_STL)/$(TARGET_ARCH_ABI))
	$(call host-cp,$(1)/libshaderc_combined.a \
		,$(NDK_APP_LIBS_OUT)/$(APP_STL)/$(TARGET_ARCH_ABI)/libshaderc.a)

ifndef HEADER_TARGET
HEADER_TARGET=1
$(NDK_APP_LIBS_OUT)/../include/shaderc/shaderc.hpp: \
		$(ROOT_SHADERC_PATH)/libshaderc/include/shaderc/shaderc.hpp
	$(call host-mkdir,$(NDK_APP_LIBS_OUT)/../include/shaderc)
	$(call host-cp,$(ROOT_SHADERC_PATH)/libshaderc/include/shaderc/shaderc.hpp \
		,$(NDK_APP_LIBS_OUT)/../include/shaderc/shaderc.hpp)

$(NDK_APP_LIBS_OUT)/../include/shaderc/shaderc.h: \
	$(ROOT_SHADERC_PATH)/libshaderc/include/shaderc/shaderc.h
	$(call host-mkdir,$(NDK_APP_LIBS_OUT)/../include/shaderc)
	$(call host-cp,$(ROOT_SHADERC_PATH)/libshaderc/include/shaderc/shaderc.h \
		,$(NDK_APP_LIBS_OUT)/../include/shaderc/shaderc.h)
endif

libshaderc_combined: \
	$(NDK_APP_LIBS_OUT)/$(APP_STL)/$(TARGET_ARCH_ABI)/libshaderc.a

endef
libshaderc_combined: $(NDK_APP_LIBS_OUT)/../include/shaderc/shaderc.hpp \
	$(NDK_APP_LIBS_OUT)/../include/shaderc/shaderc.h

$(eval $(call gen_libshaderc,$(TARGET_OUT),$(TOOLCHAIN_PREFIX)))
