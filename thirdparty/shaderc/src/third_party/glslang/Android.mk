LOCAL_PATH := $(call my-dir)

GLSLANG_OS_FLAGS := -DGLSLANG_OSINCLUDE_UNIX
# AMD and NV extensions are turned on by default in upstream Glslang.
GLSLANG_DEFINES:= -DAMD_EXTENSIONS -DNV_EXTENSIONS -DENABLE_HLSL $(GLSLANG_OS_FLAGS)

include $(CLEAR_VARS)
LOCAL_MODULE:=OSDependent
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti $(GLSLANG_DEFINES)
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)
LOCAL_SRC_FILES:=glslang/OSDependent/Unix/ossource.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH) $(LOCAL_PATH)/glslang/OSDependent/Unix/
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)/glslang/OSDependent/Unix/
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:=OGLCompiler
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti $(GLSLANG_DEFINES)
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)
LOCAL_SRC_FILES:=OGLCompilersDLL/InitializeDll.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH)/OGLCompiler
LOCAL_STATIC_LIBRARIES:=OSDependent
include $(BUILD_STATIC_LIBRARY)

# Build Glslang's HLSL parser library.
include $(CLEAR_VARS)
LOCAL_MODULE:=HLSL
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti $(GLSLANG_DEFINES)
LOCAL_SRC_FILES:= \
		hlsl/hlslAttributes.cpp \
		hlsl/hlslGrammar.cpp \
		hlsl/hlslOpMap.cpp \
		hlsl/hlslParseables.cpp \
		hlsl/hlslParseHelper.cpp \
		hlsl/hlslScanContext.cpp \
		hlsl/hlslTokenStream.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH) \
	$(LOCAL_PATH)/hlsl
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
GLSLANG_OUT_PATH=$(if $(call host-path-is-absolute,$(TARGET_OUT)),$(TARGET_OUT),$(abspath $(TARGET_OUT)))

LOCAL_MODULE:=glslang
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti $(GLSLANG_DEFINES)
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)
LOCAL_SRC_FILES:= \
		glslang/GenericCodeGen/CodeGen.cpp \
		glslang/GenericCodeGen/Link.cpp \
		glslang/MachineIndependent/attribute.cpp \
		glslang/MachineIndependent/Constant.cpp \
		glslang/MachineIndependent/glslang_tab.cpp \
		glslang/MachineIndependent/InfoSink.cpp \
		glslang/MachineIndependent/Initialize.cpp \
		glslang/MachineIndependent/Intermediate.cpp \
		glslang/MachineIndependent/intermOut.cpp \
		glslang/MachineIndependent/IntermTraverse.cpp \
		glslang/MachineIndependent/iomapper.cpp \
		glslang/MachineIndependent/limits.cpp \
		glslang/MachineIndependent/linkValidate.cpp \
		glslang/MachineIndependent/parseConst.cpp \
		glslang/MachineIndependent/ParseContextBase.cpp \
		glslang/MachineIndependent/ParseHelper.cpp \
		glslang/MachineIndependent/PoolAlloc.cpp \
		glslang/MachineIndependent/propagateNoContraction.cpp \
		glslang/MachineIndependent/reflection.cpp \
		glslang/MachineIndependent/RemoveTree.cpp \
		glslang/MachineIndependent/Scan.cpp \
		glslang/MachineIndependent/ShaderLang.cpp \
		glslang/MachineIndependent/SymbolTable.cpp \
		glslang/MachineIndependent/Versions.cpp \
		glslang/MachineIndependent/preprocessor/PpAtom.cpp \
		glslang/MachineIndependent/preprocessor/PpContext.cpp \
		glslang/MachineIndependent/preprocessor/Pp.cpp \
		glslang/MachineIndependent/preprocessor/PpScanner.cpp \
		glslang/MachineIndependent/preprocessor/PpTokens.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH) \
	$(LOCAL_PATH)/glslang/MachineIndependent \
	$(GLSLANG_OUT_PATH)
LOCAL_STATIC_LIBRARIES:=OSDependent OGLCompiler HLSL
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:=SPIRV
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti -Werror $(GLSLANG_DEFINES)
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)
LOCAL_SRC_FILES:= \
	SPIRV/GlslangToSpv.cpp \
	SPIRV/InReadableOrder.cpp \
	SPIRV/Logger.cpp \
	SPIRV/SPVRemapper.cpp \
	SPIRV/SpvBuilder.cpp \
	SPIRV/SpvPostProcess.cpp \
	SPIRV/SpvTools.cpp \
	SPIRV/disassemble.cpp \
	SPIRV/doc.cpp
LOCAL_C_INCLUDES:=$(LOCAL_PATH) $(LOCAL_PATH)/glslang/SPIRV
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)/glslang/SPIRV
LOCAL_STATIC_LIBRARIES:=glslang
include $(BUILD_STATIC_LIBRARY)
