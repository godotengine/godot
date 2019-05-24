LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_CPP_EXTENSION := .cc .cpp .cxx
LOCAL_SRC_FILES:=test.cpp
LOCAL_MODULE:=shaderc_test
LOCAL_LDLIBS:=-landroid
LOCAL_STATIC_LIBRARIES=shaderc android_native_app_glue
include $(BUILD_SHARED_LIBRARY)

include $(LOCAL_PATH)/../Android.mk
$(call import-module,android/native_app_glue)
