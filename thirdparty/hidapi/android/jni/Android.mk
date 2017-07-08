LOCAL_PATH:= $(call my-dir)

HIDAPI_ROOT_REL:= ../..
HIDAPI_ROOT_ABS:= $(LOCAL_PATH)/../..

include $(CLEAR_VARS)

LOCAL_SRC_FILES := \
  $(HIDAPI_ROOT_REL)/libusb/hid.c

LOCAL_C_INCLUDES += \
  $(HIDAPI_ROOT_ABS)/hidapi \
  $(HIDAPI_ROOT_ABS)/android

LOCAL_SHARED_LIBRARIES := libusb1.0

LOCAL_MODULE := libhidapi

include $(BUILD_SHARED_LIBRARY)
