AUTOMAKE_OPTIONS = subdir-objects
library_includedir=$(includedir)/openhmd
library_include_HEADERS = $(top_srcdir)/include/openhmd.h

lib_LTLIBRARIES = libopenhmd.la
libopenhmd_la_SOURCES = \
	openhmd.c \
	platform-win32.c \
	drv_dummy/dummy.c \
	omath.c \
	platform-posix.c \
	fusion.c \
	shaders.c

libopenhmd_la_LDFLAGS = -no-undefined -version-info 0:0:0
libopenhmd_la_CPPFLAGS = -fPIC -I$(top_srcdir)/include -Wall

if BUILD_DRIVER_OCULUS_RIFT

libopenhmd_la_SOURCES += \
	drv_oculus_rift/rift.c \
	drv_oculus_rift/packet.c

libopenhmd_la_CPPFLAGS += $(hidapi_CFLAGS) -DDRIVER_OCULUS_RIFT
libopenhmd_la_LDFLAGS += $(hidapi_LIBS)

endif

if BUILD_DRIVER_HTC_VIVE

libopenhmd_la_SOURCES += \
	drv_htc_vive/vive.c \
	drv_htc_vive/packet.c \
	ext_deps/mjson.c

libopenhmd_la_CPPFLAGS += $(hidapi_CFLAGS) -DDRIVER_HTC_VIVE
libopenhmd_la_LDFLAGS += $(hidapi_LIBS)

endif

if BUILD_DRIVER_DEEPOON

libopenhmd_la_SOURCES += \
	drv_deepoon/deepoon.c \
	drv_deepoon/packet.c

libopenhmd_la_CPPFLAGS += $(hidapi_CFLAGS) -DDRIVER_DEEPOON
libopenhmd_la_LDFLAGS += $(hidapi_LIBS)

endif

if BUILD_DRIVER_PSVR

libopenhmd_la_SOURCES += \
	drv_psvr/psvr.c \
	drv_psvr/packet.c

libopenhmd_la_CPPFLAGS += $(hidapi_CFLAGS) -DDRIVER_PSVR
libopenhmd_la_LDFLAGS += $(hidapi_LIBS)

endif

if BUILD_DRIVER_NOLO

libopenhmd_la_SOURCES += \
	drv_nolo/nolo.c \
	drv_nolo/packet.c

libopenhmd_la_CPPFLAGS += $(hidapi_CFLAGS) -DDRIVER_NOLO
libopenhmd_la_LDFLAGS += $(hidapi_LIBS)

endif

if BUILD_DRIVER_EXTERNAL

libopenhmd_la_SOURCES += \
	drv_external/external.c
endif

if BUILD_DRIVER_ANDROID

libopenhmd_la_SOURCES += \
	drv_android/android.c
endif

libopenhmd_la_LDFLAGS += $(EXTRA_LD_FLAGS)

