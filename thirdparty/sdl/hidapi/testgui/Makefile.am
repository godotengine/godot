
AM_CPPFLAGS = -I$(top_srcdir)/hidapi/ $(CFLAGS_TESTGUI)

if OS_LINUX
## Linux
bin_PROGRAMS = hidapi-hidraw-testgui hidapi-libusb-testgui

hidapi_hidraw_testgui_SOURCES = test.cpp
hidapi_hidraw_testgui_LDADD = $(top_builddir)/linux/libhidapi-hidraw.la $(LIBS_TESTGUI)

hidapi_libusb_testgui_SOURCES = test.cpp
hidapi_libusb_testgui_LDADD = $(top_builddir)/libusb/libhidapi-libusb.la $(LIBS_TESTGUI)
else
## Other OS's
bin_PROGRAMS = hidapi-testgui

hidapi_testgui_SOURCES = test.cpp
hidapi_testgui_LDADD = $(top_builddir)/$(backend)/libhidapi.la $(LIBS_TESTGUI)
endif

if OS_DARWIN
hidapi_testgui_SOURCES = test.cpp mac_support_cocoa.m mac_support.h
# Rules for copying the binary and its dependencies into the app bundle.
TestGUI.app/Contents/MacOS/hidapi-testgui$(EXEEXT): hidapi-testgui$(EXEEXT)
	$(srcdir)/copy_to_bundle.sh

all: all-am TestGUI.app/Contents/MacOS/hidapi-testgui$(EXEEXT)

endif

EXTRA_DIST = \
 copy_to_bundle.sh \
 Makefile-manual \
 Makefile.freebsd \
 Makefile.linux \
 Makefile.mac \
 Makefile.mingw \
 TestGUI.app.in \
 testgui.sln \
 testgui.vcproj

distclean-local:
	rm -rf TestGUI.app
