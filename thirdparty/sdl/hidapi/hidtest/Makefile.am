AM_CPPFLAGS = -I$(top_srcdir)/hidapi/

## Linux
if OS_LINUX
noinst_PROGRAMS = hidtest-libusb hidtest-hidraw

hidtest_hidraw_SOURCES = test.c
hidtest_hidraw_LDADD = $(top_builddir)/linux/libhidapi-hidraw.la

hidtest_libusb_SOURCES = test.c
hidtest_libusb_LDADD = $(top_builddir)/libusb/libhidapi-libusb.la
else

# Other OS's
noinst_PROGRAMS = hidtest

hidtest_SOURCES = test.c
hidtest_LDADD = $(top_builddir)/$(backend)/libhidapi.la

endif

if OS_DARWIN
AM_CPPFLAGS += -I$(top_srcdir)/mac/
endif

if OS_WINDOWS
AM_CPPFLAGS += -I$(top_srcdir)/windows/
endif
