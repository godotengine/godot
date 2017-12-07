lib_LTLIBRARIES = libhidapi-hidraw.la
libhidapi_hidraw_la_SOURCES = hid.c
libhidapi_hidraw_la_LDFLAGS = $(LTLDFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/hidapi/ $(CFLAGS_HIDRAW)
libhidapi_hidraw_la_LIBADD = $(LIBS_HIDRAW)

hdrdir = $(includedir)/hidapi
hdr_HEADERS = $(top_srcdir)/hidapi/hidapi.h

EXTRA_DIST = Makefile-manual
