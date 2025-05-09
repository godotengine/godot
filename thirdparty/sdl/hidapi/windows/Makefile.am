lib_LTLIBRARIES = libhidapi.la
libhidapi_la_SOURCES = hid.c
libhidapi_la_LDFLAGS = $(LTLDFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/hidapi/
libhidapi_la_LIBADD = $(LIBS)

hdrdir = $(includedir)/hidapi
hdr_HEADERS = $(top_srcdir)/hidapi/hidapi.h

EXTRA_DIST = \
  hidapi.vcproj \
  hidtest.vcproj \
  Makefile-manual \
  Makefile.mingw \
  hidapi.sln
