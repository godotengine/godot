all: libusb-1.0.la libusb-1.0.dll

AUTOMAKE_OPTIONS = subdir-objects

lib_LTLIBRARIES = libusb-1.0.la

POSIX_POLL_SRC = os/poll_posix.h os/poll_posix.c
POSIX_THREADS_SRC = os/threads_posix.h os/threads_posix.c
WINDOWS_POLL_SRC = os/poll_windows.h os/poll_windows.c
WINDOWS_THREADS_SRC = os/threads_windows.h os/threads_windows.c
LINUX_USBFS_SRC = os/linux_usbfs.h os/linux_usbfs.c
DARWIN_USB_SRC = os/darwin_usb.h os/darwin_usb.c
OPENBSD_USB_SRC = os/openbsd_usb.c
NETBSD_USB_SRC = os/netbsd_usb.c
SUNOS_USB_SRC = os/sunos_usb.c os/sunos_usb.h
WINDOWS_COMMON_SRC = os/windows_nt_common.h os/windows_nt_common.c \
                     os/windows_common.h libusb-1.0.rc libusb-1.0.def
WINDOWS_USB_SRC = os/windows_winusb.h os/windows_winusb.c
WINDOWS_USBDK_SRC = os/windows_usbdk.h os/windows_usbdk.c
WINCE_USB_SRC = os/wince_usb.h os/wince_usb.c
HAIKU_USB_SRC = os/haiku_usb.h os/haiku_usb_backend.cpp \
		os/haiku_usb_raw.h os/haiku_usb_raw.cpp os/haiku_pollfs.cpp

EXTRA_DIST = $(POSIX_POLL_SRC) $(POSIX_THREADS_SRC) \
	$(WINDOWS_POLL_SRC) $(WINDOWS_THREADS_SRC) \
	$(LINUX_USBFS_SRC) $(DARWIN_USB_SRC) \
	$(OPENBSD_USB_SRC) $(NETBSD_USB_SRC) \
	$(WINDOWS_COMMON_SRC) $(WINDOWS_USB_SRC) $(WINDOWS_USBDK_SRC) \
        $(WINCE_USB_SRC) $(HAIKU_USB_SRC) \
	os/linux_udev.c os/linux_netlink.c

if OS_LINUX

if USE_UDEV
OS_SRC = $(LINUX_USBFS_SRC) os/linux_udev.c
else
OS_SRC = $(LINUX_USBFS_SRC) os/linux_netlink.c
endif

endif

if OS_DARWIN
OS_SRC = $(DARWIN_USB_SRC)
AM_CFLAGS_EXT = -no-cpp-precomp
endif

if OS_OPENBSD
OS_SRC = $(OPENBSD_USB_SRC)
endif

if OS_NETBSD
OS_SRC = $(NETBSD_USB_SRC)
endif

if OS_SUNOS
OS_SRC = $(SUNOS_USB_SRC)
endif

if OS_HAIKU
noinst_LTLIBRARIES = libusb_haiku.la
libusb_haiku_la_SOURCES = $(HAIKU_USB_SRC)
libusb_1_0_la_LIBADD = libusb_haiku.la
endif

if OS_WINDOWS

if USE_USBDK
OS_SRC = $(WINDOWS_USBDK_SRC) $(WINDOWS_COMMON_SRC)
else
OS_SRC = $(WINDOWS_USB_SRC) $(WINDOWS_COMMON_SRC)
endif

.rc.lo:
	$(AM_V_GEN)$(LIBTOOL) $(AM_V_lt) $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --tag=RC --mode=compile $(RC) $(RCFLAGS) -i $< -o $@

libusb-1.0.rc: version.h version_nano.h
endif

libusb-1.0.dll: libusb-1.0.def libusb-1.0.la
if CREATE_IMPORT_LIB
# Rebuild the import lib from the .def so that MS and MinGW DLLs can be interchanged
	$(AM_V_GEN)$(DLLTOOL) $(DLLTOOLFLAGS) --kill-at --input-def $(srcdir)/libusb-1.0.def --dllname $@ --output-lib .libs/$@.a
endif

if OS_WINDOWS
POLL_SRC = $(WINDOWS_POLL_SRC)
else
POLL_SRC = $(POSIX_POLL_SRC)
endif

if THREADS_POSIX
THREADS_SRC = $(POSIX_THREADS_SRC)
else
THREADS_SRC = $(WINDOWS_THREADS_SRC)
endif

libusb_1_0_la_CFLAGS = $(AM_CFLAGS)
libusb_1_0_la_LDFLAGS = $(LTLDFLAGS)
libusb_1_0_la_SOURCES = libusbi.h libusb.h version.h version_nano.h \
	core.c descriptor.c hotplug.h hotplug.c io.c strerror.c sync.c \
	$(POLL_SRC) $(THREADS_SRC) $(OS_SRC)

hdrdir = $(includedir)/libusb-1.0
hdr_HEADERS = libusb.h
