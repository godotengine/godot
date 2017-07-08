

OS=$(shell uname)

ifeq ($(OS), Linux)
	FILE=Makefile.linux
endif

ifeq ($(OS), FreeBSD)
	FILE=Makefile.freebsd
endif

ifeq ($(FILE), )
all:
	$(error Your platform ${OS} is not supported by hidapi/libusb at this time.)
endif

include $(FILE)
