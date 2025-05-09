

OS=$(shell uname)

ifeq ($(OS), Darwin)
	FILE=Makefile.mac
endif

ifneq (,$(findstring MINGW,$(OS)))
	FILE=Makefile.mingw
endif

ifeq ($(OS), Linux)
	FILE=Makefile.linux
endif

ifeq ($(OS), FreeBSD)
	FILE=Makefile.freebsd
endif

ifeq ($(FILE), )
all:
	$(error Your platform ${OS} is not supported at this time.)
endif

include $(FILE)
