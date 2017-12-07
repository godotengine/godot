

OS=$(shell uname)

ifneq (,$(findstring MINGW,$(OS)))
	FILE=Makefile.mingw
endif

ifeq ($(FILE), )
all:
	$(error Your platform ${OS} is not supported at this time.)
endif

include $(FILE)
