GENERATE_BINDINGS = no
HEADERS = godot_headers
TARGET = debug
USE_CLANG = no

BASE = scons use_llvm=$(USE_CLANG) generate_bindings=$(GENERATE_BINDINGS) target=$(TARGET) headers=$(HEADERS) -j4
LINUX = $(BASE) platform=linux
WINDOWS = $(BASE) platform=windows
OSX = $(BASE) platform=osx


all:
	make linux
	make windows


linux:
	make linux32
	make linux64

linux32: SConstruct
	$(LINUX) bits=32

linux64: SConstruct
	$(LINUX) bits=64


windows:
	make windows32
	make windows64

windows32: SConstruct
	$(WINDOWS) bits=32

windows64: SConstruct
	$(WINDOWS) bits=64


osx:
	make osx32
	make osx64

osx32: SConstruct
	$(OSX) bits=32

osx64: SConstruct
	$(OSX) bits=64
