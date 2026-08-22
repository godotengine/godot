TARGET := game

BUILD := build

SOURCES := ../template/3ds/main.cpp

INCLUDE := \
	-I$(DEVKITPRO)/libctru/include

LIBS := \
	-L$(DEVKITPRO)/libctru/lib \
	-lctru

CXX := arm-none-eabi-g++

CXXFLAGS := \
	-march=armv6k \
	-mtune=mpcore \
	-mfloat-abi=hard \
	-mfpu=vfpv2 \
	-O2 \
	$(INCLUDE)

LDFLAGS := \
	-march=armv6k \
	-mtune=mpcore \
	-mfloat-abi=hard \
	-mfpu=vfpv2

all:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) \
		$(SOURCES) \
		$(LIBS) \
		-o $(BUILD)/$(TARGET).elf
