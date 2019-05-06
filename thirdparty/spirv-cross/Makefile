TARGET := spirv-cross

SOURCES := $(wildcard spirv_*.cpp)
CLI_SOURCES := main.cpp

OBJECTS := $(SOURCES:.cpp=.o)
CLI_OBJECTS := $(CLI_SOURCES:.cpp=.o)

STATIC_LIB := lib$(TARGET).a

DEPS := $(OBJECTS:.o=.d) $(CLI_OBJECTS:.o=.d)

CXXFLAGS += -std=c++11 -Wall -Wextra -Wshadow

ifeq ($(DEBUG), 1)
	CXXFLAGS += -O0 -g
else
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS), 1)
	CXXFLAGS += -DSPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS -fno-exceptions
endif

all: $(TARGET)

-include $(DEPS)

$(TARGET): $(CLI_OBJECTS) $(STATIC_LIB)
	$(CXX) -o $@ $(CLI_OBJECTS) $(STATIC_LIB) $(LDFLAGS)

$(STATIC_LIB): $(OBJECTS)
	$(AR) rcs $@ $(OBJECTS)

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS) -MMD

clean:
	rm -f $(TARGET) $(OBJECTS) $(CLI_OBJECTS) $(STATIC_LIB) $(DEPS)

.PHONY: clean
