# XXX Use the common Thrust Makefiles instead of this.

EXECUTABLE := bench
BUILD_SRC  := $(ROOTDIR)/thrust/internal/benchmark/bench.cu

ifeq ($(OS),Linux)
  LIBRARIES += m
endif

# XXX Why is this needed?
ifeq ($(OS),Linux)
  ifeq ($(ABITYPE), androideabi)
    override ALL_SASS_ARCHITECTURES := 32
  endif
endif

ARCH_NEG_FILTER += 20 21

include $(ROOTDIR)/thrust/internal/build/common_detect.mk
include $(ROOTDIR)/thrust/internal/build/common_build.mk
