PLATFORM=arm64
O=b/m_$(PLATFORM)
IS_X64=
IS_X86=
IS_ARM64=1
CROSS_COMPILE=
#use this code to reduce features
MY_ARCH=-arch arm64 -march=armv8-a
MY_ARCH=-arch arm64
USE_ASM=1
CC=$(CROSS_COMPILE)clang
CXX=$(CROSS_COMPILE)clang++
USE_CLANG=1
