include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMIntelJITEvents
  IntelJITEventListener.cpp
  jitprofiling.c

  LINK_LIBS pthread ${CMAKE_DL_LIBS}
)
