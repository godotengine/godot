add_llvm_library(LLVMMCJIT
  MCJIT.cpp

  DEPENDS
  intrinsics_gen
  )
