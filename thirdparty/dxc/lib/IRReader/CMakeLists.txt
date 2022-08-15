add_llvm_library(LLVMIRReader
  IRReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/IRReader

  DEPENDS
  intrinsics_gen
  )
