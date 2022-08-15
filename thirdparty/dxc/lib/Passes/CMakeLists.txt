add_llvm_library(LLVMPasses
  PassBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Passes
  )

add_dependencies(LLVMPasses intrinsics_gen)
