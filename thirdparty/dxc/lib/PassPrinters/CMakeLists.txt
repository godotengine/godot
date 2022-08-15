add_llvm_library(LLVMPassPrinters
  PassPrinters.cpp
  ADDITIONAL_HEADER_DIRS
)

add_dependencies(LLVMPassPrinters intrinsics_gen)
