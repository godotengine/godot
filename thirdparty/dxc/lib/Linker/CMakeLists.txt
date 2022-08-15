add_llvm_library(LLVMLinker
  LinkModules.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Linker

  DEPENDS
  intrinsics_gen
  )
