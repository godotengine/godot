add_llvm_library(LLVMDxcBindingTable
  DxcBindingTable.cpp
)

add_dependencies(LLVMDxcBindingTable intrinsics_gen)
