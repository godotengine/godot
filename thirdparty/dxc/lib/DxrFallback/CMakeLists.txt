add_llvm_library(LLVMDxrFallback
  DxrFallbackCompiler.cpp
  FunctionBuilder.h
  LiveValues.cpp
  LiveValues.h
  LLVMUtils.cpp
  LLVMUtils.h
  Reducibility.h
  Reducibility.cpp
  StateFunctionTransform.cpp
  StateFunctionTransform.h
)

add_dependencies(LLVMDxrFallback intrinsics_gen)
