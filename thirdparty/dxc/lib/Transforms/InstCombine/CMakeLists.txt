add_llvm_library(LLVMInstCombine
  InstructionCombining.cpp
  InstCombineAddSub.cpp
  InstCombineAndOrXor.cpp
  InstCombineCalls.cpp
  InstCombineCasts.cpp
  InstCombineCompares.cpp
  InstCombineLoadStoreAlloca.cpp
  InstCombineMulDivRem.cpp
  InstCombinePHI.cpp
  InstCombineSelect.cpp
  InstCombineShifts.cpp
  InstCombineSimplifyDemanded.cpp
  InstCombineVectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/InstCombine
  )

add_dependencies(LLVMInstCombine intrinsics_gen)

target_link_libraries(LLVMInstCombine PUBLIC LLVMDXIL) # HLSL Change