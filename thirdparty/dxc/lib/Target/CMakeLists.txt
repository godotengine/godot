list(APPEND LLVM_COMMON_DEPENDS intrinsics_gen)

add_llvm_library(LLVMTarget
  Target.cpp
  TargetIntrinsicInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachine.cpp
  TargetMachineC.cpp
  TargetRecip.cpp
  TargetSubtargetInfo.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Target
  )

foreach(t ${LLVM_TARGETS_TO_BUILD})
  if (NOT t STREQUAL "None") # HLSL Change
  message(STATUS "Targeting ${t}")
  add_subdirectory(${t})
  endif() # HLSL Change
endforeach()
