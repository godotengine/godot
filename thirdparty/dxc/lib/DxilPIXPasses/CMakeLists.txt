# Copyright (C) Microsoft Corporation. All rights reserved.
# This file is distributed under the University of Illinois Open Source License.
# See LICENSE.TXT for details.

add_hlsl_hctgen(DxilPIXPasses OUTPUT DxilPIXPasses.inc BUILD_DIR)

add_llvm_library(LLVMDxilPIXPasses
  DxilAddPixelHitInstrumentation.cpp
  DxilAnnotateWithVirtualRegister.cpp
  DxilDbgValueToDbgDeclare.cpp
  DxilDebugInstrumentation.cpp
  DxilForceEarlyZ.cpp
  DxilOutputColorBecomesConstant.cpp
  DxilPIXMeshShaderOutputInstrumentation.cpp
  DxilRemoveDiscards.cpp
  DxilReduceMSAAToSingleSample.cpp
  DxilShaderAccessTracking.cpp
  DxilPIXPasses.cpp
  DxilPIXVirtualRegisters.cpp
  PixPassHelpers.cpp
  DxilPIXAddTidToAmplificationShaderPayload.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/IR
)

add_dependencies(LLVMDxilPIXPasses intrinsics_gen)
