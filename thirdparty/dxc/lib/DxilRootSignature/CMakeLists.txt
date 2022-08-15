# Copyright (C) Microsoft Corporation. All rights reserved.
# This file is distributed under the University of Illinois Open Source License. See LICENSE.TXT for details.
add_llvm_library(LLVMDxilRootSignature
  DxilRootSignature.cpp
  DxilRootSignatureConvert.cpp
  DxilRootSignatureSerializer.cpp
  DxilRootSignatureValidator.cpp


  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/IR
)

add_dependencies(LLVMDxilRootSignature intrinsics_gen)
