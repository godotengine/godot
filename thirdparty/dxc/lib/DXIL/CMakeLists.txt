# Copyright (C) Microsoft Corporation. All rights reserved.
# This file is distributed under the University of Illinois Open Source License.
# See LICENSE.TXT for details.

add_hlsl_hctgen(DxilOperations OUTPUT DxilOperations.cpp CODE_TAG)
add_hlsl_hctgen(DxilShaderModel OUTPUT DxilShaderModel.cpp CODE_TAG)
add_hlsl_hctgen(DxilMetadata OUTPUT DxilMetadataHelper.cpp CODE_TAG)

add_llvm_library(LLVMDXIL
  DxilCBuffer.cpp
  DxilCompType.cpp
  DxilCounters.cpp
  DxilInterpolationMode.cpp
  DxilMetadataHelper.cpp
  DxilModule.cpp
  DxilModuleHelper.cpp
  DxilOperations.cpp
  DxilResource.cpp
  DxilResourceBase.cpp
  DxilResourceBinding.cpp
  DxilResourceProperties.cpp
  DxilSampler.cpp
  DxilSemantic.cpp
  DxilShaderFlags.cpp
  DxilShaderModel.cpp
  DxilSignature.cpp
  DxilSignatureElement.cpp
  DxilSubobject.cpp
  DxilTypeSystem.cpp
  DxilUtil.cpp
  DxilUtilDbgInfoAndMisc.cpp
  DxilPDB.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/IR
)

add_dependencies(LLVMDXIL intrinsics_gen)
