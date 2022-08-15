///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilFunctionProps.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Function properties for a dxil shader function.                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

// for memset dependency:
#include <cstring>

#include "dxc/DXIL/DxilConstants.h"

namespace llvm {
class Function;
class Constant;
}

namespace hlsl {
struct DxilFunctionProps {
  DxilFunctionProps() {
    memset(&ShaderProps, 0, sizeof(ShaderProps));
    shaderKind = DXIL::ShaderKind::Invalid;
    waveSize = 0;
  }
  union {
    // Compute shader.
    struct {
      unsigned numThreads[3];
    } CS;
    // Geometry shader.
    struct {
      DXIL::InputPrimitive inputPrimitive;
      unsigned maxVertexCount;
      unsigned instanceCount;
      DXIL::PrimitiveTopology
          streamPrimitiveTopologies[DXIL::kNumOutputStreams];
    } GS;
    // Hull shader.
    struct {
      llvm::Function *patchConstantFunc;
      DXIL::TessellatorDomain domain;
      DXIL::TessellatorPartitioning partition;
      DXIL::TessellatorOutputPrimitive outputPrimitive;
      unsigned inputControlPoints;
      unsigned outputControlPoints;
      float maxTessFactor;
    } HS;
    // Domain shader.
    struct {
      DXIL::TessellatorDomain domain;
      unsigned inputControlPoints;
    } DS;
    // Vertex shader.
    struct {
      llvm::Constant *clipPlanes[DXIL::kNumClipPlanes];
    } VS;
    // Pixel shader.
    struct {
      bool EarlyDepthStencil;
    } PS;
    // Ray Tracing shaders
    struct {
      union {
        unsigned payloadSizeInBytes;
        unsigned paramSizeInBytes;
      };
      unsigned attributeSizeInBytes;
    } Ray;
    // Mesh shader.
    struct {
      unsigned numThreads[3];
      unsigned maxVertexCount;
      unsigned maxPrimitiveCount;
      DXIL::MeshOutputTopology outputTopology;
      unsigned payloadSizeInBytes;
    } MS;
    // Amplification shader.
    struct {
      unsigned numThreads[3];
      unsigned payloadSizeInBytes;
    } AS;
  } ShaderProps;
  DXIL::ShaderKind shaderKind;
  // WaveSize is currently allowed only on compute shaders, but could be supported on other shader types in the future
  unsigned waveSize;
  // Save root signature for lib profile entry.
  std::vector<uint8_t> serializedRootSignature;
  void SetSerializedRootSignature(const uint8_t *pData, unsigned size) {
    serializedRootSignature.assign(pData, pData+size);
  }

  // TODO: Should we have an unmangled name here for ray tracing shaders?
  bool IsPS() const     { return shaderKind == DXIL::ShaderKind::Pixel; }
  bool IsVS() const     { return shaderKind == DXIL::ShaderKind::Vertex; }
  bool IsGS() const     { return shaderKind == DXIL::ShaderKind::Geometry; }
  bool IsHS() const     { return shaderKind == DXIL::ShaderKind::Hull; }
  bool IsDS() const     { return shaderKind == DXIL::ShaderKind::Domain; }
  bool IsCS() const     { return shaderKind == DXIL::ShaderKind::Compute; }
  bool IsGraphics() const {
    return (shaderKind >= DXIL::ShaderKind::Pixel && shaderKind <= DXIL::ShaderKind::Domain) ||
           shaderKind == DXIL::ShaderKind::Mesh || shaderKind == DXIL::ShaderKind::Amplification;
  }
  bool IsRayGeneration() const { return shaderKind == DXIL::ShaderKind::RayGeneration; }
  bool IsIntersection() const { return shaderKind == DXIL::ShaderKind::Intersection; }
  bool IsAnyHit() const { return shaderKind == DXIL::ShaderKind::AnyHit; }
  bool IsClosestHit() const { return shaderKind == DXIL::ShaderKind::ClosestHit; }
  bool IsMiss() const { return shaderKind == DXIL::ShaderKind::Miss; }
  bool IsCallable() const { return shaderKind == DXIL::ShaderKind::Callable; }
  bool IsRay() const {
    return (shaderKind >= DXIL::ShaderKind::RayGeneration && shaderKind <= DXIL::ShaderKind::Callable);
  }
  bool IsMS() const { return shaderKind == DXIL::ShaderKind::Mesh; }
  bool IsAS() const { return shaderKind == DXIL::ShaderKind::Amplification; }
};

} // namespace hlsl
