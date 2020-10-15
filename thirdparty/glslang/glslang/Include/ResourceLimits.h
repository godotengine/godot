//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _RESOURCE_LIMITS_INCLUDED_
#define _RESOURCE_LIMITS_INCLUDED_

struct TLimits {
    bool nonInductiveForLoops;
    bool whileLoops;
    bool doWhileLoops;
    bool generalUniformIndexing;
    bool generalAttributeMatrixVectorIndexing;
    bool generalVaryingIndexing;
    bool generalSamplerIndexing;
    bool generalVariableIndexing;
    bool generalConstantMatrixVectorIndexing;
};

struct TBuiltInResource {
    int maxLights;
    int maxClipPlanes;
    int maxTextureUnits;
    int maxTextureCoords;
    int maxVertexAttribs;
    int maxVertexUniformComponents;
    int maxVaryingFloats;
    int maxVertexTextureImageUnits;
    int maxCombinedTextureImageUnits;
    int maxTextureImageUnits;
    int maxFragmentUniformComponents;
    int maxDrawBuffers;
    int maxVertexUniformVectors;
    int maxVaryingVectors;
    int maxFragmentUniformVectors;
    int maxVertexOutputVectors;
    int maxFragmentInputVectors;
    int minProgramTexelOffset;
    int maxProgramTexelOffset;
    int maxClipDistances;
    int maxComputeWorkGroupCountX;
    int maxComputeWorkGroupCountY;
    int maxComputeWorkGroupCountZ;
    int maxComputeWorkGroupSizeX;
    int maxComputeWorkGroupSizeY;
    int maxComputeWorkGroupSizeZ;
    int maxComputeUniformComponents;
    int maxComputeTextureImageUnits;
    int maxComputeImageUniforms;
    int maxComputeAtomicCounters;
    int maxComputeAtomicCounterBuffers;
    int maxVaryingComponents;
    int maxVertexOutputComponents;
    int maxGeometryInputComponents;
    int maxGeometryOutputComponents;
    int maxFragmentInputComponents;
    int maxImageUnits;
    int maxCombinedImageUnitsAndFragmentOutputs;
    int maxCombinedShaderOutputResources;
    int maxImageSamples;
    int maxVertexImageUniforms;
    int maxTessControlImageUniforms;
    int maxTessEvaluationImageUniforms;
    int maxGeometryImageUniforms;
    int maxFragmentImageUniforms;
    int maxCombinedImageUniforms;
    int maxGeometryTextureImageUnits;
    int maxGeometryOutputVertices;
    int maxGeometryTotalOutputComponents;
    int maxGeometryUniformComponents;
    int maxGeometryVaryingComponents;
    int maxTessControlInputComponents;
    int maxTessControlOutputComponents;
    int maxTessControlTextureImageUnits;
    int maxTessControlUniformComponents;
    int maxTessControlTotalOutputComponents;
    int maxTessEvaluationInputComponents;
    int maxTessEvaluationOutputComponents;
    int maxTessEvaluationTextureImageUnits;
    int maxTessEvaluationUniformComponents;
    int maxTessPatchComponents;
    int maxPatchVertices;
    int maxTessGenLevel;
    int maxViewports;
    int maxVertexAtomicCounters;
    int maxTessControlAtomicCounters;
    int maxTessEvaluationAtomicCounters;
    int maxGeometryAtomicCounters;
    int maxFragmentAtomicCounters;
    int maxCombinedAtomicCounters;
    int maxAtomicCounterBindings;
    int maxVertexAtomicCounterBuffers;
    int maxTessControlAtomicCounterBuffers;
    int maxTessEvaluationAtomicCounterBuffers;
    int maxGeometryAtomicCounterBuffers;
    int maxFragmentAtomicCounterBuffers;
    int maxCombinedAtomicCounterBuffers;
    int maxAtomicCounterBufferSize;
    int maxTransformFeedbackBuffers;
    int maxTransformFeedbackInterleavedComponents;
    int maxCullDistances;
    int maxCombinedClipAndCullDistances;
    int maxSamples;
    int maxMeshOutputVerticesNV;
    int maxMeshOutputPrimitivesNV;
    int maxMeshWorkGroupSizeX_NV;
    int maxMeshWorkGroupSizeY_NV;
    int maxMeshWorkGroupSizeZ_NV;
    int maxTaskWorkGroupSizeX_NV;
    int maxTaskWorkGroupSizeY_NV;
    int maxTaskWorkGroupSizeZ_NV;
    int maxMeshViewCountNV;
    int maxDualSourceDrawBuffersEXT;

    TLimits limits;
};

#endif // _RESOURCE_LIMITS_INCLUDED_
