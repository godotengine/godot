//
// Copyright (C) 2016 Google, Inc.
// Copyright (C) 2016 LunarG, Inc.
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
//    Neither the name of Google, Inc., nor the names of its
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

//
// HLSL scanning, leveraging the scanning done by the preprocessor.
//

#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "../glslang/Include/Types.h"
#include "../glslang/MachineIndependent/SymbolTable.h"
#include "../glslang/MachineIndependent/ParseHelper.h"
#include "hlslScanContext.h"
#include "hlslTokens.h"

// preprocessor includes
#include "../glslang/MachineIndependent/preprocessor/PpContext.h"
#include "../glslang/MachineIndependent/preprocessor/PpTokens.h"

namespace {

struct str_eq
{
    bool operator()(const char* lhs, const char* rhs) const
    {
        return strcmp(lhs, rhs) == 0;
    }
};

struct str_hash
{
    size_t operator()(const char* str) const
    {
        // djb2
        unsigned long hash = 5381;
        int c;

        while ((c = *str++) != 0)
            hash = ((hash << 5) + hash) + c;

        return hash;
    }
};

// A single global usable by all threads, by all versions, by all languages.
// After a single process-level initialization, this is read only and thread safe
std::unordered_map<const char*, glslang::EHlslTokenClass, str_hash, str_eq>* KeywordMap = nullptr;
std::unordered_set<const char*, str_hash, str_eq>* ReservedSet = nullptr;
std::unordered_map<const char*, glslang::TBuiltInVariable, str_hash, str_eq>* SemanticMap = nullptr;

};

namespace glslang {

void HlslScanContext::fillInKeywordMap()
{
    if (KeywordMap != nullptr) {
        // this is really an error, as this should called only once per process
        // but, the only risk is if two threads called simultaneously
        return;
    }
    KeywordMap = new std::unordered_map<const char*, EHlslTokenClass, str_hash, str_eq>;

    (*KeywordMap)["static"] =                  EHTokStatic;
    (*KeywordMap)["const"] =                   EHTokConst;
    (*KeywordMap)["unorm"] =                   EHTokUnorm;
    (*KeywordMap)["snorm"] =                   EHTokSNorm;
    (*KeywordMap)["extern"] =                  EHTokExtern;
    (*KeywordMap)["uniform"] =                 EHTokUniform;
    (*KeywordMap)["volatile"] =                EHTokVolatile;
    (*KeywordMap)["precise"] =                 EHTokPrecise;
    (*KeywordMap)["shared"] =                  EHTokShared;
    (*KeywordMap)["groupshared"] =             EHTokGroupShared;
    (*KeywordMap)["linear"] =                  EHTokLinear;
    (*KeywordMap)["centroid"] =                EHTokCentroid;
    (*KeywordMap)["nointerpolation"] =         EHTokNointerpolation;
    (*KeywordMap)["noperspective"] =           EHTokNoperspective;
    (*KeywordMap)["sample"] =                  EHTokSample;
    (*KeywordMap)["row_major"] =               EHTokRowMajor;
    (*KeywordMap)["column_major"] =            EHTokColumnMajor;
    (*KeywordMap)["packoffset"] =              EHTokPackOffset;
    (*KeywordMap)["in"] =                      EHTokIn;
    (*KeywordMap)["out"] =                     EHTokOut;
    (*KeywordMap)["inout"] =                   EHTokInOut;
    (*KeywordMap)["layout"] =                  EHTokLayout;
    (*KeywordMap)["globallycoherent"] =        EHTokGloballyCoherent;
    (*KeywordMap)["inline"] =                  EHTokInline;

    (*KeywordMap)["point"] =                   EHTokPoint;
    (*KeywordMap)["line"] =                    EHTokLine;
    (*KeywordMap)["triangle"] =                EHTokTriangle;
    (*KeywordMap)["lineadj"] =                 EHTokLineAdj;
    (*KeywordMap)["triangleadj"] =             EHTokTriangleAdj;

    (*KeywordMap)["PointStream"] =             EHTokPointStream;
    (*KeywordMap)["LineStream"] =              EHTokLineStream;
    (*KeywordMap)["TriangleStream"] =          EHTokTriangleStream;

    (*KeywordMap)["InputPatch"] =              EHTokInputPatch;
    (*KeywordMap)["OutputPatch"] =             EHTokOutputPatch;

    (*KeywordMap)["Buffer"] =                  EHTokBuffer;
    (*KeywordMap)["vector"] =                  EHTokVector;
    (*KeywordMap)["matrix"] =                  EHTokMatrix;

    (*KeywordMap)["void"] =                    EHTokVoid;
    (*KeywordMap)["string"] =                  EHTokString;
    (*KeywordMap)["bool"] =                    EHTokBool;
    (*KeywordMap)["int"] =                     EHTokInt;
    (*KeywordMap)["uint"] =                    EHTokUint;
    (*KeywordMap)["uint64_t"] =                EHTokUint64;
    (*KeywordMap)["dword"] =                   EHTokDword;
    (*KeywordMap)["half"] =                    EHTokHalf;
    (*KeywordMap)["float"] =                   EHTokFloat;
    (*KeywordMap)["double"] =                  EHTokDouble;
    (*KeywordMap)["min16float"] =              EHTokMin16float;
    (*KeywordMap)["min10float"] =              EHTokMin10float;
    (*KeywordMap)["min16int"] =                EHTokMin16int;
    (*KeywordMap)["min12int"] =                EHTokMin12int;
    (*KeywordMap)["min16uint"] =               EHTokMin16uint;

    (*KeywordMap)["bool1"] =                   EHTokBool1;
    (*KeywordMap)["bool2"] =                   EHTokBool2;
    (*KeywordMap)["bool3"] =                   EHTokBool3;
    (*KeywordMap)["bool4"] =                   EHTokBool4;
    (*KeywordMap)["float1"] =                  EHTokFloat1;
    (*KeywordMap)["float2"] =                  EHTokFloat2;
    (*KeywordMap)["float3"] =                  EHTokFloat3;
    (*KeywordMap)["float4"] =                  EHTokFloat4;
    (*KeywordMap)["int1"] =                    EHTokInt1;
    (*KeywordMap)["int2"] =                    EHTokInt2;
    (*KeywordMap)["int3"] =                    EHTokInt3;
    (*KeywordMap)["int4"] =                    EHTokInt4;
    (*KeywordMap)["double1"] =                 EHTokDouble1;
    (*KeywordMap)["double2"] =                 EHTokDouble2;
    (*KeywordMap)["double3"] =                 EHTokDouble3;
    (*KeywordMap)["double4"] =                 EHTokDouble4;
    (*KeywordMap)["uint1"] =                   EHTokUint1;
    (*KeywordMap)["uint2"] =                   EHTokUint2;
    (*KeywordMap)["uint3"] =                   EHTokUint3;
    (*KeywordMap)["uint4"] =                   EHTokUint4;

    (*KeywordMap)["half1"] =                   EHTokHalf1;
    (*KeywordMap)["half2"] =                   EHTokHalf2;
    (*KeywordMap)["half3"] =                   EHTokHalf3;
    (*KeywordMap)["half4"] =                   EHTokHalf4;
    (*KeywordMap)["min16float1"] =             EHTokMin16float1;
    (*KeywordMap)["min16float2"] =             EHTokMin16float2;
    (*KeywordMap)["min16float3"] =             EHTokMin16float3;
    (*KeywordMap)["min16float4"] =             EHTokMin16float4;
    (*KeywordMap)["min10float1"] =             EHTokMin10float1;
    (*KeywordMap)["min10float2"] =             EHTokMin10float2;
    (*KeywordMap)["min10float3"] =             EHTokMin10float3;
    (*KeywordMap)["min10float4"] =             EHTokMin10float4;
    (*KeywordMap)["min16int1"] =               EHTokMin16int1;
    (*KeywordMap)["min16int2"] =               EHTokMin16int2;
    (*KeywordMap)["min16int3"] =               EHTokMin16int3;
    (*KeywordMap)["min16int4"] =               EHTokMin16int4;
    (*KeywordMap)["min12int1"] =               EHTokMin12int1;
    (*KeywordMap)["min12int2"] =               EHTokMin12int2;
    (*KeywordMap)["min12int3"] =               EHTokMin12int3;
    (*KeywordMap)["min12int4"] =               EHTokMin12int4;
    (*KeywordMap)["min16uint1"] =              EHTokMin16uint1;
    (*KeywordMap)["min16uint2"] =              EHTokMin16uint2;
    (*KeywordMap)["min16uint3"] =              EHTokMin16uint3;
    (*KeywordMap)["min16uint4"] =              EHTokMin16uint4;

    (*KeywordMap)["bool1x1"] =                 EHTokBool1x1;
    (*KeywordMap)["bool1x2"] =                 EHTokBool1x2;
    (*KeywordMap)["bool1x3"] =                 EHTokBool1x3;
    (*KeywordMap)["bool1x4"] =                 EHTokBool1x4;
    (*KeywordMap)["bool2x1"] =                 EHTokBool2x1;
    (*KeywordMap)["bool2x2"] =                 EHTokBool2x2;
    (*KeywordMap)["bool2x3"] =                 EHTokBool2x3;
    (*KeywordMap)["bool2x4"] =                 EHTokBool2x4;
    (*KeywordMap)["bool3x1"] =                 EHTokBool3x1;
    (*KeywordMap)["bool3x2"] =                 EHTokBool3x2;
    (*KeywordMap)["bool3x3"] =                 EHTokBool3x3;
    (*KeywordMap)["bool3x4"] =                 EHTokBool3x4;
    (*KeywordMap)["bool4x1"] =                 EHTokBool4x1;
    (*KeywordMap)["bool4x2"] =                 EHTokBool4x2;
    (*KeywordMap)["bool4x3"] =                 EHTokBool4x3;
    (*KeywordMap)["bool4x4"] =                 EHTokBool4x4;
    (*KeywordMap)["int1x1"] =                  EHTokInt1x1;
    (*KeywordMap)["int1x2"] =                  EHTokInt1x2;
    (*KeywordMap)["int1x3"] =                  EHTokInt1x3;
    (*KeywordMap)["int1x4"] =                  EHTokInt1x4;
    (*KeywordMap)["int2x1"] =                  EHTokInt2x1;
    (*KeywordMap)["int2x2"] =                  EHTokInt2x2;
    (*KeywordMap)["int2x3"] =                  EHTokInt2x3;
    (*KeywordMap)["int2x4"] =                  EHTokInt2x4;
    (*KeywordMap)["int3x1"] =                  EHTokInt3x1;
    (*KeywordMap)["int3x2"] =                  EHTokInt3x2;
    (*KeywordMap)["int3x3"] =                  EHTokInt3x3;
    (*KeywordMap)["int3x4"] =                  EHTokInt3x4;
    (*KeywordMap)["int4x1"] =                  EHTokInt4x1;
    (*KeywordMap)["int4x2"] =                  EHTokInt4x2;
    (*KeywordMap)["int4x3"] =                  EHTokInt4x3;
    (*KeywordMap)["int4x4"] =                  EHTokInt4x4;
    (*KeywordMap)["uint1x1"] =                 EHTokUint1x1;
    (*KeywordMap)["uint1x2"] =                 EHTokUint1x2;
    (*KeywordMap)["uint1x3"] =                 EHTokUint1x3;
    (*KeywordMap)["uint1x4"] =                 EHTokUint1x4;
    (*KeywordMap)["uint2x1"] =                 EHTokUint2x1;
    (*KeywordMap)["uint2x2"] =                 EHTokUint2x2;
    (*KeywordMap)["uint2x3"] =                 EHTokUint2x3;
    (*KeywordMap)["uint2x4"] =                 EHTokUint2x4;
    (*KeywordMap)["uint3x1"] =                 EHTokUint3x1;
    (*KeywordMap)["uint3x2"] =                 EHTokUint3x2;
    (*KeywordMap)["uint3x3"] =                 EHTokUint3x3;
    (*KeywordMap)["uint3x4"] =                 EHTokUint3x4;
    (*KeywordMap)["uint4x1"] =                 EHTokUint4x1;
    (*KeywordMap)["uint4x2"] =                 EHTokUint4x2;
    (*KeywordMap)["uint4x3"] =                 EHTokUint4x3;
    (*KeywordMap)["uint4x4"] =                 EHTokUint4x4;
    (*KeywordMap)["bool1x1"] =                 EHTokBool1x1;
    (*KeywordMap)["bool1x2"] =                 EHTokBool1x2;
    (*KeywordMap)["bool1x3"] =                 EHTokBool1x3;
    (*KeywordMap)["bool1x4"] =                 EHTokBool1x4;
    (*KeywordMap)["bool2x1"] =                 EHTokBool2x1;
    (*KeywordMap)["bool2x2"] =                 EHTokBool2x2;
    (*KeywordMap)["bool2x3"] =                 EHTokBool2x3;
    (*KeywordMap)["bool2x4"] =                 EHTokBool2x4;
    (*KeywordMap)["bool3x1"] =                 EHTokBool3x1;
    (*KeywordMap)["bool3x2"] =                 EHTokBool3x2;
    (*KeywordMap)["bool3x3"] =                 EHTokBool3x3;
    (*KeywordMap)["bool3x4"] =                 EHTokBool3x4;
    (*KeywordMap)["bool4x1"] =                 EHTokBool4x1;
    (*KeywordMap)["bool4x2"] =                 EHTokBool4x2;
    (*KeywordMap)["bool4x3"] =                 EHTokBool4x3;
    (*KeywordMap)["bool4x4"] =                 EHTokBool4x4;
    (*KeywordMap)["float1x1"] =                EHTokFloat1x1;
    (*KeywordMap)["float1x2"] =                EHTokFloat1x2;
    (*KeywordMap)["float1x3"] =                EHTokFloat1x3;
    (*KeywordMap)["float1x4"] =                EHTokFloat1x4;
    (*KeywordMap)["float2x1"] =                EHTokFloat2x1;
    (*KeywordMap)["float2x2"] =                EHTokFloat2x2;
    (*KeywordMap)["float2x3"] =                EHTokFloat2x3;
    (*KeywordMap)["float2x4"] =                EHTokFloat2x4;
    (*KeywordMap)["float3x1"] =                EHTokFloat3x1;
    (*KeywordMap)["float3x2"] =                EHTokFloat3x2;
    (*KeywordMap)["float3x3"] =                EHTokFloat3x3;
    (*KeywordMap)["float3x4"] =                EHTokFloat3x4;
    (*KeywordMap)["float4x1"] =                EHTokFloat4x1;
    (*KeywordMap)["float4x2"] =                EHTokFloat4x2;
    (*KeywordMap)["float4x3"] =                EHTokFloat4x3;
    (*KeywordMap)["float4x4"] =                EHTokFloat4x4;
    (*KeywordMap)["half1x1"] =                 EHTokHalf1x1;
    (*KeywordMap)["half1x2"] =                 EHTokHalf1x2;
    (*KeywordMap)["half1x3"] =                 EHTokHalf1x3;
    (*KeywordMap)["half1x4"] =                 EHTokHalf1x4;
    (*KeywordMap)["half2x1"] =                 EHTokHalf2x1;
    (*KeywordMap)["half2x2"] =                 EHTokHalf2x2;
    (*KeywordMap)["half2x3"] =                 EHTokHalf2x3;
    (*KeywordMap)["half2x4"] =                 EHTokHalf2x4;
    (*KeywordMap)["half3x1"] =                 EHTokHalf3x1;
    (*KeywordMap)["half3x2"] =                 EHTokHalf3x2;
    (*KeywordMap)["half3x3"] =                 EHTokHalf3x3;
    (*KeywordMap)["half3x4"] =                 EHTokHalf3x4;
    (*KeywordMap)["half4x1"] =                 EHTokHalf4x1;
    (*KeywordMap)["half4x2"] =                 EHTokHalf4x2;
    (*KeywordMap)["half4x3"] =                 EHTokHalf4x3;
    (*KeywordMap)["half4x4"] =                 EHTokHalf4x4;
    (*KeywordMap)["double1x1"] =               EHTokDouble1x1;
    (*KeywordMap)["double1x2"] =               EHTokDouble1x2;
    (*KeywordMap)["double1x3"] =               EHTokDouble1x3;
    (*KeywordMap)["double1x4"] =               EHTokDouble1x4;
    (*KeywordMap)["double2x1"] =               EHTokDouble2x1;
    (*KeywordMap)["double2x2"] =               EHTokDouble2x2;
    (*KeywordMap)["double2x3"] =               EHTokDouble2x3;
    (*KeywordMap)["double2x4"] =               EHTokDouble2x4;
    (*KeywordMap)["double3x1"] =               EHTokDouble3x1;
    (*KeywordMap)["double3x2"] =               EHTokDouble3x2;
    (*KeywordMap)["double3x3"] =               EHTokDouble3x3;
    (*KeywordMap)["double3x4"] =               EHTokDouble3x4;
    (*KeywordMap)["double4x1"] =               EHTokDouble4x1;
    (*KeywordMap)["double4x2"] =               EHTokDouble4x2;
    (*KeywordMap)["double4x3"] =               EHTokDouble4x3;
    (*KeywordMap)["double4x4"] =               EHTokDouble4x4;

    (*KeywordMap)["sampler"] =                 EHTokSampler;
    (*KeywordMap)["sampler1D"] =               EHTokSampler1d;
    (*KeywordMap)["sampler2D"] =               EHTokSampler2d;
    (*KeywordMap)["sampler3D"] =               EHTokSampler3d;
    (*KeywordMap)["samplerCube"] =             EHTokSamplerCube;
    (*KeywordMap)["sampler_state"] =           EHTokSamplerState;
    (*KeywordMap)["SamplerState"] =            EHTokSamplerState;
    (*KeywordMap)["SamplerComparisonState"] =  EHTokSamplerComparisonState;
    (*KeywordMap)["texture"] =                 EHTokTexture;
    (*KeywordMap)["Texture1D"] =               EHTokTexture1d;
    (*KeywordMap)["Texture1DArray"] =          EHTokTexture1darray;
    (*KeywordMap)["Texture2D"] =               EHTokTexture2d;
    (*KeywordMap)["Texture2DArray"] =          EHTokTexture2darray;
    (*KeywordMap)["Texture3D"] =               EHTokTexture3d;
    (*KeywordMap)["TextureCube"] =             EHTokTextureCube;
    (*KeywordMap)["TextureCubeArray"] =        EHTokTextureCubearray;
    (*KeywordMap)["Texture2DMS"] =             EHTokTexture2DMS;
    (*KeywordMap)["Texture2DMSArray"] =        EHTokTexture2DMSarray;
    (*KeywordMap)["RWTexture1D"] =             EHTokRWTexture1d;
    (*KeywordMap)["RWTexture1DArray"] =        EHTokRWTexture1darray;
    (*KeywordMap)["RWTexture2D"] =             EHTokRWTexture2d;
    (*KeywordMap)["RWTexture2DArray"] =        EHTokRWTexture2darray;
    (*KeywordMap)["RWTexture3D"] =             EHTokRWTexture3d;
    (*KeywordMap)["RWBuffer"] =                EHTokRWBuffer;
    (*KeywordMap)["SubpassInput"] =            EHTokSubpassInput;
    (*KeywordMap)["SubpassInputMS"] =          EHTokSubpassInputMS;

    (*KeywordMap)["AppendStructuredBuffer"] =  EHTokAppendStructuredBuffer;
    (*KeywordMap)["ByteAddressBuffer"] =       EHTokByteAddressBuffer;
    (*KeywordMap)["ConsumeStructuredBuffer"] = EHTokConsumeStructuredBuffer;
    (*KeywordMap)["RWByteAddressBuffer"] =     EHTokRWByteAddressBuffer;
    (*KeywordMap)["RWStructuredBuffer"] =      EHTokRWStructuredBuffer;
    (*KeywordMap)["StructuredBuffer"] =        EHTokStructuredBuffer;
    (*KeywordMap)["TextureBuffer"] =           EHTokTextureBuffer;

    (*KeywordMap)["class"] =                   EHTokClass;
    (*KeywordMap)["struct"] =                  EHTokStruct;
    (*KeywordMap)["cbuffer"] =                 EHTokCBuffer;
    (*KeywordMap)["ConstantBuffer"] =          EHTokConstantBuffer;
    (*KeywordMap)["tbuffer"] =                 EHTokTBuffer;
    (*KeywordMap)["typedef"] =                 EHTokTypedef;
    (*KeywordMap)["this"] =                    EHTokThis;
    (*KeywordMap)["namespace"] =               EHTokNamespace;

    (*KeywordMap)["true"] =                    EHTokBoolConstant;
    (*KeywordMap)["false"] =                   EHTokBoolConstant;

    (*KeywordMap)["for"] =                     EHTokFor;
    (*KeywordMap)["do"] =                      EHTokDo;
    (*KeywordMap)["while"] =                   EHTokWhile;
    (*KeywordMap)["break"] =                   EHTokBreak;
    (*KeywordMap)["continue"] =                EHTokContinue;
    (*KeywordMap)["if"] =                      EHTokIf;
    (*KeywordMap)["else"] =                    EHTokElse;
    (*KeywordMap)["discard"] =                 EHTokDiscard;
    (*KeywordMap)["return"] =                  EHTokReturn;
    (*KeywordMap)["switch"] =                  EHTokSwitch;
    (*KeywordMap)["case"] =                    EHTokCase;
    (*KeywordMap)["default"] =                 EHTokDefault;

    // TODO: get correct set here
    ReservedSet = new std::unordered_set<const char*, str_hash, str_eq>;

    ReservedSet->insert("auto");
    ReservedSet->insert("catch");
    ReservedSet->insert("char");
    ReservedSet->insert("const_cast");
    ReservedSet->insert("enum");
    ReservedSet->insert("explicit");
    ReservedSet->insert("friend");
    ReservedSet->insert("goto");
    ReservedSet->insert("long");
    ReservedSet->insert("mutable");
    ReservedSet->insert("new");
    ReservedSet->insert("operator");
    ReservedSet->insert("private");
    ReservedSet->insert("protected");
    ReservedSet->insert("public");
    ReservedSet->insert("reinterpret_cast");
    ReservedSet->insert("short");
    ReservedSet->insert("signed");
    ReservedSet->insert("sizeof");
    ReservedSet->insert("static_cast");
    ReservedSet->insert("template");
    ReservedSet->insert("throw");
    ReservedSet->insert("try");
    ReservedSet->insert("typename");
    ReservedSet->insert("union");
    ReservedSet->insert("unsigned");
    ReservedSet->insert("using");
    ReservedSet->insert("virtual");

    SemanticMap = new std::unordered_map<const char*, glslang::TBuiltInVariable, str_hash, str_eq>;

    // in DX9, all outputs had to have a semantic associated with them, that was either consumed
    // by the system or was a specific register assignment
    // in DX10+, only semantics with the SV_ prefix have any meaning beyond decoration
    // Fxc will only accept DX9 style semantics in compat mode
    // Also, in DX10 if a SV value is present as the input of a stage, but isn't appropriate for that
    // stage, it would just be ignored as it is likely there as part of an output struct from one stage
    // to the next
    bool bParseDX9 = false;
    if (bParseDX9) {
        (*SemanticMap)["PSIZE"] = EbvPointSize;
        (*SemanticMap)["FOG"] =   EbvFogFragCoord;
        (*SemanticMap)["DEPTH"] = EbvFragDepth;
        (*SemanticMap)["VFACE"] = EbvFace;
        (*SemanticMap)["VPOS"] =  EbvFragCoord;
    }

    (*SemanticMap)["SV_POSITION"] =               EbvPosition;
    (*SemanticMap)["SV_VERTEXID"] =               EbvVertexIndex;
    (*SemanticMap)["SV_VIEWPORTARRAYINDEX"] =     EbvViewportIndex;
    (*SemanticMap)["SV_TESSFACTOR"] =             EbvTessLevelOuter;
    (*SemanticMap)["SV_SAMPLEINDEX"] =            EbvSampleId;
    (*SemanticMap)["SV_RENDERTARGETARRAYINDEX"] = EbvLayer;
    (*SemanticMap)["SV_PRIMITIVEID"] =            EbvPrimitiveId;
    (*SemanticMap)["SV_OUTPUTCONTROLPOINTID"] =   EbvInvocationId;
    (*SemanticMap)["SV_ISFRONTFACE"] =            EbvFace;
    (*SemanticMap)["SV_INSTANCEID"] =             EbvInstanceIndex;
    (*SemanticMap)["SV_INSIDETESSFACTOR"] =       EbvTessLevelInner;
    (*SemanticMap)["SV_GSINSTANCEID"] =           EbvInvocationId;
    (*SemanticMap)["SV_DISPATCHTHREADID"] =       EbvGlobalInvocationId;
    (*SemanticMap)["SV_GROUPTHREADID"] =          EbvLocalInvocationId;
    (*SemanticMap)["SV_GROUPINDEX"] =             EbvLocalInvocationIndex;
    (*SemanticMap)["SV_GROUPID"] =                EbvWorkGroupId;
    (*SemanticMap)["SV_DOMAINLOCATION"] =         EbvTessCoord;
    (*SemanticMap)["SV_DEPTH"] =                  EbvFragDepth;
    (*SemanticMap)["SV_COVERAGE"] =               EbvSampleMask;
    (*SemanticMap)["SV_DEPTHGREATEREQUAL"] =      EbvFragDepthGreater;
    (*SemanticMap)["SV_DEPTHLESSEQUAL"] =         EbvFragDepthLesser;
    (*SemanticMap)["SV_STENCILREF"] =             EbvFragStencilRef;
}

void HlslScanContext::deleteKeywordMap()
{
    delete KeywordMap;
    KeywordMap = nullptr;
    delete ReservedSet;
    ReservedSet = nullptr;
    delete SemanticMap;
    SemanticMap = nullptr;
}

// Wrapper for tokenizeClass() to get everything inside the token.
void HlslScanContext::tokenize(HlslToken& token)
{
    EHlslTokenClass tokenClass = tokenizeClass(token);
    token.tokenClass = tokenClass;
}

glslang::TBuiltInVariable HlslScanContext::mapSemantic(const char* upperCase)
{
    auto it = SemanticMap->find(upperCase);
    if (it != SemanticMap->end())
        return it->second;
    else
        return glslang::EbvNone;
}

//
// Fill in token information for the next token, except for the token class.
// Returns the enum value of the token class of the next token found.
// Return 0 (EndOfTokens) on end of input.
//
EHlslTokenClass HlslScanContext::tokenizeClass(HlslToken& token)
{
    do {
        parserToken = &token;
        TPpToken ppToken;
        int token = ppContext.tokenize(ppToken);
        if (token == EndOfInput)
            return EHTokNone;

        tokenText = ppToken.name;
        loc = ppToken.loc;
        parserToken->loc = loc;
        switch (token) {
        case ';':                       return EHTokSemicolon;
        case ',':                       return EHTokComma;
        case ':':                       return EHTokColon;
        case '=':                       return EHTokAssign;
        case '(':                       return EHTokLeftParen;
        case ')':                       return EHTokRightParen;
        case '.':                       return EHTokDot;
        case '!':                       return EHTokBang;
        case '-':                       return EHTokDash;
        case '~':                       return EHTokTilde;
        case '+':                       return EHTokPlus;
        case '*':                       return EHTokStar;
        case '/':                       return EHTokSlash;
        case '%':                       return EHTokPercent;
        case '<':                       return EHTokLeftAngle;
        case '>':                       return EHTokRightAngle;
        case '|':                       return EHTokVerticalBar;
        case '^':                       return EHTokCaret;
        case '&':                       return EHTokAmpersand;
        case '?':                       return EHTokQuestion;
        case '[':                       return EHTokLeftBracket;
        case ']':                       return EHTokRightBracket;
        case '{':                       return EHTokLeftBrace;
        case '}':                       return EHTokRightBrace;
        case '\\':
            parseContext.error(loc, "illegal use of escape character", "\\", "");
            break;

        case PPAtomAddAssign:          return EHTokAddAssign;
        case PPAtomSubAssign:          return EHTokSubAssign;
        case PPAtomMulAssign:          return EHTokMulAssign;
        case PPAtomDivAssign:          return EHTokDivAssign;
        case PPAtomModAssign:          return EHTokModAssign;

        case PpAtomRight:              return EHTokRightOp;
        case PpAtomLeft:               return EHTokLeftOp;

        case PpAtomRightAssign:        return EHTokRightAssign;
        case PpAtomLeftAssign:         return EHTokLeftAssign;
        case PpAtomAndAssign:          return EHTokAndAssign;
        case PpAtomOrAssign:           return EHTokOrAssign;
        case PpAtomXorAssign:          return EHTokXorAssign;

        case PpAtomAnd:                return EHTokAndOp;
        case PpAtomOr:                 return EHTokOrOp;
        case PpAtomXor:                return EHTokXorOp;

        case PpAtomEQ:                 return EHTokEqOp;
        case PpAtomGE:                 return EHTokGeOp;
        case PpAtomNE:                 return EHTokNeOp;
        case PpAtomLE:                 return EHTokLeOp;

        case PpAtomDecrement:          return EHTokDecOp;
        case PpAtomIncrement:          return EHTokIncOp;

        case PpAtomColonColon:         return EHTokColonColon;

        case PpAtomConstInt:           parserToken->i = ppToken.ival;       return EHTokIntConstant;
        case PpAtomConstUint:          parserToken->i = ppToken.ival;       return EHTokUintConstant;
        case PpAtomConstFloat16:       parserToken->d = ppToken.dval;       return EHTokFloat16Constant;
        case PpAtomConstFloat:         parserToken->d = ppToken.dval;       return EHTokFloatConstant;
        case PpAtomConstDouble:        parserToken->d = ppToken.dval;       return EHTokDoubleConstant;
        case PpAtomIdentifier:
        {
            EHlslTokenClass token = tokenizeIdentifier();
            return token;
        }

        case PpAtomConstString: {
            parserToken->string = NewPoolTString(tokenText);
            return EHTokStringConstant;
        }

        case EndOfInput:               return EHTokNone;

        default:
            if (token < PpAtomMaxSingle) {
                char buf[2];
                buf[0] = (char)token;
                buf[1] = 0;
                parseContext.error(loc, "unexpected token", buf, "");
            } else if (tokenText[0] != 0)
                parseContext.error(loc, "unexpected token", tokenText, "");
            else
                parseContext.error(loc, "unexpected token", "", "");
            break;
        }
    } while (true);
}

EHlslTokenClass HlslScanContext::tokenizeIdentifier()
{
    if (ReservedSet->find(tokenText) != ReservedSet->end())
        return reservedWord();

    auto it = KeywordMap->find(tokenText);
    if (it == KeywordMap->end()) {
        // Should have an identifier of some sort
        return identifierOrType();
    }
    keyword = it->second;

    switch (keyword) {

    // qualifiers
    case EHTokStatic:
    case EHTokConst:
    case EHTokSNorm:
    case EHTokUnorm:
    case EHTokExtern:
    case EHTokUniform:
    case EHTokVolatile:
    case EHTokShared:
    case EHTokGroupShared:
    case EHTokLinear:
    case EHTokCentroid:
    case EHTokNointerpolation:
    case EHTokNoperspective:
    case EHTokSample:
    case EHTokRowMajor:
    case EHTokColumnMajor:
    case EHTokPackOffset:
    case EHTokIn:
    case EHTokOut:
    case EHTokInOut:
    case EHTokPrecise:
    case EHTokLayout:
    case EHTokGloballyCoherent:
    case EHTokInline:
        return keyword;

    // primitive types
    case EHTokPoint:
    case EHTokLine:
    case EHTokTriangle:
    case EHTokLineAdj:
    case EHTokTriangleAdj:
        return keyword;

    // stream out types
    case EHTokPointStream:
    case EHTokLineStream:
    case EHTokTriangleStream:
        return keyword;

    // Tessellation patches
    case EHTokInputPatch:
    case EHTokOutputPatch:
        return keyword;

    case EHTokBuffer:
    case EHTokVector:
    case EHTokMatrix:
        return keyword;

    // scalar types
    case EHTokVoid:
    case EHTokString:
    case EHTokBool:
    case EHTokInt:
    case EHTokUint:
    case EHTokUint64:
    case EHTokDword:
    case EHTokHalf:
    case EHTokFloat:
    case EHTokDouble:
    case EHTokMin16float:
    case EHTokMin10float:
    case EHTokMin16int:
    case EHTokMin12int:
    case EHTokMin16uint:

    // vector types
    case EHTokBool1:
    case EHTokBool2:
    case EHTokBool3:
    case EHTokBool4:
    case EHTokFloat1:
    case EHTokFloat2:
    case EHTokFloat3:
    case EHTokFloat4:
    case EHTokInt1:
    case EHTokInt2:
    case EHTokInt3:
    case EHTokInt4:
    case EHTokDouble1:
    case EHTokDouble2:
    case EHTokDouble3:
    case EHTokDouble4:
    case EHTokUint1:
    case EHTokUint2:
    case EHTokUint3:
    case EHTokUint4:
    case EHTokHalf1:
    case EHTokHalf2:
    case EHTokHalf3:
    case EHTokHalf4:
    case EHTokMin16float1:
    case EHTokMin16float2:
    case EHTokMin16float3:
    case EHTokMin16float4:
    case EHTokMin10float1:
    case EHTokMin10float2:
    case EHTokMin10float3:
    case EHTokMin10float4:
    case EHTokMin16int1:
    case EHTokMin16int2:
    case EHTokMin16int3:
    case EHTokMin16int4:
    case EHTokMin12int1:
    case EHTokMin12int2:
    case EHTokMin12int3:
    case EHTokMin12int4:
    case EHTokMin16uint1:
    case EHTokMin16uint2:
    case EHTokMin16uint3:
    case EHTokMin16uint4:

    // matrix types
    case EHTokBool1x1:
    case EHTokBool1x2:
    case EHTokBool1x3:
    case EHTokBool1x4:
    case EHTokBool2x1:
    case EHTokBool2x2:
    case EHTokBool2x3:
    case EHTokBool2x4:
    case EHTokBool3x1:
    case EHTokBool3x2:
    case EHTokBool3x3:
    case EHTokBool3x4:
    case EHTokBool4x1:
    case EHTokBool4x2:
    case EHTokBool4x3:
    case EHTokBool4x4:
    case EHTokInt1x1:
    case EHTokInt1x2:
    case EHTokInt1x3:
    case EHTokInt1x4:
    case EHTokInt2x1:
    case EHTokInt2x2:
    case EHTokInt2x3:
    case EHTokInt2x4:
    case EHTokInt3x1:
    case EHTokInt3x2:
    case EHTokInt3x3:
    case EHTokInt3x4:
    case EHTokInt4x1:
    case EHTokInt4x2:
    case EHTokInt4x3:
    case EHTokInt4x4:
    case EHTokUint1x1:
    case EHTokUint1x2:
    case EHTokUint1x3:
    case EHTokUint1x4:
    case EHTokUint2x1:
    case EHTokUint2x2:
    case EHTokUint2x3:
    case EHTokUint2x4:
    case EHTokUint3x1:
    case EHTokUint3x2:
    case EHTokUint3x3:
    case EHTokUint3x4:
    case EHTokUint4x1:
    case EHTokUint4x2:
    case EHTokUint4x3:
    case EHTokUint4x4:
    case EHTokFloat1x1:
    case EHTokFloat1x2:
    case EHTokFloat1x3:
    case EHTokFloat1x4:
    case EHTokFloat2x1:
    case EHTokFloat2x2:
    case EHTokFloat2x3:
    case EHTokFloat2x4:
    case EHTokFloat3x1:
    case EHTokFloat3x2:
    case EHTokFloat3x3:
    case EHTokFloat3x4:
    case EHTokFloat4x1:
    case EHTokFloat4x2:
    case EHTokFloat4x3:
    case EHTokFloat4x4:
    case EHTokHalf1x1:
    case EHTokHalf1x2:
    case EHTokHalf1x3:
    case EHTokHalf1x4:
    case EHTokHalf2x1:
    case EHTokHalf2x2:
    case EHTokHalf2x3:
    case EHTokHalf2x4:
    case EHTokHalf3x1:
    case EHTokHalf3x2:
    case EHTokHalf3x3:
    case EHTokHalf3x4:
    case EHTokHalf4x1:
    case EHTokHalf4x2:
    case EHTokHalf4x3:
    case EHTokHalf4x4:
    case EHTokDouble1x1:
    case EHTokDouble1x2:
    case EHTokDouble1x3:
    case EHTokDouble1x4:
    case EHTokDouble2x1:
    case EHTokDouble2x2:
    case EHTokDouble2x3:
    case EHTokDouble2x4:
    case EHTokDouble3x1:
    case EHTokDouble3x2:
    case EHTokDouble3x3:
    case EHTokDouble3x4:
    case EHTokDouble4x1:
    case EHTokDouble4x2:
    case EHTokDouble4x3:
    case EHTokDouble4x4:
        return keyword;

    // texturing types
    case EHTokSampler:
    case EHTokSampler1d:
    case EHTokSampler2d:
    case EHTokSampler3d:
    case EHTokSamplerCube:
    case EHTokSamplerState:
    case EHTokSamplerComparisonState:
    case EHTokTexture:
    case EHTokTexture1d:
    case EHTokTexture1darray:
    case EHTokTexture2d:
    case EHTokTexture2darray:
    case EHTokTexture3d:
    case EHTokTextureCube:
    case EHTokTextureCubearray:
    case EHTokTexture2DMS:
    case EHTokTexture2DMSarray:
    case EHTokRWTexture1d:
    case EHTokRWTexture1darray:
    case EHTokRWTexture2d:
    case EHTokRWTexture2darray:
    case EHTokRWTexture3d:
    case EHTokRWBuffer:
    case EHTokAppendStructuredBuffer:
    case EHTokByteAddressBuffer:
    case EHTokConsumeStructuredBuffer:
    case EHTokRWByteAddressBuffer:
    case EHTokRWStructuredBuffer:
    case EHTokStructuredBuffer:
    case EHTokTextureBuffer:
    case EHTokSubpassInput:
    case EHTokSubpassInputMS:
        return keyword;

    // variable, user type, ...
    case EHTokClass:
    case EHTokStruct:
    case EHTokTypedef:
    case EHTokCBuffer:
    case EHTokConstantBuffer:
    case EHTokTBuffer:
    case EHTokThis:
    case EHTokNamespace:
        return keyword;

    case EHTokBoolConstant:
        if (strcmp("true", tokenText) == 0)
            parserToken->b = true;
        else
            parserToken->b = false;
        return keyword;

    // control flow
    case EHTokFor:
    case EHTokDo:
    case EHTokWhile:
    case EHTokBreak:
    case EHTokContinue:
    case EHTokIf:
    case EHTokElse:
    case EHTokDiscard:
    case EHTokReturn:
    case EHTokCase:
    case EHTokSwitch:
    case EHTokDefault:
        return keyword;

    default:
        parseContext.infoSink.info.message(EPrefixInternalError, "Unknown glslang keyword", loc);
        return EHTokNone;
    }
}

EHlslTokenClass HlslScanContext::identifierOrType()
{
    parserToken->string = NewPoolTString(tokenText);

    return EHTokIdentifier;
}

// Give an error for use of a reserved symbol.
// However, allow built-in declarations to use reserved words, to allow
// extension support before the extension is enabled.
EHlslTokenClass HlslScanContext::reservedWord()
{
    if (! parseContext.symbolTable.atBuiltInLevel())
        parseContext.error(loc, "Reserved word.", tokenText, "", "");

    return EHTokNone;
}

} // end namespace glslang
