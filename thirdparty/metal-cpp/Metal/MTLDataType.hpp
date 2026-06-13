//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLDataType.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL
{
_MTL_ENUM(NS::UInteger, DataType) {
    DataTypeNone = 0,
    DataTypeStruct = 1,
    DataTypeArray = 2,
    DataTypeFloat = 3,
    DataTypeFloat2 = 4,
    DataTypeFloat3 = 5,
    DataTypeFloat4 = 6,
    DataTypeFloat2x2 = 7,
    DataTypeFloat2x3 = 8,
    DataTypeFloat2x4 = 9,
    DataTypeFloat3x2 = 10,
    DataTypeFloat3x3 = 11,
    DataTypeFloat3x4 = 12,
    DataTypeFloat4x2 = 13,
    DataTypeFloat4x3 = 14,
    DataTypeFloat4x4 = 15,
    DataTypeHalf = 16,
    DataTypeHalf2 = 17,
    DataTypeHalf3 = 18,
    DataTypeHalf4 = 19,
    DataTypeHalf2x2 = 20,
    DataTypeHalf2x3 = 21,
    DataTypeHalf2x4 = 22,
    DataTypeHalf3x2 = 23,
    DataTypeHalf3x3 = 24,
    DataTypeHalf3x4 = 25,
    DataTypeHalf4x2 = 26,
    DataTypeHalf4x3 = 27,
    DataTypeHalf4x4 = 28,
    DataTypeInt = 29,
    DataTypeInt2 = 30,
    DataTypeInt3 = 31,
    DataTypeInt4 = 32,
    DataTypeUInt = 33,
    DataTypeUInt2 = 34,
    DataTypeUInt3 = 35,
    DataTypeUInt4 = 36,
    DataTypeShort = 37,
    DataTypeShort2 = 38,
    DataTypeShort3 = 39,
    DataTypeShort4 = 40,
    DataTypeUShort = 41,
    DataTypeUShort2 = 42,
    DataTypeUShort3 = 43,
    DataTypeUShort4 = 44,
    DataTypeChar = 45,
    DataTypeChar2 = 46,
    DataTypeChar3 = 47,
    DataTypeChar4 = 48,
    DataTypeUChar = 49,
    DataTypeUChar2 = 50,
    DataTypeUChar3 = 51,
    DataTypeUChar4 = 52,
    DataTypeBool = 53,
    DataTypeBool2 = 54,
    DataTypeBool3 = 55,
    DataTypeBool4 = 56,
    DataTypeTexture = 58,
    DataTypeSampler = 59,
    DataTypePointer = 60,
    DataTypeR8Unorm = 62,
    DataTypeR8Snorm = 63,
    DataTypeR16Unorm = 64,
    DataTypeR16Snorm = 65,
    DataTypeRG8Unorm = 66,
    DataTypeRG8Snorm = 67,
    DataTypeRG16Unorm = 68,
    DataTypeRG16Snorm = 69,
    DataTypeRGBA8Unorm = 70,
    DataTypeRGBA8Unorm_sRGB = 71,
    DataTypeRGBA8Snorm = 72,
    DataTypeRGBA16Unorm = 73,
    DataTypeRGBA16Snorm = 74,
    DataTypeRGB10A2Unorm = 75,
    DataTypeRG11B10Float = 76,
    DataTypeRGB9E5Float = 77,
    DataTypeRenderPipeline = 78,
    DataTypeComputePipeline = 79,
    DataTypeIndirectCommandBuffer = 80,
    DataTypeLong = 81,
    DataTypeLong2 = 82,
    DataTypeLong3 = 83,
    DataTypeLong4 = 84,
    DataTypeULong = 85,
    DataTypeULong2 = 86,
    DataTypeULong3 = 87,
    DataTypeULong4 = 88,
    DataTypeVisibleFunctionTable = 115,
    DataTypeIntersectionFunctionTable = 116,
    DataTypePrimitiveAccelerationStructure = 117,
    DataTypeInstanceAccelerationStructure = 118,
    DataTypeBFloat = 121,
    DataTypeBFloat2 = 122,
    DataTypeBFloat3 = 123,
    DataTypeBFloat4 = 124,
    DataTypeDepthStencilState = 139,
    DataTypeTensor = 140,
};

}
