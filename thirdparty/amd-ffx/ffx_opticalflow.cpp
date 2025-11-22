// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <algorithm>    // for max used inside SPD CPU code.
#include <cmath>        // for fabs, abs, sinf, sqrt, etc.
#include <string>       // for memset
#include <cfloat>       // for FLT_EPSILON
#include "ffx_opticalflow.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#endif

#define FFX_CPU
#include "gpu/ffx_core.h"
#include "gpu/spd/ffx_spd.h"
#include "gpu/opticalflow/ffx_opticalflow_callbacks_hlsl.h"
#include "ffx_object_management.h"

#define FFX_OPTICALFLOW_MAX_QUEUED_FRAMES 16

#include "ffx_opticalflow_private.h"

typedef struct Binding
{
    uint32_t    index;
    wchar_t     name[64];
}Binding;

static const Binding srvBindingNames[] =
{
    {FFX_OF_BINDING_IDENTIFIER_INPUT_COLOR,                           L"r_input_color"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT,                    L"r_optical_flow_input"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS_INPUT,           L"r_optical_flow_previous_input"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW,                          L"r_optical_flow"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS,                 L"r_optical_flow_previous"},
};

static const Binding uavBindingNames[] =
{
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT,                      L"rw_optical_flow_input"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_1,              L"rw_optical_flow_input_level_1"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_2,              L"rw_optical_flow_input_level_2"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_3,              L"rw_optical_flow_input_level_3"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_4,              L"rw_optical_flow_input_level_4"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_5,              L"rw_optical_flow_input_level_5"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_6,              L"rw_optical_flow_input_level_6"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW,                            L"rw_optical_flow"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_NEXT_LEVEL,                 L"rw_optical_flow_next_level"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM,              L"rw_optical_flow_scd_histogram"}, // scene change detection histogram
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM,     L"rw_optical_flow_scd_previous_histogram"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP,                   L"rw_optical_flow_scd_temp"},
    {FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT,                 L"rw_optical_flow_scd_output"},
};

static const Binding cbBindingNames[] =
{
    {FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER,       L"cbOF"},
    {FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER_SPD,   L"cbOF_SPD"}
};

// Broad structure of the root signature.
typedef enum OpticalFlowRootSignatureLayout {

    OPTICALFLOW_ROOT_SIGNATURE_LAYOUT_UAVS,
    OPTICALFLOW_ROOT_SIGNATURE_LAYOUT_SRVS,
    OPTICALFLOW_ROOT_SIGNATURE_LAYOUT_CONSTANTS,
    OPTICALFLOW_ROOT_SIGNATURE_LAYOUT_CONSTANTS_REGISTER_1,
    OPTICALFLOW_ROOT_SIGNATURE_LAYOUT_PARAMETER_COUNT
} OpticalFlowRootSignatureLayout;

typedef struct OpticalFlowSpdConstants
{
    uint32_t                    mips;
    uint32_t                    numworkGroups;
    uint32_t                    workGroupOffset[2];

    uint32_t                    numworkGroupsOpticalFlowInputPyramid;
    uint32_t pad0_;
    uint32_t pad1_;
    uint32_t pad2_;

} OpticalFlowSpdConstants;

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(srvBindingNames); ++mapIndex)
        {
            if (0 == wcscmp(srvBindingNames[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        FFX_ASSERT(mapIndex < _countof(srvBindingNames));
        if (mapIndex == _countof(srvBindingNames))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvBindingNames[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(uavBindingNames); ++mapIndex)
        {
            if (0 == wcscmp(uavBindingNames[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        FFX_ASSERT(mapIndex < _countof(uavBindingNames));
        if (mapIndex == _countof(uavBindingNames))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = uavBindingNames[mapIndex].index;
    }

    for (uint32_t cbIndex = 0; cbIndex < inoutPipeline->constCount; ++cbIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(cbBindingNames); ++mapIndex)
        {
            if (0 == wcscmp(cbBindingNames[mapIndex].name, inoutPipeline->constantBufferBindings[cbIndex].name))
                break;
        }
        FFX_ASSERT(mapIndex < _countof(cbBindingNames));
        if (mapIndex == _countof(cbBindingNames))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->constantBufferBindings[cbIndex].resourceIdentifier = cbBindingNames[mapIndex].index;
    }

    return FFX_OK;
}

static uint32_t getPipelinePermutationFlags(uint32_t, FfxPass, bool fp16, bool force64, bool)
{
    uint32_t flags = 0;
    flags |= (force64) ? OPTICALFLOW_SHADER_PERMUTATION_FORCE_WAVE64 : 0;
    flags |= (fp16) ? OPTICALFLOW_SHADER_PERMUTATION_ALLOW_FP16 : 0;
    return flags;
}

static FfxErrorCode createPipelineStates(FfxOpticalflowContext_Private* context)
{
    FFX_ASSERT(context);

    constexpr size_t samplerCount = 2;
    FfxSamplerDescription samplerDescs[samplerCount] = {
        {FFX_FILTER_TYPE_MINMAGMIP_POINT, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE},
        {FFX_FILTER_TYPE_MINMAGMIP_LINEAR, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE} };

    const size_t rootConstantCount = 2;
    FfxRootConstantDescription rootConstantDescs[2] = { {sizeof(OpticalflowConstants) / sizeof(uint32_t),    FFX_BIND_COMPUTE_SHADER_STAGE},
                                                        {sizeof(OpticalFlowSpdConstants) / sizeof(uint32_t), FFX_BIND_COMPUTE_SHADER_STAGE} };
    FfxPipelineDescription pipelineDescription  = {};
    pipelineDescription.stage                   = FFX_BIND_COMPUTE_SHADER_STAGE;
    pipelineDescription.contextFlags            = context->contextDescription.flags;
    pipelineDescription.samplerCount            = samplerCount;
    pipelineDescription.samplers                = samplerDescs;
    pipelineDescription.rootConstantBufferCount = rootConstantCount;
    pipelineDescription.rootConstants           = rootConstantDescs;

    FfxDeviceCapabilities capabilities;
    context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities);

    bool haveShaderModel66 = capabilities.maximumSupportedShaderModel >= FFX_SHADER_MODEL_6_6;
    bool supportedFP16     = capabilities.fp16Supported;
    bool canForceWave64    = false;
    bool useLut            = false;

    const uint32_t waveLaneCountMin = capabilities.waveLaneCountMin;
    const uint32_t waveLaneCountMax = capabilities.waveLaneCountMax;
    if (waveLaneCountMin == 32 && waveLaneCountMax == 64)
    {
        useLut         = true;
        canForceWave64 = haveShaderModel66;
    }
    else
        canForceWave64 = false;

    uint32_t contextFlags = context->contextDescription.flags;

    auto CreateComputePipeline = [&](FfxPass pass, const wchar_t* name, FfxPipelineState* pipeline) -> FfxErrorCode {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, pipeline, context->effectContextId);
        wcscpy_s(pipelineDescription.name, name);
        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(
            &context->contextDescription.backendInterface,
            FFX_EFFECT_OPTICALFLOW,
            pass,
            getPipelinePermutationFlags(contextFlags, pass, supportedFP16, canForceWave64, useLut),
            &pipelineDescription,
            context->effectContextId,
            pipeline));

        patchResourceBindings(pipeline);
        return FFX_OK;
    };

    CreateComputePipeline(FFX_OPTICALFLOW_PASS_GENERATE_OPTICAL_FLOW_INPUT_PYRAMID, L"Opticalflow_InputPyramid", & context->pipelineGenerateOpticalFlowInputPyramid);
    pipelineDescription.rootConstantBufferCount = 1;
    CreateComputePipeline(FFX_OPTICALFLOW_PASS_PREPARE_LUMA, L"Opticalflow_Luma", &context->pipelinePrepareLuma);
    CreateComputePipeline(FFX_OPTICALFLOW_PASS_GENERATE_SCD_HISTOGRAM, L"Opticalflow_SCD_Histogram", &context->pipelineGenerateSCDHistogram);
    CreateComputePipeline(FFX_OPTICALFLOW_PASS_COMPUTE_SCD_DIVERGENCE, L"Opticalflow_SCD_Divergence", &context->pipelineComputeSCDDivergence);
    CreateComputePipeline(FFX_OPTICALFLOW_PASS_COMPUTE_OPTICAL_FLOW_ADVANCED_V5, L"Opticalflow_Search", &context->pipelineComputeOpticalFlowAdvancedV5);
    CreateComputePipeline(FFX_OPTICALFLOW_PASS_FILTER_OPTICAL_FLOW_V5, L"Opticalflow_Filter", &context->pipelineFilterOpticalFlowV5);
    CreateComputePipeline(FFX_OPTICALFLOW_PASS_SCALE_OPTICAL_FLOW_ADVANCED_V5, L"Opticalflow_Upscale", &context->pipelineScaleOpticalFlowAdvancedV5);

    return FFX_OK;
}

constexpr uint32_t OpticalFlowMaxPyramidLevels = 7;
constexpr uint32_t HistogramBins = 256;
constexpr uint32_t HistogramsPerDim = 3;
constexpr uint32_t HistogramShifts = 3;

static FfxDimensions2D GetOpticalFlowTextureSize(const FfxDimensions2D& displaySize, const uint32_t opticalFlowBlockSize)
{
    uint32_t width = (displaySize.width + opticalFlowBlockSize - 1) / opticalFlowBlockSize;
    uint32_t height = (displaySize.height + opticalFlowBlockSize - 1) / opticalFlowBlockSize;
    return { width, height };
}

static FfxDimensions2D GetOpticalFlowHistogramSize(int level)
{
    const uint32_t searchRadius = 8;
    uint32_t maxVelocity = searchRadius * (1 << (OpticalFlowMaxPyramidLevels - 1 - level));
    uint32_t binsPerDimension = 2 * maxVelocity + 1;
    return { binsPerDimension, binsPerDimension };
}

static FfxDimensions2D GetGlobalMotionSearchDispatchSize(int level)
{
    const uint32_t threadGroupSizeX = 16;
    const uint32_t threadGroupSizeY = 16;
    const FfxDimensions2D opticalFlowHistogramSize = GetOpticalFlowHistogramSize(level);
    const uint32_t additionalElementsDueToShiftsX = opticalFlowHistogramSize.width / threadGroupSizeX;
    const uint32_t additionalElementsDueToShiftsY = opticalFlowHistogramSize.height / threadGroupSizeY;
    const uint32_t dispatchX = (opticalFlowHistogramSize.width + additionalElementsDueToShiftsX + threadGroupSizeX - 1) / threadGroupSizeX;
    const uint32_t dispatchY = (opticalFlowHistogramSize.height + additionalElementsDueToShiftsY + threadGroupSizeY - 1) / threadGroupSizeY;
    return { dispatchX, dispatchY };
}

static uint32_t GetSCDHistogramTextureWidth()
{
    return HistogramBins * (HistogramsPerDim * HistogramsPerDim);
}

static FfxErrorCode opticalflowCreate(FfxOpticalflowContext_Private* context, const FfxOpticalflowContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);
    FfxErrorCode errorCode = FFX_OK;

    memset(context, 0, sizeof(FfxOpticalflowContext_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxOpticalflowContextDescription));

    // Check version info - make sure we are linked with the right backend version
    FfxVersionNumber version = context->contextDescription.backendInterface.fpGetSDKVersion(&context->contextDescription.backendInterface);
    FFX_RETURN_ON_ERROR(version == FFX_SDK_MAKE_VERSION(1, 1, 4), FFX_ERROR_INVALID_VERSION);

    errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(&context->contextDescription.backendInterface, FFX_EFFECT_OPTICALFLOW, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    errorCode = context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &context->deviceCapabilities);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    context->firstExecution = true;
    context->resourceFrameIndex = 0;

    context->constants.inputLumaResolution[0] = context->contextDescription.resolution.width;
    context->constants.inputLumaResolution[1] = context->contextDescription.resolution.height;

    FfxDimensions2D opticalFlowInputTextureSize = context->contextDescription.resolution;

    const FfxResourceType texture1dResourceType = (context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_TEXTURE1D_USAGE) ? FFX_RESOURCE_TYPE_TEXTURE1D : FFX_RESOURCE_TYPE_TEXTURE2D;

    uint32_t minBlockSize = 8;
    const FfxDimensions2D opticalFlowTextureSize = GetOpticalFlowTextureSize(contextDescription->resolution, minBlockSize);

    const FfxDimensions2D opticalFlowLevel1TextureSize = { FFX_ALIGN_UP(opticalFlowTextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowTextureSize.height, 2) / 2 };
    const FfxDimensions2D opticalFlowLevel2TextureSize = { FFX_ALIGN_UP(opticalFlowLevel1TextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowLevel1TextureSize.height, 2) / 2 };
    const FfxDimensions2D opticalFlowLevel3TextureSize = { FFX_ALIGN_UP(opticalFlowLevel2TextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowLevel2TextureSize.height, 2) / 2 };
    const FfxDimensions2D opticalFlowLevel4TextureSize = { FFX_ALIGN_UP(opticalFlowLevel3TextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowLevel3TextureSize.height, 2) / 2 };
    const FfxDimensions2D opticalFlowLevel5TextureSize = { FFX_ALIGN_UP(opticalFlowLevel4TextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowLevel4TextureSize.height, 2) / 2 };
    const FfxDimensions2D opticalFlowLevel6TextureSize = { FFX_ALIGN_UP(opticalFlowLevel5TextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowLevel5TextureSize.height, 2) / 2 };
    const FfxDimensions2D opticalFlowLevel7TextureSize = { FFX_ALIGN_UP(opticalFlowLevel6TextureSize.width, 2) / 2, FFX_ALIGN_UP(opticalFlowLevel6TextureSize.height, 2) / 2 };

    const FfxDimensions2D opticalFlowHistogramTextureSize = GetOpticalFlowHistogramSize(0);

    const FfxDimensions2D globalMotionSearchMaxDispatchSize = GetGlobalMotionSearchDispatchSize(0);
    const uint32_t globalMotionSearchTextureWidth = 4 + (globalMotionSearchMaxDispatchSize.width * globalMotionSearchMaxDispatchSize.height);

    const FfxInternalResourceDescription internalSurfaceDesc[] =    {
        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1, L"OPTICALFLOW_OpticalFlowInput1", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width, opticalFlowInputTextureSize.height, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_1, L"OPTICALFLOW_OpticalFlowInput1Level1", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 1, opticalFlowInputTextureSize.height >> 1, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_2, L"OPTICALFLOW_OpticalFlowInput1Level2", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 2, opticalFlowInputTextureSize.height >> 2, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_3, L"OPTICALFLOW_OpticalFlowInput1Level3", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 3, opticalFlowInputTextureSize.height >> 3, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_4, L"OPTICALFLOW_OpticalFlowInput1Level4", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 4, opticalFlowInputTextureSize.height >> 4, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_5, L"OPTICALFLOW_OpticalFlowInput1Level5", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 5, opticalFlowInputTextureSize.height >> 5, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_6, L"OPTICALFLOW_OpticalFlowInput1Level6", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 6, opticalFlowInputTextureSize.height >> 6, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2, L"OPTICALFLOW_OpticalFlowInput2", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width, opticalFlowInputTextureSize.height, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_1, L"OPTICALFLOW_OpticalFlowInput2Level1", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 1, opticalFlowInputTextureSize.height >> 1, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_2, L"OPTICALFLOW_OpticalFlowInput2Level2", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 2, opticalFlowInputTextureSize.height >> 2, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_3, L"OPTICALFLOW_OpticalFlowInput2Level3", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 3, opticalFlowInputTextureSize.height >> 3, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_4, L"OPTICALFLOW_OpticalFlowInput2Level4", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 4, opticalFlowInputTextureSize.height >> 4, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_5, L"OPTICALFLOW_OpticalFlowInput2Level5", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 5, opticalFlowInputTextureSize.height >> 5, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_6, L"OPTICALFLOW_OpticalFlowInput2Level6", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UINT, opticalFlowInputTextureSize.width >> 6, opticalFlowInputTextureSize.height >> 6, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1, L"OPTICALFLOW_OpticalFlow1", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowTextureSize.width, opticalFlowTextureSize.height, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1_LEVEL_1, L"OPTICALFLOW_OpticalFlow1Level1", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel1TextureSize.width, opticalFlowLevel1TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1_LEVEL_2, L"OPTICALFLOW_OpticalFlow1Level2", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel2TextureSize.width, opticalFlowLevel2TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1_LEVEL_3, L"OPTICALFLOW_OpticalFlow1Level3", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel3TextureSize.width, opticalFlowLevel3TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1_LEVEL_4, L"OPTICALFLOW_OpticalFlow1Level4", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel4TextureSize.width, opticalFlowLevel4TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1_LEVEL_5, L"OPTICALFLOW_OpticalFlow1Level5", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel5TextureSize.width, opticalFlowLevel5TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1_LEVEL_6, L"OPTICALFLOW_OpticalFlow1Level6", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel6TextureSize.width, opticalFlowLevel6TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2, L"OPTICALFLOW_OpticalFlow2", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowTextureSize.width, opticalFlowTextureSize.height, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2_LEVEL_1, L"OPTICALFLOW_OpticalFlow2Level1", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel1TextureSize.width, opticalFlowLevel1TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2_LEVEL_2, L"OPTICALFLOW_OpticalFlow2Level2", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel2TextureSize.width, opticalFlowLevel2TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2_LEVEL_3, L"OPTICALFLOW_OpticalFlow2Level3", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel3TextureSize.width, opticalFlowLevel3TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2_LEVEL_4, L"OPTICALFLOW_OpticalFlow2Level4", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel4TextureSize.width, opticalFlowLevel4TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2_LEVEL_5, L"OPTICALFLOW_OpticalFlow2Level5", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel5TextureSize.width, opticalFlowLevel5TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2_LEVEL_6, L"OPTICALFLOW_OpticalFlow2Level6", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowLevel6TextureSize.width, opticalFlowLevel6TextureSize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM, L"OPTICALFLOW_OpticalFlowSCDHistogram", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, GetSCDHistogramTextureWidth(), 1, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM, L"OPTICALFLOW_OpticalFlowSCDPreviousHistogram", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_FLOAT, GetSCDHistogramTextureWidth(), 1, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },

        {   FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP, L"OPTICALFLOW_OpticalFlowSCDTemp", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, 3, 1, 1,  FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} },
    };

    memset(context->resources, 0, sizeof(context->resources));

    for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(internalSurfaceDesc); ++currentSurfaceIndex) {

        const FfxInternalResourceDescription* currentSurfaceDescription = &internalSurfaceDesc[currentSurfaceIndex];
        const FfxResourceType resourceType = currentSurfaceDescription->height > 1 ? FFX_RESOURCE_TYPE_TEXTURE2D : texture1dResourceType;
        const FfxResourceDescription resourceDescription = {
            resourceType, currentSurfaceDescription->format,
            currentSurfaceDescription->width, currentSurfaceDescription->height, 1,
            currentSurfaceDescription->mipCount, FFX_RESOURCE_FLAGS_NONE, currentSurfaceDescription->usage };
        const FfxResourceStates initialState = FFX_RESOURCE_STATE_UNORDERED_ACCESS;
        const FfxCreateResourceDescription createResourceDescription = {
            FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, currentSurfaceDescription->name, currentSurfaceDescription->id, currentSurfaceDescription->initData };

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateResource(
            &context->contextDescription.backendInterface,
            &createResourceDescription,
            context->effectContextId,
            &context->resources[currentSurfaceDescription->id]));
    }

    memset(context->srvBindings, 0, sizeof(context->srvBindings));
    memset(context->uavBindings, 0, sizeof(context->uavBindings));

    {
        context->refreshPipelineStates = false;
        errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }

    return FFX_OK;
}

static FfxErrorCode opticalflowRelease(FfxOpticalflowContext_Private* context)
{
    FFX_ASSERT(context);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelinePrepareLuma, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineGenerateOpticalFlowInputPyramid, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineGenerateSCDHistogram, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineComputeSCDDivergence, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineComputeOpticalFlowAdvancedV5, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFilterOpticalFlowV5, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineScaleOpticalFlowAdvancedV5, context->effectContextId);

    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_OF_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex) {

        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->resources[currentResourceIndex], context->effectContextId);
    }

    context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId);

    return FFX_OK;
}

static void scheduleDispatch(FfxOpticalflowContext_Private* context, const FfxPipelineState* pipeline, const wchar_t* pipelineName, uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ = 1)
{
    FfxComputeJobDescription jobDescriptor = {};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex) {

        const uint32_t bindingIdentifier = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = context->srvBindings[bindingIdentifier];
        jobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.srvTextures[currentShaderResourceViewIndex].name, pipeline->srvTextureBindings[currentShaderResourceViewIndex].name);
#endif

        FFX_ASSERT(bindingIdentifier != FFX_OF_BINDING_IDENTIFIER_NULL);
        FFX_ASSERT(bindingIdentifier < FFX_OF_BINDING_IDENTIFIER_COUNT);
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex) {

        const uint32_t bindingIdentifier = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = context->uavBindings[bindingIdentifier];
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip = 0;
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name, pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name);
#endif

        FFX_ASSERT(bindingIdentifier != FFX_OF_BINDING_IDENTIFIER_NULL);
        FFX_ASSERT(bindingIdentifier < FFX_OF_BINDING_IDENTIFIER_COUNT);
    }

    jobDescriptor.dimensions[0] = dispatchX;
    jobDescriptor.dimensions[1] = dispatchY;
    jobDescriptor.dimensions[2] = dispatchZ;
    jobDescriptor.pipeline = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex) {
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.cbNames[currentRootConstantIndex], pipeline->constantBufferBindings[currentRootConstantIndex].name);
#endif
        jobDescriptor.cbs[currentRootConstantIndex] = context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    FfxGpuJobDescription dispatchJob = { FFX_GPU_JOB_COMPUTE };
    wcscpy_s(dispatchJob.jobLabel, pipelineName);
    dispatchJob.computeJobDescriptor = jobDescriptor;

    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode dispatch(FfxOpticalflowContext_Private* context, const FfxOpticalflowDispatchDescription* params)
{
    context->contextDescription.backendInterface.fpRegisterResource(
        &context->contextDescription.backendInterface,
        &params->opticalFlowVector,
        context->effectContextId,
        &context->uavBindings[FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_VECTOR]);
    context->contextDescription.backendInterface.fpRegisterResource(
        &context->contextDescription.backendInterface,
        &params->opticalFlowSCD,
        context->effectContextId,
        &context->uavBindings[FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_SCD_OUTPUT]);

    context->contextDescription.backendInterface.fpRegisterResource(
        &context->contextDescription.backendInterface,
        &params->color,
        context->effectContextId,
        &context->srvBindings[FFX_OF_BINDING_IDENTIFIER_INPUT_COLOR]);

    FfxCommandList commandList = params->commandList;
    int advancedAlgorithmIterations = 7;
    uint32_t opticalFlowBlockSize = 8;

    if (context->refreshPipelineStates) {

        context->refreshPipelineStates = false;

        const FfxErrorCode errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }

    const FfxResourceDescription resourceDescInputColor = context->contextDescription.backendInterface.fpGetResourceDescription(
        &context->contextDescription.backendInterface,
        context->srvBindings[FFX_OF_BINDING_IDENTIFIER_INPUT_COLOR]);
    FFX_ASSERT(resourceDescInputColor.type == FFX_RESOURCE_TYPE_TEXTURE2D);

    context->constants.backbufferTransferFunction = params->backbufferTransferFunction;
    context->constants.minMaxLuminance[0] = params->minMaxLuminance.x;
    context->constants.minMaxLuminance[1] = params->minMaxLuminance.y;

    const bool resetAccumulation = params->reset || context->firstExecution;
    context->firstExecution = false;

    if (resetAccumulation) {
        context->constants.frameIndex = 0;
    }
    else {
        context->constants.frameIndex++;
    }

    if (resetAccumulation)
    {
        const float clearValuesToZeroFloat[]{ 0.f, 0.f, 0.f, 0.f };
        FfxGpuJobDescription clearJob = { FFX_GPU_JOB_CLEAR_FLOAT };
        memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));

        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow SCD Temp");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        clearJob.clearJobDescriptor.target = context->uavBindings[FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_SCD_OUTPUT];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow SCD Histogram");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow SCD Previous histogram");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1 Level 1");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_1];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1 Level 2");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_2];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1 Level 3");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_3];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1 Level 4");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_4];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1 Level 5");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_5];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 1 Level 6");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_6];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2 Level 1");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_1];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2 Level 2");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_2];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2 Level 3");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_3];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2 Level 4");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_4];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2 Level 5");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_5];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
        wcscpy_s(clearJob.jobLabel, L"Clear Optical Flow Input 2 Level 6");
        clearJob.clearJobDescriptor.target = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_6];
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob);
    }

    uint32_t resolutionMultiplier = 1;

    FfxUInt32x2 threadGroupSizeOpticalFlowInputPyramid;
    FfxUInt32x2 workGroupOffset;
    FfxUInt32x2 numWorkGroupsAndMips;
    FfxUInt32x4 rectInfo = { 0, 0,
        context->contextDescription.resolution.width * resolutionMultiplier,
        context->contextDescription.resolution.height * resolutionMultiplier };
    ffxSpdSetup(threadGroupSizeOpticalFlowInputPyramid, workGroupOffset, numWorkGroupsAndMips, rectInfo, 4);

    OpticalFlowSpdConstants luminancePyramidConstants;
    luminancePyramidConstants.numworkGroups = numWorkGroupsAndMips[0];
    luminancePyramidConstants.mips = numWorkGroupsAndMips[1];
    luminancePyramidConstants.workGroupOffset[0] = workGroupOffset[0];
    luminancePyramidConstants.workGroupOffset[1] = workGroupOffset[1];
    luminancePyramidConstants.numworkGroupsOpticalFlowInputPyramid = numWorkGroupsAndMips[0];

    context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface, &context->constants,        sizeof(context->constants),        &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]);
    context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface, &luminancePyramidConstants, sizeof(luminancePyramidConstants), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER_SPD]);

    {
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM] = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM] = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP] = context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT] = context->uavBindings[FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_SCD_OUTPUT];

        const bool isOddFrame = !!(context->resourceFrameIndex & 1);

        uint32_t opticalFlowInputResourceIndex = isOddFrame ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1;
        uint32_t opticalFlowPreviousInputResourceIndex = isOddFrame ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2;

        uint32_t opticalFlowResourceIndex = isOddFrame ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1;
        uint32_t opticalFlowPreviousResourceIndex = isOddFrame ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2;

        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT] = context->resources[opticalFlowInputResourceIndex];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_1] = context->resources[opticalFlowInputResourceIndex + 1];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_2] = context->resources[opticalFlowInputResourceIndex + 2];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_3] = context->resources[opticalFlowInputResourceIndex + 3];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_4] = context->resources[opticalFlowInputResourceIndex + 4];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_5] = context->resources[opticalFlowInputResourceIndex + 5];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_6] = context->resources[opticalFlowInputResourceIndex + 6];

        context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT] = context->resources[opticalFlowInputResourceIndex];
        context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS_INPUT] = context->resources[opticalFlowPreviousInputResourceIndex];

        context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW] = context->resources[opticalFlowResourceIndex];
        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW] = context->resources[opticalFlowResourceIndex];
        context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS] = context->resources[opticalFlowPreviousResourceIndex];

        {
            int32_t threadGroupSizeX = 16;
            int32_t threadGroupSizeY = 16;
            uint32_t threadPixelsX = 2;
            uint32_t threadPixelsY = 2;
            int32_t dispatchX = ((context->contextDescription.resolution.width + (threadPixelsX - 1)) / threadPixelsX + (threadGroupSizeX - 1)) / threadGroupSizeX;
            int32_t dispatchY = ((context->contextDescription.resolution.height + (threadPixelsY - 1)) / threadPixelsY + (threadGroupSizeY - 1)) / threadGroupSizeY;
            scheduleDispatch(context, &context->pipelinePrepareLuma, L"OF PrepareLuma", dispatchX, dispatchY);
        }

        {
            {
                scheduleDispatch(context,
                                 &context->pipelineGenerateOpticalFlowInputPyramid,
                                 L"OF GenerateOpticalFlowInputPyramid",
                                 threadGroupSizeOpticalFlowInputPyramid[0],
                                 threadGroupSizeOpticalFlowInputPyramid[1]
                );
            }

            {
                {
                    const uint32_t threadGroupSizeX = 32;
                    const uint32_t threadGroupSizeY = 8;
                    const uint32_t strataWidth = (context->contextDescription.resolution.width / 4) / HistogramsPerDim;
                    const uint32_t strataHeight = context->contextDescription.resolution.height / HistogramsPerDim;
                    const uint32_t dispatchX = (strataWidth + threadGroupSizeX - 1) / threadGroupSizeX;
                    const uint32_t dispatchY = 16;
                    const uint32_t dispatchZ = HistogramsPerDim * HistogramsPerDim;
                    scheduleDispatch(context, &context->pipelineGenerateSCDHistogram, L"OF GenerateSCDHistogram", dispatchX, dispatchY, dispatchZ);
                }
                {
                    const uint32_t dispatchX = HistogramsPerDim * HistogramsPerDim;
                    const uint32_t dispatchY = HistogramShifts;
                    scheduleDispatch(context, &context->pipelineComputeSCDDivergence, L"OF ComputeSCDDivergence", dispatchX, dispatchY);
                }
            }

            FfxDimensions2D opticalFlowTextureSizes[OpticalFlowMaxPyramidLevels];
            const int pyramidMaxIterations = advancedAlgorithmIterations;
            FFX_ASSERT(pyramidMaxIterations <= OpticalFlowMaxPyramidLevels);

            opticalFlowTextureSizes[0] = GetOpticalFlowTextureSize(context->contextDescription.resolution, opticalFlowBlockSize);
            for (int i = 1; i < pyramidMaxIterations; i++)
            {
                opticalFlowTextureSizes[i] = {
                    (opticalFlowTextureSizes[i - 1].width + 1) / 2,
                    (opticalFlowTextureSizes[i - 1].height + 1) / 2
                };
            }

            for (int level = pyramidMaxIterations - 1; level >= 0; level--)
            {
                bool isOddLevel = !!(level & 1);

                uint32_t opticalFlowInputResourceIndexA = isOddFrame ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1;
                uint32_t opticalFlowInputResourceIndexB = isOddFrame ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2;
                uint32_t opticalFlowResourceIndexA = (isOddFrame != isOddLevel) ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1;
                uint32_t opticalFlowResourceIndexB = (isOddFrame != isOddLevel) ? FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_1 : FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_2;
                context->constants.opticalFlowPyramidLevel = level;
                context->constants.opticalFlowPyramidLevelCount = pyramidMaxIterations;

                context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface, &context->constants, sizeof(context->constants), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]);

                context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT] = context->resources[opticalFlowInputResourceIndexA + level];
                context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS_INPUT] = context->resources[opticalFlowInputResourceIndexB + level];
                context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW] = context->resources[opticalFlowResourceIndexA + level];

                context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS] = context->resources[opticalFlowResourceIndexB + level];

                {
                    const FfxUInt32 inputLumaWidth = ffxMax(context->contextDescription.resolution.width >> level, 1);
                    const FfxUInt32 inputLumaHeight = ffxMax(context->contextDescription.resolution.height >> level, 1);
                    std::wstring pipelineName = L"OF " + std::to_wstring(level) + L" Search";

                    {
                        uint32_t threadPixels = 4;
                        FFX_ASSERT(opticalFlowBlockSize >= threadPixels);
                        uint32_t threadGroupSizeY = 16;
                        uint32_t threadGroupSize = 64;
                        uint32_t dispatchX = ((inputLumaWidth + threadPixels - 1) / threadPixels * threadGroupSizeY + (threadGroupSize - 1)) / threadGroupSize;
                        uint32_t dispatchY = (inputLumaHeight + (threadGroupSizeY - 1)) / threadGroupSizeY;
                        scheduleDispatch(context, &context->pipelineComputeOpticalFlowAdvancedV5, pipelineName.c_str(), dispatchX, dispatchY);
                    }
                }

                {
                    context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS] = context->resources[opticalFlowResourceIndexA + level];
                    context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW] = context->resources[opticalFlowResourceIndexB + level];
                }

                {
                    if (level == 0)
                    {
                        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW] = context->uavBindings[FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_VECTOR];
                    }

                    const uint32_t levelWidth = opticalFlowTextureSizes[level].width;
                    const uint32_t levelHeight = opticalFlowTextureSizes[level].height;

                    const uint32_t threadGroupSizeX = 16;
                    const uint32_t threadGroupSizeY = 4;
                    const uint32_t dispatchX = (levelWidth + threadGroupSizeX - 1) / threadGroupSizeX;
                    const uint32_t dispatchY = (levelHeight + threadGroupSizeY - 1) / threadGroupSizeY;
                    std::wstring pipelineName = L"OF " + std::to_wstring(level) + L" Filter";

                    {
                        scheduleDispatch(context, &context->pipelineFilterOpticalFlowV5, pipelineName.c_str(), dispatchX, dispatchY);
                    }
                }

                if (level > 0)
                {
                    context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_1 + level - 1] = context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW];
                }

                if (level > 0)
                {
                    {
                        context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW] = context->resources[opticalFlowResourceIndexB + level];
                        context->uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_NEXT_LEVEL] = level > 0 ? context->resources[opticalFlowResourceIndexB + level - 1] : FfxResourceInternal{ FFX_OF_RESOURCE_IDENTIFIER_NULL };
                    }

                    FFX_ASSERT(opticalFlowBlockSize >= 2);
                    const uint32_t nextLevelWidth = opticalFlowTextureSizes[level - 1].width;
                    const uint32_t nextLevelHeight = opticalFlowTextureSizes[level - 1].height;

                    const uint32_t threadGroupSizeX = opticalFlowBlockSize / 2;
                    const uint32_t threadGroupSizeY = opticalFlowBlockSize / 2;
                    const uint32_t threadGroupSizeZ = 4;
                    const uint32_t dispatchX = (nextLevelWidth + threadGroupSizeX - 1) / threadGroupSizeX;
                    const uint32_t dispatchY = (nextLevelHeight + threadGroupSizeY - 1) / threadGroupSizeY;
                    const uint32_t dispatchZ = 1;
                    std::wstring pipelineName = L"OF " + std::to_wstring(level) + L" Scale";

                    {
                        const uint32_t dispatchX = (nextLevelWidth + 3) / 4;
                        const uint32_t dispatchY = (nextLevelHeight + 3) / 4;
                        scheduleDispatch(context, &context->pipelineScaleOpticalFlowAdvancedV5, pipelineName.c_str(), dispatchX, dispatchY, dispatchZ);
                    }

                    {
                        FfxGpuJobDescription barrierJob = {FFX_GPU_JOB_BARRIER};
                        barrierJob.barrierDescriptor = { context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
                        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
                    }
                }

                {
                    FfxGpuJobDescription barrierJob = {FFX_GPU_JOB_BARRIER};
                    barrierJob.barrierDescriptor = { context->srvBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
                    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
                }
            }
        }
    }

    {
        FfxGpuJobDescription barrierJob = {FFX_GPU_JOB_BARRIER};

        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1 Level 1");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_1], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1 Level 2");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_2], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1 Level 3");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_3], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1 Level 4");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_4], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1 Level 5");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_5], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 1 Level 6");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_1_LEVEL_6], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2 Level 1");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_1], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2 Level 2");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_2], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2 Level 3");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_3], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2 Level 4");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_4], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2 Level 5");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_5], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
        wcscpy_s(barrierJob.jobLabel, L"Transition Optical Flow Input 2 Level 6");
        barrierJob.barrierDescriptor = { context->resources[FFX_OF_RESOURCE_IDENTIFIER_OPTICAL_FLOW_INPUT_2_LEVEL_6], FFX_BARRIER_TYPE_TRANSITION, FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_STATE_UNORDERED_ACCESS, 0};
        context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &barrierJob);
    }

    context->resourceFrameIndex = (context->resourceFrameIndex + 1) % FFX_OPTICALFLOW_MAX_QUEUED_FRAMES;

    FFX_VALIDATE(context->contextDescription.backendInterface.fpExecuteGpuJobs(&context->contextDescription.backendInterface, commandList, context->effectContextId));

    context->contextDescription.backendInterface.fpUnregisterResources(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    return FFX_OK;
}

FfxErrorCode ffxOpticalflowContextCreate(FfxOpticalflowContext* context, FfxOpticalflowContextDescription* contextDescription)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(contextDescription, FFX_ERROR_INVALID_POINTER);

    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetSDKVersion, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

    if (contextDescription->backendInterface.scratchBuffer) {

        FFX_RETURN_ON_ERROR(contextDescription->backendInterface.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    FFX_STATIC_ASSERT(sizeof(FfxOpticalflowContext) >= sizeof(FfxOpticalflowContext_Private));

    FfxOpticalflowContext_Private* contextPrivate = (FfxOpticalflowContext_Private*)(context);
    FfxErrorCode errorCode = opticalflowCreate(contextPrivate, contextDescription);

    return errorCode;
}

FFX_API FfxErrorCode ffxOpticalflowContextGetGpuMemoryUsage(FfxOpticalflowContext* context, FfxEffectMemoryUsage* vramUsage)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(vramUsage, FFX_ERROR_INVALID_POINTER);
    FfxOpticalflowContext_Private* contextPrivate = (FfxOpticalflowContext_Private*)(context);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);

    FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->effectContextId, vramUsage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

FfxErrorCode ffxOpticalflowContextDestroy(FfxOpticalflowContext* context)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);

    FfxOpticalflowContext_Private* contextPrivate = (FfxOpticalflowContext_Private*)(context);
    const FfxErrorCode errorCode = opticalflowRelease(contextPrivate);

    return errorCode;
}

FFX_API bool ffxOpticalflowResourceIsNull(FfxResource resource)
{
    return resource.resource == NULL;
}

FFX_API FfxErrorCode ffxOpticalflowGetSharedResourceDescriptions(FfxOpticalflowContext* context, FfxOpticalflowSharedResourceDescriptions* SharedResources)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(SharedResources, FFX_ERROR_INVALID_POINTER);

    FfxOpticalflowContext_Private* contextPrivate = (FfxOpticalflowContext_Private*)(context);
    const FfxDimensions2D opticalFlowTextureSize = GetOpticalFlowTextureSize(contextPrivate->contextDescription.resolution, 8);
    const FfxDimensions2D globalMotionSearchMaxDispatchSize = GetGlobalMotionSearchDispatchSize(0);
    const uint32_t globalMotionSearchTextureWidth = 4 /* predefined slots */ + (globalMotionSearchMaxDispatchSize.width * globalMotionSearchMaxDispatchSize.height);

    SharedResources->opticalFlowVector = {
        FFX_HEAP_TYPE_DEFAULT,
        { FFX_RESOURCE_TYPE_TEXTURE2D, FFX_SURFACE_FORMAT_R16G16_SINT, opticalFlowTextureSize.width, opticalFlowTextureSize.height, 1, 1, FFX_RESOURCE_FLAGS_NONE, FFX_RESOURCE_USAGE_UAV },
        FFX_RESOURCE_STATE_UNORDERED_ACCESS, L"OPTICALFLOW_Result", 0, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} };

    SharedResources->opticalFlowSCD = {
        FFX_HEAP_TYPE_DEFAULT,
        { FFX_RESOURCE_TYPE_TEXTURE2D, FFX_SURFACE_FORMAT_R32_UINT, 3, 1, 1, 1, FFX_RESOURCE_FLAGS_NONE, FFX_RESOURCE_USAGE_UAV },
        FFX_RESOURCE_STATE_UNORDERED_ACCESS, L"OPTICALFLOW_SCDOutput", 0, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} };

    return FFX_OK;
}

FfxErrorCode ffxOpticalflowContextDispatch(FfxOpticalflowContext* context, const FfxOpticalflowDispatchDescription* dispatchParams)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->commandList, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(!ffxOpticalflowResourceIsNull(dispatchParams->color), FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.type == FFX_RESOURCE_TYPE_TEXTURE2D, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(!ffxOpticalflowResourceIsNull(dispatchParams->opticalFlowVector), FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(!ffxOpticalflowResourceIsNull(dispatchParams->opticalFlowSCD), FFX_ERROR_INVALID_POINTER);

    FfxOpticalflowContext_Private* contextPrivate = (FfxOpticalflowContext_Private*)(context);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.width <= contextPrivate->contextDescription.resolution.width, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.height <= contextPrivate->contextDescription.resolution.height, FFX_ERROR_INVALID_ARGUMENT);

    const FfxErrorCode errorCode = dispatch(contextPrivate, dispatchParams);
    return errorCode;
}

FFX_API FfxVersionNumber ffxOpticalflowGetEffectVersion()
{
    return FFX_SDK_MAKE_VERSION(FFX_OPTICALFLOW_VERSION_MAJOR, FFX_OPTICALFLOW_VERSION_MINOR, FFX_OPTICALFLOW_VERSION_PATCH);
}
