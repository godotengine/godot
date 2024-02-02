#ifndef VULKAN_BETA_H_
#define VULKAN_BETA_H_ 1

/*
** Copyright 2015-2023 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

/*
** This header is generated from the Khronos Vulkan XML API Registry.
**
*/


#ifdef __cplusplus
extern "C" {
#endif



// VK_KHR_portability_subset is a preprocessor guard. Do not pass it to API calls.
#define VK_KHR_portability_subset 1
#define VK_KHR_PORTABILITY_SUBSET_SPEC_VERSION 1
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
typedef struct VkPhysicalDevicePortabilitySubsetFeaturesKHR {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           constantAlphaColorBlendFactors;
    VkBool32           events;
    VkBool32           imageViewFormatReinterpretation;
    VkBool32           imageViewFormatSwizzle;
    VkBool32           imageView2DOn3DImage;
    VkBool32           multisampleArrayImage;
    VkBool32           mutableComparisonSamplers;
    VkBool32           pointPolygons;
    VkBool32           samplerMipLodBias;
    VkBool32           separateStencilMaskRef;
    VkBool32           shaderSampleRateInterpolationFunctions;
    VkBool32           tessellationIsolines;
    VkBool32           tessellationPointMode;
    VkBool32           triangleFans;
    VkBool32           vertexAttributeAccessBeyondStride;
} VkPhysicalDevicePortabilitySubsetFeaturesKHR;

typedef struct VkPhysicalDevicePortabilitySubsetPropertiesKHR {
    VkStructureType    sType;
    void*              pNext;
    uint32_t           minVertexInputBindingStrideAlignment;
} VkPhysicalDevicePortabilitySubsetPropertiesKHR;



// VK_KHR_video_encode_queue is a preprocessor guard. Do not pass it to API calls.
#define VK_KHR_video_encode_queue 1
#define VK_KHR_VIDEO_ENCODE_QUEUE_SPEC_VERSION 10
#define VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME "VK_KHR_video_encode_queue"

typedef enum VkVideoEncodeTuningModeKHR {
    VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR = 0,
    VK_VIDEO_ENCODE_TUNING_MODE_HIGH_QUALITY_KHR = 1,
    VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR = 2,
    VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR = 3,
    VK_VIDEO_ENCODE_TUNING_MODE_LOSSLESS_KHR = 4,
    VK_VIDEO_ENCODE_TUNING_MODE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeTuningModeKHR;
typedef VkFlags VkVideoEncodeFlagsKHR;

typedef enum VkVideoEncodeCapabilityFlagBitsKHR {
    VK_VIDEO_ENCODE_CAPABILITY_PRECEDING_EXTERNALLY_ENCODED_BYTES_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_CAPABILITY_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION_BIT_KHR = 0x00000002,
    VK_VIDEO_ENCODE_CAPABILITY_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeCapabilityFlagBitsKHR;
typedef VkFlags VkVideoEncodeCapabilityFlagsKHR;

typedef enum VkVideoEncodeRateControlModeFlagBitsKHR {
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR = 0,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR = 0x00000002,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR = 0x00000004,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeRateControlModeFlagBitsKHR;
typedef VkFlags VkVideoEncodeRateControlModeFlagsKHR;

typedef enum VkVideoEncodeFeedbackFlagBitsKHR {
    VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR = 0x00000002,
    VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_HAS_OVERRIDES_BIT_KHR = 0x00000004,
    VK_VIDEO_ENCODE_FEEDBACK_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeFeedbackFlagBitsKHR;
typedef VkFlags VkVideoEncodeFeedbackFlagsKHR;

typedef enum VkVideoEncodeUsageFlagBitsKHR {
    VK_VIDEO_ENCODE_USAGE_DEFAULT_KHR = 0,
    VK_VIDEO_ENCODE_USAGE_TRANSCODING_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR = 0x00000002,
    VK_VIDEO_ENCODE_USAGE_RECORDING_BIT_KHR = 0x00000004,
    VK_VIDEO_ENCODE_USAGE_CONFERENCING_BIT_KHR = 0x00000008,
    VK_VIDEO_ENCODE_USAGE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeUsageFlagBitsKHR;
typedef VkFlags VkVideoEncodeUsageFlagsKHR;

typedef enum VkVideoEncodeContentFlagBitsKHR {
    VK_VIDEO_ENCODE_CONTENT_DEFAULT_KHR = 0,
    VK_VIDEO_ENCODE_CONTENT_CAMERA_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_CONTENT_DESKTOP_BIT_KHR = 0x00000002,
    VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR = 0x00000004,
    VK_VIDEO_ENCODE_CONTENT_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeContentFlagBitsKHR;
typedef VkFlags VkVideoEncodeContentFlagsKHR;
typedef VkFlags VkVideoEncodeRateControlFlagsKHR;
typedef struct VkVideoEncodeInfoKHR {
    VkStructureType                       sType;
    const void*                           pNext;
    VkVideoEncodeFlagsKHR                 flags;
    VkBuffer                              dstBuffer;
    VkDeviceSize                          dstBufferOffset;
    VkDeviceSize                          dstBufferRange;
    VkVideoPictureResourceInfoKHR         srcPictureResource;
    const VkVideoReferenceSlotInfoKHR*    pSetupReferenceSlot;
    uint32_t                              referenceSlotCount;
    const VkVideoReferenceSlotInfoKHR*    pReferenceSlots;
    uint32_t                              precedingExternallyEncodedBytes;
} VkVideoEncodeInfoKHR;

typedef struct VkVideoEncodeCapabilitiesKHR {
    VkStructureType                         sType;
    void*                                   pNext;
    VkVideoEncodeCapabilityFlagsKHR         flags;
    VkVideoEncodeRateControlModeFlagsKHR    rateControlModes;
    uint32_t                                maxRateControlLayers;
    uint64_t                                maxBitrate;
    uint32_t                                maxQualityLevels;
    VkExtent2D                              encodeInputPictureGranularity;
    VkVideoEncodeFeedbackFlagsKHR           supportedEncodeFeedbackFlags;
} VkVideoEncodeCapabilitiesKHR;

typedef struct VkQueryPoolVideoEncodeFeedbackCreateInfoKHR {
    VkStructureType                  sType;
    const void*                      pNext;
    VkVideoEncodeFeedbackFlagsKHR    encodeFeedbackFlags;
} VkQueryPoolVideoEncodeFeedbackCreateInfoKHR;

typedef struct VkVideoEncodeUsageInfoKHR {
    VkStructureType                 sType;
    const void*                     pNext;
    VkVideoEncodeUsageFlagsKHR      videoUsageHints;
    VkVideoEncodeContentFlagsKHR    videoContentHints;
    VkVideoEncodeTuningModeKHR      tuningMode;
} VkVideoEncodeUsageInfoKHR;

typedef struct VkVideoEncodeRateControlLayerInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint64_t           averageBitrate;
    uint64_t           maxBitrate;
    uint32_t           frameRateNumerator;
    uint32_t           frameRateDenominator;
} VkVideoEncodeRateControlLayerInfoKHR;

typedef struct VkVideoEncodeRateControlInfoKHR {
    VkStructureType                                sType;
    const void*                                    pNext;
    VkVideoEncodeRateControlFlagsKHR               flags;
    VkVideoEncodeRateControlModeFlagBitsKHR        rateControlMode;
    uint32_t                                       layerCount;
    const VkVideoEncodeRateControlLayerInfoKHR*    pLayers;
    uint32_t                                       virtualBufferSizeInMs;
    uint32_t                                       initialVirtualBufferSizeInMs;
} VkVideoEncodeRateControlInfoKHR;

typedef struct VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR {
    VkStructureType                 sType;
    const void*                     pNext;
    const VkVideoProfileInfoKHR*    pVideoProfile;
    uint32_t                        qualityLevel;
} VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR;

typedef struct VkVideoEncodeQualityLevelPropertiesKHR {
    VkStructureType                            sType;
    void*                                      pNext;
    VkVideoEncodeRateControlModeFlagBitsKHR    preferredRateControlMode;
    uint32_t                                   preferredRateControlLayerCount;
} VkVideoEncodeQualityLevelPropertiesKHR;

typedef struct VkVideoEncodeQualityLevelInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           qualityLevel;
} VkVideoEncodeQualityLevelInfoKHR;

typedef struct VkVideoEncodeSessionParametersGetInfoKHR {
    VkStructureType                sType;
    const void*                    pNext;
    VkVideoSessionParametersKHR    videoSessionParameters;
} VkVideoEncodeSessionParametersGetInfoKHR;

typedef struct VkVideoEncodeSessionParametersFeedbackInfoKHR {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           hasOverrides;
} VkVideoEncodeSessionParametersFeedbackInfoKHR;

typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR* pQualityLevelInfo, VkVideoEncodeQualityLevelPropertiesKHR* pQualityLevelProperties);
typedef VkResult (VKAPI_PTR *PFN_vkGetEncodedVideoSessionParametersKHR)(VkDevice device, const VkVideoEncodeSessionParametersGetInfoKHR* pVideoSessionParametersInfo, VkVideoEncodeSessionParametersFeedbackInfoKHR* pFeedbackInfo, size_t* pDataSize, void* pData);
typedef void (VKAPI_PTR *PFN_vkCmdEncodeVideoKHR)(VkCommandBuffer commandBuffer, const VkVideoEncodeInfoKHR* pEncodeInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR* pQualityLevelInfo,
    VkVideoEncodeQualityLevelPropertiesKHR*     pQualityLevelProperties);

VKAPI_ATTR VkResult VKAPI_CALL vkGetEncodedVideoSessionParametersKHR(
    VkDevice                                    device,
    const VkVideoEncodeSessionParametersGetInfoKHR* pVideoSessionParametersInfo,
    VkVideoEncodeSessionParametersFeedbackInfoKHR* pFeedbackInfo,
    size_t*                                     pDataSize,
    void*                                       pData);

VKAPI_ATTR void VKAPI_CALL vkCmdEncodeVideoKHR(
    VkCommandBuffer                             commandBuffer,
    const VkVideoEncodeInfoKHR*                 pEncodeInfo);
#endif


// VK_EXT_video_encode_h264 is a preprocessor guard. Do not pass it to API calls.
#define VK_EXT_video_encode_h264 1
#include "vk_video/vulkan_video_codec_h264std.h"
#include "vk_video/vulkan_video_codec_h264std_encode.h"
#define VK_EXT_VIDEO_ENCODE_H264_SPEC_VERSION 12
#define VK_EXT_VIDEO_ENCODE_H264_EXTENSION_NAME "VK_EXT_video_encode_h264"

typedef enum VkVideoEncodeH264CapabilityFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_CAPABILITY_ROW_UNALIGNED_SLICE_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H264_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H264_CAPABILITY_PER_SLICE_CONSTANT_QP_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H264_CAPABILITY_GENERATE_PREFIX_NALU_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H264_CAPABILITY_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264CapabilityFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264CapabilityFlagsEXT;

typedef enum VkVideoEncodeH264StdFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_STD_QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_STD_SCALING_MATRIX_PRESENT_FLAG_SET_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_STD_CHROMA_QP_INDEX_OFFSET_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H264_STD_SECOND_CHROMA_QP_INDEX_OFFSET_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H264_STD_PIC_INIT_QP_MINUS26_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H264_STD_WEIGHTED_PRED_FLAG_SET_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_EXPLICIT_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_IMPLICIT_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_EXT = 0x00000200,
    VK_VIDEO_ENCODE_H264_STD_DIRECT_SPATIAL_MV_PRED_FLAG_UNSET_BIT_EXT = 0x00000400,
    VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_EXT = 0x00000800,
    VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_EXT = 0x00001000,
    VK_VIDEO_ENCODE_H264_STD_DIRECT_8X8_INFERENCE_FLAG_UNSET_BIT_EXT = 0x00002000,
    VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_EXT = 0x00004000,
    VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_DISABLED_BIT_EXT = 0x00008000,
    VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_ENABLED_BIT_EXT = 0x00010000,
    VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_PARTIAL_BIT_EXT = 0x00020000,
    VK_VIDEO_ENCODE_H264_STD_SLICE_QP_DELTA_BIT_EXT = 0x00080000,
    VK_VIDEO_ENCODE_H264_STD_DIFFERENT_SLICE_QP_DELTA_BIT_EXT = 0x00100000,
    VK_VIDEO_ENCODE_H264_STD_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264StdFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264StdFlagsEXT;

typedef enum VkVideoEncodeH264RateControlFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_TEMPORAL_LAYER_PATTERN_DYADIC_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264RateControlFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264RateControlFlagsEXT;
typedef struct VkVideoEncodeH264CapabilitiesEXT {
    VkStructureType                        sType;
    void*                                  pNext;
    VkVideoEncodeH264CapabilityFlagsEXT    flags;
    StdVideoH264LevelIdc                   maxLevelIdc;
    uint32_t                               maxSliceCount;
    uint32_t                               maxPPictureL0ReferenceCount;
    uint32_t                               maxBPictureL0ReferenceCount;
    uint32_t                               maxL1ReferenceCount;
    uint32_t                               maxTemporalLayerCount;
    VkBool32                               expectDyadicTemporalLayerPattern;
    int32_t                                minQp;
    int32_t                                maxQp;
    VkBool32                               prefersGopRemainingFrames;
    VkBool32                               requiresGopRemainingFrames;
    VkVideoEncodeH264StdFlagsEXT           stdSyntaxFlags;
} VkVideoEncodeH264CapabilitiesEXT;

typedef struct VkVideoEncodeH264QpEXT {
    int32_t    qpI;
    int32_t    qpP;
    int32_t    qpB;
} VkVideoEncodeH264QpEXT;

typedef struct VkVideoEncodeH264QualityLevelPropertiesEXT {
    VkStructureType                         sType;
    void*                                   pNext;
    VkVideoEncodeH264RateControlFlagsEXT    preferredRateControlFlags;
    uint32_t                                preferredGopFrameCount;
    uint32_t                                preferredIdrPeriod;
    uint32_t                                preferredConsecutiveBFrameCount;
    uint32_t                                preferredTemporalLayerCount;
    VkVideoEncodeH264QpEXT                  preferredConstantQp;
    uint32_t                                preferredMaxL0ReferenceCount;
    uint32_t                                preferredMaxL1ReferenceCount;
    VkBool32                                preferredStdEntropyCodingModeFlag;
} VkVideoEncodeH264QualityLevelPropertiesEXT;

typedef struct VkVideoEncodeH264SessionCreateInfoEXT {
    VkStructureType         sType;
    const void*             pNext;
    VkBool32                useMaxLevelIdc;
    StdVideoH264LevelIdc    maxLevelIdc;
} VkVideoEncodeH264SessionCreateInfoEXT;

typedef struct VkVideoEncodeH264SessionParametersAddInfoEXT {
    VkStructureType                            sType;
    const void*                                pNext;
    uint32_t                                   stdSPSCount;
    const StdVideoH264SequenceParameterSet*    pStdSPSs;
    uint32_t                                   stdPPSCount;
    const StdVideoH264PictureParameterSet*     pStdPPSs;
} VkVideoEncodeH264SessionParametersAddInfoEXT;

typedef struct VkVideoEncodeH264SessionParametersCreateInfoEXT {
    VkStructureType                                        sType;
    const void*                                            pNext;
    uint32_t                                               maxStdSPSCount;
    uint32_t                                               maxStdPPSCount;
    const VkVideoEncodeH264SessionParametersAddInfoEXT*    pParametersAddInfo;
} VkVideoEncodeH264SessionParametersCreateInfoEXT;

typedef struct VkVideoEncodeH264SessionParametersGetInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    VkBool32           writeStdSPS;
    VkBool32           writeStdPPS;
    uint32_t           stdSPSId;
    uint32_t           stdPPSId;
} VkVideoEncodeH264SessionParametersGetInfoEXT;

typedef struct VkVideoEncodeH264SessionParametersFeedbackInfoEXT {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           hasStdSPSOverrides;
    VkBool32           hasStdPPSOverrides;
} VkVideoEncodeH264SessionParametersFeedbackInfoEXT;

typedef struct VkVideoEncodeH264NaluSliceInfoEXT {
    VkStructureType                         sType;
    const void*                             pNext;
    int32_t                                 constantQp;
    const StdVideoEncodeH264SliceHeader*    pStdSliceHeader;
} VkVideoEncodeH264NaluSliceInfoEXT;

typedef struct VkVideoEncodeH264PictureInfoEXT {
    VkStructureType                             sType;
    const void*                                 pNext;
    uint32_t                                    naluSliceEntryCount;
    const VkVideoEncodeH264NaluSliceInfoEXT*    pNaluSliceEntries;
    const StdVideoEncodeH264PictureInfo*        pStdPictureInfo;
    VkBool32                                    generatePrefixNalu;
} VkVideoEncodeH264PictureInfoEXT;

typedef struct VkVideoEncodeH264DpbSlotInfoEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    const StdVideoEncodeH264ReferenceInfo*    pStdReferenceInfo;
} VkVideoEncodeH264DpbSlotInfoEXT;

typedef struct VkVideoEncodeH264ProfileInfoEXT {
    VkStructureType           sType;
    const void*               pNext;
    StdVideoH264ProfileIdc    stdProfileIdc;
} VkVideoEncodeH264ProfileInfoEXT;

typedef struct VkVideoEncodeH264RateControlInfoEXT {
    VkStructureType                         sType;
    const void*                             pNext;
    VkVideoEncodeH264RateControlFlagsEXT    flags;
    uint32_t                                gopFrameCount;
    uint32_t                                idrPeriod;
    uint32_t                                consecutiveBFrameCount;
    uint32_t                                temporalLayerCount;
} VkVideoEncodeH264RateControlInfoEXT;

typedef struct VkVideoEncodeH264FrameSizeEXT {
    uint32_t    frameISize;
    uint32_t    framePSize;
    uint32_t    frameBSize;
} VkVideoEncodeH264FrameSizeEXT;

typedef struct VkVideoEncodeH264RateControlLayerInfoEXT {
    VkStructureType                  sType;
    const void*                      pNext;
    VkBool32                         useMinQp;
    VkVideoEncodeH264QpEXT           minQp;
    VkBool32                         useMaxQp;
    VkVideoEncodeH264QpEXT           maxQp;
    VkBool32                         useMaxFrameSize;
    VkVideoEncodeH264FrameSizeEXT    maxFrameSize;
} VkVideoEncodeH264RateControlLayerInfoEXT;

typedef struct VkVideoEncodeH264GopRemainingFrameInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    VkBool32           useGopRemainingFrames;
    uint32_t           gopRemainingI;
    uint32_t           gopRemainingP;
    uint32_t           gopRemainingB;
} VkVideoEncodeH264GopRemainingFrameInfoEXT;



// VK_EXT_video_encode_h265 is a preprocessor guard. Do not pass it to API calls.
#define VK_EXT_video_encode_h265 1
#include "vk_video/vulkan_video_codec_h265std.h"
#include "vk_video/vulkan_video_codec_h265std_encode.h"
#define VK_EXT_VIDEO_ENCODE_H265_SPEC_VERSION 12
#define VK_EXT_VIDEO_ENCODE_H265_EXTENSION_NAME "VK_EXT_video_encode_h265"

typedef enum VkVideoEncodeH265CapabilityFlagBitsEXT {
    VK_VIDEO_ENCODE_H265_CAPABILITY_HRD_COMPLIANCE_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H265_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H265_CAPABILITY_ROW_UNALIGNED_SLICE_SEGMENT_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H265_CAPABILITY_DIFFERENT_SLICE_SEGMENT_TYPE_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H265_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H265_CAPABILITY_PER_SLICE_SEGMENT_CONSTANT_QP_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILES_PER_SLICE_SEGMENT_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_SLICE_SEGMENTS_PER_TILE_BIT_EXT = 0x00000200,
    VK_VIDEO_ENCODE_H265_CAPABILITY_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265CapabilityFlagBitsEXT;
typedef VkFlags VkVideoEncodeH265CapabilityFlagsEXT;

typedef enum VkVideoEncodeH265StdFlagBitsEXT {
    VK_VIDEO_ENCODE_H265_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H265_STD_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H265_STD_SCALING_LIST_DATA_PRESENT_FLAG_SET_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H265_STD_PCM_ENABLED_FLAG_SET_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H265_STD_SPS_TEMPORAL_MVP_ENABLED_FLAG_SET_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H265_STD_INIT_QP_MINUS26_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H265_STD_WEIGHTED_PRED_FLAG_SET_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H265_STD_WEIGHTED_BIPRED_FLAG_SET_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H265_STD_LOG2_PARALLEL_MERGE_LEVEL_MINUS2_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H265_STD_SIGN_DATA_HIDING_ENABLED_FLAG_SET_BIT_EXT = 0x00000200,
    VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_SET_BIT_EXT = 0x00000400,
    VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_UNSET_BIT_EXT = 0x00000800,
    VK_VIDEO_ENCODE_H265_STD_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET_BIT_EXT = 0x00001000,
    VK_VIDEO_ENCODE_H265_STD_TRANSQUANT_BYPASS_ENABLED_FLAG_SET_BIT_EXT = 0x00002000,
    VK_VIDEO_ENCODE_H265_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_EXT = 0x00004000,
    VK_VIDEO_ENCODE_H265_STD_ENTROPY_CODING_SYNC_ENABLED_FLAG_SET_BIT_EXT = 0x00008000,
    VK_VIDEO_ENCODE_H265_STD_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET_BIT_EXT = 0x00010000,
    VK_VIDEO_ENCODE_H265_STD_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET_BIT_EXT = 0x00020000,
    VK_VIDEO_ENCODE_H265_STD_DEPENDENT_SLICE_SEGMENT_FLAG_SET_BIT_EXT = 0x00040000,
    VK_VIDEO_ENCODE_H265_STD_SLICE_QP_DELTA_BIT_EXT = 0x00080000,
    VK_VIDEO_ENCODE_H265_STD_DIFFERENT_SLICE_QP_DELTA_BIT_EXT = 0x00100000,
    VK_VIDEO_ENCODE_H265_STD_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265StdFlagBitsEXT;
typedef VkFlags VkVideoEncodeH265StdFlagsEXT;

typedef enum VkVideoEncodeH265CtbSizeFlagBitsEXT {
    VK_VIDEO_ENCODE_H265_CTB_SIZE_16_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H265_CTB_SIZE_32_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H265_CTB_SIZE_64_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H265_CTB_SIZE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265CtbSizeFlagBitsEXT;
typedef VkFlags VkVideoEncodeH265CtbSizeFlagsEXT;

typedef enum VkVideoEncodeH265TransformBlockSizeFlagBitsEXT {
    VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_4_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_16_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_32_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265TransformBlockSizeFlagBitsEXT;
typedef VkFlags VkVideoEncodeH265TransformBlockSizeFlagsEXT;

typedef enum VkVideoEncodeH265RateControlFlagBitsEXT {
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_REGULAR_GOP_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_TEMPORAL_SUB_LAYER_PATTERN_DYADIC_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265RateControlFlagBitsEXT;
typedef VkFlags VkVideoEncodeH265RateControlFlagsEXT;
typedef struct VkVideoEncodeH265CapabilitiesEXT {
    VkStructureType                                sType;
    void*                                          pNext;
    VkVideoEncodeH265CapabilityFlagsEXT            flags;
    StdVideoH265LevelIdc                           maxLevelIdc;
    uint32_t                                       maxSliceSegmentCount;
    VkExtent2D                                     maxTiles;
    VkVideoEncodeH265CtbSizeFlagsEXT               ctbSizes;
    VkVideoEncodeH265TransformBlockSizeFlagsEXT    transformBlockSizes;
    uint32_t                                       maxPPictureL0ReferenceCount;
    uint32_t                                       maxBPictureL0ReferenceCount;
    uint32_t                                       maxL1ReferenceCount;
    uint32_t                                       maxSubLayerCount;
    VkBool32                                       expectDyadicTemporalSubLayerPattern;
    int32_t                                        minQp;
    int32_t                                        maxQp;
    VkBool32                                       prefersGopRemainingFrames;
    VkBool32                                       requiresGopRemainingFrames;
    VkVideoEncodeH265StdFlagsEXT                   stdSyntaxFlags;
} VkVideoEncodeH265CapabilitiesEXT;

typedef struct VkVideoEncodeH265SessionCreateInfoEXT {
    VkStructureType         sType;
    const void*             pNext;
    VkBool32                useMaxLevelIdc;
    StdVideoH265LevelIdc    maxLevelIdc;
} VkVideoEncodeH265SessionCreateInfoEXT;

typedef struct VkVideoEncodeH265QpEXT {
    int32_t    qpI;
    int32_t    qpP;
    int32_t    qpB;
} VkVideoEncodeH265QpEXT;

typedef struct VkVideoEncodeH265QualityLevelPropertiesEXT {
    VkStructureType                         sType;
    void*                                   pNext;
    VkVideoEncodeH265RateControlFlagsEXT    preferredRateControlFlags;
    uint32_t                                preferredGopFrameCount;
    uint32_t                                preferredIdrPeriod;
    uint32_t                                preferredConsecutiveBFrameCount;
    uint32_t                                preferredSubLayerCount;
    VkVideoEncodeH265QpEXT                  preferredConstantQp;
    uint32_t                                preferredMaxL0ReferenceCount;
    uint32_t                                preferredMaxL1ReferenceCount;
} VkVideoEncodeH265QualityLevelPropertiesEXT;

typedef struct VkVideoEncodeH265SessionParametersAddInfoEXT {
    VkStructureType                            sType;
    const void*                                pNext;
    uint32_t                                   stdVPSCount;
    const StdVideoH265VideoParameterSet*       pStdVPSs;
    uint32_t                                   stdSPSCount;
    const StdVideoH265SequenceParameterSet*    pStdSPSs;
    uint32_t                                   stdPPSCount;
    const StdVideoH265PictureParameterSet*     pStdPPSs;
} VkVideoEncodeH265SessionParametersAddInfoEXT;

typedef struct VkVideoEncodeH265SessionParametersCreateInfoEXT {
    VkStructureType                                        sType;
    const void*                                            pNext;
    uint32_t                                               maxStdVPSCount;
    uint32_t                                               maxStdSPSCount;
    uint32_t                                               maxStdPPSCount;
    const VkVideoEncodeH265SessionParametersAddInfoEXT*    pParametersAddInfo;
} VkVideoEncodeH265SessionParametersCreateInfoEXT;

typedef struct VkVideoEncodeH265SessionParametersGetInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    VkBool32           writeStdVPS;
    VkBool32           writeStdSPS;
    VkBool32           writeStdPPS;
    uint32_t           stdVPSId;
    uint32_t           stdSPSId;
    uint32_t           stdPPSId;
} VkVideoEncodeH265SessionParametersGetInfoEXT;

typedef struct VkVideoEncodeH265SessionParametersFeedbackInfoEXT {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           hasStdVPSOverrides;
    VkBool32           hasStdSPSOverrides;
    VkBool32           hasStdPPSOverrides;
} VkVideoEncodeH265SessionParametersFeedbackInfoEXT;

typedef struct VkVideoEncodeH265NaluSliceSegmentInfoEXT {
    VkStructureType                                sType;
    const void*                                    pNext;
    int32_t                                        constantQp;
    const StdVideoEncodeH265SliceSegmentHeader*    pStdSliceSegmentHeader;
} VkVideoEncodeH265NaluSliceSegmentInfoEXT;

typedef struct VkVideoEncodeH265PictureInfoEXT {
    VkStructureType                                    sType;
    const void*                                        pNext;
    uint32_t                                           naluSliceSegmentEntryCount;
    const VkVideoEncodeH265NaluSliceSegmentInfoEXT*    pNaluSliceSegmentEntries;
    const StdVideoEncodeH265PictureInfo*               pStdPictureInfo;
} VkVideoEncodeH265PictureInfoEXT;

typedef struct VkVideoEncodeH265DpbSlotInfoEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    const StdVideoEncodeH265ReferenceInfo*    pStdReferenceInfo;
} VkVideoEncodeH265DpbSlotInfoEXT;

typedef struct VkVideoEncodeH265ProfileInfoEXT {
    VkStructureType           sType;
    const void*               pNext;
    StdVideoH265ProfileIdc    stdProfileIdc;
} VkVideoEncodeH265ProfileInfoEXT;

typedef struct VkVideoEncodeH265RateControlInfoEXT {
    VkStructureType                         sType;
    const void*                             pNext;
    VkVideoEncodeH265RateControlFlagsEXT    flags;
    uint32_t                                gopFrameCount;
    uint32_t                                idrPeriod;
    uint32_t                                consecutiveBFrameCount;
    uint32_t                                subLayerCount;
} VkVideoEncodeH265RateControlInfoEXT;

typedef struct VkVideoEncodeH265FrameSizeEXT {
    uint32_t    frameISize;
    uint32_t    framePSize;
    uint32_t    frameBSize;
} VkVideoEncodeH265FrameSizeEXT;

typedef struct VkVideoEncodeH265RateControlLayerInfoEXT {
    VkStructureType                  sType;
    const void*                      pNext;
    VkBool32                         useMinQp;
    VkVideoEncodeH265QpEXT           minQp;
    VkBool32                         useMaxQp;
    VkVideoEncodeH265QpEXT           maxQp;
    VkBool32                         useMaxFrameSize;
    VkVideoEncodeH265FrameSizeEXT    maxFrameSize;
} VkVideoEncodeH265RateControlLayerInfoEXT;

typedef struct VkVideoEncodeH265GopRemainingFrameInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    VkBool32           useGopRemainingFrames;
    uint32_t           gopRemainingI;
    uint32_t           gopRemainingP;
    uint32_t           gopRemainingB;
} VkVideoEncodeH265GopRemainingFrameInfoEXT;



// VK_AMDX_shader_enqueue is a preprocessor guard. Do not pass it to API calls.
#define VK_AMDX_shader_enqueue 1
#define VK_AMDX_SHADER_ENQUEUE_SPEC_VERSION 1
#define VK_AMDX_SHADER_ENQUEUE_EXTENSION_NAME "VK_AMDX_shader_enqueue"
#define VK_SHADER_INDEX_UNUSED_AMDX       (~0U)
typedef struct VkPhysicalDeviceShaderEnqueueFeaturesAMDX {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderEnqueue;
} VkPhysicalDeviceShaderEnqueueFeaturesAMDX;

typedef struct VkPhysicalDeviceShaderEnqueuePropertiesAMDX {
    VkStructureType    sType;
    void*              pNext;
    uint32_t           maxExecutionGraphDepth;
    uint32_t           maxExecutionGraphShaderOutputNodes;
    uint32_t           maxExecutionGraphShaderPayloadSize;
    uint32_t           maxExecutionGraphShaderPayloadCount;
    uint32_t           executionGraphDispatchAddressAlignment;
} VkPhysicalDeviceShaderEnqueuePropertiesAMDX;

typedef struct VkExecutionGraphPipelineScratchSizeAMDX {
    VkStructureType    sType;
    void*              pNext;
    VkDeviceSize       size;
} VkExecutionGraphPipelineScratchSizeAMDX;

typedef struct VkExecutionGraphPipelineCreateInfoAMDX {
    VkStructureType                           sType;
    const void*                               pNext;
    VkPipelineCreateFlags                     flags;
    uint32_t                                  stageCount;
    const VkPipelineShaderStageCreateInfo*    pStages;
    const VkPipelineLibraryCreateInfoKHR*     pLibraryInfo;
    VkPipelineLayout                          layout;
    VkPipeline                                basePipelineHandle;
    int32_t                                   basePipelineIndex;
} VkExecutionGraphPipelineCreateInfoAMDX;

typedef union VkDeviceOrHostAddressConstAMDX {
    VkDeviceAddress    deviceAddress;
    const void*        hostAddress;
} VkDeviceOrHostAddressConstAMDX;

typedef struct VkDispatchGraphInfoAMDX {
    uint32_t                          nodeIndex;
    uint32_t                          payloadCount;
    VkDeviceOrHostAddressConstAMDX    payloads;
    uint64_t                          payloadStride;
} VkDispatchGraphInfoAMDX;

typedef struct VkDispatchGraphCountInfoAMDX {
    uint32_t                          count;
    VkDeviceOrHostAddressConstAMDX    infos;
    uint64_t                          stride;
} VkDispatchGraphCountInfoAMDX;

typedef struct VkPipelineShaderStageNodeCreateInfoAMDX {
      VkStructureType    sType;
    const void*          pNext;
    const char*          pName;
    uint32_t             index;
} VkPipelineShaderStageNodeCreateInfoAMDX;

typedef VkResult (VKAPI_PTR *PFN_vkCreateExecutionGraphPipelinesAMDX)(VkDevice                                        device, VkPipelineCache                 pipelineCache, uint32_t                                        createInfoCount, const VkExecutionGraphPipelineCreateInfoAMDX* pCreateInfos, const VkAllocationCallbacks*    pAllocator, VkPipeline*               pPipelines);
typedef VkResult (VKAPI_PTR *PFN_vkGetExecutionGraphPipelineScratchSizeAMDX)(VkDevice                                        device, VkPipeline                                      executionGraph, VkExecutionGraphPipelineScratchSizeAMDX*         pSizeInfo);
typedef VkResult (VKAPI_PTR *PFN_vkGetExecutionGraphPipelineNodeIndexAMDX)(VkDevice                                        device, VkPipeline                                      executionGraph, const VkPipelineShaderStageNodeCreateInfoAMDX*   pNodeInfo, uint32_t*                                       pNodeIndex);
typedef void (VKAPI_PTR *PFN_vkCmdInitializeGraphScratchMemoryAMDX)(VkCommandBuffer                                 commandBuffer, VkDeviceAddress                                 scratch);
typedef void (VKAPI_PTR *PFN_vkCmdDispatchGraphAMDX)(VkCommandBuffer                                 commandBuffer, VkDeviceAddress                                 scratch, const VkDispatchGraphCountInfoAMDX*              pCountInfo);
typedef void (VKAPI_PTR *PFN_vkCmdDispatchGraphIndirectAMDX)(VkCommandBuffer                                 commandBuffer, VkDeviceAddress                                 scratch, const VkDispatchGraphCountInfoAMDX*              pCountInfo);
typedef void (VKAPI_PTR *PFN_vkCmdDispatchGraphIndirectCountAMDX)(VkCommandBuffer                                 commandBuffer, VkDeviceAddress                                 scratch, VkDeviceAddress                                 countInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR VkResult VKAPI_CALL vkCreateExecutionGraphPipelinesAMDX(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkExecutionGraphPipelineCreateInfoAMDX* pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines);

VKAPI_ATTR VkResult VKAPI_CALL vkGetExecutionGraphPipelineScratchSizeAMDX(
    VkDevice                                    device,
    VkPipeline                                  executionGraph,
    VkExecutionGraphPipelineScratchSizeAMDX*    pSizeInfo);

VKAPI_ATTR VkResult VKAPI_CALL vkGetExecutionGraphPipelineNodeIndexAMDX(
    VkDevice                                    device,
    VkPipeline                                  executionGraph,
    const VkPipelineShaderStageNodeCreateInfoAMDX* pNodeInfo,
    uint32_t*                                   pNodeIndex);

VKAPI_ATTR void VKAPI_CALL vkCmdInitializeGraphScratchMemoryAMDX(
    VkCommandBuffer                             commandBuffer,
    VkDeviceAddress                             scratch);

VKAPI_ATTR void VKAPI_CALL vkCmdDispatchGraphAMDX(
    VkCommandBuffer                             commandBuffer,
    VkDeviceAddress                             scratch,
    const VkDispatchGraphCountInfoAMDX*         pCountInfo);

VKAPI_ATTR void VKAPI_CALL vkCmdDispatchGraphIndirectAMDX(
    VkCommandBuffer                             commandBuffer,
    VkDeviceAddress                             scratch,
    const VkDispatchGraphCountInfoAMDX*         pCountInfo);

VKAPI_ATTR void VKAPI_CALL vkCmdDispatchGraphIndirectCountAMDX(
    VkCommandBuffer                             commandBuffer,
    VkDeviceAddress                             scratch,
    VkDeviceAddress                             countInfo);
#endif


// VK_NV_displacement_micromap is a preprocessor guard. Do not pass it to API calls.
#define VK_NV_displacement_micromap 1
#define VK_NV_DISPLACEMENT_MICROMAP_SPEC_VERSION 2
#define VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME "VK_NV_displacement_micromap"

typedef enum VkDisplacementMicromapFormatNV {
    VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV = 1,
    VK_DISPLACEMENT_MICROMAP_FORMAT_256_TRIANGLES_128_BYTES_NV = 2,
    VK_DISPLACEMENT_MICROMAP_FORMAT_1024_TRIANGLES_128_BYTES_NV = 3,
    VK_DISPLACEMENT_MICROMAP_FORMAT_MAX_ENUM_NV = 0x7FFFFFFF
} VkDisplacementMicromapFormatNV;
typedef struct VkPhysicalDeviceDisplacementMicromapFeaturesNV {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           displacementMicromap;
} VkPhysicalDeviceDisplacementMicromapFeaturesNV;

typedef struct VkPhysicalDeviceDisplacementMicromapPropertiesNV {
    VkStructureType    sType;
    void*              pNext;
    uint32_t           maxDisplacementMicromapSubdivisionLevel;
} VkPhysicalDeviceDisplacementMicromapPropertiesNV;

typedef struct VkAccelerationStructureTrianglesDisplacementMicromapNV {
    VkStructureType                     sType;
    void*                               pNext;
    VkFormat                            displacementBiasAndScaleFormat;
    VkFormat                            displacementVectorFormat;
    VkDeviceOrHostAddressConstKHR       displacementBiasAndScaleBuffer;
    VkDeviceSize                        displacementBiasAndScaleStride;
    VkDeviceOrHostAddressConstKHR       displacementVectorBuffer;
    VkDeviceSize                        displacementVectorStride;
    VkDeviceOrHostAddressConstKHR       displacedMicromapPrimitiveFlags;
    VkDeviceSize                        displacedMicromapPrimitiveFlagsStride;
    VkIndexType                         indexType;
    VkDeviceOrHostAddressConstKHR       indexBuffer;
    VkDeviceSize                        indexStride;
    uint32_t                            baseTriangle;
    uint32_t                            usageCountsCount;
    const VkMicromapUsageEXT*           pUsageCounts;
    const VkMicromapUsageEXT* const*    ppUsageCounts;
    VkMicromapEXT                       micromap;
} VkAccelerationStructureTrianglesDisplacementMicromapNV;


#ifdef __cplusplus
}
#endif

#endif
