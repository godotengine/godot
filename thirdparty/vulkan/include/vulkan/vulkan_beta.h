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



#define VK_KHR_video_encode_queue 1
#define VK_KHR_VIDEO_ENCODE_QUEUE_SPEC_VERSION 8
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
    uint32_t                              qualityLevel;
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
    uint32_t                                maxQualityLevels;
    VkExtent2D                              inputImageDataFillAlignment;
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
    uint32_t           virtualBufferSizeInMs;
    uint32_t           initialVirtualBufferSizeInMs;
} VkVideoEncodeRateControlLayerInfoKHR;

typedef struct VkVideoEncodeRateControlInfoKHR {
    VkStructureType                                sType;
    const void*                                    pNext;
    VkVideoEncodeRateControlFlagsKHR               flags;
    VkVideoEncodeRateControlModeFlagBitsKHR        rateControlMode;
    uint32_t                                       layerCount;
    const VkVideoEncodeRateControlLayerInfoKHR*    pLayers;
} VkVideoEncodeRateControlInfoKHR;

typedef void (VKAPI_PTR *PFN_vkCmdEncodeVideoKHR)(VkCommandBuffer commandBuffer, const VkVideoEncodeInfoKHR* pEncodeInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkCmdEncodeVideoKHR(
    VkCommandBuffer                             commandBuffer,
    const VkVideoEncodeInfoKHR*                 pEncodeInfo);
#endif


#define VK_EXT_video_encode_h264 1
#include "vk_video/vulkan_video_codec_h264std.h"
#include "vk_video/vulkan_video_codec_h264std_encode.h"
#define VK_EXT_VIDEO_ENCODE_H264_SPEC_VERSION 10
#define VK_EXT_VIDEO_ENCODE_H264_EXTENSION_NAME "VK_EXT_video_encode_h264"

typedef enum VkVideoEncodeH264RateControlStructureEXT {
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_UNKNOWN_EXT = 0,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_FLAT_EXT = 1,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_DYADIC_EXT = 2,
    VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264RateControlStructureEXT;

typedef enum VkVideoEncodeH264CapabilityFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_CAPABILITY_DIRECT_8X8_INFERENCE_ENABLED_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DIRECT_8X8_INFERENCE_DISABLED_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_CAPABILITY_SEPARATE_COLOUR_PLANE_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_CAPABILITY_QPPRIME_Y_ZERO_TRANSFORM_BYPASS_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H264_CAPABILITY_SCALING_LISTS_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H264_CAPABILITY_CHROMA_QP_OFFSET_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H264_CAPABILITY_SECOND_CHROMA_QP_OFFSET_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H264_CAPABILITY_PIC_INIT_QP_MINUS26_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_PRED_BIT_EXT = 0x00000200,
    VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_BIPRED_EXPLICIT_BIT_EXT = 0x00000400,
    VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_BIPRED_IMPLICIT_BIT_EXT = 0x00000800,
    VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_PRED_NO_TABLE_BIT_EXT = 0x00001000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_TRANSFORM_8X8_BIT_EXT = 0x00002000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_CABAC_BIT_EXT = 0x00004000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_CAVLC_BIT_EXT = 0x00008000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_DISABLED_BIT_EXT = 0x00010000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_ENABLED_BIT_EXT = 0x00020000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_PARTIAL_BIT_EXT = 0x00040000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DISABLE_DIRECT_SPATIAL_MV_PRED_BIT_EXT = 0x00080000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_MULTIPLE_SLICE_PER_FRAME_BIT_EXT = 0x00100000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_SLICE_MB_COUNT_BIT_EXT = 0x00200000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_ROW_UNALIGNED_SLICE_BIT_EXT = 0x00400000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_EXT = 0x00800000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT = 0x01000000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_REFERENCE_FINAL_LISTS_BIT_EXT = 0x02000000,
    VK_VIDEO_ENCODE_H264_CAPABILITY_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264CapabilityFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264CapabilityFlagsEXT;
typedef struct VkVideoEncodeH264CapabilitiesEXT {
    VkStructureType                        sType;
    void*                                  pNext;
    VkVideoEncodeH264CapabilityFlagsEXT    flags;
    uint32_t                               maxPPictureL0ReferenceCount;
    uint32_t                               maxBPictureL0ReferenceCount;
    uint32_t                               maxL1ReferenceCount;
    VkBool32                               motionVectorsOverPicBoundariesFlag;
    uint32_t                               maxBytesPerPicDenom;
    uint32_t                               maxBitsPerMbDenom;
    uint32_t                               log2MaxMvLengthHorizontal;
    uint32_t                               log2MaxMvLengthVertical;
} VkVideoEncodeH264CapabilitiesEXT;

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

typedef struct VkVideoEncodeH264NaluSliceInfoEXT {
    VkStructureType                                sType;
    const void*                                    pNext;
    uint32_t                                       mbCount;
    const StdVideoEncodeH264ReferenceListsInfo*    pStdReferenceFinalLists;
    const StdVideoEncodeH264SliceHeader*           pStdSliceHeader;
} VkVideoEncodeH264NaluSliceInfoEXT;

typedef struct VkVideoEncodeH264VclFrameInfoEXT {
    VkStructureType                                sType;
    const void*                                    pNext;
    const StdVideoEncodeH264ReferenceListsInfo*    pStdReferenceFinalLists;
    uint32_t                                       naluSliceEntryCount;
    const VkVideoEncodeH264NaluSliceInfoEXT*       pNaluSliceEntries;
    const StdVideoEncodeH264PictureInfo*           pStdPictureInfo;
} VkVideoEncodeH264VclFrameInfoEXT;

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
    VkStructureType                             sType;
    const void*                                 pNext;
    uint32_t                                    gopFrameCount;
    uint32_t                                    idrPeriod;
    uint32_t                                    consecutiveBFrameCount;
    VkVideoEncodeH264RateControlStructureEXT    rateControlStructure;
    uint32_t                                    temporalLayerCount;
} VkVideoEncodeH264RateControlInfoEXT;

typedef struct VkVideoEncodeH264QpEXT {
    int32_t    qpI;
    int32_t    qpP;
    int32_t    qpB;
} VkVideoEncodeH264QpEXT;

typedef struct VkVideoEncodeH264FrameSizeEXT {
    uint32_t    frameISize;
    uint32_t    framePSize;
    uint32_t    frameBSize;
} VkVideoEncodeH264FrameSizeEXT;

typedef struct VkVideoEncodeH264RateControlLayerInfoEXT {
    VkStructureType                  sType;
    const void*                      pNext;
    uint32_t                         temporalLayerId;
    VkBool32                         useInitialRcQp;
    VkVideoEncodeH264QpEXT           initialRcQp;
    VkBool32                         useMinQp;
    VkVideoEncodeH264QpEXT           minQp;
    VkBool32                         useMaxQp;
    VkVideoEncodeH264QpEXT           maxQp;
    VkBool32                         useMaxFrameSize;
    VkVideoEncodeH264FrameSizeEXT    maxFrameSize;
} VkVideoEncodeH264RateControlLayerInfoEXT;



#define VK_EXT_video_encode_h265 1
#include "vk_video/vulkan_video_codec_h265std.h"
#include "vk_video/vulkan_video_codec_h265std_encode.h"
#define VK_EXT_VIDEO_ENCODE_H265_SPEC_VERSION 10
#define VK_EXT_VIDEO_ENCODE_H265_EXTENSION_NAME "VK_EXT_video_encode_h265"

typedef enum VkVideoEncodeH265RateControlStructureEXT {
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_UNKNOWN_EXT = 0,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_FLAT_EXT = 1,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_DYADIC_EXT = 2,
    VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265RateControlStructureEXT;

typedef enum VkVideoEncodeH265CapabilityFlagBitsEXT {
    VK_VIDEO_ENCODE_H265_CAPABILITY_SEPARATE_COLOUR_PLANE_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H265_CAPABILITY_SCALING_LISTS_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H265_CAPABILITY_SAMPLE_ADAPTIVE_OFFSET_ENABLED_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H265_CAPABILITY_PCM_ENABLE_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H265_CAPABILITY_SPS_TEMPORAL_MVP_ENABLED_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H265_CAPABILITY_HRD_COMPLIANCE_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H265_CAPABILITY_INIT_QP_MINUS26_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H265_CAPABILITY_LOG2_PARALLEL_MERGE_LEVEL_MINUS2_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H265_CAPABILITY_SIGN_DATA_HIDING_ENABLED_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H265_CAPABILITY_TRANSFORM_SKIP_ENABLED_BIT_EXT = 0x00000200,
    VK_VIDEO_ENCODE_H265_CAPABILITY_TRANSFORM_SKIP_DISABLED_BIT_EXT = 0x00000400,
    VK_VIDEO_ENCODE_H265_CAPABILITY_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_BIT_EXT = 0x00000800,
    VK_VIDEO_ENCODE_H265_CAPABILITY_WEIGHTED_PRED_BIT_EXT = 0x00001000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_WEIGHTED_BIPRED_BIT_EXT = 0x00002000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_WEIGHTED_PRED_NO_TABLE_BIT_EXT = 0x00004000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_TRANSQUANT_BYPASS_ENABLED_BIT_EXT = 0x00008000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_ENTROPY_CODING_SYNC_ENABLED_BIT_EXT = 0x00010000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_DEBLOCKING_FILTER_OVERRIDE_ENABLED_BIT_EXT = 0x00020000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILE_PER_FRAME_BIT_EXT = 0x00040000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_SLICE_PER_TILE_BIT_EXT = 0x00080000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILE_PER_SLICE_BIT_EXT = 0x00100000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_SLICE_SEGMENT_CTB_COUNT_BIT_EXT = 0x00200000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_ROW_UNALIGNED_SLICE_SEGMENT_BIT_EXT = 0x00400000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_DEPENDENT_SLICE_SEGMENT_BIT_EXT = 0x00800000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_EXT = 0x01000000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT = 0x02000000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_DIFFERENT_REFERENCE_FINAL_LISTS_BIT_EXT = 0x04000000,
    VK_VIDEO_ENCODE_H265_CAPABILITY_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH265CapabilityFlagBitsEXT;
typedef VkFlags VkVideoEncodeH265CapabilityFlagsEXT;

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
typedef struct VkVideoEncodeH265CapabilitiesEXT {
    VkStructureType                                sType;
    void*                                          pNext;
    VkVideoEncodeH265CapabilityFlagsEXT            flags;
    VkVideoEncodeH265CtbSizeFlagsEXT               ctbSizes;
    VkVideoEncodeH265TransformBlockSizeFlagsEXT    transformBlockSizes;
    uint32_t                                       maxPPictureL0ReferenceCount;
    uint32_t                                       maxBPictureL0ReferenceCount;
    uint32_t                                       maxL1ReferenceCount;
    uint32_t                                       maxSubLayersCount;
    uint32_t                                       minLog2MinLumaCodingBlockSizeMinus3;
    uint32_t                                       maxLog2MinLumaCodingBlockSizeMinus3;
    uint32_t                                       minLog2MinLumaTransformBlockSizeMinus2;
    uint32_t                                       maxLog2MinLumaTransformBlockSizeMinus2;
    uint32_t                                       minMaxTransformHierarchyDepthInter;
    uint32_t                                       maxMaxTransformHierarchyDepthInter;
    uint32_t                                       minMaxTransformHierarchyDepthIntra;
    uint32_t                                       maxMaxTransformHierarchyDepthIntra;
    uint32_t                                       maxDiffCuQpDeltaDepth;
    uint32_t                                       minMaxNumMergeCand;
    uint32_t                                       maxMaxNumMergeCand;
} VkVideoEncodeH265CapabilitiesEXT;

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

typedef struct VkVideoEncodeH265NaluSliceSegmentInfoEXT {
    VkStructureType                                sType;
    const void*                                    pNext;
    uint32_t                                       ctbCount;
    const StdVideoEncodeH265ReferenceListsInfo*    pStdReferenceFinalLists;
    const StdVideoEncodeH265SliceSegmentHeader*    pStdSliceSegmentHeader;
} VkVideoEncodeH265NaluSliceSegmentInfoEXT;

typedef struct VkVideoEncodeH265VclFrameInfoEXT {
    VkStructureType                                    sType;
    const void*                                        pNext;
    const StdVideoEncodeH265ReferenceListsInfo*        pStdReferenceFinalLists;
    uint32_t                                           naluSliceSegmentEntryCount;
    const VkVideoEncodeH265NaluSliceSegmentInfoEXT*    pNaluSliceSegmentEntries;
    const StdVideoEncodeH265PictureInfo*               pStdPictureInfo;
} VkVideoEncodeH265VclFrameInfoEXT;

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
    VkStructureType                             sType;
    const void*                                 pNext;
    uint32_t                                    gopFrameCount;
    uint32_t                                    idrPeriod;
    uint32_t                                    consecutiveBFrameCount;
    VkVideoEncodeH265RateControlStructureEXT    rateControlStructure;
    uint32_t                                    subLayerCount;
} VkVideoEncodeH265RateControlInfoEXT;

typedef struct VkVideoEncodeH265QpEXT {
    int32_t    qpI;
    int32_t    qpP;
    int32_t    qpB;
} VkVideoEncodeH265QpEXT;

typedef struct VkVideoEncodeH265FrameSizeEXT {
    uint32_t    frameISize;
    uint32_t    framePSize;
    uint32_t    frameBSize;
} VkVideoEncodeH265FrameSizeEXT;

typedef struct VkVideoEncodeH265RateControlLayerInfoEXT {
    VkStructureType                  sType;
    const void*                      pNext;
    uint32_t                         temporalId;
    VkBool32                         useInitialRcQp;
    VkVideoEncodeH265QpEXT           initialRcQp;
    VkBool32                         useMinQp;
    VkVideoEncodeH265QpEXT           minQp;
    VkBool32                         useMaxQp;
    VkVideoEncodeH265QpEXT           maxQp;
    VkBool32                         useMaxFrameSize;
    VkVideoEncodeH265FrameSizeEXT    maxFrameSize;
} VkVideoEncodeH265RateControlLayerInfoEXT;



#define VK_NV_displacement_micromap 1
#define VK_NV_DISPLACEMENT_MICROMAP_SPEC_VERSION 1
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
