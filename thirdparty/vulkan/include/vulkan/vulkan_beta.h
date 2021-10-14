#ifndef VULKAN_BETA_H_
#define VULKAN_BETA_H_ 1

/*
** Copyright 2015-2021 The Khronos Group Inc.
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



#define VK_KHR_video_queue 1
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkVideoSessionKHR)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkVideoSessionParametersKHR)
#define VK_KHR_VIDEO_QUEUE_SPEC_VERSION   2
#define VK_KHR_VIDEO_QUEUE_EXTENSION_NAME "VK_KHR_video_queue"

typedef enum VkQueryResultStatusKHR {
    VK_QUERY_RESULT_STATUS_ERROR_KHR = -1,
    VK_QUERY_RESULT_STATUS_NOT_READY_KHR = 0,
    VK_QUERY_RESULT_STATUS_COMPLETE_KHR = 1,
    VK_QUERY_RESULT_STATUS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkQueryResultStatusKHR;

typedef enum VkVideoCodecOperationFlagBitsKHR {
    VK_VIDEO_CODEC_OPERATION_INVALID_BIT_KHR = 0,
#ifdef VK_ENABLE_BETA_EXTENSIONS
    VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_EXT = 0x00010000,
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_EXT = 0x00000001,
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
    VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_EXT = 0x00000002,
#endif
    VK_VIDEO_CODEC_OPERATION_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoCodecOperationFlagBitsKHR;
typedef VkFlags VkVideoCodecOperationFlagsKHR;

typedef enum VkVideoChromaSubsamplingFlagBitsKHR {
    VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_BIT_KHR = 0,
    VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR = 0x00000001,
    VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR = 0x00000002,
    VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR = 0x00000004,
    VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR = 0x00000008,
    VK_VIDEO_CHROMA_SUBSAMPLING_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoChromaSubsamplingFlagBitsKHR;
typedef VkFlags VkVideoChromaSubsamplingFlagsKHR;

typedef enum VkVideoComponentBitDepthFlagBitsKHR {
    VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR = 0,
    VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR = 0x00000001,
    VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR = 0x00000004,
    VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR = 0x00000010,
    VK_VIDEO_COMPONENT_BIT_DEPTH_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoComponentBitDepthFlagBitsKHR;
typedef VkFlags VkVideoComponentBitDepthFlagsKHR;

typedef enum VkVideoCapabilityFlagBitsKHR {
    VK_VIDEO_CAPABILITY_PROTECTED_CONTENT_BIT_KHR = 0x00000001,
    VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR = 0x00000002,
    VK_VIDEO_CAPABILITY_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoCapabilityFlagBitsKHR;
typedef VkFlags VkVideoCapabilityFlagsKHR;

typedef enum VkVideoSessionCreateFlagBitsKHR {
    VK_VIDEO_SESSION_CREATE_DEFAULT_KHR = 0,
    VK_VIDEO_SESSION_CREATE_PROTECTED_CONTENT_BIT_KHR = 0x00000001,
    VK_VIDEO_SESSION_CREATE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoSessionCreateFlagBitsKHR;
typedef VkFlags VkVideoSessionCreateFlagsKHR;
typedef VkFlags VkVideoBeginCodingFlagsKHR;
typedef VkFlags VkVideoEndCodingFlagsKHR;

typedef enum VkVideoCodingControlFlagBitsKHR {
    VK_VIDEO_CODING_CONTROL_DEFAULT_KHR = 0,
    VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR = 0x00000001,
    VK_VIDEO_CODING_CONTROL_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoCodingControlFlagBitsKHR;
typedef VkFlags VkVideoCodingControlFlagsKHR;

typedef enum VkVideoCodingQualityPresetFlagBitsKHR {
    VK_VIDEO_CODING_QUALITY_PRESET_DEFAULT_BIT_KHR = 0,
    VK_VIDEO_CODING_QUALITY_PRESET_NORMAL_BIT_KHR = 0x00000001,
    VK_VIDEO_CODING_QUALITY_PRESET_POWER_BIT_KHR = 0x00000002,
    VK_VIDEO_CODING_QUALITY_PRESET_QUALITY_BIT_KHR = 0x00000004,
    VK_VIDEO_CODING_QUALITY_PRESET_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoCodingQualityPresetFlagBitsKHR;
typedef VkFlags VkVideoCodingQualityPresetFlagsKHR;
typedef struct VkVideoQueueFamilyProperties2KHR {
    VkStructureType                  sType;
    void*                            pNext;
    VkVideoCodecOperationFlagsKHR    videoCodecOperations;
} VkVideoQueueFamilyProperties2KHR;

typedef struct VkVideoProfileKHR {
    VkStructureType                     sType;
    void*                               pNext;
    VkVideoCodecOperationFlagBitsKHR    videoCodecOperation;
    VkVideoChromaSubsamplingFlagsKHR    chromaSubsampling;
    VkVideoComponentBitDepthFlagsKHR    lumaBitDepth;
    VkVideoComponentBitDepthFlagsKHR    chromaBitDepth;
} VkVideoProfileKHR;

typedef struct VkVideoProfilesKHR {
    VkStructureType             sType;
    void*                       pNext;
    uint32_t                    profileCount;
    const VkVideoProfileKHR*    pProfiles;
} VkVideoProfilesKHR;

typedef struct VkVideoCapabilitiesKHR {
    VkStructureType              sType;
    void*                        pNext;
    VkVideoCapabilityFlagsKHR    capabilityFlags;
    VkDeviceSize                 minBitstreamBufferOffsetAlignment;
    VkDeviceSize                 minBitstreamBufferSizeAlignment;
    VkExtent2D                   videoPictureExtentGranularity;
    VkExtent2D                   minExtent;
    VkExtent2D                   maxExtent;
    uint32_t                     maxReferencePicturesSlotsCount;
    uint32_t                     maxReferencePicturesActiveCount;
} VkVideoCapabilitiesKHR;

typedef struct VkPhysicalDeviceVideoFormatInfoKHR {
    VkStructureType              sType;
    void*                        pNext;
    VkImageUsageFlags            imageUsage;
    const VkVideoProfilesKHR*    pVideoProfiles;
} VkPhysicalDeviceVideoFormatInfoKHR;

typedef struct VkVideoFormatPropertiesKHR {
    VkStructureType    sType;
    void*              pNext;
    VkFormat           format;
} VkVideoFormatPropertiesKHR;

typedef struct VkVideoPictureResourceKHR {
    VkStructureType    sType;
    const void*        pNext;
    VkOffset2D         codedOffset;
    VkExtent2D         codedExtent;
    uint32_t           baseArrayLayer;
    VkImageView        imageViewBinding;
} VkVideoPictureResourceKHR;

typedef struct VkVideoReferenceSlotKHR {
    VkStructureType                     sType;
    const void*                         pNext;
    int8_t                              slotIndex;
    const VkVideoPictureResourceKHR*    pPictureResource;
} VkVideoReferenceSlotKHR;

typedef struct VkVideoGetMemoryPropertiesKHR {
    VkStructureType           sType;
    const void*               pNext;
    uint32_t                  memoryBindIndex;
    VkMemoryRequirements2*    pMemoryRequirements;
} VkVideoGetMemoryPropertiesKHR;

typedef struct VkVideoBindMemoryKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           memoryBindIndex;
    VkDeviceMemory     memory;
    VkDeviceSize       memoryOffset;
    VkDeviceSize       memorySize;
} VkVideoBindMemoryKHR;

typedef struct VkVideoSessionCreateInfoKHR {
    VkStructureType                 sType;
    const void*                     pNext;
    uint32_t                        queueFamilyIndex;
    VkVideoSessionCreateFlagsKHR    flags;
    const VkVideoProfileKHR*        pVideoProfile;
    VkFormat                        pictureFormat;
    VkExtent2D                      maxCodedExtent;
    VkFormat                        referencePicturesFormat;
    uint32_t                        maxReferencePicturesSlotsCount;
    uint32_t                        maxReferencePicturesActiveCount;
} VkVideoSessionCreateInfoKHR;

typedef struct VkVideoSessionParametersCreateInfoKHR {
    VkStructureType                sType;
    const void*                    pNext;
    VkVideoSessionParametersKHR    videoSessionParametersTemplate;
    VkVideoSessionKHR              videoSession;
} VkVideoSessionParametersCreateInfoKHR;

typedef struct VkVideoSessionParametersUpdateInfoKHR {
    VkStructureType    sType;
    const void*        pNext;
    uint32_t           updateSequenceCount;
} VkVideoSessionParametersUpdateInfoKHR;

typedef struct VkVideoBeginCodingInfoKHR {
    VkStructureType                       sType;
    const void*                           pNext;
    VkVideoBeginCodingFlagsKHR            flags;
    VkVideoCodingQualityPresetFlagsKHR    codecQualityPreset;
    VkVideoSessionKHR                     videoSession;
    VkVideoSessionParametersKHR           videoSessionParameters;
    uint32_t                              referenceSlotCount;
    const VkVideoReferenceSlotKHR*        pReferenceSlots;
} VkVideoBeginCodingInfoKHR;

typedef struct VkVideoEndCodingInfoKHR {
    VkStructureType             sType;
    const void*                 pNext;
    VkVideoEndCodingFlagsKHR    flags;
} VkVideoEndCodingInfoKHR;

typedef struct VkVideoCodingControlInfoKHR {
    VkStructureType                 sType;
    const void*                     pNext;
    VkVideoCodingControlFlagsKHR    flags;
} VkVideoCodingControlInfoKHR;

typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)(VkPhysicalDevice physicalDevice, const VkVideoProfileKHR* pVideoProfile, VkVideoCapabilitiesKHR* pCapabilities);
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceVideoFormatInfoKHR* pVideoFormatInfo, uint32_t* pVideoFormatPropertyCount, VkVideoFormatPropertiesKHR* pVideoFormatProperties);
typedef VkResult (VKAPI_PTR *PFN_vkCreateVideoSessionKHR)(VkDevice device, const VkVideoSessionCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkVideoSessionKHR* pVideoSession);
typedef void (VKAPI_PTR *PFN_vkDestroyVideoSessionKHR)(VkDevice device, VkVideoSessionKHR videoSession, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkGetVideoSessionMemoryRequirementsKHR)(VkDevice device, VkVideoSessionKHR videoSession, uint32_t* pVideoSessionMemoryRequirementsCount, VkVideoGetMemoryPropertiesKHR* pVideoSessionMemoryRequirements);
typedef VkResult (VKAPI_PTR *PFN_vkBindVideoSessionMemoryKHR)(VkDevice device, VkVideoSessionKHR videoSession, uint32_t videoSessionBindMemoryCount, const VkVideoBindMemoryKHR* pVideoSessionBindMemories);
typedef VkResult (VKAPI_PTR *PFN_vkCreateVideoSessionParametersKHR)(VkDevice device, const VkVideoSessionParametersCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkVideoSessionParametersKHR* pVideoSessionParameters);
typedef VkResult (VKAPI_PTR *PFN_vkUpdateVideoSessionParametersKHR)(VkDevice device, VkVideoSessionParametersKHR videoSessionParameters, const VkVideoSessionParametersUpdateInfoKHR* pUpdateInfo);
typedef void (VKAPI_PTR *PFN_vkDestroyVideoSessionParametersKHR)(VkDevice device, VkVideoSessionParametersKHR videoSessionParameters, const VkAllocationCallbacks* pAllocator);
typedef void (VKAPI_PTR *PFN_vkCmdBeginVideoCodingKHR)(VkCommandBuffer commandBuffer, const VkVideoBeginCodingInfoKHR* pBeginInfo);
typedef void (VKAPI_PTR *PFN_vkCmdEndVideoCodingKHR)(VkCommandBuffer commandBuffer, const VkVideoEndCodingInfoKHR* pEndCodingInfo);
typedef void (VKAPI_PTR *PFN_vkCmdControlVideoCodingKHR)(VkCommandBuffer commandBuffer, const VkVideoCodingControlInfoKHR* pCodingControlInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceVideoCapabilitiesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkVideoProfileKHR*                    pVideoProfile,
    VkVideoCapabilitiesKHR*                     pCapabilities);

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceVideoFormatPropertiesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceVideoFormatInfoKHR*   pVideoFormatInfo,
    uint32_t*                                   pVideoFormatPropertyCount,
    VkVideoFormatPropertiesKHR*                 pVideoFormatProperties);

VKAPI_ATTR VkResult VKAPI_CALL vkCreateVideoSessionKHR(
    VkDevice                                    device,
    const VkVideoSessionCreateInfoKHR*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkVideoSessionKHR*                          pVideoSession);

VKAPI_ATTR void VKAPI_CALL vkDestroyVideoSessionKHR(
    VkDevice                                    device,
    VkVideoSessionKHR                           videoSession,
    const VkAllocationCallbacks*                pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL vkGetVideoSessionMemoryRequirementsKHR(
    VkDevice                                    device,
    VkVideoSessionKHR                           videoSession,
    uint32_t*                                   pVideoSessionMemoryRequirementsCount,
    VkVideoGetMemoryPropertiesKHR*              pVideoSessionMemoryRequirements);

VKAPI_ATTR VkResult VKAPI_CALL vkBindVideoSessionMemoryKHR(
    VkDevice                                    device,
    VkVideoSessionKHR                           videoSession,
    uint32_t                                    videoSessionBindMemoryCount,
    const VkVideoBindMemoryKHR*                 pVideoSessionBindMemories);

VKAPI_ATTR VkResult VKAPI_CALL vkCreateVideoSessionParametersKHR(
    VkDevice                                    device,
    const VkVideoSessionParametersCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkVideoSessionParametersKHR*                pVideoSessionParameters);

VKAPI_ATTR VkResult VKAPI_CALL vkUpdateVideoSessionParametersKHR(
    VkDevice                                    device,
    VkVideoSessionParametersKHR                 videoSessionParameters,
    const VkVideoSessionParametersUpdateInfoKHR* pUpdateInfo);

VKAPI_ATTR void VKAPI_CALL vkDestroyVideoSessionParametersKHR(
    VkDevice                                    device,
    VkVideoSessionParametersKHR                 videoSessionParameters,
    const VkAllocationCallbacks*                pAllocator);

VKAPI_ATTR void VKAPI_CALL vkCmdBeginVideoCodingKHR(
    VkCommandBuffer                             commandBuffer,
    const VkVideoBeginCodingInfoKHR*            pBeginInfo);

VKAPI_ATTR void VKAPI_CALL vkCmdEndVideoCodingKHR(
    VkCommandBuffer                             commandBuffer,
    const VkVideoEndCodingInfoKHR*              pEndCodingInfo);

VKAPI_ATTR void VKAPI_CALL vkCmdControlVideoCodingKHR(
    VkCommandBuffer                             commandBuffer,
    const VkVideoCodingControlInfoKHR*          pCodingControlInfo);
#endif


#define VK_KHR_video_decode_queue 1
#define VK_KHR_VIDEO_DECODE_QUEUE_SPEC_VERSION 1
#define VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME "VK_KHR_video_decode_queue"

typedef enum VkVideoDecodeFlagBitsKHR {
    VK_VIDEO_DECODE_DEFAULT_KHR = 0,
    VK_VIDEO_DECODE_RESERVED_0_BIT_KHR = 0x00000001,
    VK_VIDEO_DECODE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoDecodeFlagBitsKHR;
typedef VkFlags VkVideoDecodeFlagsKHR;
typedef struct VkVideoDecodeInfoKHR {
    VkStructureType                   sType;
    const void*                       pNext;
    VkVideoDecodeFlagsKHR             flags;
    VkOffset2D                        codedOffset;
    VkExtent2D                        codedExtent;
    VkBuffer                          srcBuffer;
    VkDeviceSize                      srcBufferOffset;
    VkDeviceSize                      srcBufferRange;
    VkVideoPictureResourceKHR         dstPictureResource;
    const VkVideoReferenceSlotKHR*    pSetupReferenceSlot;
    uint32_t                          referenceSlotCount;
    const VkVideoReferenceSlotKHR*    pReferenceSlots;
} VkVideoDecodeInfoKHR;

typedef void (VKAPI_PTR *PFN_vkCmdDecodeVideoKHR)(VkCommandBuffer commandBuffer, const VkVideoDecodeInfoKHR* pFrameInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkCmdDecodeVideoKHR(
    VkCommandBuffer                             commandBuffer,
    const VkVideoDecodeInfoKHR*                 pFrameInfo);
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
#define VK_KHR_VIDEO_ENCODE_QUEUE_SPEC_VERSION 2
#define VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME "VK_KHR_video_encode_queue"

typedef enum VkVideoEncodeFlagBitsKHR {
    VK_VIDEO_ENCODE_DEFAULT_KHR = 0,
    VK_VIDEO_ENCODE_RESERVED_0_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeFlagBitsKHR;
typedef VkFlags VkVideoEncodeFlagsKHR;

typedef enum VkVideoEncodeRateControlFlagBitsKHR {
    VK_VIDEO_ENCODE_RATE_CONTROL_DEFAULT_KHR = 0,
    VK_VIDEO_ENCODE_RATE_CONTROL_RESET_BIT_KHR = 0x00000001,
    VK_VIDEO_ENCODE_RATE_CONTROL_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeRateControlFlagBitsKHR;
typedef VkFlags VkVideoEncodeRateControlFlagsKHR;

typedef enum VkVideoEncodeRateControlModeFlagBitsKHR {
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_NONE_BIT_KHR = 0,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR = 1,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR = 2,
    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_FLAG_BITS_MAX_ENUM_KHR = 0x7FFFFFFF
} VkVideoEncodeRateControlModeFlagBitsKHR;
typedef VkFlags VkVideoEncodeRateControlModeFlagsKHR;
typedef struct VkVideoEncodeInfoKHR {
    VkStructureType                   sType;
    const void*                       pNext;
    VkVideoEncodeFlagsKHR             flags;
    uint32_t                          qualityLevel;
    VkExtent2D                        codedExtent;
    VkBuffer                          dstBitstreamBuffer;
    VkDeviceSize                      dstBitstreamBufferOffset;
    VkDeviceSize                      dstBitstreamBufferMaxRange;
    VkVideoPictureResourceKHR         srcPictureResource;
    const VkVideoReferenceSlotKHR*    pSetupReferenceSlot;
    uint32_t                          referenceSlotCount;
    const VkVideoReferenceSlotKHR*    pReferenceSlots;
} VkVideoEncodeInfoKHR;

typedef struct VkVideoEncodeRateControlInfoKHR {
    VkStructureType                            sType;
    const void*                                pNext;
    VkVideoEncodeRateControlFlagsKHR           flags;
    VkVideoEncodeRateControlModeFlagBitsKHR    rateControlMode;
    uint32_t                                   averageBitrate;
    uint16_t                                   peakToAverageBitrateRatio;
    uint16_t                                   frameRateNumerator;
    uint16_t                                   frameRateDenominator;
    uint32_t                                   virtualBufferSizeInMs;
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
#define VK_EXT_VIDEO_ENCODE_H264_SPEC_VERSION 2
#define VK_EXT_VIDEO_ENCODE_H264_EXTENSION_NAME "VK_EXT_video_encode_h264"

typedef enum VkVideoEncodeH264CapabilityFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_CAPABILITY_CABAC_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_CAPABILITY_CAVLC_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_BI_PRED_IMPLICIT_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_CAPABILITY_TRANSFORM_8X8_BIT_EXT = 0x00000008,
    VK_VIDEO_ENCODE_H264_CAPABILITY_CHROMA_QP_OFFSET_BIT_EXT = 0x00000010,
    VK_VIDEO_ENCODE_H264_CAPABILITY_SECOND_CHROMA_QP_OFFSET_BIT_EXT = 0x00000020,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_DISABLED_BIT_EXT = 0x00000040,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_ENABLED_BIT_EXT = 0x00000080,
    VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_PARTIAL_BIT_EXT = 0x00000100,
    VK_VIDEO_ENCODE_H264_CAPABILITY_MULTIPLE_SLICE_PER_FRAME_BIT_EXT = 0x00000200,
    VK_VIDEO_ENCODE_H264_CAPABILITY_EVENLY_DISTRIBUTED_SLICE_SIZE_BIT_EXT = 0x00000400,
    VK_VIDEO_ENCODE_H264_CAPABILITY_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264CapabilityFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264CapabilityFlagsEXT;

typedef enum VkVideoEncodeH264InputModeFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_INPUT_MODE_FRAME_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_INPUT_MODE_SLICE_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_INPUT_MODE_NON_VCL_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_INPUT_MODE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264InputModeFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264InputModeFlagsEXT;

typedef enum VkVideoEncodeH264OutputModeFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_OUTPUT_MODE_FRAME_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_OUTPUT_MODE_SLICE_BIT_EXT = 0x00000002,
    VK_VIDEO_ENCODE_H264_OUTPUT_MODE_NON_VCL_BIT_EXT = 0x00000004,
    VK_VIDEO_ENCODE_H264_OUTPUT_MODE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264OutputModeFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264OutputModeFlagsEXT;

typedef enum VkVideoEncodeH264CreateFlagBitsEXT {
    VK_VIDEO_ENCODE_H264_CREATE_DEFAULT_EXT = 0,
    VK_VIDEO_ENCODE_H264_CREATE_RESERVED_0_BIT_EXT = 0x00000001,
    VK_VIDEO_ENCODE_H264_CREATE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoEncodeH264CreateFlagBitsEXT;
typedef VkFlags VkVideoEncodeH264CreateFlagsEXT;
typedef struct VkVideoEncodeH264CapabilitiesEXT {
    VkStructureType                        sType;
    const void*                            pNext;
    VkVideoEncodeH264CapabilityFlagsEXT    flags;
    VkVideoEncodeH264InputModeFlagsEXT     inputModeFlags;
    VkVideoEncodeH264OutputModeFlagsEXT    outputModeFlags;
    VkExtent2D                             minPictureSizeInMbs;
    VkExtent2D                             maxPictureSizeInMbs;
    VkExtent2D                             inputImageDataAlignment;
    uint8_t                                maxNumL0ReferenceForP;
    uint8_t                                maxNumL0ReferenceForB;
    uint8_t                                maxNumL1Reference;
    uint8_t                                qualityLevelCount;
    VkExtensionProperties                  stdExtensionVersion;
} VkVideoEncodeH264CapabilitiesEXT;

typedef struct VkVideoEncodeH264SessionCreateInfoEXT {
    VkStructureType                    sType;
    const void*                        pNext;
    VkVideoEncodeH264CreateFlagsEXT    flags;
    VkExtent2D                         maxPictureSizeInMbs;
    const VkExtensionProperties*       pStdExtensionVersion;
} VkVideoEncodeH264SessionCreateInfoEXT;

typedef struct VkVideoEncodeH264SessionParametersAddInfoEXT {
    VkStructureType                            sType;
    const void*                                pNext;
    uint32_t                                   spsStdCount;
    const StdVideoH264SequenceParameterSet*    pSpsStd;
    uint32_t                                   ppsStdCount;
    const StdVideoH264PictureParameterSet*     pPpsStd;
} VkVideoEncodeH264SessionParametersAddInfoEXT;

typedef struct VkVideoEncodeH264SessionParametersCreateInfoEXT {
    VkStructureType                                        sType;
    const void*                                            pNext;
    uint32_t                                               maxSpsStdCount;
    uint32_t                                               maxPpsStdCount;
    const VkVideoEncodeH264SessionParametersAddInfoEXT*    pParametersAddInfo;
} VkVideoEncodeH264SessionParametersCreateInfoEXT;

typedef struct VkVideoEncodeH264DpbSlotInfoEXT {
    VkStructureType                         sType;
    const void*                             pNext;
    int8_t                                  slotIndex;
    const StdVideoEncodeH264PictureInfo*    pStdPictureInfo;
} VkVideoEncodeH264DpbSlotInfoEXT;

typedef struct VkVideoEncodeH264NaluSliceEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    const StdVideoEncodeH264SliceHeader*      pSliceHeaderStd;
    uint32_t                                  mbCount;
    uint8_t                                   refFinalList0EntryCount;
    const VkVideoEncodeH264DpbSlotInfoEXT*    pRefFinalList0Entries;
    uint8_t                                   refFinalList1EntryCount;
    const VkVideoEncodeH264DpbSlotInfoEXT*    pRefFinalList1Entries;
    uint32_t                                  precedingNaluBytes;
    uint8_t                                   minQp;
    uint8_t                                   maxQp;
} VkVideoEncodeH264NaluSliceEXT;

typedef struct VkVideoEncodeH264VclFrameInfoEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    uint8_t                                   refDefaultFinalList0EntryCount;
    const VkVideoEncodeH264DpbSlotInfoEXT*    pRefDefaultFinalList0Entries;
    uint8_t                                   refDefaultFinalList1EntryCount;
    const VkVideoEncodeH264DpbSlotInfoEXT*    pRefDefaultFinalList1Entries;
    uint32_t                                  naluSliceEntryCount;
    const VkVideoEncodeH264NaluSliceEXT*      pNaluSliceEntries;
    const VkVideoEncodeH264DpbSlotInfoEXT*    pCurrentPictureInfo;
} VkVideoEncodeH264VclFrameInfoEXT;

typedef struct VkVideoEncodeH264EmitPictureParametersEXT {
    VkStructureType    sType;
    const void*        pNext;
    uint8_t            spsId;
    VkBool32           emitSpsEnable;
    uint32_t           ppsIdEntryCount;
    const uint8_t*     ppsIdEntries;
} VkVideoEncodeH264EmitPictureParametersEXT;

typedef struct VkVideoEncodeH264ProfileEXT {
    VkStructureType           sType;
    const void*               pNext;
    StdVideoH264ProfileIdc    stdProfileIdc;
} VkVideoEncodeH264ProfileEXT;



#define VK_EXT_video_decode_h264 1
#include "vk_video/vulkan_video_codec_h264std_decode.h"
#define VK_EXT_VIDEO_DECODE_H264_SPEC_VERSION 3
#define VK_EXT_VIDEO_DECODE_H264_EXTENSION_NAME "VK_EXT_video_decode_h264"

typedef enum VkVideoDecodeH264PictureLayoutFlagBitsEXT {
    VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_EXT = 0,
    VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED_LINES_BIT_EXT = 0x00000001,
    VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_EXT = 0x00000002,
    VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkVideoDecodeH264PictureLayoutFlagBitsEXT;
typedef VkFlags VkVideoDecodeH264PictureLayoutFlagsEXT;
typedef VkFlags VkVideoDecodeH264CreateFlagsEXT;
typedef struct VkVideoDecodeH264ProfileEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    StdVideoH264ProfileIdc                    stdProfileIdc;
    VkVideoDecodeH264PictureLayoutFlagsEXT    pictureLayout;
} VkVideoDecodeH264ProfileEXT;

typedef struct VkVideoDecodeH264CapabilitiesEXT {
    VkStructureType          sType;
    void*                    pNext;
    uint32_t                 maxLevel;
    VkOffset2D               fieldOffsetGranularity;
    VkExtensionProperties    stdExtensionVersion;
} VkVideoDecodeH264CapabilitiesEXT;

typedef struct VkVideoDecodeH264SessionCreateInfoEXT {
    VkStructureType                    sType;
    const void*                        pNext;
    VkVideoDecodeH264CreateFlagsEXT    flags;
    const VkExtensionProperties*       pStdExtensionVersion;
} VkVideoDecodeH264SessionCreateInfoEXT;

typedef struct VkVideoDecodeH264SessionParametersAddInfoEXT {
    VkStructureType                            sType;
    const void*                                pNext;
    uint32_t                                   spsStdCount;
    const StdVideoH264SequenceParameterSet*    pSpsStd;
    uint32_t                                   ppsStdCount;
    const StdVideoH264PictureParameterSet*     pPpsStd;
} VkVideoDecodeH264SessionParametersAddInfoEXT;

typedef struct VkVideoDecodeH264SessionParametersCreateInfoEXT {
    VkStructureType                                        sType;
    const void*                                            pNext;
    uint32_t                                               maxSpsStdCount;
    uint32_t                                               maxPpsStdCount;
    const VkVideoDecodeH264SessionParametersAddInfoEXT*    pParametersAddInfo;
} VkVideoDecodeH264SessionParametersCreateInfoEXT;

typedef struct VkVideoDecodeH264PictureInfoEXT {
    VkStructureType                         sType;
    const void*                             pNext;
    const StdVideoDecodeH264PictureInfo*    pStdPictureInfo;
    uint32_t                                slicesCount;
    const uint32_t*                         pSlicesDataOffsets;
} VkVideoDecodeH264PictureInfoEXT;

typedef struct VkVideoDecodeH264MvcEXT {
    VkStructureType                 sType;
    const void*                     pNext;
    const StdVideoDecodeH264Mvc*    pStdMvc;
} VkVideoDecodeH264MvcEXT;

typedef struct VkVideoDecodeH264DpbSlotInfoEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    const StdVideoDecodeH264ReferenceInfo*    pStdReferenceInfo;
} VkVideoDecodeH264DpbSlotInfoEXT;



#define VK_EXT_video_decode_h265 1
#include "vk_video/vulkan_video_codec_h265std.h"
#include "vk_video/vulkan_video_codec_h265std_decode.h"
#define VK_EXT_VIDEO_DECODE_H265_SPEC_VERSION 1
#define VK_EXT_VIDEO_DECODE_H265_EXTENSION_NAME "VK_EXT_video_decode_h265"
typedef VkFlags VkVideoDecodeH265CreateFlagsEXT;
typedef struct VkVideoDecodeH265ProfileEXT {
    VkStructureType           sType;
    const void*               pNext;
    StdVideoH265ProfileIdc    stdProfileIdc;
} VkVideoDecodeH265ProfileEXT;

typedef struct VkVideoDecodeH265CapabilitiesEXT {
    VkStructureType          sType;
    void*                    pNext;
    uint32_t                 maxLevel;
    VkExtensionProperties    stdExtensionVersion;
} VkVideoDecodeH265CapabilitiesEXT;

typedef struct VkVideoDecodeH265SessionCreateInfoEXT {
    VkStructureType                    sType;
    const void*                        pNext;
    VkVideoDecodeH265CreateFlagsEXT    flags;
    const VkExtensionProperties*       pStdExtensionVersion;
} VkVideoDecodeH265SessionCreateInfoEXT;

typedef struct VkVideoDecodeH265SessionParametersAddInfoEXT {
    VkStructureType                            sType;
    const void*                                pNext;
    uint32_t                                   spsStdCount;
    const StdVideoH265SequenceParameterSet*    pSpsStd;
    uint32_t                                   ppsStdCount;
    const StdVideoH265PictureParameterSet*     pPpsStd;
} VkVideoDecodeH265SessionParametersAddInfoEXT;

typedef struct VkVideoDecodeH265SessionParametersCreateInfoEXT {
    VkStructureType                                        sType;
    const void*                                            pNext;
    uint32_t                                               maxSpsStdCount;
    uint32_t                                               maxPpsStdCount;
    const VkVideoDecodeH265SessionParametersAddInfoEXT*    pParametersAddInfo;
} VkVideoDecodeH265SessionParametersCreateInfoEXT;

typedef struct VkVideoDecodeH265PictureInfoEXT {
    VkStructureType                   sType;
    const void*                       pNext;
    StdVideoDecodeH265PictureInfo*    pStdPictureInfo;
    uint32_t                          slicesCount;
    const uint32_t*                   pSlicesDataOffsets;
} VkVideoDecodeH265PictureInfoEXT;

typedef struct VkVideoDecodeH265DpbSlotInfoEXT {
    VkStructureType                           sType;
    const void*                               pNext;
    const StdVideoDecodeH265ReferenceInfo*    pStdReferenceInfo;
} VkVideoDecodeH265DpbSlotInfoEXT;


#ifdef __cplusplus
}
#endif

#endif
