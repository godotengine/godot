#ifndef OPENXR_REFLECTION_PARENT_STRUCTS_H_
#define OPENXR_REFLECTION_PARENT_STRUCTS_H_ 1

/*
** Copyright (c) 2017-2023, The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0 OR MIT
*/

/*
** This header is generated from the Khronos OpenXR XML API Registry.
**
*/

#include "openxr.h"

/*
This file contains expansion macros (X Macros) for OpenXR structures that have a parent type.
*/


/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrCompositionLayerBaseHeader
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrCompositionLayerBaseHeader(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrCompositionLayerBaseHeader_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrCompositionLayerBaseHeader()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrCompositionLayerBaseHeader_CORE(_avail, _unavail) \
    _avail(XrCompositionLayerProjection, XR_TYPE_COMPOSITION_LAYER_PROJECTION) \
    _avail(XrCompositionLayerQuad, XR_TYPE_COMPOSITION_LAYER_QUAD) \
    _avail(XrCompositionLayerCubeKHR, XR_TYPE_COMPOSITION_LAYER_CUBE_KHR) \
    _avail(XrCompositionLayerCylinderKHR, XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR) \
    _avail(XrCompositionLayerEquirectKHR, XR_TYPE_COMPOSITION_LAYER_EQUIRECT_KHR) \
    _avail(XrCompositionLayerEquirect2KHR, XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR) \
    _avail(XrCompositionLayerPassthroughHTC, XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_HTC) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrEventDataBaseHeader
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrEventDataBaseHeader(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrEventDataBaseHeader_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrEventDataBaseHeader()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrEventDataBaseHeader_CORE(_avail, _unavail) \
    _avail(XrEventDataEventsLost, XR_TYPE_EVENT_DATA_EVENTS_LOST) \
    _avail(XrEventDataInstanceLossPending, XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) \
    _avail(XrEventDataSessionStateChanged, XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) \
    _avail(XrEventDataReferenceSpaceChangePending, XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) \
    _avail(XrEventDataInteractionProfileChanged, XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) \
    _avail(XrEventDataVisibilityMaskChangedKHR, XR_TYPE_EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR) \
    _avail(XrEventDataPerfSettingsEXT, XR_TYPE_EVENT_DATA_PERF_SETTINGS_EXT) \
    _avail(XrEventDataMainSessionVisibilityChangedEXTX, XR_TYPE_EVENT_DATA_MAIN_SESSION_VISIBILITY_CHANGED_EXTX) \
    _avail(XrEventDataDisplayRefreshRateChangedFB, XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB) \
    _avail(XrEventDataViveTrackerConnectedHTCX, XR_TYPE_EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX) \
    _avail(XrEventDataSpatialAnchorCreateCompleteFB, XR_TYPE_EVENT_DATA_SPATIAL_ANCHOR_CREATE_COMPLETE_FB) \
    _avail(XrEventDataSpaceSetStatusCompleteFB, XR_TYPE_EVENT_DATA_SPACE_SET_STATUS_COMPLETE_FB) \
    _avail(XrEventDataMarkerTrackingUpdateVARJO, XR_TYPE_EVENT_DATA_MARKER_TRACKING_UPDATE_VARJO) \
    _avail(XrEventDataSpaceQueryResultsAvailableFB, XR_TYPE_EVENT_DATA_SPACE_QUERY_RESULTS_AVAILABLE_FB) \
    _avail(XrEventDataSpaceQueryCompleteFB, XR_TYPE_EVENT_DATA_SPACE_QUERY_COMPLETE_FB) \
    _avail(XrEventDataSpaceSaveCompleteFB, XR_TYPE_EVENT_DATA_SPACE_SAVE_COMPLETE_FB) \
    _avail(XrEventDataSpaceEraseCompleteFB, XR_TYPE_EVENT_DATA_SPACE_ERASE_COMPLETE_FB) \
    _avail(XrEventDataSpaceShareCompleteFB, XR_TYPE_EVENT_DATA_SPACE_SHARE_COMPLETE_FB) \
    _avail(XrEventDataSpaceListSaveCompleteFB, XR_TYPE_EVENT_DATA_SPACE_LIST_SAVE_COMPLETE_FB) \
    _avail(XrEventDataHeadsetFitChangedML, XR_TYPE_EVENT_DATA_HEADSET_FIT_CHANGED_ML) \
    _avail(XrEventDataEyeCalibrationChangedML, XR_TYPE_EVENT_DATA_EYE_CALIBRATION_CHANGED_ML) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrHapticBaseHeader
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrHapticBaseHeader(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrHapticBaseHeader_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrHapticBaseHeader()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrHapticBaseHeader_CORE(_avail, _unavail) \
    _avail(XrHapticVibration, XR_TYPE_HAPTIC_VIBRATION) \
    _avail(XrHapticAmplitudeEnvelopeVibrationFB, XR_TYPE_HAPTIC_AMPLITUDE_ENVELOPE_VIBRATION_FB) \
    _avail(XrHapticPcmVibrationFB, XR_TYPE_HAPTIC_PCM_VIBRATION_FB) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSwapchainImageBaseHeader
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_CORE(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_D3D11(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_D3D12(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_OPENGL(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_OPENGL_ES(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_VULKAN(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_CORE(_avail, _unavail) \


#if defined(XR_USE_GRAPHICS_API_D3D11)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_D3D11(_avail, _unavail) \
    _avail(XrSwapchainImageD3D11KHR, XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_D3D11(_avail, _unavail) \
    _unavail(XrSwapchainImageD3D11KHR, XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR) \

#endif

#if defined(XR_USE_GRAPHICS_API_D3D12)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_D3D12(_avail, _unavail) \
    _avail(XrSwapchainImageD3D12KHR, XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_D3D12(_avail, _unavail) \
    _unavail(XrSwapchainImageD3D12KHR, XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR) \

#endif

#if defined(XR_USE_GRAPHICS_API_OPENGL)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_OPENGL(_avail, _unavail) \
    _avail(XrSwapchainImageOpenGLKHR, XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_OPENGL(_avail, _unavail) \
    _unavail(XrSwapchainImageOpenGLKHR, XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR) \

#endif

#if defined(XR_USE_GRAPHICS_API_OPENGL_ES)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_OPENGL_ES(_avail, _unavail) \
    _avail(XrSwapchainImageOpenGLESKHR, XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_OPENGL_ES(_avail, _unavail) \
    _unavail(XrSwapchainImageOpenGLESKHR, XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR) \

#endif

#if defined(XR_USE_GRAPHICS_API_VULKAN)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_VULKAN(_avail, _unavail) \
    _avail(XrSwapchainImageVulkanKHR, XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_VULKAN(_avail, _unavail) \
    _unavail(XrSwapchainImageVulkanKHR, XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR) \

#endif




/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrLoaderInitInfoBaseHeaderKHR
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR_CORE(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR_XR_USE_PLATFORM_ANDROID(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR_CORE(_avail, _unavail) \


#if defined(XR_USE_PLATFORM_ANDROID)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR_XR_USE_PLATFORM_ANDROID(_avail, _unavail) \
    _avail(XrLoaderInitInfoAndroidKHR, XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrLoaderInitInfoBaseHeaderKHR_XR_USE_PLATFORM_ANDROID(_avail, _unavail) \
    _unavail(XrLoaderInitInfoAndroidKHR, XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR) \

#endif




/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrBindingModificationBaseHeaderKHR
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrBindingModificationBaseHeaderKHR(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrBindingModificationBaseHeaderKHR_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrBindingModificationBaseHeaderKHR()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrBindingModificationBaseHeaderKHR_CORE(_avail, _unavail) \
    _avail(XrInteractionProfileDpadBindingEXT, XR_TYPE_INTERACTION_PROFILE_DPAD_BINDING_EXT) \
    _avail(XrInteractionProfileAnalogThresholdVALVE, XR_TYPE_INTERACTION_PROFILE_ANALOG_THRESHOLD_VALVE) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSwapchainStateBaseHeaderFB
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_CORE(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_GRAPHICS_API_OPENGL_ES(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_GRAPHICS_API_VULKAN(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_PLATFORM_ANDROID(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_CORE(_avail, _unavail) \
    _avail(XrSwapchainStateFoveationFB, XR_TYPE_SWAPCHAIN_STATE_FOVEATION_FB) \


#if defined(XR_USE_GRAPHICS_API_OPENGL_ES)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_GRAPHICS_API_OPENGL_ES(_avail, _unavail) \
    _avail(XrSwapchainStateSamplerOpenGLESFB, XR_TYPE_SWAPCHAIN_STATE_SAMPLER_OPENGL_ES_FB) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_GRAPHICS_API_OPENGL_ES(_avail, _unavail) \
    _unavail(XrSwapchainStateSamplerOpenGLESFB, XR_TYPE_SWAPCHAIN_STATE_SAMPLER_OPENGL_ES_FB) \

#endif

#if defined(XR_USE_GRAPHICS_API_VULKAN)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_GRAPHICS_API_VULKAN(_avail, _unavail) \
    _avail(XrSwapchainStateSamplerVulkanFB, XR_TYPE_SWAPCHAIN_STATE_SAMPLER_VULKAN_FB) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_GRAPHICS_API_VULKAN(_avail, _unavail) \
    _unavail(XrSwapchainStateSamplerVulkanFB, XR_TYPE_SWAPCHAIN_STATE_SAMPLER_VULKAN_FB) \

#endif

#if defined(XR_USE_PLATFORM_ANDROID)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_PLATFORM_ANDROID(_avail, _unavail) \
    _avail(XrSwapchainStateAndroidSurfaceDimensionsFB, XR_TYPE_SWAPCHAIN_STATE_ANDROID_SURFACE_DIMENSIONS_FB) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainStateBaseHeaderFB_XR_USE_PLATFORM_ANDROID(_avail, _unavail) \
    _unavail(XrSwapchainStateAndroidSurfaceDimensionsFB, XR_TYPE_SWAPCHAIN_STATE_ANDROID_SURFACE_DIMENSIONS_FB) \

#endif




/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSpaceQueryInfoBaseHeaderFB
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceQueryInfoBaseHeaderFB(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceQueryInfoBaseHeaderFB_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceQueryInfoBaseHeaderFB()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceQueryInfoBaseHeaderFB_CORE(_avail, _unavail) \
    _avail(XrSpaceQueryInfoFB, XR_TYPE_SPACE_QUERY_INFO_FB) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSpaceFilterInfoBaseHeaderFB
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterInfoBaseHeaderFB(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterInfoBaseHeaderFB_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterInfoBaseHeaderFB()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterInfoBaseHeaderFB_CORE(_avail, _unavail) \
    _avail(XrSpaceUuidFilterInfoFB, XR_TYPE_SPACE_UUID_FILTER_INFO_FB) \
    _avail(XrSpaceComponentFilterInfoFB, XR_TYPE_SPACE_COMPONENT_FILTER_INFO_FB) \





#endif

