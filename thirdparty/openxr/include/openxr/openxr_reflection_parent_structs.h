#ifndef OPENXR_REFLECTION_PARENT_STRUCTS_H_
#define OPENXR_REFLECTION_PARENT_STRUCTS_H_ 1

/*
** Copyright (c) 2017-2025 The Khronos Group Inc.
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
    _avail(XrCompositionLayerPassthroughFB, XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB) \
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
    _avail(XrEventDataPassthroughStateChangedFB, XR_TYPE_EVENT_DATA_PASSTHROUGH_STATE_CHANGED_FB) \
    _avail(XrEventDataMarkerTrackingUpdateVARJO, XR_TYPE_EVENT_DATA_MARKER_TRACKING_UPDATE_VARJO) \
    _avail(XrEventDataLocalizationChangedML, XR_TYPE_EVENT_DATA_LOCALIZATION_CHANGED_ML) \
    _avail(XrEventDataSpaceQueryResultsAvailableFB, XR_TYPE_EVENT_DATA_SPACE_QUERY_RESULTS_AVAILABLE_FB) \
    _avail(XrEventDataSpaceQueryCompleteFB, XR_TYPE_EVENT_DATA_SPACE_QUERY_COMPLETE_FB) \
    _avail(XrEventDataSpaceSaveCompleteFB, XR_TYPE_EVENT_DATA_SPACE_SAVE_COMPLETE_FB) \
    _avail(XrEventDataSpaceEraseCompleteFB, XR_TYPE_EVENT_DATA_SPACE_ERASE_COMPLETE_FB) \
    _avail(XrEventDataSpaceShareCompleteFB, XR_TYPE_EVENT_DATA_SPACE_SHARE_COMPLETE_FB) \
    _avail(XrEventDataSceneCaptureCompleteFB, XR_TYPE_EVENT_DATA_SCENE_CAPTURE_COMPLETE_FB) \
    _avail(XrEventDataVirtualKeyboardCommitTextMETA, XR_TYPE_EVENT_DATA_VIRTUAL_KEYBOARD_COMMIT_TEXT_META) \
    _avail(XrEventDataVirtualKeyboardBackspaceMETA, XR_TYPE_EVENT_DATA_VIRTUAL_KEYBOARD_BACKSPACE_META) \
    _avail(XrEventDataVirtualKeyboardEnterMETA, XR_TYPE_EVENT_DATA_VIRTUAL_KEYBOARD_ENTER_META) \
    _avail(XrEventDataVirtualKeyboardShownMETA, XR_TYPE_EVENT_DATA_VIRTUAL_KEYBOARD_SHOWN_META) \
    _avail(XrEventDataVirtualKeyboardHiddenMETA, XR_TYPE_EVENT_DATA_VIRTUAL_KEYBOARD_HIDDEN_META) \
    _avail(XrEventDataSpaceListSaveCompleteFB, XR_TYPE_EVENT_DATA_SPACE_LIST_SAVE_COMPLETE_FB) \
    _avail(XrEventDataSpaceDiscoveryResultsAvailableMETA, XR_TYPE_EVENT_DATA_SPACE_DISCOVERY_RESULTS_AVAILABLE_META) \
    _avail(XrEventDataSpaceDiscoveryCompleteMETA, XR_TYPE_EVENT_DATA_SPACE_DISCOVERY_COMPLETE_META) \
    _avail(XrEventDataSpacesSaveResultMETA, XR_TYPE_EVENT_DATA_SPACES_SAVE_RESULT_META) \
    _avail(XrEventDataSpacesEraseResultMETA, XR_TYPE_EVENT_DATA_SPACES_ERASE_RESULT_META) \
    _avail(XrEventDataPassthroughLayerResumedMETA, XR_TYPE_EVENT_DATA_PASSTHROUGH_LAYER_RESUMED_META) \
    _avail(XrEventDataShareSpacesCompleteMETA, XR_TYPE_EVENT_DATA_SHARE_SPACES_COMPLETE_META) \
    _avail(XrEventDataInteractionRenderModelsChangedEXT, XR_TYPE_EVENT_DATA_INTERACTION_RENDER_MODELS_CHANGED_EXT) \
    _avail(XrEventDataSenseDataProviderStateChangedBD, XR_TYPE_EVENT_DATA_SENSE_DATA_PROVIDER_STATE_CHANGED_BD) \
    _avail(XrEventDataSenseDataUpdatedBD, XR_TYPE_EVENT_DATA_SENSE_DATA_UPDATED_BD) \
    _avail(XrEventDataUserPresenceChangedEXT, XR_TYPE_EVENT_DATA_USER_PRESENCE_CHANGED_EXT) \
    _avail(XrEventDataHeadsetFitChangedML, XR_TYPE_EVENT_DATA_HEADSET_FIT_CHANGED_ML) \
    _avail(XrEventDataEyeCalibrationChangedML, XR_TYPE_EVENT_DATA_EYE_CALIBRATION_CHANGED_ML) \
    _avail(XrEventDataStartColocationAdvertisementCompleteMETA, XR_TYPE_EVENT_DATA_START_COLOCATION_ADVERTISEMENT_COMPLETE_META) \
    _avail(XrEventDataStopColocationAdvertisementCompleteMETA, XR_TYPE_EVENT_DATA_STOP_COLOCATION_ADVERTISEMENT_COMPLETE_META) \
    _avail(XrEventDataColocationAdvertisementCompleteMETA, XR_TYPE_EVENT_DATA_COLOCATION_ADVERTISEMENT_COMPLETE_META) \
    _avail(XrEventDataStartColocationDiscoveryCompleteMETA, XR_TYPE_EVENT_DATA_START_COLOCATION_DISCOVERY_COMPLETE_META) \
    _avail(XrEventDataColocationDiscoveryResultMETA, XR_TYPE_EVENT_DATA_COLOCATION_DISCOVERY_RESULT_META) \
    _avail(XrEventDataColocationDiscoveryCompleteMETA, XR_TYPE_EVENT_DATA_COLOCATION_DISCOVERY_COMPLETE_META) \
    _avail(XrEventDataStopColocationDiscoveryCompleteMETA, XR_TYPE_EVENT_DATA_STOP_COLOCATION_DISCOVERY_COMPLETE_META) \
    _avail(XrEventDataSpatialDiscoveryRecommendedEXT, XR_TYPE_EVENT_DATA_SPATIAL_DISCOVERY_RECOMMENDED_EXT) \





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
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_METAL(_avail, _unavail) \
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

#if defined(XR_USE_GRAPHICS_API_METAL)
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_METAL(_avail, _unavail) \
    _avail(XrSwapchainImageMetalKHR, XR_TYPE_SWAPCHAIN_IMAGE_METAL_KHR) \

#else
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSwapchainImageBaseHeader_XR_USE_GRAPHICS_API_METAL(_avail, _unavail) \
    _unavail(XrSwapchainImageMetalKHR, XR_TYPE_SWAPCHAIN_IMAGE_METAL_KHR) \

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
    _avail(XrLoaderInitInfoPropertiesEXT, XR_TYPE_LOADER_INIT_INFO_PROPERTIES_EXT) \


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




/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSpatialAnchorsCreateInfoBaseHeaderML
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsCreateInfoBaseHeaderML(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsCreateInfoBaseHeaderML_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsCreateInfoBaseHeaderML()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsCreateInfoBaseHeaderML_CORE(_avail, _unavail) \
    _avail(XrSpatialAnchorsCreateInfoFromPoseML, XR_TYPE_SPATIAL_ANCHORS_CREATE_INFO_FROM_POSE_ML) \
    _avail(XrSpatialAnchorsCreateInfoFromUuidsML, XR_TYPE_SPATIAL_ANCHORS_CREATE_INFO_FROM_UUIDS_ML) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrFutureCompletionBaseHeaderEXT
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrFutureCompletionBaseHeaderEXT(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrFutureCompletionBaseHeaderEXT_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrFutureCompletionBaseHeaderEXT()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrFutureCompletionBaseHeaderEXT_CORE(_avail, _unavail) \
    _avail(XrCreateSpatialAnchorsCompletionML, XR_TYPE_CREATE_SPATIAL_ANCHORS_COMPLETION_ML) \
    _avail(XrSpatialAnchorsQueryCompletionML, XR_TYPE_SPATIAL_ANCHORS_QUERY_COMPLETION_ML) \
    _avail(XrSpatialAnchorsPublishCompletionML, XR_TYPE_SPATIAL_ANCHORS_PUBLISH_COMPLETION_ML) \
    _avail(XrSpatialAnchorsDeleteCompletionML, XR_TYPE_SPATIAL_ANCHORS_DELETE_COMPLETION_ML) \
    _avail(XrSpatialAnchorsUpdateExpirationCompletionML, XR_TYPE_SPATIAL_ANCHORS_UPDATE_EXPIRATION_COMPLETION_ML) \
    _avail(XrSenseDataQueryCompletionBD, XR_TYPE_SENSE_DATA_QUERY_COMPLETION_BD) \
    _avail(XrFutureCompletionEXT, XR_TYPE_FUTURE_COMPLETION_EXT) \
    _avail(XrSpatialAnchorCreateCompletionBD, XR_TYPE_SPATIAL_ANCHOR_CREATE_COMPLETION_BD) \
    _avail(XrWorldMeshStateRequestCompletionML, XR_TYPE_WORLD_MESH_STATE_REQUEST_COMPLETION_ML) \
    _avail(XrWorldMeshRequestCompletionML, XR_TYPE_WORLD_MESH_REQUEST_COMPLETION_ML) \
    _avail(XrCreateSpatialContextCompletionEXT, XR_TYPE_CREATE_SPATIAL_CONTEXT_COMPLETION_EXT) \
    _avail(XrCreateSpatialDiscoverySnapshotCompletionEXT, XR_TYPE_CREATE_SPATIAL_DISCOVERY_SNAPSHOT_COMPLETION_EXT) \
    _avail(XrCreateSpatialPersistenceContextCompletionEXT, XR_TYPE_CREATE_SPATIAL_PERSISTENCE_CONTEXT_COMPLETION_EXT) \
    _avail(XrPersistSpatialEntityCompletionEXT, XR_TYPE_PERSIST_SPATIAL_ENTITY_COMPLETION_EXT) \
    _avail(XrUnpersistSpatialEntityCompletionEXT, XR_TYPE_UNPERSIST_SPATIAL_ENTITY_COMPLETION_EXT) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSpatialAnchorsQueryInfoBaseHeaderML
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsQueryInfoBaseHeaderML(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsQueryInfoBaseHeaderML_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsQueryInfoBaseHeaderML()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialAnchorsQueryInfoBaseHeaderML_CORE(_avail, _unavail) \
    _avail(XrSpatialAnchorsQueryInfoRadiusML, XR_TYPE_SPATIAL_ANCHORS_QUERY_INFO_RADIUS_ML) \





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





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSpaceFilterBaseHeaderMETA
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterBaseHeaderMETA(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterBaseHeaderMETA_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterBaseHeaderMETA()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpaceFilterBaseHeaderMETA_CORE(_avail, _unavail) \
    _avail(XrSpaceFilterUuidMETA, XR_TYPE_SPACE_FILTER_UUID_META) \
    _avail(XrSpaceFilterComponentMETA, XR_TYPE_SPACE_FILTER_COMPONENT_META) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrShareSpacesRecipientBaseHeaderMETA
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrShareSpacesRecipientBaseHeaderMETA(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrShareSpacesRecipientBaseHeaderMETA_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrShareSpacesRecipientBaseHeaderMETA()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrShareSpacesRecipientBaseHeaderMETA_CORE(_avail, _unavail) \
    _avail(XrShareSpacesRecipientGroupsMETA, XR_TYPE_SHARE_SPACES_RECIPIENT_GROUPS_META) \





/// Like XR_LIST_ALL_STRUCTURE_TYPES, but only includes types whose parent struct type is XrSpatialCapabilityConfigurationBaseHeaderEXT
#define XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialCapabilityConfigurationBaseHeaderEXT(_avail, _unavail) \
    _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialCapabilityConfigurationBaseHeaderEXT_CORE(_avail, _unavail) \


// Implementation detail of XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialCapabilityConfigurationBaseHeaderEXT()
#define _impl_XR_LIST_ALL_CHILD_STRUCTURE_TYPES_XrSpatialCapabilityConfigurationBaseHeaderEXT_CORE(_avail, _unavail) \
    _avail(XrSpatialCapabilityConfigurationPlaneTrackingEXT, XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_PLANE_TRACKING_EXT) \
    _avail(XrSpatialCapabilityConfigurationQrCodeEXT, XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_QR_CODE_EXT) \
    _avail(XrSpatialCapabilityConfigurationMicroQrCodeEXT, XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_MICRO_QR_CODE_EXT) \
    _avail(XrSpatialCapabilityConfigurationArucoMarkerEXT, XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_ARUCO_MARKER_EXT) \
    _avail(XrSpatialCapabilityConfigurationAprilTagEXT, XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_APRIL_TAG_EXT) \
    _avail(XrSpatialCapabilityConfigurationAnchorEXT, XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_ANCHOR_EXT) \





#endif

