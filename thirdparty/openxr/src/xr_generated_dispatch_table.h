// Copyright (c) 2017-2023, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
// SPDX-License-Identifier: Apache-2.0 OR MIT
// *********** THIS FILE IS GENERATED - DO NOT EDIT ***********
//     See utility_source_generator.py for modifications
// ************************************************************

// Copyright (c) 2017-2023, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0
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
// Author: Mark Young <marky@lunarg.com>
//

#pragma once
#include "xr_dependencies.h"
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>


#ifdef __cplusplus
extern "C" { 
#endif
// Generated dispatch table
struct XrGeneratedDispatchTable {

    // ---- Core 1.0 commands
    PFN_xrGetInstanceProcAddr GetInstanceProcAddr;
    PFN_xrEnumerateApiLayerProperties EnumerateApiLayerProperties;
    PFN_xrEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_xrCreateInstance CreateInstance;
    PFN_xrDestroyInstance DestroyInstance;
    PFN_xrGetInstanceProperties GetInstanceProperties;
    PFN_xrPollEvent PollEvent;
    PFN_xrResultToString ResultToString;
    PFN_xrStructureTypeToString StructureTypeToString;
    PFN_xrGetSystem GetSystem;
    PFN_xrGetSystemProperties GetSystemProperties;
    PFN_xrEnumerateEnvironmentBlendModes EnumerateEnvironmentBlendModes;
    PFN_xrCreateSession CreateSession;
    PFN_xrDestroySession DestroySession;
    PFN_xrEnumerateReferenceSpaces EnumerateReferenceSpaces;
    PFN_xrCreateReferenceSpace CreateReferenceSpace;
    PFN_xrGetReferenceSpaceBoundsRect GetReferenceSpaceBoundsRect;
    PFN_xrCreateActionSpace CreateActionSpace;
    PFN_xrLocateSpace LocateSpace;
    PFN_xrDestroySpace DestroySpace;
    PFN_xrEnumerateViewConfigurations EnumerateViewConfigurations;
    PFN_xrGetViewConfigurationProperties GetViewConfigurationProperties;
    PFN_xrEnumerateViewConfigurationViews EnumerateViewConfigurationViews;
    PFN_xrEnumerateSwapchainFormats EnumerateSwapchainFormats;
    PFN_xrCreateSwapchain CreateSwapchain;
    PFN_xrDestroySwapchain DestroySwapchain;
    PFN_xrEnumerateSwapchainImages EnumerateSwapchainImages;
    PFN_xrAcquireSwapchainImage AcquireSwapchainImage;
    PFN_xrWaitSwapchainImage WaitSwapchainImage;
    PFN_xrReleaseSwapchainImage ReleaseSwapchainImage;
    PFN_xrBeginSession BeginSession;
    PFN_xrEndSession EndSession;
    PFN_xrRequestExitSession RequestExitSession;
    PFN_xrWaitFrame WaitFrame;
    PFN_xrBeginFrame BeginFrame;
    PFN_xrEndFrame EndFrame;
    PFN_xrLocateViews LocateViews;
    PFN_xrStringToPath StringToPath;
    PFN_xrPathToString PathToString;
    PFN_xrCreateActionSet CreateActionSet;
    PFN_xrDestroyActionSet DestroyActionSet;
    PFN_xrCreateAction CreateAction;
    PFN_xrDestroyAction DestroyAction;
    PFN_xrSuggestInteractionProfileBindings SuggestInteractionProfileBindings;
    PFN_xrAttachSessionActionSets AttachSessionActionSets;
    PFN_xrGetCurrentInteractionProfile GetCurrentInteractionProfile;
    PFN_xrGetActionStateBoolean GetActionStateBoolean;
    PFN_xrGetActionStateFloat GetActionStateFloat;
    PFN_xrGetActionStateVector2f GetActionStateVector2f;
    PFN_xrGetActionStatePose GetActionStatePose;
    PFN_xrSyncActions SyncActions;
    PFN_xrEnumerateBoundSourcesForAction EnumerateBoundSourcesForAction;
    PFN_xrGetInputSourceLocalizedName GetInputSourceLocalizedName;
    PFN_xrApplyHapticFeedback ApplyHapticFeedback;
    PFN_xrStopHapticFeedback StopHapticFeedback;

    // ---- XR_KHR_android_thread_settings extension commands
#if defined(XR_USE_PLATFORM_ANDROID)
    PFN_xrSetAndroidApplicationThreadKHR SetAndroidApplicationThreadKHR;
#endif // defined(XR_USE_PLATFORM_ANDROID)

    // ---- XR_KHR_android_surface_swapchain extension commands
#if defined(XR_USE_PLATFORM_ANDROID)
    PFN_xrCreateSwapchainAndroidSurfaceKHR CreateSwapchainAndroidSurfaceKHR;
#endif // defined(XR_USE_PLATFORM_ANDROID)

    // ---- XR_KHR_opengl_enable extension commands
#if defined(XR_USE_GRAPHICS_API_OPENGL)
    PFN_xrGetOpenGLGraphicsRequirementsKHR GetOpenGLGraphicsRequirementsKHR;
#endif // defined(XR_USE_GRAPHICS_API_OPENGL)

    // ---- XR_KHR_opengl_es_enable extension commands
#if defined(XR_USE_GRAPHICS_API_OPENGL_ES)
    PFN_xrGetOpenGLESGraphicsRequirementsKHR GetOpenGLESGraphicsRequirementsKHR;
#endif // defined(XR_USE_GRAPHICS_API_OPENGL_ES)

    // ---- XR_KHR_vulkan_enable extension commands
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrGetVulkanInstanceExtensionsKHR GetVulkanInstanceExtensionsKHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrGetVulkanDeviceExtensionsKHR GetVulkanDeviceExtensionsKHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrGetVulkanGraphicsDeviceKHR GetVulkanGraphicsDeviceKHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrGetVulkanGraphicsRequirementsKHR GetVulkanGraphicsRequirementsKHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)

    // ---- XR_KHR_D3D11_enable extension commands
#if defined(XR_USE_GRAPHICS_API_D3D11)
    PFN_xrGetD3D11GraphicsRequirementsKHR GetD3D11GraphicsRequirementsKHR;
#endif // defined(XR_USE_GRAPHICS_API_D3D11)

    // ---- XR_KHR_D3D12_enable extension commands
#if defined(XR_USE_GRAPHICS_API_D3D12)
    PFN_xrGetD3D12GraphicsRequirementsKHR GetD3D12GraphicsRequirementsKHR;
#endif // defined(XR_USE_GRAPHICS_API_D3D12)

    // ---- XR_KHR_visibility_mask extension commands
    PFN_xrGetVisibilityMaskKHR GetVisibilityMaskKHR;

    // ---- XR_KHR_win32_convert_performance_counter_time extension commands
#if defined(XR_USE_PLATFORM_WIN32)
    PFN_xrConvertWin32PerformanceCounterToTimeKHR ConvertWin32PerformanceCounterToTimeKHR;
#endif // defined(XR_USE_PLATFORM_WIN32)
#if defined(XR_USE_PLATFORM_WIN32)
    PFN_xrConvertTimeToWin32PerformanceCounterKHR ConvertTimeToWin32PerformanceCounterKHR;
#endif // defined(XR_USE_PLATFORM_WIN32)

    // ---- XR_KHR_convert_timespec_time extension commands
#if defined(XR_USE_TIMESPEC)
    PFN_xrConvertTimespecTimeToTimeKHR ConvertTimespecTimeToTimeKHR;
#endif // defined(XR_USE_TIMESPEC)
#if defined(XR_USE_TIMESPEC)
    PFN_xrConvertTimeToTimespecTimeKHR ConvertTimeToTimespecTimeKHR;
#endif // defined(XR_USE_TIMESPEC)

    // ---- XR_KHR_loader_init extension commands
    PFN_xrInitializeLoaderKHR InitializeLoaderKHR;

    // ---- XR_KHR_vulkan_enable2 extension commands
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrCreateVulkanInstanceKHR CreateVulkanInstanceKHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrCreateVulkanDeviceKHR CreateVulkanDeviceKHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrGetVulkanGraphicsDevice2KHR GetVulkanGraphicsDevice2KHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    PFN_xrGetVulkanGraphicsRequirements2KHR GetVulkanGraphicsRequirements2KHR;
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)

    // ---- XR_EXT_performance_settings extension commands
    PFN_xrPerfSettingsSetPerformanceLevelEXT PerfSettingsSetPerformanceLevelEXT;

    // ---- XR_EXT_thermal_query extension commands
    PFN_xrThermalGetTemperatureTrendEXT ThermalGetTemperatureTrendEXT;

    // ---- XR_EXT_debug_utils extension commands
    PFN_xrSetDebugUtilsObjectNameEXT SetDebugUtilsObjectNameEXT;
    PFN_xrCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT;
    PFN_xrDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT;
    PFN_xrSubmitDebugUtilsMessageEXT SubmitDebugUtilsMessageEXT;
    PFN_xrSessionBeginDebugUtilsLabelRegionEXT SessionBeginDebugUtilsLabelRegionEXT;
    PFN_xrSessionEndDebugUtilsLabelRegionEXT SessionEndDebugUtilsLabelRegionEXT;
    PFN_xrSessionInsertDebugUtilsLabelEXT SessionInsertDebugUtilsLabelEXT;

    // ---- XR_MSFT_spatial_anchor extension commands
    PFN_xrCreateSpatialAnchorMSFT CreateSpatialAnchorMSFT;
    PFN_xrCreateSpatialAnchorSpaceMSFT CreateSpatialAnchorSpaceMSFT;
    PFN_xrDestroySpatialAnchorMSFT DestroySpatialAnchorMSFT;

    // ---- XR_EXT_conformance_automation extension commands
    PFN_xrSetInputDeviceActiveEXT SetInputDeviceActiveEXT;
    PFN_xrSetInputDeviceStateBoolEXT SetInputDeviceStateBoolEXT;
    PFN_xrSetInputDeviceStateFloatEXT SetInputDeviceStateFloatEXT;
    PFN_xrSetInputDeviceStateVector2fEXT SetInputDeviceStateVector2fEXT;
    PFN_xrSetInputDeviceLocationEXT SetInputDeviceLocationEXT;

    // ---- XR_MSFT_spatial_graph_bridge extension commands
    PFN_xrCreateSpatialGraphNodeSpaceMSFT CreateSpatialGraphNodeSpaceMSFT;
    PFN_xrTryCreateSpatialGraphStaticNodeBindingMSFT TryCreateSpatialGraphStaticNodeBindingMSFT;
    PFN_xrDestroySpatialGraphNodeBindingMSFT DestroySpatialGraphNodeBindingMSFT;
    PFN_xrGetSpatialGraphNodeBindingPropertiesMSFT GetSpatialGraphNodeBindingPropertiesMSFT;

    // ---- XR_EXT_hand_tracking extension commands
    PFN_xrCreateHandTrackerEXT CreateHandTrackerEXT;
    PFN_xrDestroyHandTrackerEXT DestroyHandTrackerEXT;
    PFN_xrLocateHandJointsEXT LocateHandJointsEXT;

    // ---- XR_MSFT_hand_tracking_mesh extension commands
    PFN_xrCreateHandMeshSpaceMSFT CreateHandMeshSpaceMSFT;
    PFN_xrUpdateHandMeshMSFT UpdateHandMeshMSFT;

    // ---- XR_MSFT_controller_model extension commands
    PFN_xrGetControllerModelKeyMSFT GetControllerModelKeyMSFT;
    PFN_xrLoadControllerModelMSFT LoadControllerModelMSFT;
    PFN_xrGetControllerModelPropertiesMSFT GetControllerModelPropertiesMSFT;
    PFN_xrGetControllerModelStateMSFT GetControllerModelStateMSFT;

    // ---- XR_MSFT_perception_anchor_interop extension commands
#if defined(XR_USE_PLATFORM_WIN32)
    PFN_xrCreateSpatialAnchorFromPerceptionAnchorMSFT CreateSpatialAnchorFromPerceptionAnchorMSFT;
#endif // defined(XR_USE_PLATFORM_WIN32)
#if defined(XR_USE_PLATFORM_WIN32)
    PFN_xrTryGetPerceptionAnchorFromSpatialAnchorMSFT TryGetPerceptionAnchorFromSpatialAnchorMSFT;
#endif // defined(XR_USE_PLATFORM_WIN32)

    // ---- XR_MSFT_composition_layer_reprojection extension commands
    PFN_xrEnumerateReprojectionModesMSFT EnumerateReprojectionModesMSFT;

    // ---- XR_FB_swapchain_update_state extension commands
    PFN_xrUpdateSwapchainFB UpdateSwapchainFB;
    PFN_xrGetSwapchainStateFB GetSwapchainStateFB;

    // ---- XR_FB_body_tracking extension commands
    PFN_xrCreateBodyTrackerFB CreateBodyTrackerFB;
    PFN_xrDestroyBodyTrackerFB DestroyBodyTrackerFB;
    PFN_xrLocateBodyJointsFB LocateBodyJointsFB;
    PFN_xrGetBodySkeletonFB GetBodySkeletonFB;

    // ---- XR_MSFT_scene_understanding extension commands
    PFN_xrEnumerateSceneComputeFeaturesMSFT EnumerateSceneComputeFeaturesMSFT;
    PFN_xrCreateSceneObserverMSFT CreateSceneObserverMSFT;
    PFN_xrDestroySceneObserverMSFT DestroySceneObserverMSFT;
    PFN_xrCreateSceneMSFT CreateSceneMSFT;
    PFN_xrDestroySceneMSFT DestroySceneMSFT;
    PFN_xrComputeNewSceneMSFT ComputeNewSceneMSFT;
    PFN_xrGetSceneComputeStateMSFT GetSceneComputeStateMSFT;
    PFN_xrGetSceneComponentsMSFT GetSceneComponentsMSFT;
    PFN_xrLocateSceneComponentsMSFT LocateSceneComponentsMSFT;
    PFN_xrGetSceneMeshBuffersMSFT GetSceneMeshBuffersMSFT;

    // ---- XR_MSFT_scene_understanding_serialization extension commands
    PFN_xrDeserializeSceneMSFT DeserializeSceneMSFT;
    PFN_xrGetSerializedSceneFragmentDataMSFT GetSerializedSceneFragmentDataMSFT;

    // ---- XR_FB_display_refresh_rate extension commands
    PFN_xrEnumerateDisplayRefreshRatesFB EnumerateDisplayRefreshRatesFB;
    PFN_xrGetDisplayRefreshRateFB GetDisplayRefreshRateFB;
    PFN_xrRequestDisplayRefreshRateFB RequestDisplayRefreshRateFB;

    // ---- XR_HTCX_vive_tracker_interaction extension commands
    PFN_xrEnumerateViveTrackerPathsHTCX EnumerateViveTrackerPathsHTCX;

    // ---- XR_HTC_facial_tracking extension commands
    PFN_xrCreateFacialTrackerHTC CreateFacialTrackerHTC;
    PFN_xrDestroyFacialTrackerHTC DestroyFacialTrackerHTC;
    PFN_xrGetFacialExpressionsHTC GetFacialExpressionsHTC;

    // ---- XR_FB_color_space extension commands
    PFN_xrEnumerateColorSpacesFB EnumerateColorSpacesFB;
    PFN_xrSetColorSpaceFB SetColorSpaceFB;

    // ---- XR_FB_hand_tracking_mesh extension commands
    PFN_xrGetHandMeshFB GetHandMeshFB;

    // ---- XR_FB_spatial_entity extension commands
    PFN_xrCreateSpatialAnchorFB CreateSpatialAnchorFB;
    PFN_xrGetSpaceUuidFB GetSpaceUuidFB;
    PFN_xrEnumerateSpaceSupportedComponentsFB EnumerateSpaceSupportedComponentsFB;
    PFN_xrSetSpaceComponentStatusFB SetSpaceComponentStatusFB;
    PFN_xrGetSpaceComponentStatusFB GetSpaceComponentStatusFB;

    // ---- XR_FB_foveation extension commands
    PFN_xrCreateFoveationProfileFB CreateFoveationProfileFB;
    PFN_xrDestroyFoveationProfileFB DestroyFoveationProfileFB;

    // ---- XR_FB_keyboard_tracking extension commands
    PFN_xrQuerySystemTrackedKeyboardFB QuerySystemTrackedKeyboardFB;
    PFN_xrCreateKeyboardSpaceFB CreateKeyboardSpaceFB;

    // ---- XR_FB_triangle_mesh extension commands
    PFN_xrCreateTriangleMeshFB CreateTriangleMeshFB;
    PFN_xrDestroyTriangleMeshFB DestroyTriangleMeshFB;
    PFN_xrTriangleMeshGetVertexBufferFB TriangleMeshGetVertexBufferFB;
    PFN_xrTriangleMeshGetIndexBufferFB TriangleMeshGetIndexBufferFB;
    PFN_xrTriangleMeshBeginUpdateFB TriangleMeshBeginUpdateFB;
    PFN_xrTriangleMeshEndUpdateFB TriangleMeshEndUpdateFB;
    PFN_xrTriangleMeshBeginVertexBufferUpdateFB TriangleMeshBeginVertexBufferUpdateFB;
    PFN_xrTriangleMeshEndVertexBufferUpdateFB TriangleMeshEndVertexBufferUpdateFB;

    // ---- XR_FB_passthrough extension commands
    PFN_xrCreatePassthroughFB CreatePassthroughFB;
    PFN_xrDestroyPassthroughFB DestroyPassthroughFB;
    PFN_xrPassthroughStartFB PassthroughStartFB;
    PFN_xrPassthroughPauseFB PassthroughPauseFB;
    PFN_xrCreatePassthroughLayerFB CreatePassthroughLayerFB;
    PFN_xrDestroyPassthroughLayerFB DestroyPassthroughLayerFB;
    PFN_xrPassthroughLayerPauseFB PassthroughLayerPauseFB;
    PFN_xrPassthroughLayerResumeFB PassthroughLayerResumeFB;
    PFN_xrPassthroughLayerSetStyleFB PassthroughLayerSetStyleFB;
    PFN_xrCreateGeometryInstanceFB CreateGeometryInstanceFB;
    PFN_xrDestroyGeometryInstanceFB DestroyGeometryInstanceFB;
    PFN_xrGeometryInstanceSetTransformFB GeometryInstanceSetTransformFB;

    // ---- XR_FB_render_model extension commands
    PFN_xrEnumerateRenderModelPathsFB EnumerateRenderModelPathsFB;
    PFN_xrGetRenderModelPropertiesFB GetRenderModelPropertiesFB;
    PFN_xrLoadRenderModelFB LoadRenderModelFB;

    // ---- XR_VARJO_environment_depth_estimation extension commands
    PFN_xrSetEnvironmentDepthEstimationVARJO SetEnvironmentDepthEstimationVARJO;

    // ---- XR_VARJO_marker_tracking extension commands
    PFN_xrSetMarkerTrackingVARJO SetMarkerTrackingVARJO;
    PFN_xrSetMarkerTrackingTimeoutVARJO SetMarkerTrackingTimeoutVARJO;
    PFN_xrSetMarkerTrackingPredictionVARJO SetMarkerTrackingPredictionVARJO;
    PFN_xrGetMarkerSizeVARJO GetMarkerSizeVARJO;
    PFN_xrCreateMarkerSpaceVARJO CreateMarkerSpaceVARJO;

    // ---- XR_VARJO_view_offset extension commands
    PFN_xrSetViewOffsetVARJO SetViewOffsetVARJO;

    // ---- XR_ML_compat extension commands
#if defined(XR_USE_PLATFORM_ML)
    PFN_xrCreateSpaceFromCoordinateFrameUIDML CreateSpaceFromCoordinateFrameUIDML;
#endif // defined(XR_USE_PLATFORM_ML)

    // ---- XR_MSFT_spatial_anchor_persistence extension commands
    PFN_xrCreateSpatialAnchorStoreConnectionMSFT CreateSpatialAnchorStoreConnectionMSFT;
    PFN_xrDestroySpatialAnchorStoreConnectionMSFT DestroySpatialAnchorStoreConnectionMSFT;
    PFN_xrPersistSpatialAnchorMSFT PersistSpatialAnchorMSFT;
    PFN_xrEnumeratePersistedSpatialAnchorNamesMSFT EnumeratePersistedSpatialAnchorNamesMSFT;
    PFN_xrCreateSpatialAnchorFromPersistedNameMSFT CreateSpatialAnchorFromPersistedNameMSFT;
    PFN_xrUnpersistSpatialAnchorMSFT UnpersistSpatialAnchorMSFT;
    PFN_xrClearSpatialAnchorStoreMSFT ClearSpatialAnchorStoreMSFT;

    // ---- XR_FB_spatial_entity_query extension commands
    PFN_xrQuerySpacesFB QuerySpacesFB;
    PFN_xrRetrieveSpaceQueryResultsFB RetrieveSpaceQueryResultsFB;

    // ---- XR_FB_spatial_entity_storage extension commands
    PFN_xrSaveSpaceFB SaveSpaceFB;
    PFN_xrEraseSpaceFB EraseSpaceFB;

    // ---- XR_OCULUS_audio_device_guid extension commands
#if defined(XR_USE_PLATFORM_WIN32)
    PFN_xrGetAudioOutputDeviceGuidOculus GetAudioOutputDeviceGuidOculus;
#endif // defined(XR_USE_PLATFORM_WIN32)
#if defined(XR_USE_PLATFORM_WIN32)
    PFN_xrGetAudioInputDeviceGuidOculus GetAudioInputDeviceGuidOculus;
#endif // defined(XR_USE_PLATFORM_WIN32)

    // ---- XR_FB_spatial_entity_sharing extension commands
    PFN_xrShareSpacesFB ShareSpacesFB;

    // ---- XR_FB_scene extension commands
    PFN_xrGetSpaceBoundingBox2DFB GetSpaceBoundingBox2DFB;
    PFN_xrGetSpaceBoundingBox3DFB GetSpaceBoundingBox3DFB;
    PFN_xrGetSpaceSemanticLabelsFB GetSpaceSemanticLabelsFB;
    PFN_xrGetSpaceBoundary2DFB GetSpaceBoundary2DFB;
    PFN_xrGetSpaceRoomLayoutFB GetSpaceRoomLayoutFB;

    // ---- XR_ALMALENCE_digital_lens_control extension commands
    PFN_xrSetDigitalLensControlALMALENCE SetDigitalLensControlALMALENCE;

    // ---- XR_FB_scene_capture extension commands
    PFN_xrRequestSceneCaptureFB RequestSceneCaptureFB;

    // ---- XR_FB_spatial_entity_container extension commands
    PFN_xrGetSpaceContainerFB GetSpaceContainerFB;

    // ---- XR_META_foveation_eye_tracked extension commands
    PFN_xrGetFoveationEyeTrackedStateMETA GetFoveationEyeTrackedStateMETA;

    // ---- XR_FB_face_tracking extension commands
    PFN_xrCreateFaceTrackerFB CreateFaceTrackerFB;
    PFN_xrDestroyFaceTrackerFB DestroyFaceTrackerFB;
    PFN_xrGetFaceExpressionWeightsFB GetFaceExpressionWeightsFB;

    // ---- XR_FB_eye_tracking_social extension commands
    PFN_xrCreateEyeTrackerFB CreateEyeTrackerFB;
    PFN_xrDestroyEyeTrackerFB DestroyEyeTrackerFB;
    PFN_xrGetEyeGazesFB GetEyeGazesFB;

    // ---- XR_FB_passthrough_keyboard_hands extension commands
    PFN_xrPassthroughLayerSetKeyboardHandsIntensityFB PassthroughLayerSetKeyboardHandsIntensityFB;

    // ---- XR_FB_haptic_pcm extension commands
    PFN_xrGetDeviceSampleRateFB GetDeviceSampleRateFB;

    // ---- XR_META_virtual_keyboard extension commands
    PFN_xrCreateVirtualKeyboardMETA CreateVirtualKeyboardMETA;
    PFN_xrDestroyVirtualKeyboardMETA DestroyVirtualKeyboardMETA;
    PFN_xrCreateVirtualKeyboardSpaceMETA CreateVirtualKeyboardSpaceMETA;
    PFN_xrSuggestVirtualKeyboardLocationMETA SuggestVirtualKeyboardLocationMETA;
    PFN_xrGetVirtualKeyboardScaleMETA GetVirtualKeyboardScaleMETA;
    PFN_xrSetVirtualKeyboardModelVisibilityMETA SetVirtualKeyboardModelVisibilityMETA;
    PFN_xrGetVirtualKeyboardModelAnimationStatesMETA GetVirtualKeyboardModelAnimationStatesMETA;
    PFN_xrGetVirtualKeyboardDirtyTexturesMETA GetVirtualKeyboardDirtyTexturesMETA;
    PFN_xrGetVirtualKeyboardTextureDataMETA GetVirtualKeyboardTextureDataMETA;
    PFN_xrSendVirtualKeyboardInputMETA SendVirtualKeyboardInputMETA;
    PFN_xrChangeVirtualKeyboardTextContextMETA ChangeVirtualKeyboardTextContextMETA;

    // ---- XR_OCULUS_external_camera extension commands
    PFN_xrEnumerateExternalCamerasOCULUS EnumerateExternalCamerasOCULUS;

    // ---- XR_META_performance_metrics extension commands
    PFN_xrEnumeratePerformanceMetricsCounterPathsMETA EnumeratePerformanceMetricsCounterPathsMETA;
    PFN_xrSetPerformanceMetricsStateMETA SetPerformanceMetricsStateMETA;
    PFN_xrGetPerformanceMetricsStateMETA GetPerformanceMetricsStateMETA;
    PFN_xrQueryPerformanceMetricsCounterMETA QueryPerformanceMetricsCounterMETA;

    // ---- XR_FB_spatial_entity_storage_batch extension commands
    PFN_xrSaveSpaceListFB SaveSpaceListFB;

    // ---- XR_FB_spatial_entity_user extension commands
    PFN_xrCreateSpaceUserFB CreateSpaceUserFB;
    PFN_xrGetSpaceUserIdFB GetSpaceUserIdFB;
    PFN_xrDestroySpaceUserFB DestroySpaceUserFB;

    // ---- XR_META_passthrough_color_lut extension commands
    PFN_xrCreatePassthroughColorLutMETA CreatePassthroughColorLutMETA;
    PFN_xrDestroyPassthroughColorLutMETA DestroyPassthroughColorLutMETA;
    PFN_xrUpdatePassthroughColorLutMETA UpdatePassthroughColorLutMETA;

    // ---- XR_QCOM_tracking_optimization_settings extension commands
    PFN_xrSetTrackingOptimizationSettingsHintQCOM SetTrackingOptimizationSettingsHintQCOM;

    // ---- XR_HTC_passthrough extension commands
    PFN_xrCreatePassthroughHTC CreatePassthroughHTC;
    PFN_xrDestroyPassthroughHTC DestroyPassthroughHTC;

    // ---- XR_HTC_foveation extension commands
    PFN_xrApplyFoveationHTC ApplyFoveationHTC;

    // ---- XR_MNDX_force_feedback_curl extension commands
    PFN_xrApplyForceFeedbackCurlMNDX ApplyForceFeedbackCurlMNDX;

    // ---- XR_EXT_plane_detection extension commands
    PFN_xrCreatePlaneDetectorEXT CreatePlaneDetectorEXT;
    PFN_xrDestroyPlaneDetectorEXT DestroyPlaneDetectorEXT;
    PFN_xrBeginPlaneDetectionEXT BeginPlaneDetectionEXT;
    PFN_xrGetPlaneDetectionStateEXT GetPlaneDetectionStateEXT;
    PFN_xrGetPlaneDetectionsEXT GetPlaneDetectionsEXT;
    PFN_xrGetPlanePolygonBufferEXT GetPlanePolygonBufferEXT;
};


// Prototype for dispatch table helper function
void GeneratedXrPopulateDispatchTable(struct XrGeneratedDispatchTable *table,
                                      XrInstance instance,
                                      PFN_xrGetInstanceProcAddr get_inst_proc_addr);

#ifdef __cplusplus
} // extern "C"
#endif

