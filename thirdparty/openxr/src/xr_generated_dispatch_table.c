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

#include <time.h>
#include "xr_generated_dispatch_table.h"
#include "xr_dependencies.h"
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>


#ifdef __cplusplus
extern "C" { 
#endif
// Helper function to populate an instance dispatch table
void GeneratedXrPopulateDispatchTable(struct XrGeneratedDispatchTable *table,
                                      XrInstance instance,
                                      PFN_xrGetInstanceProcAddr get_inst_proc_addr) {

    // ---- Core 1.0 commands
    table->GetInstanceProcAddr = get_inst_proc_addr;
    (get_inst_proc_addr(instance, "xrCreateInstance", (PFN_xrVoidFunction*)&table->CreateInstance));
    (get_inst_proc_addr(instance, "xrDestroyInstance", (PFN_xrVoidFunction*)&table->DestroyInstance));
    (get_inst_proc_addr(instance, "xrGetInstanceProperties", (PFN_xrVoidFunction*)&table->GetInstanceProperties));
    (get_inst_proc_addr(instance, "xrPollEvent", (PFN_xrVoidFunction*)&table->PollEvent));
    (get_inst_proc_addr(instance, "xrResultToString", (PFN_xrVoidFunction*)&table->ResultToString));
    (get_inst_proc_addr(instance, "xrStructureTypeToString", (PFN_xrVoidFunction*)&table->StructureTypeToString));
    (get_inst_proc_addr(instance, "xrGetSystem", (PFN_xrVoidFunction*)&table->GetSystem));
    (get_inst_proc_addr(instance, "xrGetSystemProperties", (PFN_xrVoidFunction*)&table->GetSystemProperties));
    (get_inst_proc_addr(instance, "xrEnumerateEnvironmentBlendModes", (PFN_xrVoidFunction*)&table->EnumerateEnvironmentBlendModes));
    (get_inst_proc_addr(instance, "xrCreateSession", (PFN_xrVoidFunction*)&table->CreateSession));
    (get_inst_proc_addr(instance, "xrDestroySession", (PFN_xrVoidFunction*)&table->DestroySession));
    (get_inst_proc_addr(instance, "xrEnumerateReferenceSpaces", (PFN_xrVoidFunction*)&table->EnumerateReferenceSpaces));
    (get_inst_proc_addr(instance, "xrCreateReferenceSpace", (PFN_xrVoidFunction*)&table->CreateReferenceSpace));
    (get_inst_proc_addr(instance, "xrGetReferenceSpaceBoundsRect", (PFN_xrVoidFunction*)&table->GetReferenceSpaceBoundsRect));
    (get_inst_proc_addr(instance, "xrCreateActionSpace", (PFN_xrVoidFunction*)&table->CreateActionSpace));
    (get_inst_proc_addr(instance, "xrLocateSpace", (PFN_xrVoidFunction*)&table->LocateSpace));
    (get_inst_proc_addr(instance, "xrDestroySpace", (PFN_xrVoidFunction*)&table->DestroySpace));
    (get_inst_proc_addr(instance, "xrEnumerateViewConfigurations", (PFN_xrVoidFunction*)&table->EnumerateViewConfigurations));
    (get_inst_proc_addr(instance, "xrGetViewConfigurationProperties", (PFN_xrVoidFunction*)&table->GetViewConfigurationProperties));
    (get_inst_proc_addr(instance, "xrEnumerateViewConfigurationViews", (PFN_xrVoidFunction*)&table->EnumerateViewConfigurationViews));
    (get_inst_proc_addr(instance, "xrEnumerateSwapchainFormats", (PFN_xrVoidFunction*)&table->EnumerateSwapchainFormats));
    (get_inst_proc_addr(instance, "xrCreateSwapchain", (PFN_xrVoidFunction*)&table->CreateSwapchain));
    (get_inst_proc_addr(instance, "xrDestroySwapchain", (PFN_xrVoidFunction*)&table->DestroySwapchain));
    (get_inst_proc_addr(instance, "xrEnumerateSwapchainImages", (PFN_xrVoidFunction*)&table->EnumerateSwapchainImages));
    (get_inst_proc_addr(instance, "xrAcquireSwapchainImage", (PFN_xrVoidFunction*)&table->AcquireSwapchainImage));
    (get_inst_proc_addr(instance, "xrWaitSwapchainImage", (PFN_xrVoidFunction*)&table->WaitSwapchainImage));
    (get_inst_proc_addr(instance, "xrReleaseSwapchainImage", (PFN_xrVoidFunction*)&table->ReleaseSwapchainImage));
    (get_inst_proc_addr(instance, "xrBeginSession", (PFN_xrVoidFunction*)&table->BeginSession));
    (get_inst_proc_addr(instance, "xrEndSession", (PFN_xrVoidFunction*)&table->EndSession));
    (get_inst_proc_addr(instance, "xrRequestExitSession", (PFN_xrVoidFunction*)&table->RequestExitSession));
    (get_inst_proc_addr(instance, "xrWaitFrame", (PFN_xrVoidFunction*)&table->WaitFrame));
    (get_inst_proc_addr(instance, "xrBeginFrame", (PFN_xrVoidFunction*)&table->BeginFrame));
    (get_inst_proc_addr(instance, "xrEndFrame", (PFN_xrVoidFunction*)&table->EndFrame));
    (get_inst_proc_addr(instance, "xrLocateViews", (PFN_xrVoidFunction*)&table->LocateViews));
    (get_inst_proc_addr(instance, "xrStringToPath", (PFN_xrVoidFunction*)&table->StringToPath));
    (get_inst_proc_addr(instance, "xrPathToString", (PFN_xrVoidFunction*)&table->PathToString));
    (get_inst_proc_addr(instance, "xrCreateActionSet", (PFN_xrVoidFunction*)&table->CreateActionSet));
    (get_inst_proc_addr(instance, "xrDestroyActionSet", (PFN_xrVoidFunction*)&table->DestroyActionSet));
    (get_inst_proc_addr(instance, "xrCreateAction", (PFN_xrVoidFunction*)&table->CreateAction));
    (get_inst_proc_addr(instance, "xrDestroyAction", (PFN_xrVoidFunction*)&table->DestroyAction));
    (get_inst_proc_addr(instance, "xrSuggestInteractionProfileBindings", (PFN_xrVoidFunction*)&table->SuggestInteractionProfileBindings));
    (get_inst_proc_addr(instance, "xrAttachSessionActionSets", (PFN_xrVoidFunction*)&table->AttachSessionActionSets));
    (get_inst_proc_addr(instance, "xrGetCurrentInteractionProfile", (PFN_xrVoidFunction*)&table->GetCurrentInteractionProfile));
    (get_inst_proc_addr(instance, "xrGetActionStateBoolean", (PFN_xrVoidFunction*)&table->GetActionStateBoolean));
    (get_inst_proc_addr(instance, "xrGetActionStateFloat", (PFN_xrVoidFunction*)&table->GetActionStateFloat));
    (get_inst_proc_addr(instance, "xrGetActionStateVector2f", (PFN_xrVoidFunction*)&table->GetActionStateVector2f));
    (get_inst_proc_addr(instance, "xrGetActionStatePose", (PFN_xrVoidFunction*)&table->GetActionStatePose));
    (get_inst_proc_addr(instance, "xrSyncActions", (PFN_xrVoidFunction*)&table->SyncActions));
    (get_inst_proc_addr(instance, "xrEnumerateBoundSourcesForAction", (PFN_xrVoidFunction*)&table->EnumerateBoundSourcesForAction));
    (get_inst_proc_addr(instance, "xrGetInputSourceLocalizedName", (PFN_xrVoidFunction*)&table->GetInputSourceLocalizedName));
    (get_inst_proc_addr(instance, "xrApplyHapticFeedback", (PFN_xrVoidFunction*)&table->ApplyHapticFeedback));
    (get_inst_proc_addr(instance, "xrStopHapticFeedback", (PFN_xrVoidFunction*)&table->StopHapticFeedback));

    // ---- XR_KHR_android_thread_settings extension commands
#if defined(XR_USE_PLATFORM_ANDROID)
    (get_inst_proc_addr(instance, "xrSetAndroidApplicationThreadKHR", (PFN_xrVoidFunction*)&table->SetAndroidApplicationThreadKHR));
#endif // defined(XR_USE_PLATFORM_ANDROID)

    // ---- XR_KHR_android_surface_swapchain extension commands
#if defined(XR_USE_PLATFORM_ANDROID)
    (get_inst_proc_addr(instance, "xrCreateSwapchainAndroidSurfaceKHR", (PFN_xrVoidFunction*)&table->CreateSwapchainAndroidSurfaceKHR));
#endif // defined(XR_USE_PLATFORM_ANDROID)

    // ---- XR_KHR_opengl_enable extension commands
#if defined(XR_USE_GRAPHICS_API_OPENGL)
    (get_inst_proc_addr(instance, "xrGetOpenGLGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&table->GetOpenGLGraphicsRequirementsKHR));
#endif // defined(XR_USE_GRAPHICS_API_OPENGL)

    // ---- XR_KHR_opengl_es_enable extension commands
#if defined(XR_USE_GRAPHICS_API_OPENGL_ES)
    (get_inst_proc_addr(instance, "xrGetOpenGLESGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&table->GetOpenGLESGraphicsRequirementsKHR));
#endif // defined(XR_USE_GRAPHICS_API_OPENGL_ES)

    // ---- XR_KHR_vulkan_enable extension commands
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrGetVulkanInstanceExtensionsKHR", (PFN_xrVoidFunction*)&table->GetVulkanInstanceExtensionsKHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrGetVulkanDeviceExtensionsKHR", (PFN_xrVoidFunction*)&table->GetVulkanDeviceExtensionsKHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrGetVulkanGraphicsDeviceKHR", (PFN_xrVoidFunction*)&table->GetVulkanGraphicsDeviceKHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrGetVulkanGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&table->GetVulkanGraphicsRequirementsKHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)

    // ---- XR_KHR_D3D11_enable extension commands
#if defined(XR_USE_GRAPHICS_API_D3D11)
    (get_inst_proc_addr(instance, "xrGetD3D11GraphicsRequirementsKHR", (PFN_xrVoidFunction*)&table->GetD3D11GraphicsRequirementsKHR));
#endif // defined(XR_USE_GRAPHICS_API_D3D11)

    // ---- XR_KHR_D3D12_enable extension commands
#if defined(XR_USE_GRAPHICS_API_D3D12)
    (get_inst_proc_addr(instance, "xrGetD3D12GraphicsRequirementsKHR", (PFN_xrVoidFunction*)&table->GetD3D12GraphicsRequirementsKHR));
#endif // defined(XR_USE_GRAPHICS_API_D3D12)

    // ---- XR_KHR_visibility_mask extension commands
    (get_inst_proc_addr(instance, "xrGetVisibilityMaskKHR", (PFN_xrVoidFunction*)&table->GetVisibilityMaskKHR));

    // ---- XR_KHR_win32_convert_performance_counter_time extension commands
#if defined(XR_USE_PLATFORM_WIN32)
    (get_inst_proc_addr(instance, "xrConvertWin32PerformanceCounterToTimeKHR", (PFN_xrVoidFunction*)&table->ConvertWin32PerformanceCounterToTimeKHR));
#endif // defined(XR_USE_PLATFORM_WIN32)
#if defined(XR_USE_PLATFORM_WIN32)
    (get_inst_proc_addr(instance, "xrConvertTimeToWin32PerformanceCounterKHR", (PFN_xrVoidFunction*)&table->ConvertTimeToWin32PerformanceCounterKHR));
#endif // defined(XR_USE_PLATFORM_WIN32)

    // ---- XR_KHR_convert_timespec_time extension commands
#if defined(XR_USE_TIMESPEC)
    (get_inst_proc_addr(instance, "xrConvertTimespecTimeToTimeKHR", (PFN_xrVoidFunction*)&table->ConvertTimespecTimeToTimeKHR));
#endif // defined(XR_USE_TIMESPEC)
#if defined(XR_USE_TIMESPEC)
    (get_inst_proc_addr(instance, "xrConvertTimeToTimespecTimeKHR", (PFN_xrVoidFunction*)&table->ConvertTimeToTimespecTimeKHR));
#endif // defined(XR_USE_TIMESPEC)

    // ---- XR_KHR_vulkan_enable2 extension commands
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrCreateVulkanInstanceKHR", (PFN_xrVoidFunction*)&table->CreateVulkanInstanceKHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrCreateVulkanDeviceKHR", (PFN_xrVoidFunction*)&table->CreateVulkanDeviceKHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrGetVulkanGraphicsDevice2KHR", (PFN_xrVoidFunction*)&table->GetVulkanGraphicsDevice2KHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)
#if defined(XR_USE_GRAPHICS_API_VULKAN)
    (get_inst_proc_addr(instance, "xrGetVulkanGraphicsRequirements2KHR", (PFN_xrVoidFunction*)&table->GetVulkanGraphicsRequirements2KHR));
#endif // defined(XR_USE_GRAPHICS_API_VULKAN)

    // ---- XR_EXT_performance_settings extension commands
    (get_inst_proc_addr(instance, "xrPerfSettingsSetPerformanceLevelEXT", (PFN_xrVoidFunction*)&table->PerfSettingsSetPerformanceLevelEXT));

    // ---- XR_EXT_thermal_query extension commands
    (get_inst_proc_addr(instance, "xrThermalGetTemperatureTrendEXT", (PFN_xrVoidFunction*)&table->ThermalGetTemperatureTrendEXT));

    // ---- XR_EXT_debug_utils extension commands
    (get_inst_proc_addr(instance, "xrSetDebugUtilsObjectNameEXT", (PFN_xrVoidFunction*)&table->SetDebugUtilsObjectNameEXT));
    (get_inst_proc_addr(instance, "xrCreateDebugUtilsMessengerEXT", (PFN_xrVoidFunction*)&table->CreateDebugUtilsMessengerEXT));
    (get_inst_proc_addr(instance, "xrDestroyDebugUtilsMessengerEXT", (PFN_xrVoidFunction*)&table->DestroyDebugUtilsMessengerEXT));
    (get_inst_proc_addr(instance, "xrSubmitDebugUtilsMessageEXT", (PFN_xrVoidFunction*)&table->SubmitDebugUtilsMessageEXT));
    (get_inst_proc_addr(instance, "xrSessionBeginDebugUtilsLabelRegionEXT", (PFN_xrVoidFunction*)&table->SessionBeginDebugUtilsLabelRegionEXT));
    (get_inst_proc_addr(instance, "xrSessionEndDebugUtilsLabelRegionEXT", (PFN_xrVoidFunction*)&table->SessionEndDebugUtilsLabelRegionEXT));
    (get_inst_proc_addr(instance, "xrSessionInsertDebugUtilsLabelEXT", (PFN_xrVoidFunction*)&table->SessionInsertDebugUtilsLabelEXT));

    // ---- XR_MSFT_spatial_anchor extension commands
    (get_inst_proc_addr(instance, "xrCreateSpatialAnchorMSFT", (PFN_xrVoidFunction*)&table->CreateSpatialAnchorMSFT));
    (get_inst_proc_addr(instance, "xrCreateSpatialAnchorSpaceMSFT", (PFN_xrVoidFunction*)&table->CreateSpatialAnchorSpaceMSFT));
    (get_inst_proc_addr(instance, "xrDestroySpatialAnchorMSFT", (PFN_xrVoidFunction*)&table->DestroySpatialAnchorMSFT));

    // ---- XR_EXT_conformance_automation extension commands
    (get_inst_proc_addr(instance, "xrSetInputDeviceActiveEXT", (PFN_xrVoidFunction*)&table->SetInputDeviceActiveEXT));
    (get_inst_proc_addr(instance, "xrSetInputDeviceStateBoolEXT", (PFN_xrVoidFunction*)&table->SetInputDeviceStateBoolEXT));
    (get_inst_proc_addr(instance, "xrSetInputDeviceStateFloatEXT", (PFN_xrVoidFunction*)&table->SetInputDeviceStateFloatEXT));
    (get_inst_proc_addr(instance, "xrSetInputDeviceStateVector2fEXT", (PFN_xrVoidFunction*)&table->SetInputDeviceStateVector2fEXT));
    (get_inst_proc_addr(instance, "xrSetInputDeviceLocationEXT", (PFN_xrVoidFunction*)&table->SetInputDeviceLocationEXT));

    // ---- XR_MSFT_spatial_graph_bridge extension commands
    (get_inst_proc_addr(instance, "xrCreateSpatialGraphNodeSpaceMSFT", (PFN_xrVoidFunction*)&table->CreateSpatialGraphNodeSpaceMSFT));
    (get_inst_proc_addr(instance, "xrTryCreateSpatialGraphStaticNodeBindingMSFT", (PFN_xrVoidFunction*)&table->TryCreateSpatialGraphStaticNodeBindingMSFT));
    (get_inst_proc_addr(instance, "xrDestroySpatialGraphNodeBindingMSFT", (PFN_xrVoidFunction*)&table->DestroySpatialGraphNodeBindingMSFT));
    (get_inst_proc_addr(instance, "xrGetSpatialGraphNodeBindingPropertiesMSFT", (PFN_xrVoidFunction*)&table->GetSpatialGraphNodeBindingPropertiesMSFT));

    // ---- XR_EXT_hand_tracking extension commands
    (get_inst_proc_addr(instance, "xrCreateHandTrackerEXT", (PFN_xrVoidFunction*)&table->CreateHandTrackerEXT));
    (get_inst_proc_addr(instance, "xrDestroyHandTrackerEXT", (PFN_xrVoidFunction*)&table->DestroyHandTrackerEXT));
    (get_inst_proc_addr(instance, "xrLocateHandJointsEXT", (PFN_xrVoidFunction*)&table->LocateHandJointsEXT));

    // ---- XR_MSFT_hand_tracking_mesh extension commands
    (get_inst_proc_addr(instance, "xrCreateHandMeshSpaceMSFT", (PFN_xrVoidFunction*)&table->CreateHandMeshSpaceMSFT));
    (get_inst_proc_addr(instance, "xrUpdateHandMeshMSFT", (PFN_xrVoidFunction*)&table->UpdateHandMeshMSFT));

    // ---- XR_MSFT_controller_model extension commands
    (get_inst_proc_addr(instance, "xrGetControllerModelKeyMSFT", (PFN_xrVoidFunction*)&table->GetControllerModelKeyMSFT));
    (get_inst_proc_addr(instance, "xrLoadControllerModelMSFT", (PFN_xrVoidFunction*)&table->LoadControllerModelMSFT));
    (get_inst_proc_addr(instance, "xrGetControllerModelPropertiesMSFT", (PFN_xrVoidFunction*)&table->GetControllerModelPropertiesMSFT));
    (get_inst_proc_addr(instance, "xrGetControllerModelStateMSFT", (PFN_xrVoidFunction*)&table->GetControllerModelStateMSFT));

    // ---- XR_MSFT_perception_anchor_interop extension commands
#if defined(XR_USE_PLATFORM_WIN32)
    (get_inst_proc_addr(instance, "xrCreateSpatialAnchorFromPerceptionAnchorMSFT", (PFN_xrVoidFunction*)&table->CreateSpatialAnchorFromPerceptionAnchorMSFT));
#endif // defined(XR_USE_PLATFORM_WIN32)
#if defined(XR_USE_PLATFORM_WIN32)
    (get_inst_proc_addr(instance, "xrTryGetPerceptionAnchorFromSpatialAnchorMSFT", (PFN_xrVoidFunction*)&table->TryGetPerceptionAnchorFromSpatialAnchorMSFT));
#endif // defined(XR_USE_PLATFORM_WIN32)

    // ---- XR_MSFT_composition_layer_reprojection extension commands
    (get_inst_proc_addr(instance, "xrEnumerateReprojectionModesMSFT", (PFN_xrVoidFunction*)&table->EnumerateReprojectionModesMSFT));

    // ---- XR_FB_swapchain_update_state extension commands
    (get_inst_proc_addr(instance, "xrUpdateSwapchainFB", (PFN_xrVoidFunction*)&table->UpdateSwapchainFB));
    (get_inst_proc_addr(instance, "xrGetSwapchainStateFB", (PFN_xrVoidFunction*)&table->GetSwapchainStateFB));

    // ---- XR_FB_body_tracking extension commands
    (get_inst_proc_addr(instance, "xrCreateBodyTrackerFB", (PFN_xrVoidFunction*)&table->CreateBodyTrackerFB));
    (get_inst_proc_addr(instance, "xrDestroyBodyTrackerFB", (PFN_xrVoidFunction*)&table->DestroyBodyTrackerFB));
    (get_inst_proc_addr(instance, "xrLocateBodyJointsFB", (PFN_xrVoidFunction*)&table->LocateBodyJointsFB));
    (get_inst_proc_addr(instance, "xrGetBodySkeletonFB", (PFN_xrVoidFunction*)&table->GetBodySkeletonFB));

    // ---- XR_MSFT_scene_understanding extension commands
    (get_inst_proc_addr(instance, "xrEnumerateSceneComputeFeaturesMSFT", (PFN_xrVoidFunction*)&table->EnumerateSceneComputeFeaturesMSFT));
    (get_inst_proc_addr(instance, "xrCreateSceneObserverMSFT", (PFN_xrVoidFunction*)&table->CreateSceneObserverMSFT));
    (get_inst_proc_addr(instance, "xrDestroySceneObserverMSFT", (PFN_xrVoidFunction*)&table->DestroySceneObserverMSFT));
    (get_inst_proc_addr(instance, "xrCreateSceneMSFT", (PFN_xrVoidFunction*)&table->CreateSceneMSFT));
    (get_inst_proc_addr(instance, "xrDestroySceneMSFT", (PFN_xrVoidFunction*)&table->DestroySceneMSFT));
    (get_inst_proc_addr(instance, "xrComputeNewSceneMSFT", (PFN_xrVoidFunction*)&table->ComputeNewSceneMSFT));
    (get_inst_proc_addr(instance, "xrGetSceneComputeStateMSFT", (PFN_xrVoidFunction*)&table->GetSceneComputeStateMSFT));
    (get_inst_proc_addr(instance, "xrGetSceneComponentsMSFT", (PFN_xrVoidFunction*)&table->GetSceneComponentsMSFT));
    (get_inst_proc_addr(instance, "xrLocateSceneComponentsMSFT", (PFN_xrVoidFunction*)&table->LocateSceneComponentsMSFT));
    (get_inst_proc_addr(instance, "xrGetSceneMeshBuffersMSFT", (PFN_xrVoidFunction*)&table->GetSceneMeshBuffersMSFT));

    // ---- XR_MSFT_scene_understanding_serialization extension commands
    (get_inst_proc_addr(instance, "xrDeserializeSceneMSFT", (PFN_xrVoidFunction*)&table->DeserializeSceneMSFT));
    (get_inst_proc_addr(instance, "xrGetSerializedSceneFragmentDataMSFT", (PFN_xrVoidFunction*)&table->GetSerializedSceneFragmentDataMSFT));

    // ---- XR_FB_display_refresh_rate extension commands
    (get_inst_proc_addr(instance, "xrEnumerateDisplayRefreshRatesFB", (PFN_xrVoidFunction*)&table->EnumerateDisplayRefreshRatesFB));
    (get_inst_proc_addr(instance, "xrGetDisplayRefreshRateFB", (PFN_xrVoidFunction*)&table->GetDisplayRefreshRateFB));
    (get_inst_proc_addr(instance, "xrRequestDisplayRefreshRateFB", (PFN_xrVoidFunction*)&table->RequestDisplayRefreshRateFB));

    // ---- XR_HTCX_vive_tracker_interaction extension commands
    (get_inst_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX", (PFN_xrVoidFunction*)&table->EnumerateViveTrackerPathsHTCX));

    // ---- XR_HTC_facial_tracking extension commands
    (get_inst_proc_addr(instance, "xrCreateFacialTrackerHTC", (PFN_xrVoidFunction*)&table->CreateFacialTrackerHTC));
    (get_inst_proc_addr(instance, "xrDestroyFacialTrackerHTC", (PFN_xrVoidFunction*)&table->DestroyFacialTrackerHTC));
    (get_inst_proc_addr(instance, "xrGetFacialExpressionsHTC", (PFN_xrVoidFunction*)&table->GetFacialExpressionsHTC));

    // ---- XR_FB_color_space extension commands
    (get_inst_proc_addr(instance, "xrEnumerateColorSpacesFB", (PFN_xrVoidFunction*)&table->EnumerateColorSpacesFB));
    (get_inst_proc_addr(instance, "xrSetColorSpaceFB", (PFN_xrVoidFunction*)&table->SetColorSpaceFB));

    // ---- XR_FB_hand_tracking_mesh extension commands
    (get_inst_proc_addr(instance, "xrGetHandMeshFB", (PFN_xrVoidFunction*)&table->GetHandMeshFB));

    // ---- XR_FB_spatial_entity extension commands
    (get_inst_proc_addr(instance, "xrCreateSpatialAnchorFB", (PFN_xrVoidFunction*)&table->CreateSpatialAnchorFB));
    (get_inst_proc_addr(instance, "xrGetSpaceUuidFB", (PFN_xrVoidFunction*)&table->GetSpaceUuidFB));
    (get_inst_proc_addr(instance, "xrEnumerateSpaceSupportedComponentsFB", (PFN_xrVoidFunction*)&table->EnumerateSpaceSupportedComponentsFB));
    (get_inst_proc_addr(instance, "xrSetSpaceComponentStatusFB", (PFN_xrVoidFunction*)&table->SetSpaceComponentStatusFB));
    (get_inst_proc_addr(instance, "xrGetSpaceComponentStatusFB", (PFN_xrVoidFunction*)&table->GetSpaceComponentStatusFB));

    // ---- XR_FB_foveation extension commands
    (get_inst_proc_addr(instance, "xrCreateFoveationProfileFB", (PFN_xrVoidFunction*)&table->CreateFoveationProfileFB));
    (get_inst_proc_addr(instance, "xrDestroyFoveationProfileFB", (PFN_xrVoidFunction*)&table->DestroyFoveationProfileFB));

    // ---- XR_FB_keyboard_tracking extension commands
    (get_inst_proc_addr(instance, "xrQuerySystemTrackedKeyboardFB", (PFN_xrVoidFunction*)&table->QuerySystemTrackedKeyboardFB));
    (get_inst_proc_addr(instance, "xrCreateKeyboardSpaceFB", (PFN_xrVoidFunction*)&table->CreateKeyboardSpaceFB));

    // ---- XR_FB_triangle_mesh extension commands
    (get_inst_proc_addr(instance, "xrCreateTriangleMeshFB", (PFN_xrVoidFunction*)&table->CreateTriangleMeshFB));
    (get_inst_proc_addr(instance, "xrDestroyTriangleMeshFB", (PFN_xrVoidFunction*)&table->DestroyTriangleMeshFB));
    (get_inst_proc_addr(instance, "xrTriangleMeshGetVertexBufferFB", (PFN_xrVoidFunction*)&table->TriangleMeshGetVertexBufferFB));
    (get_inst_proc_addr(instance, "xrTriangleMeshGetIndexBufferFB", (PFN_xrVoidFunction*)&table->TriangleMeshGetIndexBufferFB));
    (get_inst_proc_addr(instance, "xrTriangleMeshBeginUpdateFB", (PFN_xrVoidFunction*)&table->TriangleMeshBeginUpdateFB));
    (get_inst_proc_addr(instance, "xrTriangleMeshEndUpdateFB", (PFN_xrVoidFunction*)&table->TriangleMeshEndUpdateFB));
    (get_inst_proc_addr(instance, "xrTriangleMeshBeginVertexBufferUpdateFB", (PFN_xrVoidFunction*)&table->TriangleMeshBeginVertexBufferUpdateFB));
    (get_inst_proc_addr(instance, "xrTriangleMeshEndVertexBufferUpdateFB", (PFN_xrVoidFunction*)&table->TriangleMeshEndVertexBufferUpdateFB));

    // ---- XR_FB_passthrough extension commands
    (get_inst_proc_addr(instance, "xrCreatePassthroughFB", (PFN_xrVoidFunction*)&table->CreatePassthroughFB));
    (get_inst_proc_addr(instance, "xrDestroyPassthroughFB", (PFN_xrVoidFunction*)&table->DestroyPassthroughFB));
    (get_inst_proc_addr(instance, "xrPassthroughStartFB", (PFN_xrVoidFunction*)&table->PassthroughStartFB));
    (get_inst_proc_addr(instance, "xrPassthroughPauseFB", (PFN_xrVoidFunction*)&table->PassthroughPauseFB));
    (get_inst_proc_addr(instance, "xrCreatePassthroughLayerFB", (PFN_xrVoidFunction*)&table->CreatePassthroughLayerFB));
    (get_inst_proc_addr(instance, "xrDestroyPassthroughLayerFB", (PFN_xrVoidFunction*)&table->DestroyPassthroughLayerFB));
    (get_inst_proc_addr(instance, "xrPassthroughLayerPauseFB", (PFN_xrVoidFunction*)&table->PassthroughLayerPauseFB));
    (get_inst_proc_addr(instance, "xrPassthroughLayerResumeFB", (PFN_xrVoidFunction*)&table->PassthroughLayerResumeFB));
    (get_inst_proc_addr(instance, "xrPassthroughLayerSetStyleFB", (PFN_xrVoidFunction*)&table->PassthroughLayerSetStyleFB));
    (get_inst_proc_addr(instance, "xrCreateGeometryInstanceFB", (PFN_xrVoidFunction*)&table->CreateGeometryInstanceFB));
    (get_inst_proc_addr(instance, "xrDestroyGeometryInstanceFB", (PFN_xrVoidFunction*)&table->DestroyGeometryInstanceFB));
    (get_inst_proc_addr(instance, "xrGeometryInstanceSetTransformFB", (PFN_xrVoidFunction*)&table->GeometryInstanceSetTransformFB));

    // ---- XR_FB_render_model extension commands
    (get_inst_proc_addr(instance, "xrEnumerateRenderModelPathsFB", (PFN_xrVoidFunction*)&table->EnumerateRenderModelPathsFB));
    (get_inst_proc_addr(instance, "xrGetRenderModelPropertiesFB", (PFN_xrVoidFunction*)&table->GetRenderModelPropertiesFB));
    (get_inst_proc_addr(instance, "xrLoadRenderModelFB", (PFN_xrVoidFunction*)&table->LoadRenderModelFB));

    // ---- XR_VARJO_environment_depth_estimation extension commands
    (get_inst_proc_addr(instance, "xrSetEnvironmentDepthEstimationVARJO", (PFN_xrVoidFunction*)&table->SetEnvironmentDepthEstimationVARJO));

    // ---- XR_VARJO_marker_tracking extension commands
    (get_inst_proc_addr(instance, "xrSetMarkerTrackingVARJO", (PFN_xrVoidFunction*)&table->SetMarkerTrackingVARJO));
    (get_inst_proc_addr(instance, "xrSetMarkerTrackingTimeoutVARJO", (PFN_xrVoidFunction*)&table->SetMarkerTrackingTimeoutVARJO));
    (get_inst_proc_addr(instance, "xrSetMarkerTrackingPredictionVARJO", (PFN_xrVoidFunction*)&table->SetMarkerTrackingPredictionVARJO));
    (get_inst_proc_addr(instance, "xrGetMarkerSizeVARJO", (PFN_xrVoidFunction*)&table->GetMarkerSizeVARJO));
    (get_inst_proc_addr(instance, "xrCreateMarkerSpaceVARJO", (PFN_xrVoidFunction*)&table->CreateMarkerSpaceVARJO));

    // ---- XR_VARJO_view_offset extension commands
    (get_inst_proc_addr(instance, "xrSetViewOffsetVARJO", (PFN_xrVoidFunction*)&table->SetViewOffsetVARJO));

    // ---- XR_ML_compat extension commands
#if defined(XR_USE_PLATFORM_ML)
    (get_inst_proc_addr(instance, "xrCreateSpaceFromCoordinateFrameUIDML", (PFN_xrVoidFunction*)&table->CreateSpaceFromCoordinateFrameUIDML));
#endif // defined(XR_USE_PLATFORM_ML)

    // ---- XR_MSFT_spatial_anchor_persistence extension commands
    (get_inst_proc_addr(instance, "xrCreateSpatialAnchorStoreConnectionMSFT", (PFN_xrVoidFunction*)&table->CreateSpatialAnchorStoreConnectionMSFT));
    (get_inst_proc_addr(instance, "xrDestroySpatialAnchorStoreConnectionMSFT", (PFN_xrVoidFunction*)&table->DestroySpatialAnchorStoreConnectionMSFT));
    (get_inst_proc_addr(instance, "xrPersistSpatialAnchorMSFT", (PFN_xrVoidFunction*)&table->PersistSpatialAnchorMSFT));
    (get_inst_proc_addr(instance, "xrEnumeratePersistedSpatialAnchorNamesMSFT", (PFN_xrVoidFunction*)&table->EnumeratePersistedSpatialAnchorNamesMSFT));
    (get_inst_proc_addr(instance, "xrCreateSpatialAnchorFromPersistedNameMSFT", (PFN_xrVoidFunction*)&table->CreateSpatialAnchorFromPersistedNameMSFT));
    (get_inst_proc_addr(instance, "xrUnpersistSpatialAnchorMSFT", (PFN_xrVoidFunction*)&table->UnpersistSpatialAnchorMSFT));
    (get_inst_proc_addr(instance, "xrClearSpatialAnchorStoreMSFT", (PFN_xrVoidFunction*)&table->ClearSpatialAnchorStoreMSFT));

    // ---- XR_FB_spatial_entity_query extension commands
    (get_inst_proc_addr(instance, "xrQuerySpacesFB", (PFN_xrVoidFunction*)&table->QuerySpacesFB));
    (get_inst_proc_addr(instance, "xrRetrieveSpaceQueryResultsFB", (PFN_xrVoidFunction*)&table->RetrieveSpaceQueryResultsFB));

    // ---- XR_FB_spatial_entity_storage extension commands
    (get_inst_proc_addr(instance, "xrSaveSpaceFB", (PFN_xrVoidFunction*)&table->SaveSpaceFB));
    (get_inst_proc_addr(instance, "xrEraseSpaceFB", (PFN_xrVoidFunction*)&table->EraseSpaceFB));

    // ---- XR_OCULUS_audio_device_guid extension commands
#if defined(XR_USE_PLATFORM_WIN32)
    (get_inst_proc_addr(instance, "xrGetAudioOutputDeviceGuidOculus", (PFN_xrVoidFunction*)&table->GetAudioOutputDeviceGuidOculus));
#endif // defined(XR_USE_PLATFORM_WIN32)
#if defined(XR_USE_PLATFORM_WIN32)
    (get_inst_proc_addr(instance, "xrGetAudioInputDeviceGuidOculus", (PFN_xrVoidFunction*)&table->GetAudioInputDeviceGuidOculus));
#endif // defined(XR_USE_PLATFORM_WIN32)

    // ---- XR_FB_spatial_entity_sharing extension commands
    (get_inst_proc_addr(instance, "xrShareSpacesFB", (PFN_xrVoidFunction*)&table->ShareSpacesFB));

    // ---- XR_FB_scene extension commands
    (get_inst_proc_addr(instance, "xrGetSpaceBoundingBox2DFB", (PFN_xrVoidFunction*)&table->GetSpaceBoundingBox2DFB));
    (get_inst_proc_addr(instance, "xrGetSpaceBoundingBox3DFB", (PFN_xrVoidFunction*)&table->GetSpaceBoundingBox3DFB));
    (get_inst_proc_addr(instance, "xrGetSpaceSemanticLabelsFB", (PFN_xrVoidFunction*)&table->GetSpaceSemanticLabelsFB));
    (get_inst_proc_addr(instance, "xrGetSpaceBoundary2DFB", (PFN_xrVoidFunction*)&table->GetSpaceBoundary2DFB));
    (get_inst_proc_addr(instance, "xrGetSpaceRoomLayoutFB", (PFN_xrVoidFunction*)&table->GetSpaceRoomLayoutFB));

    // ---- XR_ALMALENCE_digital_lens_control extension commands
    (get_inst_proc_addr(instance, "xrSetDigitalLensControlALMALENCE", (PFN_xrVoidFunction*)&table->SetDigitalLensControlALMALENCE));

    // ---- XR_FB_scene_capture extension commands
    (get_inst_proc_addr(instance, "xrRequestSceneCaptureFB", (PFN_xrVoidFunction*)&table->RequestSceneCaptureFB));

    // ---- XR_FB_spatial_entity_container extension commands
    (get_inst_proc_addr(instance, "xrGetSpaceContainerFB", (PFN_xrVoidFunction*)&table->GetSpaceContainerFB));

    // ---- XR_META_foveation_eye_tracked extension commands
    (get_inst_proc_addr(instance, "xrGetFoveationEyeTrackedStateMETA", (PFN_xrVoidFunction*)&table->GetFoveationEyeTrackedStateMETA));

    // ---- XR_FB_face_tracking extension commands
    (get_inst_proc_addr(instance, "xrCreateFaceTrackerFB", (PFN_xrVoidFunction*)&table->CreateFaceTrackerFB));
    (get_inst_proc_addr(instance, "xrDestroyFaceTrackerFB", (PFN_xrVoidFunction*)&table->DestroyFaceTrackerFB));
    (get_inst_proc_addr(instance, "xrGetFaceExpressionWeightsFB", (PFN_xrVoidFunction*)&table->GetFaceExpressionWeightsFB));

    // ---- XR_FB_eye_tracking_social extension commands
    (get_inst_proc_addr(instance, "xrCreateEyeTrackerFB", (PFN_xrVoidFunction*)&table->CreateEyeTrackerFB));
    (get_inst_proc_addr(instance, "xrDestroyEyeTrackerFB", (PFN_xrVoidFunction*)&table->DestroyEyeTrackerFB));
    (get_inst_proc_addr(instance, "xrGetEyeGazesFB", (PFN_xrVoidFunction*)&table->GetEyeGazesFB));

    // ---- XR_FB_passthrough_keyboard_hands extension commands
    (get_inst_proc_addr(instance, "xrPassthroughLayerSetKeyboardHandsIntensityFB", (PFN_xrVoidFunction*)&table->PassthroughLayerSetKeyboardHandsIntensityFB));

    // ---- XR_FB_haptic_pcm extension commands
    (get_inst_proc_addr(instance, "xrGetDeviceSampleRateFB", (PFN_xrVoidFunction*)&table->GetDeviceSampleRateFB));

    // ---- XR_META_virtual_keyboard extension commands
    (get_inst_proc_addr(instance, "xrCreateVirtualKeyboardMETA", (PFN_xrVoidFunction*)&table->CreateVirtualKeyboardMETA));
    (get_inst_proc_addr(instance, "xrDestroyVirtualKeyboardMETA", (PFN_xrVoidFunction*)&table->DestroyVirtualKeyboardMETA));
    (get_inst_proc_addr(instance, "xrCreateVirtualKeyboardSpaceMETA", (PFN_xrVoidFunction*)&table->CreateVirtualKeyboardSpaceMETA));
    (get_inst_proc_addr(instance, "xrSuggestVirtualKeyboardLocationMETA", (PFN_xrVoidFunction*)&table->SuggestVirtualKeyboardLocationMETA));
    (get_inst_proc_addr(instance, "xrGetVirtualKeyboardScaleMETA", (PFN_xrVoidFunction*)&table->GetVirtualKeyboardScaleMETA));
    (get_inst_proc_addr(instance, "xrSetVirtualKeyboardModelVisibilityMETA", (PFN_xrVoidFunction*)&table->SetVirtualKeyboardModelVisibilityMETA));
    (get_inst_proc_addr(instance, "xrGetVirtualKeyboardModelAnimationStatesMETA", (PFN_xrVoidFunction*)&table->GetVirtualKeyboardModelAnimationStatesMETA));
    (get_inst_proc_addr(instance, "xrGetVirtualKeyboardDirtyTexturesMETA", (PFN_xrVoidFunction*)&table->GetVirtualKeyboardDirtyTexturesMETA));
    (get_inst_proc_addr(instance, "xrGetVirtualKeyboardTextureDataMETA", (PFN_xrVoidFunction*)&table->GetVirtualKeyboardTextureDataMETA));
    (get_inst_proc_addr(instance, "xrSendVirtualKeyboardInputMETA", (PFN_xrVoidFunction*)&table->SendVirtualKeyboardInputMETA));
    (get_inst_proc_addr(instance, "xrChangeVirtualKeyboardTextContextMETA", (PFN_xrVoidFunction*)&table->ChangeVirtualKeyboardTextContextMETA));

    // ---- XR_OCULUS_external_camera extension commands
    (get_inst_proc_addr(instance, "xrEnumerateExternalCamerasOCULUS", (PFN_xrVoidFunction*)&table->EnumerateExternalCamerasOCULUS));

    // ---- XR_META_performance_metrics extension commands
    (get_inst_proc_addr(instance, "xrEnumeratePerformanceMetricsCounterPathsMETA", (PFN_xrVoidFunction*)&table->EnumeratePerformanceMetricsCounterPathsMETA));
    (get_inst_proc_addr(instance, "xrSetPerformanceMetricsStateMETA", (PFN_xrVoidFunction*)&table->SetPerformanceMetricsStateMETA));
    (get_inst_proc_addr(instance, "xrGetPerformanceMetricsStateMETA", (PFN_xrVoidFunction*)&table->GetPerformanceMetricsStateMETA));
    (get_inst_proc_addr(instance, "xrQueryPerformanceMetricsCounterMETA", (PFN_xrVoidFunction*)&table->QueryPerformanceMetricsCounterMETA));

    // ---- XR_FB_spatial_entity_storage_batch extension commands
    (get_inst_proc_addr(instance, "xrSaveSpaceListFB", (PFN_xrVoidFunction*)&table->SaveSpaceListFB));

    // ---- XR_FB_spatial_entity_user extension commands
    (get_inst_proc_addr(instance, "xrCreateSpaceUserFB", (PFN_xrVoidFunction*)&table->CreateSpaceUserFB));
    (get_inst_proc_addr(instance, "xrGetSpaceUserIdFB", (PFN_xrVoidFunction*)&table->GetSpaceUserIdFB));
    (get_inst_proc_addr(instance, "xrDestroySpaceUserFB", (PFN_xrVoidFunction*)&table->DestroySpaceUserFB));

    // ---- XR_META_passthrough_color_lut extension commands
    (get_inst_proc_addr(instance, "xrCreatePassthroughColorLutMETA", (PFN_xrVoidFunction*)&table->CreatePassthroughColorLutMETA));
    (get_inst_proc_addr(instance, "xrDestroyPassthroughColorLutMETA", (PFN_xrVoidFunction*)&table->DestroyPassthroughColorLutMETA));
    (get_inst_proc_addr(instance, "xrUpdatePassthroughColorLutMETA", (PFN_xrVoidFunction*)&table->UpdatePassthroughColorLutMETA));

    // ---- XR_QCOM_tracking_optimization_settings extension commands
    (get_inst_proc_addr(instance, "xrSetTrackingOptimizationSettingsHintQCOM", (PFN_xrVoidFunction*)&table->SetTrackingOptimizationSettingsHintQCOM));

    // ---- XR_HTC_passthrough extension commands
    (get_inst_proc_addr(instance, "xrCreatePassthroughHTC", (PFN_xrVoidFunction*)&table->CreatePassthroughHTC));
    (get_inst_proc_addr(instance, "xrDestroyPassthroughHTC", (PFN_xrVoidFunction*)&table->DestroyPassthroughHTC));

    // ---- XR_HTC_foveation extension commands
    (get_inst_proc_addr(instance, "xrApplyFoveationHTC", (PFN_xrVoidFunction*)&table->ApplyFoveationHTC));

    // ---- XR_MNDX_force_feedback_curl extension commands
    (get_inst_proc_addr(instance, "xrApplyForceFeedbackCurlMNDX", (PFN_xrVoidFunction*)&table->ApplyForceFeedbackCurlMNDX));

    // ---- XR_EXT_plane_detection extension commands
    (get_inst_proc_addr(instance, "xrCreatePlaneDetectorEXT", (PFN_xrVoidFunction*)&table->CreatePlaneDetectorEXT));
    (get_inst_proc_addr(instance, "xrDestroyPlaneDetectorEXT", (PFN_xrVoidFunction*)&table->DestroyPlaneDetectorEXT));
    (get_inst_proc_addr(instance, "xrBeginPlaneDetectionEXT", (PFN_xrVoidFunction*)&table->BeginPlaneDetectionEXT));
    (get_inst_proc_addr(instance, "xrGetPlaneDetectionStateEXT", (PFN_xrVoidFunction*)&table->GetPlaneDetectionStateEXT));
    (get_inst_proc_addr(instance, "xrGetPlaneDetectionsEXT", (PFN_xrVoidFunction*)&table->GetPlaneDetectionsEXT));
    (get_inst_proc_addr(instance, "xrGetPlanePolygonBufferEXT", (PFN_xrVoidFunction*)&table->GetPlanePolygonBufferEXT));
}


#ifdef __cplusplus
} // extern "C"
#endif

