// Copyright (c) 2017-2025 The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
// SPDX-License-Identifier: Apache-2.0 OR MIT
// *********** THIS FILE IS GENERATED - DO NOT EDIT ***********
//     See loader_source_generator.py for modifications
// ************************************************************

// Copyright (c) 2017-2025 The Khronos Group Inc.
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

#include "xr_generated_loader.hpp"

#include "api_layer_interface.hpp"
#include "exception_handling.hpp"
#include "hex_and_handles.h"
#include "loader_instance.hpp"
#include "loader_logger.hpp"
#include "loader_platform.hpp"
#include "runtime_interface.hpp"
#include "xr_generated_dispatch_table_core.h"

#include "xr_dependencies.h"
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>


// Automatically generated instance trampolines and terminators
extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetInstanceProperties(
    XrInstance                                  instance,
    XrInstanceProperties*                       instanceProperties) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetInstanceProperties");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetInstanceProperties(instance, instanceProperties);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrPollEvent(
    XrInstance                                  instance,
    XrEventDataBuffer*                          eventData) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrPollEvent");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->PollEvent(instance, eventData);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrResultToString(
    XrInstance                                  instance,
    XrResult                                    value,
    char                                        buffer[XR_MAX_RESULT_STRING_SIZE]) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrResultToString");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->ResultToString(instance, value, buffer);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrStructureTypeToString(
    XrInstance                                  instance,
    XrStructureType                             value,
    char                                        buffer[XR_MAX_STRUCTURE_NAME_SIZE]) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrStructureTypeToString");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->StructureTypeToString(instance, value, buffer);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetSystem(
    XrInstance                                  instance,
    const XrSystemGetInfo*                      getInfo,
    XrSystemId*                                 systemId) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetSystem");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetSystem(instance, getInfo, systemId);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetSystemProperties(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrSystemProperties*                         properties) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetSystemProperties");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetSystemProperties(instance, systemId, properties);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateEnvironmentBlendModes(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrViewConfigurationType                     viewConfigurationType,
    uint32_t                                    environmentBlendModeCapacityInput,
    uint32_t*                                   environmentBlendModeCountOutput,
    XrEnvironmentBlendMode*                     environmentBlendModes) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateEnvironmentBlendModes");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateEnvironmentBlendModes(instance, systemId, viewConfigurationType, environmentBlendModeCapacityInput, environmentBlendModeCountOutput, environmentBlendModes);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrCreateSession(
    XrInstance                                  instance,
    const XrSessionCreateInfo*                  createInfo,
    XrSession*                                  session) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrCreateSession");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->CreateSession(instance, createInfo, session);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrDestroySession(
    XrSession                                   session) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrDestroySession");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->DestroySession(session);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateReferenceSpaces(
    XrSession                                   session,
    uint32_t                                    spaceCapacityInput,
    uint32_t*                                   spaceCountOutput,
    XrReferenceSpaceType*                       spaces) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateReferenceSpaces");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateReferenceSpaces(session, spaceCapacityInput, spaceCountOutput, spaces);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrCreateReferenceSpace(
    XrSession                                   session,
    const XrReferenceSpaceCreateInfo*           createInfo,
    XrSpace*                                    space) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrCreateReferenceSpace");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->CreateReferenceSpace(session, createInfo, space);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetReferenceSpaceBoundsRect(
    XrSession                                   session,
    XrReferenceSpaceType                        referenceSpaceType,
    XrExtent2Df*                                bounds) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetReferenceSpaceBoundsRect");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetReferenceSpaceBoundsRect(session, referenceSpaceType, bounds);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrCreateActionSpace(
    XrSession                                   session,
    const XrActionSpaceCreateInfo*              createInfo,
    XrSpace*                                    space) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrCreateActionSpace");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->CreateActionSpace(session, createInfo, space);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrLocateSpace(
    XrSpace                                     space,
    XrSpace                                     baseSpace,
    XrTime                                      time,
    XrSpaceLocation*                            location) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrLocateSpace");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->LocateSpace(space, baseSpace, time, location);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrDestroySpace(
    XrSpace                                     space) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrDestroySpace");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->DestroySpace(space);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateViewConfigurations(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    uint32_t                                    viewConfigurationTypeCapacityInput,
    uint32_t*                                   viewConfigurationTypeCountOutput,
    XrViewConfigurationType*                    viewConfigurationTypes) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateViewConfigurations");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateViewConfigurations(instance, systemId, viewConfigurationTypeCapacityInput, viewConfigurationTypeCountOutput, viewConfigurationTypes);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetViewConfigurationProperties(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrViewConfigurationType                     viewConfigurationType,
    XrViewConfigurationProperties*              configurationProperties) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetViewConfigurationProperties");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetViewConfigurationProperties(instance, systemId, viewConfigurationType, configurationProperties);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateViewConfigurationViews(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrViewConfigurationType                     viewConfigurationType,
    uint32_t                                    viewCapacityInput,
    uint32_t*                                   viewCountOutput,
    XrViewConfigurationView*                    views) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateViewConfigurationViews");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateViewConfigurationViews(instance, systemId, viewConfigurationType, viewCapacityInput, viewCountOutput, views);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateSwapchainFormats(
    XrSession                                   session,
    uint32_t                                    formatCapacityInput,
    uint32_t*                                   formatCountOutput,
    int64_t*                                    formats) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateSwapchainFormats");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateSwapchainFormats(session, formatCapacityInput, formatCountOutput, formats);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrCreateSwapchain(
    XrSession                                   session,
    const XrSwapchainCreateInfo*                createInfo,
    XrSwapchain*                                swapchain) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrCreateSwapchain");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->CreateSwapchain(session, createInfo, swapchain);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrDestroySwapchain(
    XrSwapchain                                 swapchain) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrDestroySwapchain");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->DestroySwapchain(swapchain);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateSwapchainImages(
    XrSwapchain                                 swapchain,
    uint32_t                                    imageCapacityInput,
    uint32_t*                                   imageCountOutput,
    XrSwapchainImageBaseHeader*                 images) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateSwapchainImages");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateSwapchainImages(swapchain, imageCapacityInput, imageCountOutput, images);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrAcquireSwapchainImage(
    XrSwapchain                                 swapchain,
    const XrSwapchainImageAcquireInfo*          acquireInfo,
    uint32_t*                                   index) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrAcquireSwapchainImage");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->AcquireSwapchainImage(swapchain, acquireInfo, index);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrWaitSwapchainImage(
    XrSwapchain                                 swapchain,
    const XrSwapchainImageWaitInfo*             waitInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrWaitSwapchainImage");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->WaitSwapchainImage(swapchain, waitInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrReleaseSwapchainImage(
    XrSwapchain                                 swapchain,
    const XrSwapchainImageReleaseInfo*          releaseInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrReleaseSwapchainImage");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->ReleaseSwapchainImage(swapchain, releaseInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrBeginSession(
    XrSession                                   session,
    const XrSessionBeginInfo*                   beginInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrBeginSession");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->BeginSession(session, beginInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEndSession(
    XrSession                                   session) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEndSession");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EndSession(session);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrRequestExitSession(
    XrSession                                   session) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrRequestExitSession");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->RequestExitSession(session);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrWaitFrame(
    XrSession                                   session,
    const XrFrameWaitInfo*                      frameWaitInfo,
    XrFrameState*                               frameState) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrWaitFrame");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->WaitFrame(session, frameWaitInfo, frameState);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrBeginFrame(
    XrSession                                   session,
    const XrFrameBeginInfo*                     frameBeginInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrBeginFrame");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->BeginFrame(session, frameBeginInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEndFrame(
    XrSession                                   session,
    const XrFrameEndInfo*                       frameEndInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEndFrame");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EndFrame(session, frameEndInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrLocateViews(
    XrSession                                   session,
    const XrViewLocateInfo*                     viewLocateInfo,
    XrViewState*                                viewState,
    uint32_t                                    viewCapacityInput,
    uint32_t*                                   viewCountOutput,
    XrView*                                     views) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrLocateViews");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->LocateViews(session, viewLocateInfo, viewState, viewCapacityInput, viewCountOutput, views);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrStringToPath(
    XrInstance                                  instance,
    const char*                                 pathString,
    XrPath*                                     path) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrStringToPath");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->StringToPath(instance, pathString, path);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrPathToString(
    XrInstance                                  instance,
    XrPath                                      path,
    uint32_t                                    bufferCapacityInput,
    uint32_t*                                   bufferCountOutput,
    char*                                       buffer) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrPathToString");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->PathToString(instance, path, bufferCapacityInput, bufferCountOutput, buffer);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrCreateActionSet(
    XrInstance                                  instance,
    const XrActionSetCreateInfo*                createInfo,
    XrActionSet*                                actionSet) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrCreateActionSet");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->CreateActionSet(instance, createInfo, actionSet);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrDestroyActionSet(
    XrActionSet                                 actionSet) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrDestroyActionSet");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->DestroyActionSet(actionSet);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrCreateAction(
    XrActionSet                                 actionSet,
    const XrActionCreateInfo*                   createInfo,
    XrAction*                                   action) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrCreateAction");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->CreateAction(actionSet, createInfo, action);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrDestroyAction(
    XrAction                                    action) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrDestroyAction");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->DestroyAction(action);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrSuggestInteractionProfileBindings(
    XrInstance                                  instance,
    const XrInteractionProfileSuggestedBinding* suggestedBindings) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrSuggestInteractionProfileBindings");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->SuggestInteractionProfileBindings(instance, suggestedBindings);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrAttachSessionActionSets(
    XrSession                                   session,
    const XrSessionActionSetsAttachInfo*        attachInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrAttachSessionActionSets");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->AttachSessionActionSets(session, attachInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetCurrentInteractionProfile(
    XrSession                                   session,
    XrPath                                      topLevelUserPath,
    XrInteractionProfileState*                  interactionProfile) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetCurrentInteractionProfile");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetCurrentInteractionProfile(session, topLevelUserPath, interactionProfile);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStateBoolean(
    XrSession                                   session,
    const XrActionStateGetInfo*                 getInfo,
    XrActionStateBoolean*                       state) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetActionStateBoolean");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetActionStateBoolean(session, getInfo, state);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStateFloat(
    XrSession                                   session,
    const XrActionStateGetInfo*                 getInfo,
    XrActionStateFloat*                         state) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetActionStateFloat");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetActionStateFloat(session, getInfo, state);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStateVector2f(
    XrSession                                   session,
    const XrActionStateGetInfo*                 getInfo,
    XrActionStateVector2f*                      state) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetActionStateVector2f");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetActionStateVector2f(session, getInfo, state);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStatePose(
    XrSession                                   session,
    const XrActionStateGetInfo*                 getInfo,
    XrActionStatePose*                          state) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetActionStatePose");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetActionStatePose(session, getInfo, state);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrSyncActions(
    XrSession                                   session,
    const XrActionsSyncInfo*                    syncInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrSyncActions");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->SyncActions(session, syncInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrEnumerateBoundSourcesForAction(
    XrSession                                   session,
    const XrBoundSourcesForActionEnumerateInfo* enumerateInfo,
    uint32_t                                    sourceCapacityInput,
    uint32_t*                                   sourceCountOutput,
    XrPath*                                     sources) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrEnumerateBoundSourcesForAction");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->EnumerateBoundSourcesForAction(session, enumerateInfo, sourceCapacityInput, sourceCountOutput, sources);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrGetInputSourceLocalizedName(
    XrSession                                   session,
    const XrInputSourceLocalizedNameGetInfo*    getInfo,
    uint32_t                                    bufferCapacityInput,
    uint32_t*                                   bufferCountOutput,
    char*                                       buffer) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrGetInputSourceLocalizedName");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->GetInputSourceLocalizedName(session, getInfo, bufferCapacityInput, bufferCountOutput, buffer);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrApplyHapticFeedback(
    XrSession                                   session,
    const XrHapticActionInfo*                   hapticActionInfo,
    const XrHapticBaseHeader*                   hapticFeedback) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrApplyHapticFeedback");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->ApplyHapticFeedback(session, hapticActionInfo, hapticFeedback);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrStopHapticFeedback(
    XrSession                                   session,
    const XrHapticActionInfo*                   hapticActionInfo) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrStopHapticFeedback");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->StopHapticFeedback(session, hapticActionInfo);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK

extern "C" LOADER_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrLocateSpaces(
    XrSession                                   session,
    const XrSpacesLocateInfo*                   locateInfo,
    XrSpaceLocations*                           spaceLocations) XRLOADER_ABI_TRY {
    LoaderInstance* loader_instance;
    XrResult result = ActiveLoaderInstance::Get(&loader_instance, "xrLocateSpaces");
    if (XR_SUCCEEDED(result)) {
        result = loader_instance->DispatchTable()->LocateSpaces(session, locateInfo, spaceLocations);
    }
    return result;
}
XRLOADER_ABI_CATCH_FALLBACK


