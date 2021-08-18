/*
*
* Copyright (c) 2016 The Khronos Group Inc.
* Copyright (c) 2016 Valve Corporation
* Copyright (c) 2016 LunarG, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Author: Mark Lobodzinski <mark@lunarg.com>
*
*/
#pragma once

// Linked list node for tree of debug callbacks
typedef struct VkDebugReportContent {
    VkDebugReportCallbackEXT msgCallback;
    PFN_vkDebugReportCallbackEXT pfnMsgCallback;
    VkFlags msgFlags;
} VkDebugReportContent;

typedef struct VkDebugUtilsMessengerContent {
    VkDebugUtilsMessengerEXT messenger;
    VkDebugUtilsMessageSeverityFlagsEXT messageSeverity;
    VkDebugUtilsMessageTypeFlagsEXT messageType;
    PFN_vkDebugUtilsMessengerCallbackEXT pfnUserCallback;
} VkDebugUtilsMessengerContent;

typedef struct VkLayerDbgFunctionNode_ {
    bool is_messenger;
    union {
        VkDebugReportContent report;
        VkDebugUtilsMessengerContent messenger;
    };
    void *pUserData;
    struct VkLayerDbgFunctionNode_ *pNext;
} VkLayerDbgFunctionNode;
