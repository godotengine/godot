// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#include "loader_logger.hpp"

#include "extra_algorithms.h"
#include "hex_and_handles.h"
#include "loader_logger_recorders.hpp"
#include "platform_utils.hpp"

#include <openxr/openxr.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// For routing platform_utils.hpp messages into the LoaderLogger.
void LogPlatformUtilsError(const std::string& message) { LoaderLogger::LogErrorMessage("platform_utils", message); }

bool LoaderLogRecorder::LogDebugUtilsMessage(XrDebugUtilsMessageSeverityFlagsEXT /*message_severity*/,
                                             XrDebugUtilsMessageTypeFlagsEXT /*message_type*/,
                                             const XrDebugUtilsMessengerCallbackDataEXT* /*callback_data*/) {
    return false;
}

// Utility functions for converting to/from XR_EXT_debug_utils values

XrLoaderLogMessageSeverityFlags DebugUtilsSeveritiesToLoaderLogMessageSeverities(
    XrDebugUtilsMessageSeverityFlagsEXT utils_severities) {
    XrLoaderLogMessageSeverityFlags log_severities = 0UL;
    if ((utils_severities & XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) != 0u) {
        log_severities |= XR_LOADER_LOG_MESSAGE_SEVERITY_VERBOSE_BIT;
    }
    if ((utils_severities & XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0u) {
        log_severities |= XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT;
    }
    if ((utils_severities & XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0u) {
        log_severities |= XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT;
    }
    if ((utils_severities & XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u) {
        log_severities |= XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT;
    }
    return log_severities;
}

XrDebugUtilsMessageSeverityFlagsEXT LoaderLogMessageSeveritiesToDebugUtilsMessageSeverities(
    XrLoaderLogMessageSeverityFlags log_severities) {
    XrDebugUtilsMessageSeverityFlagsEXT utils_severities = 0UL;
    if ((log_severities & XR_LOADER_LOG_MESSAGE_SEVERITY_VERBOSE_BIT) != 0u) {
        utils_severities |= XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    }
    if ((log_severities & XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT) != 0u) {
        utils_severities |= XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    }
    if ((log_severities & XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT) != 0u) {
        utils_severities |= XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    }
    if ((log_severities & XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT) != 0u) {
        utils_severities |= XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    }
    return utils_severities;
}

XrLoaderLogMessageTypeFlagBits DebugUtilsMessageTypesToLoaderLogMessageTypes(XrDebugUtilsMessageTypeFlagsEXT utils_types) {
    XrLoaderLogMessageTypeFlagBits log_types = 0UL;
    if ((utils_types & XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) != 0u) {
        log_types |= XR_LOADER_LOG_MESSAGE_TYPE_GENERAL_BIT;
    }
    if ((utils_types & XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0u) {
        log_types |= XR_LOADER_LOG_MESSAGE_TYPE_SPECIFICATION_BIT;
    }
    if ((utils_types & XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0u) {
        log_types |= XR_LOADER_LOG_MESSAGE_TYPE_PERFORMANCE_BIT;
    }
    return log_types;
}

XrDebugUtilsMessageTypeFlagsEXT LoaderLogMessageTypesToDebugUtilsMessageTypes(XrLoaderLogMessageTypeFlagBits log_types) {
    XrDebugUtilsMessageTypeFlagsEXT utils_types = 0UL;
    if ((log_types & XR_LOADER_LOG_MESSAGE_TYPE_GENERAL_BIT) != 0u) {
        utils_types |= XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    }
    if ((log_types & XR_LOADER_LOG_MESSAGE_TYPE_SPECIFICATION_BIT) != 0u) {
        utils_types |= XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    }
    if ((log_types & XR_LOADER_LOG_MESSAGE_TYPE_PERFORMANCE_BIT) != 0u) {
        utils_types |= XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    }
    return utils_types;
}

LoaderLogger::LoaderLogger() {
    std::string debug_string = PlatformUtilsGetEnv("XR_LOADER_DEBUG");

    // Add an error logger by default so that we at least get errors out to std::cerr.
    // Normally we enable stderr output. But if the XR_LOADER_DEBUG environment variable is
    // present as "none" then we don't.
    if (debug_string != "none") {
        AddLogRecorder(MakeStdErrLoaderLogRecorder(nullptr));
#ifdef __ANDROID__
        // Add a logcat logger by default.
        AddLogRecorder(MakeLogcatLoaderLogRecorder());
#endif  // __ANDROID__
    }

#ifdef _WIN32
    // Add an debugger logger by default so that we at least get errors out to the debugger.
    AddLogRecorder(MakeDebuggerLoaderLogRecorder(nullptr));
#endif

    // If the environment variable to enable loader debugging is set, then enable the
    // appropriate logging out to std::cout.
    if (!debug_string.empty()) {
        XrLoaderLogMessageSeverityFlags debug_flags = {};
        if (debug_string == "error") {
            debug_flags = XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT;
        } else if (debug_string == "warn") {
            debug_flags = XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT | XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT;
        } else if (debug_string == "info") {
            debug_flags = XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT | XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT |
                          XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT;
        } else if (debug_string == "all" || debug_string == "verbose") {
            debug_flags = XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT | XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT |
                          XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT | XR_LOADER_LOG_MESSAGE_SEVERITY_VERBOSE_BIT;
        }
        AddLogRecorder(MakeStdOutLoaderLogRecorder(nullptr, debug_flags));
    }
}

void LoaderLogger::AddLogRecorder(std::unique_ptr<LoaderLogRecorder>&& recorder) {
    std::unique_lock<std::shared_timed_mutex> lock(_mutex);
    _recorders.push_back(std::move(recorder));
}

void LoaderLogger::AddLogRecorderForXrInstance(XrInstance instance, std::unique_ptr<LoaderLogRecorder>&& recorder) {
    std::unique_lock<std::shared_timed_mutex> lock(_mutex);
    _recordersByInstance[instance].insert(recorder->UniqueId());
    _recorders.emplace_back(std::move(recorder));
}

void LoaderLogger::RemoveLogRecorder(uint64_t unique_id) {
    std::unique_lock<std::shared_timed_mutex> lock(_mutex);
    vector_remove_if_and_erase(
        _recorders, [=](std::unique_ptr<LoaderLogRecorder> const& recorder) { return recorder->UniqueId() == unique_id; });
    for (auto& recorders : _recordersByInstance) {
        auto& messengersForInstance = recorders.second;
        if (messengersForInstance.count(unique_id) > 0) {
            messengersForInstance.erase(unique_id);
        }
    }
}

void LoaderLogger::RemoveLogRecordersForXrInstance(XrInstance instance) {
    std::unique_lock<std::shared_timed_mutex> lock(_mutex);
    if (_recordersByInstance.find(instance) != _recordersByInstance.end()) {
        auto recorders = _recordersByInstance[instance];
        vector_remove_if_and_erase(_recorders, [=](std::unique_ptr<LoaderLogRecorder> const& recorder) {
            return recorders.find(recorder->UniqueId()) != recorders.end();
        });
        _recordersByInstance.erase(instance);
    }
}

bool LoaderLogger::LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity, XrLoaderLogMessageTypeFlags message_type,
                              const std::string& message_id, const std::string& command_name, const std::string& message,
                              const std::vector<XrSdkLogObjectInfo>& objects) {
    XrLoaderLogMessengerCallbackData callback_data = {};
    callback_data.message_id = message_id.c_str();
    callback_data.command_name = command_name.c_str();
    callback_data.message = message.c_str();

    auto names_and_labels = data_.PopulateNamesAndLabels(objects);
    callback_data.objects = names_and_labels.sdk_objects.empty() ? nullptr : names_and_labels.sdk_objects.data();
    callback_data.object_count = static_cast<uint8_t>(names_and_labels.objects.size());

    callback_data.session_labels = names_and_labels.labels.empty() ? nullptr : names_and_labels.labels.data();
    callback_data.session_labels_count = static_cast<uint8_t>(names_and_labels.labels.size());

    std::shared_lock<std::shared_timed_mutex> lock(_mutex);
    bool exit_app = false;
    for (std::unique_ptr<LoaderLogRecorder>& recorder : _recorders) {
        if ((recorder->MessageSeverities() & message_severity) == message_severity &&
            (recorder->MessageTypes() & message_type) == message_type) {
            exit_app |= recorder->LogMessage(message_severity, message_type, &callback_data);
        }
    }
    return exit_app;
}

// Extension-specific logging functions
bool LoaderLogger::LogDebugUtilsMessage(XrDebugUtilsMessageSeverityFlagsEXT message_severity,
                                        XrDebugUtilsMessageTypeFlagsEXT message_type,
                                        const XrDebugUtilsMessengerCallbackDataEXT* callback_data) {
    bool exit_app = false;
    XrLoaderLogMessageSeverityFlags log_message_severity = DebugUtilsSeveritiesToLoaderLogMessageSeverities(message_severity);
    XrLoaderLogMessageTypeFlags log_message_type = DebugUtilsMessageTypesToLoaderLogMessageTypes(message_type);

    AugmentedCallbackData augmented_data;
    data_.WrapCallbackData(&augmented_data, callback_data);

    // Loop through the recorders
    std::shared_lock<std::shared_timed_mutex> lock(_mutex);
    for (std::unique_ptr<LoaderLogRecorder>& recorder : _recorders) {
        // Only send the message if it's a debug utils recorder and of the type the recorder cares about.
        if (recorder->Type() != XR_LOADER_LOG_DEBUG_UTILS ||
            (recorder->MessageSeverities() & log_message_severity) != log_message_severity ||
            (recorder->MessageTypes() & log_message_type) != log_message_type) {
            continue;
        }

        exit_app |= recorder->LogDebugUtilsMessage(message_severity, message_type, augmented_data.exported_data);
    }
    return exit_app;
}

void LoaderLogger::AddObjectName(uint64_t object_handle, XrObjectType object_type, const std::string& object_name) {
    data_.AddObjectName(object_handle, object_type, object_name);
}

void LoaderLogger::BeginLabelRegion(XrSession session, const XrDebugUtilsLabelEXT* label_info) {
    data_.BeginLabelRegion(session, *label_info);
}

void LoaderLogger::EndLabelRegion(XrSession session) { data_.EndLabelRegion(session); }

void LoaderLogger::InsertLabel(XrSession session, const XrDebugUtilsLabelEXT* label_info) {
    data_.InsertLabel(session, *label_info);
}

void LoaderLogger::DeleteSessionLabels(XrSession session) { data_.DeleteSessionLabels(session); }
