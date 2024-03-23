// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Mark Young <marky@lunarg.com>
//

#include "loader_logger_recorders.hpp"

#include "hex_and_handles.h"
#include "loader_logger.hpp"

#include <openxr/openxr.h>

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#ifdef __ANDROID__
#include "android/log.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

// Anonymous namespace to keep these types private
namespace {
void OutputMessageToStream(std::ostream& os, XrLoaderLogMessageSeverityFlagBits message_severity,
                           XrLoaderLogMessageTypeFlags message_type, const XrLoaderLogMessengerCallbackData* callback_data) {
    if (XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT > message_severity) {
        os << "Verbose [";
    } else if (XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT > message_severity) {
        os << "Info [";
    } else if (XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT > message_severity) {
        os << "Warning [";
    } else {
        os << "Error [";
    }
    switch (message_type) {
        case XR_LOADER_LOG_MESSAGE_TYPE_GENERAL_BIT:
            os << "GENERAL";
            break;
        case XR_LOADER_LOG_MESSAGE_TYPE_SPECIFICATION_BIT:
            os << "SPEC";
            break;
        case XR_LOADER_LOG_MESSAGE_TYPE_PERFORMANCE_BIT:
            os << "PERF";
            break;
        default:
            os << "UNKNOWN";
            break;
    }
    os << " | " << callback_data->command_name << " | " << callback_data->message_id << "] : " << callback_data->message
       << std::endl;

    for (uint32_t obj = 0; obj < callback_data->object_count; ++obj) {
        os << "    Object[" << obj << "] = " << callback_data->objects[obj].ToString();
        os << std::endl;
    }
    for (uint32_t label = 0; label < callback_data->session_labels_count; ++label) {
        os << "    SessionLabel[" << std::to_string(label) << "] = " << callback_data->session_labels[label].labelName;
        os << std::endl;
    }
}

// With std::cerr: Standard Error logger, always on for now
// With std::cout: Standard Output logger used with XR_LOADER_DEBUG
class OstreamLoaderLogRecorder : public LoaderLogRecorder {
   public:
    OstreamLoaderLogRecorder(std::ostream& os, void* user_data, XrLoaderLogMessageSeverityFlags flags);

    bool LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity, XrLoaderLogMessageTypeFlags message_type,
                    const XrLoaderLogMessengerCallbackData* callback_data) override;

   private:
    std::ostream& os_;
};

// Debug Utils logger used with XR_EXT_debug_utils
class DebugUtilsLogRecorder : public LoaderLogRecorder {
   public:
    DebugUtilsLogRecorder(const XrDebugUtilsMessengerCreateInfoEXT* create_info, XrDebugUtilsMessengerEXT debug_messenger);

    bool LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity, XrLoaderLogMessageTypeFlags message_type,
                    const XrLoaderLogMessengerCallbackData* callback_data) override;

    // Extension-specific logging functions
    bool LogDebugUtilsMessage(XrDebugUtilsMessageSeverityFlagsEXT message_severity, XrDebugUtilsMessageTypeFlagsEXT message_type,
                              const XrDebugUtilsMessengerCallbackDataEXT* callback_data) override;

   private:
    PFN_xrDebugUtilsMessengerCallbackEXT _user_callback;
};
#ifdef __ANDROID__

class LogcatLoaderLogRecorder : public LoaderLogRecorder {
   public:
    LogcatLoaderLogRecorder();

    bool LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity, XrLoaderLogMessageTypeFlags message_type,
                    const XrLoaderLogMessengerCallbackData* callback_data) override;
};
#endif

#ifdef _WIN32
// Output to debugger
class DebuggerLoaderLogRecorder : public LoaderLogRecorder {
   public:
    DebuggerLoaderLogRecorder(void* user_data, XrLoaderLogMessageSeverityFlags flags);

    bool LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity, XrLoaderLogMessageTypeFlags message_type,
                    const XrLoaderLogMessengerCallbackData* callback_data) override;
};
#endif

// Unified stdout/stderr logger
OstreamLoaderLogRecorder::OstreamLoaderLogRecorder(std::ostream& os, void* user_data, XrLoaderLogMessageSeverityFlags flags)
    : LoaderLogRecorder(XR_LOADER_LOG_STDOUT, user_data, flags, 0xFFFFFFFFUL), os_(os) {
    // Automatically start
    Start();
}

bool OstreamLoaderLogRecorder::LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity,
                                          XrLoaderLogMessageTypeFlags message_type,
                                          const XrLoaderLogMessengerCallbackData* callback_data) {
    if (_active && 0 != (_message_severities & message_severity) && 0 != (_message_types & message_type)) {
        OutputMessageToStream(os_, message_severity, message_type, callback_data);
    }

    // Return of "true" means that we should exit the application after the logged message.  We
    // don't want to do that for our internal logging.  Only let a user return true.
    return false;
}

// A logger associated with the XR_EXT_debug_utils extension

DebugUtilsLogRecorder::DebugUtilsLogRecorder(const XrDebugUtilsMessengerCreateInfoEXT* create_info,
                                             XrDebugUtilsMessengerEXT debug_messenger)
    : LoaderLogRecorder(XR_LOADER_LOG_DEBUG_UTILS, static_cast<void*>(create_info->userData),
                        DebugUtilsSeveritiesToLoaderLogMessageSeverities(create_info->messageSeverities),
                        DebugUtilsMessageTypesToLoaderLogMessageTypes(create_info->messageTypes)),
      _user_callback(create_info->userCallback) {
    // Use the debug messenger value to uniquely identify this logger with that messenger
    _unique_id = MakeHandleGeneric(debug_messenger);
    Start();
}

// Extension-specific logging functions
bool DebugUtilsLogRecorder::LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity,
                                       XrLoaderLogMessageTypeFlags message_type,
                                       const XrLoaderLogMessengerCallbackData* callback_data) {
    bool should_exit = false;
    if (_active && 0 != (_message_severities & message_severity) && 0 != (_message_types & message_type)) {
        XrDebugUtilsMessageSeverityFlagsEXT utils_severity = DebugUtilsSeveritiesToLoaderLogMessageSeverities(message_severity);
        XrDebugUtilsMessageTypeFlagsEXT utils_type = LoaderLogMessageTypesToDebugUtilsMessageTypes(message_type);

        // Convert the loader log message into the debug utils log message information
        XrDebugUtilsMessengerCallbackDataEXT utils_callback_data{};
        utils_callback_data.type = XR_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
        utils_callback_data.messageId = callback_data->message_id;
        utils_callback_data.functionName = callback_data->command_name;
        utils_callback_data.message = callback_data->message;

        XrDebugUtilsObjectNameInfoEXT example_utils_info{};
        example_utils_info.type = XR_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        std::vector<XrDebugUtilsObjectNameInfoEXT> utils_objects(callback_data->object_count, example_utils_info);
        for (uint8_t object = 0; object < callback_data->object_count; ++object) {
            utils_objects[object].objectHandle = callback_data->objects[object].handle;
            utils_objects[object].objectType = callback_data->objects[object].type;
            utils_objects[object].objectName = callback_data->objects[object].name.c_str();
        }
        utils_callback_data.objectCount = callback_data->object_count;
        utils_callback_data.objects = utils_objects.data();
        utils_callback_data.sessionLabelCount = callback_data->session_labels_count;
        utils_callback_data.sessionLabels = callback_data->session_labels;

        // Call the user callback with the appropriate info
        // Return of "true" means that we should exit the application after the logged message.
        should_exit = (_user_callback(utils_severity, utils_type, &utils_callback_data, _user_data) == XR_TRUE);
    }

    return should_exit;
}

bool DebugUtilsLogRecorder::LogDebugUtilsMessage(XrDebugUtilsMessageSeverityFlagsEXT message_severity,
                                                 XrDebugUtilsMessageTypeFlagsEXT message_type,
                                                 const XrDebugUtilsMessengerCallbackDataEXT* callback_data) {
    // Call the user callback with the appropriate info
    // Return of "true" means that we should exit the application after the logged message.
    return (_user_callback(message_severity, message_type, callback_data, _user_data) == XR_TRUE);
}

#ifdef __ANDROID__

static inline android_LogPriority LoaderToAndroidLogPriority(XrLoaderLogMessageSeverityFlags message_severity) {
    if (0 != (message_severity & XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT)) {
        return ANDROID_LOG_ERROR;
    }
    if (0 != (message_severity & XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT)) {
        return ANDROID_LOG_WARN;
    }
    if (0 != (message_severity & XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT)) {
        return ANDROID_LOG_INFO;
    }
    return ANDROID_LOG_VERBOSE;
}

LogcatLoaderLogRecorder::LogcatLoaderLogRecorder()
    : LoaderLogRecorder(XR_LOADER_LOG_LOGCAT, nullptr,
                        XR_LOADER_LOG_MESSAGE_SEVERITY_VERBOSE_BIT | XR_LOADER_LOG_MESSAGE_SEVERITY_INFO_BIT |
                            XR_LOADER_LOG_MESSAGE_SEVERITY_WARNING_BIT | XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT,
                        0xFFFFFFFFUL) {
    // Automatically start
    Start();
}

bool LogcatLoaderLogRecorder::LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity,
                                         XrLoaderLogMessageTypeFlags message_type,
                                         const XrLoaderLogMessengerCallbackData* callback_data) {
    if (_active && 0 != (_message_severities & message_severity) && 0 != (_message_types & message_type)) {
        std::stringstream ss;
        OutputMessageToStream(ss, message_severity, message_type, callback_data);
        __android_log_write(LoaderToAndroidLogPriority(message_severity), "OpenXR-Loader", ss.str().c_str());
    }

    // Return of "true" means that we should exit the application after the logged message.  We
    // don't want to do that for our internal logging.  Only let a user return true.
    return false;
}
#endif  // __ANDROID__

#ifdef _WIN32
// Unified stdout/stderr logger
DebuggerLoaderLogRecorder::DebuggerLoaderLogRecorder(void* user_data, XrLoaderLogMessageSeverityFlags flags)
    : LoaderLogRecorder(XR_LOADER_LOG_DEBUGGER, user_data, flags, 0xFFFFFFFFUL) {
    // Automatically start
    Start();
}

bool DebuggerLoaderLogRecorder::LogMessage(XrLoaderLogMessageSeverityFlagBits message_severity,
                                           XrLoaderLogMessageTypeFlags message_type,
                                           const XrLoaderLogMessengerCallbackData* callback_data) {
    if (_active && 0 != (_message_severities & message_severity) && 0 != (_message_types & message_type)) {
        std::stringstream ss;
        OutputMessageToStream(ss, message_severity, message_type, callback_data);

        OutputDebugStringA(ss.str().c_str());
    }

    // Return of "true" means that we should exit the application after the logged message.  We
    // don't want to do that for our internal logging.  Only let a user return true.
    return false;
}
#endif
}  // namespace

std::unique_ptr<LoaderLogRecorder> MakeStdOutLoaderLogRecorder(void* user_data, XrLoaderLogMessageSeverityFlags flags) {
    std::unique_ptr<LoaderLogRecorder> recorder(new OstreamLoaderLogRecorder(std::cout, user_data, flags));
    return recorder;
}

std::unique_ptr<LoaderLogRecorder> MakeStdErrLoaderLogRecorder(void* user_data) {
    std::unique_ptr<LoaderLogRecorder> recorder(
        new OstreamLoaderLogRecorder(std::cerr, user_data, XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT));
    return recorder;
}

std::unique_ptr<LoaderLogRecorder> MakeDebugUtilsLoaderLogRecorder(const XrDebugUtilsMessengerCreateInfoEXT* create_info,
                                                                   XrDebugUtilsMessengerEXT debug_messenger) {
    std::unique_ptr<LoaderLogRecorder> recorder(new DebugUtilsLogRecorder(create_info, debug_messenger));
    return recorder;
}

#ifdef __ANDROID__
std::unique_ptr<LoaderLogRecorder> MakeLogcatLoaderLogRecorder() {
    std::unique_ptr<LoaderLogRecorder> recorder(new LogcatLoaderLogRecorder());
    return recorder;
}
#endif

#ifdef _WIN32
std::unique_ptr<LoaderLogRecorder> MakeDebuggerLoaderLogRecorder(void* user_data) {
    std::unique_ptr<LoaderLogRecorder> recorder(new DebuggerLoaderLogRecorder(user_data, XR_LOADER_LOG_MESSAGE_SEVERITY_ERROR_BIT));
    return recorder;
}
#endif
