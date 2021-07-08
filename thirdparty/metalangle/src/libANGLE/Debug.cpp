//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Debug.cpp: Defines debug state used for GL_KHR_debug

#include "libANGLE/Debug.h"

#include "common/debug.h"

#include <algorithm>
#include <tuple>

namespace
{
const char *GLSeverityToString(GLenum severity)
{
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:
            return "HIGH";
        case GL_DEBUG_SEVERITY_MEDIUM:
            return "MEDIUM";
        case GL_DEBUG_SEVERITY_LOW:
            return "LOW";
        case GL_DEBUG_SEVERITY_NOTIFICATION:
        default:
            return "NOTIFICATION";
    }
}

const char *EGLMessageTypeToString(egl::MessageType messageType)
{
    switch (messageType)
    {
        case egl::MessageType::Critical:
            return "CRITICAL";
        case egl::MessageType::Error:
            return "ERROR";
        case egl::MessageType::Warn:
            return "WARNING";
        case egl::MessageType::Info:
        default:
            return "INFO";
    }
}

const char *GLMessageTypeToString(GLenum type)
{
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:
            return "error";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            return "deprecated behavior";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            return "undefined behavior";
        case GL_DEBUG_TYPE_PORTABILITY:
            return "portability";
        case GL_DEBUG_TYPE_PERFORMANCE:
            return "performance";
        case GL_DEBUG_TYPE_MARKER:
            return "marker";
        case GL_DEBUG_TYPE_PUSH_GROUP:
            return "start of group";
        case GL_DEBUG_TYPE_POP_GROUP:
            return "end of group";
        case GL_DEBUG_TYPE_OTHER:
        default:
            return "other message";
    }
}
}  // namespace

namespace gl
{

Debug::Control::Control() {}

Debug::Control::~Control() {}

Debug::Control::Control(const Control &other) = default;

Debug::Group::Group() {}

Debug::Group::~Group() {}

Debug::Group::Group(const Group &other) = default;

Debug::Debug(bool initialDebugState)
    : mOutputEnabled(initialDebugState),
      mCallbackFunction(nullptr),
      mCallbackUserParam(nullptr),
      mMessages(),
      mMaxLoggedMessages(0),
      mOutputSynchronous(false),
      mGroups()
{
    pushDefaultGroup();
}

Debug::~Debug() {}

void Debug::setMaxLoggedMessages(GLuint maxLoggedMessages)
{
    mMaxLoggedMessages = maxLoggedMessages;
}

void Debug::setOutputEnabled(bool enabled)
{
    mOutputEnabled = enabled;
}

bool Debug::isOutputEnabled() const
{
    return mOutputEnabled;
}

void Debug::setOutputSynchronous(bool synchronous)
{
    mOutputSynchronous = synchronous;
}

bool Debug::isOutputSynchronous() const
{
    return mOutputSynchronous;
}

void Debug::setCallback(GLDEBUGPROCKHR callback, const void *userParam)
{
    mCallbackFunction  = callback;
    mCallbackUserParam = userParam;
}

GLDEBUGPROCKHR Debug::getCallback() const
{
    return mCallbackFunction;
}

const void *Debug::getUserParam() const
{
    return mCallbackUserParam;
}

void Debug::insertMessage(GLenum source,
                          GLenum type,
                          GLuint id,
                          GLenum severity,
                          const std::string &message,
                          gl::LogSeverity logSeverity) const
{
    std::string messageCopy(message);
    insertMessage(source, type, id, severity, std::move(messageCopy), logSeverity);
}

void Debug::insertMessage(GLenum source,
                          GLenum type,
                          GLuint id,
                          GLenum severity,
                          std::string &&message,
                          gl::LogSeverity logSeverity) const
{
    {
        // output all messages to the debug log
        const char *messageTypeString = GLMessageTypeToString(type);
        const char *severityString    = GLSeverityToString(severity);
        std::ostringstream messageStream;
        messageStream << "GL " << messageTypeString << ": " << severityString << ": " << message;
        switch (logSeverity)
        {
            case gl::LOG_FATAL:
                FATAL() << messageStream.str();
                break;
            case gl::LOG_ERR:
                ERR() << messageStream.str();
                break;
            case gl::LOG_WARN:
                WARN() << messageStream.str();
                break;
            case gl::LOG_INFO:
                INFO() << messageStream.str();
                break;
            case gl::LOG_EVENT:
                ANGLE_LOG(EVENT) << messageStream.str();
                break;
        }
    }

    if (!isMessageEnabled(source, type, id, severity))
    {
        return;
    }

    if (mCallbackFunction != nullptr)
    {
        // TODO(geofflang) Check the synchronous flag and potentially flush messages from another
        // thread.
        mCallbackFunction(source, type, id, severity, static_cast<GLsizei>(message.length()),
                          message.c_str(), mCallbackUserParam);
    }
    else
    {
        if (mMessages.size() >= mMaxLoggedMessages)
        {
            // Drop messages over the limit
            return;
        }

        Message m;
        m.source   = source;
        m.type     = type;
        m.id       = id;
        m.severity = severity;
        m.message  = std::move(message);

        mMessages.push_back(std::move(m));
    }
}

size_t Debug::getMessages(GLuint count,
                          GLsizei bufSize,
                          GLenum *sources,
                          GLenum *types,
                          GLuint *ids,
                          GLenum *severities,
                          GLsizei *lengths,
                          GLchar *messageLog)
{
    size_t messageCount       = 0;
    size_t messageStringIndex = 0;
    while (messageCount <= count && !mMessages.empty())
    {
        const Message &m = mMessages.front();

        if (messageLog != nullptr)
        {
            // Check that this message can fit in the message buffer
            if (messageStringIndex + m.message.length() + 1 > static_cast<size_t>(bufSize))
            {
                break;
            }

            std::copy(m.message.begin(), m.message.end(), messageLog + messageStringIndex);
            messageStringIndex += m.message.length();

            messageLog[messageStringIndex] = '\0';
            messageStringIndex += 1;
        }

        if (sources != nullptr)
        {
            sources[messageCount] = m.source;
        }

        if (types != nullptr)
        {
            types[messageCount] = m.type;
        }

        if (ids != nullptr)
        {
            ids[messageCount] = m.id;
        }

        if (severities != nullptr)
        {
            severities[messageCount] = m.severity;
        }

        if (lengths != nullptr)
        {
            lengths[messageCount] = static_cast<GLsizei>(m.message.length());
        }

        mMessages.pop_front();

        messageCount++;
    }

    return messageCount;
}

size_t Debug::getNextMessageLength() const
{
    return mMessages.empty() ? 0 : mMessages.front().message.length();
}

size_t Debug::getMessageCount() const
{
    return mMessages.size();
}

void Debug::setMessageControl(GLenum source,
                              GLenum type,
                              GLenum severity,
                              std::vector<GLuint> &&ids,
                              bool enabled)
{
    Control c;
    c.source   = source;
    c.type     = type;
    c.severity = severity;
    c.ids      = std::move(ids);
    c.enabled  = enabled;

    auto &controls = mGroups.back().controls;
    controls.push_back(std::move(c));
}

void Debug::pushGroup(GLenum source, GLuint id, std::string &&message)
{
    insertMessage(source, GL_DEBUG_TYPE_PUSH_GROUP, id, GL_DEBUG_SEVERITY_NOTIFICATION,
                  std::string(message), gl::LOG_INFO);

    Group g;
    g.source  = source;
    g.id      = id;
    g.message = std::move(message);
    mGroups.push_back(std::move(g));
}

void Debug::popGroup()
{
    // Make sure the default group is not about to be popped
    ASSERT(mGroups.size() > 1);

    Group g = mGroups.back();
    mGroups.pop_back();

    insertMessage(g.source, GL_DEBUG_TYPE_POP_GROUP, g.id, GL_DEBUG_SEVERITY_NOTIFICATION,
                  g.message, gl::LOG_INFO);
}

size_t Debug::getGroupStackDepth() const
{
    return mGroups.size();
}

bool Debug::isMessageEnabled(GLenum source, GLenum type, GLuint id, GLenum severity) const
{
    if (!mOutputEnabled)
    {
        return false;
    }

    for (auto groupIter = mGroups.rbegin(); groupIter != mGroups.rend(); groupIter++)
    {
        const auto &controls = groupIter->controls;
        for (auto controlIter = controls.rbegin(); controlIter != controls.rend(); controlIter++)
        {
            const auto &control = *controlIter;

            if (control.source != GL_DONT_CARE && control.source != source)
            {
                continue;
            }

            if (control.type != GL_DONT_CARE && control.type != type)
            {
                continue;
            }

            if (control.severity != GL_DONT_CARE && control.severity != severity)
            {
                continue;
            }

            if (!control.ids.empty() &&
                std::find(control.ids.begin(), control.ids.end(), id) == control.ids.end())
            {
                continue;
            }

            return control.enabled;
        }
    }

    return true;
}

void Debug::pushDefaultGroup()
{
    Group g;
    g.source  = GL_NONE;
    g.id      = 0;
    g.message = "";

    Control c0;
    c0.source   = GL_DONT_CARE;
    c0.type     = GL_DONT_CARE;
    c0.severity = GL_DONT_CARE;
    c0.enabled  = true;
    g.controls.push_back(std::move(c0));

    Control c1;
    c1.source   = GL_DONT_CARE;
    c1.type     = GL_DONT_CARE;
    c1.severity = GL_DEBUG_SEVERITY_LOW;
    c1.enabled  = false;
    g.controls.push_back(std::move(c1));

    mGroups.push_back(std::move(g));
}
}  // namespace gl

namespace egl
{

namespace
{
angle::PackedEnumBitSet<MessageType> GetDefaultMessageTypeBits()
{
    angle::PackedEnumBitSet<MessageType> result;
    result.set(MessageType::Critical);
    result.set(MessageType::Error);
    return result;
}
}  // anonymous namespace

Debug::Debug() : mCallback(nullptr), mEnabledMessageTypes(GetDefaultMessageTypeBits()) {}

void Debug::setCallback(EGLDEBUGPROCKHR callback, const AttributeMap &attribs)
{
    mCallback = callback;

    const angle::PackedEnumBitSet<MessageType> defaultMessageTypes = GetDefaultMessageTypeBits();
    if (mCallback != nullptr)
    {
        for (MessageType messageType : angle::AllEnums<MessageType>())
        {
            mEnabledMessageTypes[messageType] =
                (attribs.getAsInt(egl::ToEGLenum(messageType), defaultMessageTypes[messageType]) ==
                 EGL_TRUE);
        }
    }
}

EGLDEBUGPROCKHR Debug::getCallback() const
{
    return mCallback;
}

bool Debug::isMessageTypeEnabled(MessageType type) const
{
    return mEnabledMessageTypes[type];
}

void Debug::insertMessage(EGLenum error,
                          const char *command,
                          MessageType messageType,
                          EGLLabelKHR threadLabel,
                          EGLLabelKHR objectLabel,
                          const std::string &message) const
{
    {
        // output all messages to the debug log
        const char *messageTypeString = EGLMessageTypeToString(messageType);
        std::ostringstream messageStream;
        messageStream << "EGL " << messageTypeString << ": " << command << ": " << message;
        INFO() << messageStream.str();
    }

    // TODO(geofflang): Lock before checking the callback. http://anglebug.com/2464
    if (mCallback && isMessageTypeEnabled(messageType))
    {
        mCallback(error, command, egl::ToEGLenum(messageType), threadLabel, objectLabel,
                  message.c_str());
    }
}

}  // namespace egl
