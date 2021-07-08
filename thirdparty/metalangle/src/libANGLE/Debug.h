//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Debug.h: Defines debug state used for GL_KHR_debug

#ifndef LIBANGLE_DEBUG_H_
#define LIBANGLE_DEBUG_H_

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/AttributeMap.h"

#include <deque>
#include <string>
#include <vector>

namespace gl
{
class Context;

class LabeledObject
{
  public:
    virtual ~LabeledObject() {}
    virtual void setLabel(const Context *context, const std::string &label) = 0;
    virtual const std::string &getLabel() const                             = 0;
};

class Debug : angle::NonCopyable
{
  public:
    Debug(bool initialDebugState);
    ~Debug();

    void setMaxLoggedMessages(GLuint maxLoggedMessages);

    void setOutputEnabled(bool enabled);
    bool isOutputEnabled() const;

    void setOutputSynchronous(bool synchronous);
    bool isOutputSynchronous() const;

    void setCallback(GLDEBUGPROCKHR callback, const void *userParam);
    GLDEBUGPROCKHR getCallback() const;
    const void *getUserParam() const;

    void insertMessage(GLenum source,
                       GLenum type,
                       GLuint id,
                       GLenum severity,
                       const std::string &message,
                       gl::LogSeverity logSeverity) const;
    void insertMessage(GLenum source,
                       GLenum type,
                       GLuint id,
                       GLenum severity,
                       std::string &&message,
                       gl::LogSeverity logSeverity) const;

    void setMessageControl(GLenum source,
                           GLenum type,
                           GLenum severity,
                           std::vector<GLuint> &&ids,
                           bool enabled);
    size_t getMessages(GLuint count,
                       GLsizei bufSize,
                       GLenum *sources,
                       GLenum *types,
                       GLuint *ids,
                       GLenum *severities,
                       GLsizei *lengths,
                       GLchar *messageLog);
    size_t getNextMessageLength() const;
    size_t getMessageCount() const;

    void pushGroup(GLenum source, GLuint id, std::string &&message);
    void popGroup();
    size_t getGroupStackDepth() const;

  private:
    bool isMessageEnabled(GLenum source, GLenum type, GLuint id, GLenum severity) const;

    void pushDefaultGroup();

    struct Message
    {
        GLenum source;
        GLenum type;
        GLuint id;
        GLenum severity;
        std::string message;
    };

    struct Control
    {
        Control();
        ~Control();
        Control(const Control &other);

        GLenum source;
        GLenum type;
        GLenum severity;
        std::vector<GLuint> ids;
        bool enabled;
    };

    struct Group
    {
        Group();
        ~Group();
        Group(const Group &other);

        GLenum source;
        GLuint id;
        std::string message;

        std::vector<Control> controls;
    };

    bool mOutputEnabled;
    GLDEBUGPROCKHR mCallbackFunction;
    const void *mCallbackUserParam;
    mutable std::deque<Message> mMessages;
    GLuint mMaxLoggedMessages;
    bool mOutputSynchronous;
    std::vector<Group> mGroups;
};
}  // namespace gl

namespace egl
{
class LabeledObject
{
  public:
    virtual ~LabeledObject() {}
    virtual void setLabel(EGLLabelKHR label) = 0;
    virtual EGLLabelKHR getLabel() const     = 0;
};

class Debug : angle::NonCopyable
{
  public:
    Debug();

    void setCallback(EGLDEBUGPROCKHR callback, const AttributeMap &attribs);
    EGLDEBUGPROCKHR getCallback() const;
    bool isMessageTypeEnabled(MessageType type) const;

    void insertMessage(EGLenum error,
                       const char *command,
                       MessageType messageType,
                       EGLLabelKHR threadLabel,
                       EGLLabelKHR objectLabel,
                       const std::string &message) const;

  private:
    EGLDEBUGPROCKHR mCallback;
    angle::PackedEnumBitSet<MessageType> mEnabledMessageTypes;
};
}  // namespace egl
#endif  // LIBANGLE_DEBUG_H_
