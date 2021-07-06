
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FrameCapture.h:
//   ANGLE Frame capture inteface.
//

#ifndef LIBANGLE_FRAME_CAPTURE_H_
#define LIBANGLE_FRAME_CAPTURE_H_

#include "common/PackedEnums.h"
#include "libANGLE/Context.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/entry_points_utils.h"
#include "libANGLE/frame_capture_utils_autogen.h"

namespace gl
{
enum class GLenumGroup;
}

namespace angle
{
struct ParamCapture : angle::NonCopyable
{
    ParamCapture();
    ParamCapture(const char *nameIn, ParamType typeIn);
    ~ParamCapture();

    ParamCapture(ParamCapture &&other);
    ParamCapture &operator=(ParamCapture &&other);

    std::string name;
    ParamType type;
    ParamValue value;
    gl::GLenumGroup enumGroup;  // only used for param type GLenum, GLboolean and GLbitfield
    std::vector<std::vector<uint8_t>> data;
    int arrayClientPointerIndex = -1;
    size_t readBufferSizeBytes  = 0;
};

class ParamBuffer final : angle::NonCopyable
{
  public:
    ParamBuffer();
    ~ParamBuffer();

    ParamBuffer(ParamBuffer &&other);
    ParamBuffer &operator=(ParamBuffer &&other);

    template <typename T>
    void addValueParam(const char *paramName, ParamType paramType, T paramValue);
    template <typename T>
    void addEnumParam(const char *paramName,
                      gl::GLenumGroup enumGroup,
                      ParamType paramType,
                      T paramValue);

    ParamCapture &getParam(const char *paramName, ParamType paramType, int index);
    const ParamCapture &getParam(const char *paramName, ParamType paramType, int index) const;
    const ParamCapture &getReturnValue() const { return mReturnValueCapture; }

    void addParam(ParamCapture &&param);
    void addReturnValue(ParamCapture &&returnValue);
    bool hasClientArrayData() const { return mClientArrayDataParam != -1; }
    ParamCapture &getClientArrayPointerParameter();
    size_t getReadBufferSize() const { return mReadBufferSize; }

    const std::vector<ParamCapture> &getParamCaptures() const { return mParamCaptures; }

  private:
    std::vector<ParamCapture> mParamCaptures;
    ParamCapture mReturnValueCapture;
    int mClientArrayDataParam = -1;
    size_t mReadBufferSize    = 0;
};

struct CallCapture
{
    CallCapture(gl::EntryPoint entryPointIn, ParamBuffer &&paramsIn);
    CallCapture(const std::string &customFunctionNameIn, ParamBuffer &&paramsIn);
    ~CallCapture();

    CallCapture(CallCapture &&other);
    CallCapture &operator=(CallCapture &&other);

    const char *name() const;

    gl::EntryPoint entryPoint;
    std::string customFunctionName;
    ParamBuffer params;
};

class ReplayContext
{
  public:
    ReplayContext(size_t readBufferSizebytes, const gl::AttribArray<size_t> &clientArraysSizebytes);
    ~ReplayContext();

    template <typename T>
    T getReadBufferPointer(const ParamCapture &param)
    {
        ASSERT(param.readBufferSizeBytes > 0);
        ASSERT(mReadBuffer.size() >= param.readBufferSizeBytes);
        return reinterpret_cast<T>(mReadBuffer.data());
    }
    template <typename T>
    T getAsConstPointer(const ParamCapture &param)
    {
        if (param.arrayClientPointerIndex != -1)
        {
            return reinterpret_cast<T>(mClientArraysBuffer[param.arrayClientPointerIndex].data());
        }

        if (!param.data.empty())
        {
            ASSERT(param.data.size() == 1);
            return reinterpret_cast<T>(param.data[0].data());
        }

        return nullptr;
    }

    template <typename T>
    T getAsPointerConstPointer(const ParamCapture &param)
    {
        static_assert(sizeof(typename std::remove_pointer<T>::type) == sizeof(uint8_t *),
                      "pointer size not match!");

        ASSERT(!param.data.empty());
        mPointersBuffer.clear();
        mPointersBuffer.reserve(param.data.size());
        for (const std::vector<uint8_t> &data : param.data)
        {
            mPointersBuffer.emplace_back(data.data());
        }
        return reinterpret_cast<T>(mPointersBuffer.data());
    }

    gl::AttribArray<std::vector<uint8_t>> &getClientArraysBuffer() { return mClientArraysBuffer; }

  private:
    std::vector<uint8_t> mReadBuffer;
    std::vector<const uint8_t *> mPointersBuffer;
    gl::AttribArray<std::vector<uint8_t>> mClientArraysBuffer;
};

// Helper to use unique IDs for each local data variable.
class DataCounters final : angle::NonCopyable
{
  public:
    DataCounters();
    ~DataCounters();

    int getAndIncrement(gl::EntryPoint entryPoint, const std::string &paramName);

  private:
    // <CallName, ParamName>
    using Counter = std::pair<gl::EntryPoint, std::string>;
    std::map<Counter, int> mData;
};

class FrameCapture final : angle::NonCopyable
{
  public:
    FrameCapture();
    ~FrameCapture();

    void captureCall(const gl::Context *context, CallCapture &&call);
    void onEndFrame(const gl::Context *context);
    bool enabled() const;
    void replay(gl::Context *context);

  private:
    void captureClientArraySnapshot(const gl::Context *context,
                                    size_t vertexCount,
                                    size_t instanceCount);

    void reset();
    void maybeCaptureClientData(const gl::Context *context, const CallCapture &call);
    void maybeUpdateResourceIDs(const gl::Context *context, const CallCapture &call);

    template <typename IDType>
    void captureUpdateResourceIDs(const gl::Context *context,
                                  const CallCapture &call,
                                  const ParamCapture &param);

    std::vector<CallCapture> mCalls;
    gl::AttribArray<int> mClientVertexArrayMap;
    uint32_t mFrameIndex;
    gl::AttribArray<size_t> mClientArraySizes;
    size_t mReadBufferSize;

    static void ReplayCall(gl::Context *context,
                           ReplayContext *replayContext,
                           const CallCapture &call);
};

template <typename CaptureFuncT, typename... ArgsT>
void CaptureCallToFrameCapture(CaptureFuncT captureFunc,
                               bool isCallValid,
                               gl::Context *context,
                               ArgsT... captureParams)
{
    FrameCapture *frameCapture = context->getFrameCapture();
    if (!frameCapture->enabled())
        return;

    CallCapture call = captureFunc(context, isCallValid, captureParams...);
    frameCapture->captureCall(context, std::move(call));
}

template <typename T>
void ParamBuffer::addValueParam(const char *paramName, ParamType paramType, T paramValue)
{
    ParamCapture capture(paramName, paramType);
    InitParamValue(paramType, paramValue, &capture.value);
    mParamCaptures.emplace_back(std::move(capture));
}

template <typename T>
void ParamBuffer::addEnumParam(const char *paramName,
                               gl::GLenumGroup enumGroup,
                               ParamType paramType,
                               T paramValue)
{
    ParamCapture capture(paramName, paramType);
    InitParamValue(paramType, paramValue, &capture.value);
    capture.enumGroup = enumGroup;
    mParamCaptures.emplace_back(std::move(capture));
}

std::ostream &operator<<(std::ostream &os, const ParamCapture &capture);

// Pointer capture helpers.
void CaptureMemory(const void *source, size_t size, ParamCapture *paramCapture);
void CaptureString(const GLchar *str, ParamCapture *paramCapture);
void CaptureGenHandlesImpl(GLsizei n, GLuint *handles, ParamCapture *paramCapture);

template <typename T>
void CaptureGenHandles(GLsizei n, T *handles, ParamCapture *paramCapture)
{
    CaptureGenHandlesImpl(n, reinterpret_cast<GLuint *>(handles), paramCapture);
}

template <ParamType ParamT, typename T>
void WriteParamValueToStream(std::ostream &os, T value);

template <>
void WriteParamValueToStream<ParamType::TGLboolean>(std::ostream &os, GLboolean value);

template <>
void WriteParamValueToStream<ParamType::TvoidConstPointer>(std::ostream &os, const void *value);

template <>
void WriteParamValueToStream<ParamType::TGLDEBUGPROCKHR>(std::ostream &os, GLDEBUGPROCKHR value);

template <>
void WriteParamValueToStream<ParamType::TGLDEBUGPROC>(std::ostream &os, GLDEBUGPROC value);

template <>
void WriteParamValueToStream<ParamType::TBufferID>(std::ostream &os, gl::BufferID value);

template <>
void WriteParamValueToStream<ParamType::TFenceNVID>(std::ostream &os, gl::FenceNVID value);

template <>
void WriteParamValueToStream<ParamType::TFramebufferID>(std::ostream &os, gl::FramebufferID value);

template <>
void WriteParamValueToStream<ParamType::TMemoryObjectID>(std::ostream &os,
                                                         gl::MemoryObjectID value);

template <>
void WriteParamValueToStream<ParamType::TPathID>(std::ostream &os, gl::PathID value);

template <>
void WriteParamValueToStream<ParamType::TProgramPipelineID>(std::ostream &os,
                                                            gl::ProgramPipelineID value);

template <>
void WriteParamValueToStream<ParamType::TQueryID>(std::ostream &os, gl::QueryID value);

template <>
void WriteParamValueToStream<ParamType::TRenderbufferID>(std::ostream &os,
                                                         gl::RenderbufferID value);

template <>
void WriteParamValueToStream<ParamType::TSamplerID>(std::ostream &os, gl::SamplerID value);

template <>
void WriteParamValueToStream<ParamType::TSemaphoreID>(std::ostream &os, gl::SemaphoreID value);

template <>
void WriteParamValueToStream<ParamType::TShaderProgramID>(std::ostream &os,
                                                          gl::ShaderProgramID value);

template <>
void WriteParamValueToStream<ParamType::TTextureID>(std::ostream &os, gl::TextureID value);

template <>
void WriteParamValueToStream<ParamType::TTransformFeedbackID>(std::ostream &os,
                                                              gl::TransformFeedbackID value);

template <>
void WriteParamValueToStream<ParamType::TVertexArrayID>(std::ostream &os, gl::VertexArrayID value);

// General fallback for any unspecific type.
template <ParamType ParamT, typename T>
void WriteParamValueToStream(std::ostream &os, T value)
{
    os << value;
}
}  // namespace angle

#endif  // LIBANGLE_FRAME_CAPTURE_H_
