//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FrameCapture.cpp:
//   ANGLE Frame capture implementation.
//

#include "libANGLE/FrameCapture.h"

#include <cerrno>
#include <cstring>
#include <string>

#include "libANGLE/Context.h"
#include "libANGLE/VertexArray.h"
#include "libANGLE/gl_enum_utils_autogen.h"

#ifdef ANGLE_PLATFORM_ANDROID
#    define ANGLE_CAPTURE_PATH ("/sdcard/Android/data/" + CurrentAPKName() + "/")

std::string CurrentAPKName()
{
    static char sApplicationId[512] = {0};
    if (!sApplicationId[0])
    {
        // Linux interface to get application id of the running process
        FILE *cmdline = fopen("/proc/self/cmdline", "r");
        if (cmdline)
        {
            fread(sApplicationId, 1, sizeof(sApplicationId), cmdline);
            fclose(cmdline);

            // Some package may have application id as <app_name>:<cmd_name>
            char *colonSep = strchr(sApplicationId, ':');
            if (colonSep)
            {
                *colonSep = '\0';
            }
        }
        else
        {
            WARN() << "not able to lookup application id";
        }
    }
    return std::string(sApplicationId);
}

#else
#    define ANGLE_CAPTURE_PATH "./"
#endif  // ANGLE_PLATFORM_ANDROID

namespace angle
{
#if !ANGLE_CAPTURE_ENABLED
FrameCapture::FrameCapture()  = default;
FrameCapture::~FrameCapture() = default;
void FrameCapture::captureCall(const gl::Context *context, CallCapture &&call)
{
    // Do nothing.
}
void FrameCapture::onEndFrame(const gl::Context *context)
{
    // Do nothing.
}
CallCapture::~CallCapture()   = default;
ParamBuffer::~ParamBuffer()   = default;
ParamCapture::~ParamCapture() = default;

#else  // !ANGLE_CAPTURE_ENABLED
namespace
{
std::string GetCaptureFileName(int contextId, uint32_t frameIndex, const char *suffix)
{
    std::stringstream fnameStream;
    fnameStream << "angle_capture_context" << contextId << "_frame" << std::setfill('0')
                << std::setw(3) << frameIndex << suffix;
    return fnameStream.str();
}

std::string GetCaptureFilePath(int contextId, uint32_t frameIndex, const char *suffix)
{
    return ANGLE_CAPTURE_PATH + GetCaptureFileName(contextId, frameIndex, suffix);
}

void WriteParamStaticVarName(const CallCapture &call,
                             const ParamCapture &param,
                             int counter,
                             std::ostream &out)
{
    out << call.name() << "_" << param.name << "_" << counter;
}

template <typename T, typename CastT = T>
void WriteInlineData(const std::vector<uint8_t> &vec, std::ostream &out)
{
    const T *data = reinterpret_cast<const T *>(vec.data());
    size_t count  = vec.size() / sizeof(T);

    out << static_cast<CastT>(data[0]);

    for (size_t dataIndex = 1; dataIndex < count; ++dataIndex)
    {
        out << ", " << static_cast<CastT>(data[dataIndex]);
    }
}

constexpr size_t kInlineDataThreshold = 128;

void WriteStringParamReplay(std::ostream &out, const ParamCapture &param)
{
    const std::vector<uint8_t> &data = param.data[0];
    // null terminate C style string
    ASSERT(data.size() > 0 && data.back() == '\0');
    std::string str(data.begin(), data.end() - 1);
    out << "\"" << str << "\"";
}

void WriteStringPointerParamReplay(DataCounters *counters,
                                   std::ostream &out,
                                   std::ostream &header,
                                   const CallCapture &call,
                                   const ParamCapture &param)
{
    int counter = counters->getAndIncrement(call.entryPoint, param.name);

    header << "const char *";
    WriteParamStaticVarName(call, param, counter, header);
    header << "[] = { \n";

    for (const std::vector<uint8_t> &data : param.data)
    {
        // null terminate C style string
        ASSERT(data.size() > 0 && data.back() == '\0');
        std::string str(data.begin(), data.end() - 1);
        header << "    R\"(" << str << ")\",\n";
    }

    header << " };\n";
    WriteParamStaticVarName(call, param, counter, out);
}

template <typename ParamT>
void WriteResourceIDPointerParamReplay(DataCounters *counters,
                                       std::ostream &out,
                                       std::ostream &header,
                                       const CallCapture &call,
                                       const ParamCapture &param)
{
    int counter = counters->getAndIncrement(call.entryPoint, param.name);

    header << "const GLuint ";
    WriteParamStaticVarName(call, param, counter, header);
    header << "[] = { ";

    const ResourceIDType resourceIDType = GetResourceIDTypeFromParamType(param.type);
    ASSERT(resourceIDType != ResourceIDType::InvalidEnum);
    const char *name = GetResourceIDTypeName(resourceIDType);

    GLsizei n = call.params.getParam("n", ParamType::TGLsizei, 0).value.GLsizeiVal;
    ASSERT(param.data.size() == 1);
    const ParamT *returnedIDs = reinterpret_cast<const ParamT *>(param.data[0].data());
    for (GLsizei resIndex = 0; resIndex < n; ++resIndex)
    {
        ParamT id = returnedIDs[resIndex];
        if (resIndex > 0)
        {
            header << ", ";
        }
        header << "g" << name << "Map[" << id.value << "]";
    }

    header << " };\n    ";

    WriteParamStaticVarName(call, param, counter, out);
}

void WriteBinaryParamReplay(DataCounters *counters,
                            std::ostream &out,
                            std::ostream &header,
                            const CallCapture &call,
                            const ParamCapture &param,
                            std::vector<uint8_t> *binaryData)
{
    int counter = counters->getAndIncrement(call.entryPoint, param.name);

    ASSERT(param.data.size() == 1);
    const std::vector<uint8_t> &data = param.data[0];

    if (data.size() > kInlineDataThreshold)
    {
        size_t offset = binaryData->size();
        binaryData->resize(offset + data.size());
        memcpy(binaryData->data() + offset, data.data(), data.size());
        if (param.type == ParamType::TvoidConstPointer || param.type == ParamType::TvoidPointer)
        {
            out << "&gBinaryData[" << offset << "]";
        }
        else
        {
            out << "reinterpret_cast<" << ParamTypeToString(param.type) << ">(&gBinaryData["
                << offset << "])";
        }
    }
    else
    {
        ParamType overrideType = param.type;
        if (param.type == ParamType::TGLvoidConstPointer ||
            param.type == ParamType::TvoidConstPointer)
        {
            overrideType = ParamType::TGLubyteConstPointer;
        }

        std::string paramTypeString = ParamTypeToString(overrideType);
        header << paramTypeString.substr(0, paramTypeString.length() - 1);
        WriteParamStaticVarName(call, param, counter, header);

        header << "[] = { ";

        switch (overrideType)
        {
            case ParamType::TGLintConstPointer:
                WriteInlineData<GLint>(data, header);
                break;
            case ParamType::TGLshortConstPointer:
                WriteInlineData<GLshort>(data, header);
                break;
            case ParamType::TGLfloatConstPointer:
                WriteInlineData<GLfloat>(data, header);
                break;
            case ParamType::TGLubyteConstPointer:
                WriteInlineData<GLubyte, int>(data, header);
                break;
            case ParamType::TGLuintConstPointer:
            case ParamType::TGLenumConstPointer:
                WriteInlineData<GLuint>(data, header);
                break;
            default:
                UNIMPLEMENTED();
                break;
        }

        header << " };\n";

        WriteParamStaticVarName(call, param, counter, out);
    }
}
void WriteCppReplayForCall(const CallCapture &call,
                           DataCounters *counters,
                           std::ostream &out,
                           std::ostream &header,
                           std::vector<uint8_t> *binaryData)
{
    std::ostringstream callOut;

    if (call.entryPoint == gl::EntryPoint::CreateShader ||
        call.entryPoint == gl::EntryPoint::CreateProgram)
    {
        GLuint id = call.params.getReturnValue().value.GLuintVal;
        callOut << "gShaderProgramMap[" << id << "] = ";
    }

    callOut << call.name() << "(";

    bool first = true;
    for (const ParamCapture &param : call.params.getParamCaptures())
    {
        if (!first)
        {
            callOut << ", ";
        }

        if (param.arrayClientPointerIndex != -1)
        {
            callOut << "gClientArrays[" << param.arrayClientPointerIndex << "].data()";
        }
        else if (param.readBufferSizeBytes > 0)
        {
            callOut << "reinterpret_cast<" << ParamTypeToString(param.type)
                    << ">(gReadBuffer.data())";
        }
        else if (param.data.empty())
        {
            if (param.type == ParamType::TGLenum)
            {
                OutputGLenumString(callOut, param.enumGroup, param.value.GLenumVal);
            }
            else if (param.type == ParamType::TGLbitfield)
            {
                OutputGLbitfieldString(callOut, param.enumGroup, param.value.GLbitfieldVal);
            }
            else
            {
                callOut << param;
            }
        }
        else
        {
            switch (param.type)
            {
                case ParamType::TGLcharConstPointer:
                    WriteStringParamReplay(callOut, param);
                    break;
                case ParamType::TGLcharConstPointerPointer:
                    WriteStringPointerParamReplay(counters, callOut, header, call, param);
                    break;
                case ParamType::TBufferIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::BufferID>(counters, callOut, out, call,
                                                                    param);
                    break;
                case ParamType::TFenceNVIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::FenceNVID>(counters, callOut, out, call,
                                                                     param);
                    break;
                case ParamType::TFramebufferIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::FramebufferID>(counters, callOut, out,
                                                                         call, param);
                    break;
                case ParamType::TMemoryObjectIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::MemoryObjectID>(counters, callOut, out,
                                                                          call, param);
                    break;
                case ParamType::TProgramPipelineIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::ProgramPipelineID>(counters, callOut, out,
                                                                             call, param);
                    break;
                case ParamType::TQueryIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::QueryID>(counters, callOut, out, call,
                                                                   param);
                    break;
                case ParamType::TRenderbufferIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::RenderbufferID>(counters, callOut, out,
                                                                          call, param);
                    break;
                case ParamType::TSamplerIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::SamplerID>(counters, callOut, out, call,
                                                                     param);
                    break;
                case ParamType::TSemaphoreIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::SemaphoreID>(counters, callOut, out, call,
                                                                       param);
                    break;
                case ParamType::TTextureIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::TextureID>(counters, callOut, out, call,
                                                                     param);
                    break;
                case ParamType::TTransformFeedbackIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::TransformFeedbackID>(counters, callOut,
                                                                               out, call, param);
                    break;
                case ParamType::TVertexArrayIDConstPointer:
                    WriteResourceIDPointerParamReplay<gl::VertexArrayID>(counters, callOut, out,
                                                                         call, param);
                    break;
                default:
                    WriteBinaryParamReplay(counters, callOut, header, call, param, binaryData);
                    break;
            }
        }

        first = false;
    }

    callOut << ")";

    out << callOut.str();
}

bool AnyClientArray(const gl::AttribArray<size_t> &clientArraySizes)
{
    for (size_t size : clientArraySizes)
    {
        if (size > 0)
            return true;
    }

    return false;
}

void WriteCppReplay(int contextId,
                    uint32_t frameIndex,
                    const std::vector<CallCapture> &calls,
                    const gl::AttribArray<size_t> &clientArraySizes,
                    size_t readBufferSize)
{
    bool useClientArrays = AnyClientArray(clientArraySizes);

    // Count resource IDs.
    angle::PackedEnumMap<ResourceIDType, uint32_t, angle::kParamTypeCount> resourceIDCounts = {};
    for (const CallCapture &call : calls)
    {
        for (const ParamCapture &param : call.params.getParamCaptures())
        {
            ResourceIDType idType = GetResourceIDTypeFromParamType(param.type);
            if (idType != ResourceIDType::InvalidEnum)
            {
                resourceIDCounts[idType]++;
            }
        }
    }

    DataCounters counters;

    std::stringstream out;
    std::stringstream header;
    std::vector<uint8_t> binaryData;

    header << "#include \"util/gles_loader_autogen.h\"\n";
    header << "\n";
    header << "#include <cstdio>\n";
    header << "#include <cstring>\n";
    header << "#include <vector>\n";
    header << "#include <unordered_map>\n";
    header << "\n";
    header << "namespace\n";
    header << "{\n";
    if (readBufferSize > 0)
    {
        header << "std::vector<uint8_t> gReadBuffer;\n";
    }
    if (useClientArrays)
    {
        header << "std::vector<uint8_t> gClientArrays[" << gl::MAX_VERTEX_ATTRIBS << "];\n";
        header << "void UpdateClientArrayPointer(int arrayIndex, const void *data, GLuint64 size)"
               << "\n";
        header << "{\n";
        header << "    memcpy(gClientArrays[arrayIndex].data(), data, size);\n";
        header << "}\n";
    }

    header << "using ResourceMap = std::unordered_map<GLuint, GLuint>;\n";
    header << "void UpdateResourceMap(ResourceMap *resourceMap, GLuint id, GLsizei "
              "readBufferOffset)\n";
    header << "{\n";
    header << "    GLuint returnedID;\n";
    header << "    memcpy(&returnedID, &gReadBuffer[readBufferOffset], sizeof(GLuint));\n";
    header << "    (*resourceMap)[id] = returnedID;\n";
    header << "}\n";
    header << "\n";
    header << "// Resource Maps\n";

    for (ResourceIDType resourceType : AllEnums<ResourceIDType>())
    {
        if (resourceIDCounts[resourceType] == 0)
            continue;

        const char *name = GetResourceIDTypeName(resourceType);
        header << "ResourceMap g" << name << "Map;\n";
        header << "void Update" << name << "ID(GLuint id, GLsizei readBufferOffset)\n";
        header << "{\n";
        header << "    UpdateResourceMap(&g" << name << "Map, id, readBufferOffset);\n";
        header << "}\n";
    }

    out << "void ReplayFrame" << frameIndex << "()\n";
    out << "{\n";
    out << "    LoadBinaryData();\n";

    for (size_t arrayIndex = 0; arrayIndex < clientArraySizes.size(); ++arrayIndex)
    {
        if (clientArraySizes[arrayIndex] > 0)
        {
            out << "    gClientArrays[" << arrayIndex << "].resize(" << clientArraySizes[arrayIndex]
                << ");\n";
        }
    }

    if (readBufferSize > 0)
    {
        out << "    gReadBuffer.resize(" << readBufferSize << ");\n";
    }

    for (const CallCapture &call : calls)
    {
        out << "    ";
        WriteCppReplayForCall(call, &counters, out, header, &binaryData);
        out << ";\n";
    }

    if (!binaryData.empty())
    {
        std::string dataFilepath = GetCaptureFilePath(contextId, frameIndex, ".angledata");

        FILE *fp = fopen(dataFilepath.c_str(), "wb");
        if (!fp)
        {
            FATAL() << "file " << dataFilepath << " can not be created!: " << strerror(errno);
        }
        fwrite(binaryData.data(), 1, binaryData.size(), fp);
        fclose(fp);

        std::string fname = GetCaptureFileName(contextId, frameIndex, ".angledata");
        header << "std::vector<uint8_t> gBinaryData;\n";
        header << "void LoadBinaryData()\n";
        header << "{\n";
        header << "    gBinaryData.resize(" << static_cast<int>(binaryData.size()) << ");\n";
        header << "    FILE *fp = fopen(\"" << fname << "\", \"rb\");\n";
        header << "    fread(gBinaryData.data(), 1, " << static_cast<int>(binaryData.size())
               << ", fp);\n";
        header << "    fclose(fp);\n";
        header << "}\n";
    }
    else
    {
        header << "// No binary data.\n";
        header << "void LoadBinaryData() {}\n";
    }

    out << "}\n";

    header << "}  // anonymous namespace\n";

    std::string outString    = out.str();
    std::string headerString = header.str();

    std::string cppFilePath = GetCaptureFilePath(contextId, frameIndex, ".cpp");
    FILE *fp                = fopen(cppFilePath.c_str(), "w");
    if (!fp)
    {
        FATAL() << "file " << cppFilePath << " can not be created!: " << strerror(errno);
    }
    fprintf(fp, "%s\n\n%s", headerString.c_str(), outString.c_str());
    fclose(fp);

    printf("Saved '%s'.\n", cppFilePath.c_str());
}
}  // anonymous namespace

ParamCapture::ParamCapture() : type(ParamType::TGLenum), enumGroup(gl::GLenumGroup::DefaultGroup) {}

ParamCapture::ParamCapture(const char *nameIn, ParamType typeIn)
    : name(nameIn), type(typeIn), enumGroup(gl::GLenumGroup::DefaultGroup)
{}

ParamCapture::~ParamCapture() = default;

ParamCapture::ParamCapture(ParamCapture &&other)
    : type(ParamType::TGLenum), enumGroup(gl::GLenumGroup::DefaultGroup)
{
    *this = std::move(other);
}

ParamCapture &ParamCapture::operator=(ParamCapture &&other)
{
    std::swap(name, other.name);
    std::swap(type, other.type);
    std::swap(value, other.value);
    std::swap(enumGroup, other.enumGroup);
    std::swap(data, other.data);
    std::swap(arrayClientPointerIndex, other.arrayClientPointerIndex);
    std::swap(readBufferSizeBytes, other.readBufferSizeBytes);
    return *this;
}

ParamBuffer::ParamBuffer() {}

ParamBuffer::~ParamBuffer() = default;

ParamBuffer::ParamBuffer(ParamBuffer &&other)
{
    *this = std::move(other);
}

ParamBuffer &ParamBuffer::operator=(ParamBuffer &&other)
{
    std::swap(mParamCaptures, other.mParamCaptures);
    std::swap(mClientArrayDataParam, other.mClientArrayDataParam);
    std::swap(mReadBufferSize, other.mReadBufferSize);
    std::swap(mReturnValueCapture, other.mReturnValueCapture);
    return *this;
}

ParamCapture &ParamBuffer::getParam(const char *paramName, ParamType paramType, int index)
{
    ParamCapture &capture = mParamCaptures[index];
    ASSERT(capture.name == paramName);
    ASSERT(capture.type == paramType);
    return capture;
}

const ParamCapture &ParamBuffer::getParam(const char *paramName,
                                          ParamType paramType,
                                          int index) const
{
    return const_cast<ParamBuffer *>(this)->getParam(paramName, paramType, index);
}

void ParamBuffer::addParam(ParamCapture &&param)
{
    if (param.arrayClientPointerIndex != -1)
    {
        ASSERT(mClientArrayDataParam == -1);
        mClientArrayDataParam = static_cast<int>(mParamCaptures.size());
    }

    mReadBufferSize = std::max(param.readBufferSizeBytes, mReadBufferSize);
    mParamCaptures.emplace_back(std::move(param));
}

void ParamBuffer::addReturnValue(ParamCapture &&returnValue)
{
    mReturnValueCapture = std::move(returnValue);
}

ParamCapture &ParamBuffer::getClientArrayPointerParameter()
{
    ASSERT(hasClientArrayData());
    return mParamCaptures[mClientArrayDataParam];
}

CallCapture::CallCapture(gl::EntryPoint entryPointIn, ParamBuffer &&paramsIn)
    : entryPoint(entryPointIn), params(std::move(paramsIn))
{}

CallCapture::CallCapture(const std::string &customFunctionNameIn, ParamBuffer &&paramsIn)
    : entryPoint(gl::EntryPoint::Invalid),
      customFunctionName(customFunctionNameIn),
      params(std::move(paramsIn))
{}

CallCapture::~CallCapture() = default;

CallCapture::CallCapture(CallCapture &&other)
{
    *this = std::move(other);
}

CallCapture &CallCapture::operator=(CallCapture &&other)
{
    std::swap(entryPoint, other.entryPoint);
    std::swap(customFunctionName, other.customFunctionName);
    std::swap(params, other.params);
    return *this;
}

const char *CallCapture::name() const
{
    if (entryPoint == gl::EntryPoint::Invalid)
    {
        ASSERT(!customFunctionName.empty());
        return customFunctionName.c_str();
    }

    return gl::GetEntryPointName(entryPoint);
}

ReplayContext::ReplayContext(size_t readBufferSizebytes,
                             const gl::AttribArray<size_t> &clientArraysSizebytes)
{
    mReadBuffer.resize(readBufferSizebytes);

    for (uint32_t i = 0; i < clientArraysSizebytes.size(); i++)
    {
        mClientArraysBuffer[i].resize(clientArraysSizebytes[i]);
    }
}
ReplayContext::~ReplayContext() {}

FrameCapture::FrameCapture() : mFrameIndex(0), mReadBufferSize(0)
{
    reset();
}

FrameCapture::~FrameCapture() = default;

void FrameCapture::maybeCaptureClientData(const gl::Context *context, const CallCapture &call)
{
    switch (call.entryPoint)
    {
        case gl::EntryPoint::VertexAttribPointer:
        {
            // Get array location
            GLuint index = call.params.getParam("index", ParamType::TGLuint, 0).value.GLuintVal;

            if (call.params.hasClientArrayData())
            {
                mClientVertexArrayMap[index] = static_cast<int>(mCalls.size());
            }
            else
            {
                mClientVertexArrayMap[index] = -1;
            }
            break;
        }

        case gl::EntryPoint::DrawArrays:
        {
            if (context->getStateCache().hasAnyActiveClientAttrib())
            {
                // Get counts from paramBuffer.
                GLint firstVertex =
                    call.params.getParam("first", ParamType::TGLint, 1).value.GLintVal;
                GLsizei drawCount =
                    call.params.getParam("count", ParamType::TGLsizei, 2).value.GLsizeiVal;
                captureClientArraySnapshot(context, firstVertex + drawCount, 1);
            }
            break;
        }

        case gl::EntryPoint::DrawElements:
        {
            if (context->getStateCache().hasAnyActiveClientAttrib())
            {
                GLsizei count =
                    call.params.getParam("count", ParamType::TGLsizei, 1).value.GLsizeiVal;
                gl::DrawElementsType drawElementsType =
                    call.params.getParam("typePacked", ParamType::TDrawElementsType, 2)
                        .value.DrawElementsTypeVal;
                const void *indices =
                    call.params.getParam("indices", ParamType::TvoidConstPointer, 3)
                        .value.voidConstPointerVal;

                gl::IndexRange indexRange;

                bool restart = context->getState().isPrimitiveRestartEnabled();

                gl::Buffer *elementArrayBuffer =
                    context->getState().getVertexArray()->getElementArrayBuffer();
                if (elementArrayBuffer)
                {
                    size_t offset = reinterpret_cast<size_t>(indices);
                    (void)elementArrayBuffer->getIndexRange(context, drawElementsType, offset,
                                                            count, restart, &indexRange);
                }
                else
                {
                    indexRange = gl::ComputeIndexRange(drawElementsType, indices, count, restart);
                }

                // index starts from 0
                captureClientArraySnapshot(context, indexRange.end + 1, 1);
            }
            break;
        }

        default:
            break;
    }
}

void FrameCapture::captureCall(const gl::Context *context, CallCapture &&call)
{
    // Process client data snapshots.
    maybeCaptureClientData(context, call);

    mReadBufferSize = std::max(mReadBufferSize, call.params.getReadBufferSize());
    mCalls.emplace_back(std::move(call));

    // Process resource ID updates.
    maybeUpdateResourceIDs(context, mCalls.back());
}

template <typename IDType>
void FrameCapture::captureUpdateResourceIDs(const gl::Context *context,
                                            const CallCapture &call,
                                            const ParamCapture &param)
{
    GLsizei n = call.params.getParam("n", ParamType::TGLsizei, 0).value.GLsizeiVal;
    ASSERT(param.data.size() == 1);
    const IDType *returnedIDs     = reinterpret_cast<const IDType *>(param.data[0].data());
    ResourceIDType resourceIDType = GetResourceIDTypeFromParamType(param.type);
    ASSERT(resourceIDType != ResourceIDType::InvalidEnum);
    const char *resourceName = GetResourceIDTypeName(resourceIDType);

    std::stringstream updateFuncNameStr;
    updateFuncNameStr << "Update" << resourceName << "ID";
    std::string updateFuncName = updateFuncNameStr.str();

    for (GLsizei idIndex = 0; idIndex < n; ++idIndex)
    {
        IDType id                = returnedIDs[idIndex];
        GLsizei readBufferOffset = idIndex * sizeof(gl::RenderbufferID);
        ParamBuffer params;
        params.addValueParam("id", ParamType::TGLuint, id.value);
        params.addValueParam("readBufferOffset", ParamType::TGLsizei, readBufferOffset);
        mCalls.emplace_back(updateFuncName, std::move(params));
    }
}

void FrameCapture::maybeUpdateResourceIDs(const gl::Context *context, const CallCapture &call)
{
    switch (call.entryPoint)
    {
        case gl::EntryPoint::GenBuffers:
        {
            const ParamCapture &buffers =
                call.params.getParam("buffersPacked", ParamType::TBufferIDPointer, 1);
            captureUpdateResourceIDs<gl::BufferID>(context, call, buffers);
            break;
        }

        case gl::EntryPoint::GenFencesNV:
        {
            const ParamCapture &fences =
                call.params.getParam("fencesPacked", ParamType::TFenceNVIDPointer, 1);
            captureUpdateResourceIDs<gl::FenceNVID>(context, call, fences);
            break;
        }

        case gl::EntryPoint::GenFramebuffers:
        case gl::EntryPoint::GenFramebuffersOES:
        {
            const ParamCapture &framebuffers =
                call.params.getParam("framebuffersPacked", ParamType::TFramebufferIDPointer, 1);
            captureUpdateResourceIDs<gl::FramebufferID>(context, call, framebuffers);
            break;
        }

        case gl::EntryPoint::GenPathsCHROMIUM:
        {
            // TODO(jmadill): Handle path IDs. http://anglebug.com/3611
            break;
        }

        case gl::EntryPoint::GenProgramPipelines:
        {
            const ParamCapture &pipelines =
                call.params.getParam("pipelinesPacked", ParamType::TProgramPipelineIDPointer, 1);
            captureUpdateResourceIDs<gl::ProgramPipelineID>(context, call, pipelines);
            break;
        }

        case gl::EntryPoint::GenQueries:
        case gl::EntryPoint::GenQueriesEXT:
        {
            const ParamCapture &queries =
                call.params.getParam("idsPacked", ParamType::TQueryIDPointer, 1);
            captureUpdateResourceIDs<gl::QueryID>(context, call, queries);
            break;
        }

        case gl::EntryPoint::GenRenderbuffers:
        case gl::EntryPoint::GenRenderbuffersOES:
        {
            const ParamCapture &renderbuffers =
                call.params.getParam("renderbuffersPacked", ParamType::TRenderbufferIDPointer, 1);
            captureUpdateResourceIDs<gl::RenderbufferID>(context, call, renderbuffers);
            break;
        }

        case gl::EntryPoint::GenSamplers:
        {
            const ParamCapture &samplers =
                call.params.getParam("samplersPacked", ParamType::TSamplerIDPointer, 1);
            captureUpdateResourceIDs<gl::SamplerID>(context, call, samplers);
            break;
        }

        case gl::EntryPoint::GenSemaphoresEXT:
        {
            const ParamCapture &semaphores =
                call.params.getParam("semaphoresPacked", ParamType::TSemaphoreIDPointer, 1);
            captureUpdateResourceIDs<gl::SemaphoreID>(context, call, semaphores);
            break;
        }

        case gl::EntryPoint::GenTextures:
        {
            const ParamCapture &textures =
                call.params.getParam("texturesPacked", ParamType::TTextureIDPointer, 1);
            captureUpdateResourceIDs<gl::TextureID>(context, call, textures);
            break;
        }

        case gl::EntryPoint::GenTransformFeedbacks:
        {
            const ParamCapture &xfbs =
                call.params.getParam("idsPacked", ParamType::TTransformFeedbackIDPointer, 1);
            captureUpdateResourceIDs<gl::TransformFeedbackID>(context, call, xfbs);
            break;
        }

        case gl::EntryPoint::GenVertexArrays:
        case gl::EntryPoint::GenVertexArraysOES:
        {
            const ParamCapture &vertexArrays =
                call.params.getParam("vetexArraysPacked", ParamType::TVertexArrayIDPointer, 1);
            captureUpdateResourceIDs<gl::VertexArrayID>(context, call, vertexArrays);
            break;
        }

        default:
            break;
    }
}

void FrameCapture::captureClientArraySnapshot(const gl::Context *context,
                                              size_t vertexCount,
                                              size_t instanceCount)
{
    const gl::VertexArray *vao = context->getState().getVertexArray();

    // Capture client array data.
    for (size_t attribIndex : context->getStateCache().getActiveClientAttribsMask())
    {
        const gl::VertexAttribute &attrib = vao->getVertexAttribute(attribIndex);
        const gl::VertexBinding &binding  = vao->getVertexBinding(attrib.bindingIndex);

        int callIndex = mClientVertexArrayMap[attribIndex];

        if (callIndex != -1)
        {
            size_t count = vertexCount;

            if (binding.getDivisor() > 0)
            {
                count = rx::UnsignedCeilDivide(static_cast<uint32_t>(instanceCount),
                                               binding.getDivisor());
            }

            // The last capture element doesn't take up the full stride.
            size_t bytesToCapture = (count - 1) * binding.getStride() + attrib.format->pixelBytes;

            CallCapture &call   = mCalls[callIndex];
            ParamCapture &param = call.params.getClientArrayPointerParameter();
            ASSERT(param.type == ParamType::TvoidConstPointer);

            ParamBuffer updateParamBuffer;
            updateParamBuffer.addValueParam<GLint>("arrayIndex", ParamType::TGLint,
                                                   static_cast<uint32_t>(attribIndex));

            ParamCapture updateMemory("pointer", ParamType::TvoidConstPointer);
            CaptureMemory(param.value.voidConstPointerVal, bytesToCapture, &updateMemory);
            updateParamBuffer.addParam(std::move(updateMemory));

            updateParamBuffer.addValueParam<GLuint64>("size", ParamType::TGLuint64, bytesToCapture);

            mCalls.emplace_back("UpdateClientArrayPointer", std::move(updateParamBuffer));

            mClientArraySizes[attribIndex] =
                std::max(mClientArraySizes[attribIndex], bytesToCapture);
        }
    }
}

void FrameCapture::onEndFrame(const gl::Context *context)
{
    if (!mCalls.empty())
    {
        WriteCppReplay(context->id(), mFrameIndex, mCalls, mClientArraySizes, mReadBufferSize);
        reset();
        mFrameIndex++;
    }
}

DataCounters::DataCounters() = default;

DataCounters::~DataCounters() = default;

int DataCounters::getAndIncrement(gl::EntryPoint entryPoint, const std::string &paramName)
{
    Counter counterKey = {entryPoint, paramName};
    return mData[counterKey]++;
}

bool FrameCapture::enabled() const
{
    return mFrameIndex < 100;
}

void FrameCapture::replay(gl::Context *context)
{
    ReplayContext replayContext(mReadBufferSize, mClientArraySizes);
    for (const CallCapture &call : mCalls)
    {
        INFO() << "frame index: " << mFrameIndex << " " << call.name();

        if (call.entryPoint == gl::EntryPoint::Invalid)
        {
            if (call.customFunctionName == "UpdateClientArrayPointer")
            {
                GLint arrayIndex =
                    call.params.getParam("arrayIndex", ParamType::TGLint, 0).value.GLintVal;
                ASSERT(arrayIndex < gl::MAX_VERTEX_ATTRIBS);

                const ParamCapture &pointerParam =
                    call.params.getParam("pointer", ParamType::TvoidConstPointer, 1);
                ASSERT(pointerParam.data.size() == 1);
                const void *pointer = pointerParam.data[0].data();

                size_t size = static_cast<size_t>(
                    call.params.getParam("size", ParamType::TGLuint64, 2).value.GLuint64Val);

                std::vector<uint8_t> &curClientArrayBuffer =
                    replayContext.getClientArraysBuffer()[arrayIndex];
                ASSERT(curClientArrayBuffer.size() >= size);
                memcpy(curClientArrayBuffer.data(), pointer, size);
            }
            continue;
        }

        ReplayCall(context, &replayContext, call);
    }
}

void FrameCapture::reset()
{
    mCalls.clear();
    mClientVertexArrayMap.fill(-1);
    mClientArraySizes.fill(0);
    mReadBufferSize = 0;
}

std::ostream &operator<<(std::ostream &os, const ParamCapture &capture)
{
    WriteParamTypeToStream(os, capture.type, capture.value);
    return os;
}

void CaptureMemory(const void *source, size_t size, ParamCapture *paramCapture)
{
    std::vector<uint8_t> data(size);
    memcpy(data.data(), source, size);
    paramCapture->data.emplace_back(std::move(data));
}

void CaptureString(const GLchar *str, ParamCapture *paramCapture)
{
    // include the '\0' suffix
    CaptureMemory(str, strlen(str) + 1, paramCapture);
}

void CaptureGenHandlesImpl(GLsizei n, GLuint *handles, ParamCapture *paramCapture)
{
    paramCapture->readBufferSizeBytes = sizeof(GLuint) * n;
    CaptureMemory(handles, paramCapture->readBufferSizeBytes, paramCapture);
}

template <>
void WriteParamValueToStream<ParamType::TGLboolean>(std::ostream &os, GLboolean value)
{
    switch (value)
    {
        case GL_TRUE:
            os << "GL_TRUE";
            break;
        case GL_FALSE:
            os << "GL_FALSE";
            break;
        default:
            os << "GL_INVALID_ENUM";
    }
}

template <>
void WriteParamValueToStream<ParamType::TvoidConstPointer>(std::ostream &os, const void *value)
{
    if (value == 0)
    {
        os << "nullptr";
    }
    else
    {
        os << "reinterpret_cast<const void *>("
           << static_cast<int>(reinterpret_cast<uintptr_t>(value)) << ")";
    }
}

template <>
void WriteParamValueToStream<ParamType::TGLDEBUGPROCKHR>(std::ostream &os, GLDEBUGPROCKHR value)
{}

template <>
void WriteParamValueToStream<ParamType::TGLDEBUGPROC>(std::ostream &os, GLDEBUGPROC value)
{}

template <>
void WriteParamValueToStream<ParamType::TBufferID>(std::ostream &os, gl::BufferID value)
{
    os << "gBufferMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TFenceNVID>(std::ostream &os, gl::FenceNVID value)
{
    os << "gFenceMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TFramebufferID>(std::ostream &os, gl::FramebufferID value)
{
    os << "gFramebufferMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TMemoryObjectID>(std::ostream &os, gl::MemoryObjectID value)
{
    os << "gMemoryObjectMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TPathID>(std::ostream &os, gl::PathID value)
{
    os << "gPathMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TProgramPipelineID>(std::ostream &os,
                                                            gl::ProgramPipelineID value)
{
    os << "gProgramPipelineMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TQueryID>(std::ostream &os, gl::QueryID value)
{
    os << "gQueryMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TRenderbufferID>(std::ostream &os, gl::RenderbufferID value)
{
    os << "gRenderbufferMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TSamplerID>(std::ostream &os, gl::SamplerID value)
{
    os << "gSamplerMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TSemaphoreID>(std::ostream &os, gl::SemaphoreID value)
{
    os << "gSempahoreMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TShaderProgramID>(std::ostream &os,
                                                          gl::ShaderProgramID value)
{
    os << "gShaderProgramMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TTextureID>(std::ostream &os, gl::TextureID value)
{
    os << "gTextureMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TTransformFeedbackID>(std::ostream &os,
                                                              gl::TransformFeedbackID value)
{
    os << "gTransformFeedbackMap[" << value.value << "]";
}

template <>
void WriteParamValueToStream<ParamType::TVertexArrayID>(std::ostream &os, gl::VertexArrayID value)
{
    os << "gVertexArrayMap[" << value.value << "]";
}

#endif  // ANGLE_CAPTURE_ENABLED

}  // namespace angle
