//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VertexArrayMtl.mm:
//    Implements the class methods for VertexArrayMtl.
//

#include "libANGLE/renderer/metal/VertexArrayMtl.h"

#include <TargetConditionals.h>

#include "libANGLE/renderer/metal/BufferMtl.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/mtl_format_utils.h"

#include "common/debug.h"

namespace rx
{
namespace
{

angle::Result StreamVertexData(ContextMtl *contextMtl,
                               mtl::BufferPool *dynamicBuffer,
                               const uint8_t *sourceData,
                               size_t bytesToAllocate,
                               size_t destOffset,
                               size_t vertexCount,
                               size_t stride,
                               VertexCopyFunction vertexLoadFunction,
                               SimpleWeakBufferHolderMtl *bufferHolder,
                               size_t *bufferOffsetOut)
{
    ANGLE_CHECK(contextMtl, vertexLoadFunction, "Unsupported format conversion", GL_INVALID_ENUM);
    uint8_t *dst = nullptr;
    mtl::BufferRef newBuffer;
    ANGLE_TRY(dynamicBuffer->allocate(contextMtl, bytesToAllocate, &dst, &newBuffer,
                                      bufferOffsetOut, nullptr));
    bufferHolder->set(newBuffer);
    dst += destOffset;
    vertexLoadFunction(sourceData, stride, vertexCount, dst);

    ANGLE_TRY(dynamicBuffer->commit(contextMtl));
    return angle::Result::Continue;
}

template <typename SizeT>
const mtl::VertexFormat &GetVertexConversionFormat(ContextMtl *contextMtl,
                                                   angle::FormatID originalFormat,
                                                   SizeT *strideOut)
{
    // Convert to tightly packed format
    const mtl::VertexFormat &packedFormat = contextMtl->getVertexFormat(originalFormat, true);
    *strideOut                            = packedFormat.actualAngleFormat().pixelBytes;
    return packedFormat;
}

size_t GetIndexConvertedBufferSize(gl::DrawElementsType indexType, size_t indexCount)
{
    size_t elementSize = gl::GetDrawElementsTypeSize(indexType);
    if (indexType == gl::DrawElementsType::UnsignedByte)
    {
        // 8-bit indices are not supported by Metal, so they are promoted to
        // 16-bit indices below
        elementSize = sizeof(GLushort);
    }

    const size_t amount = elementSize * indexCount;

    return amount;
}

angle::Result StreamIndexData(ContextMtl *contextMtl,
                              mtl::BufferPool *dynamicBuffer,
                              const uint8_t *sourcePointer,
                              gl::DrawElementsType indexType,
                              size_t indexCount,
                              bool primitiveRestartEnabled,
                              mtl::BufferRef *bufferOut,
                              size_t *bufferOffsetOut)
{
    const size_t amount = GetIndexConvertedBufferSize(indexType, indexCount);
    GLubyte *dst        = nullptr;

    ANGLE_TRY(
        dynamicBuffer->allocate(contextMtl, amount, &dst, bufferOut, bufferOffsetOut, nullptr));

    if (indexType == gl::DrawElementsType::UnsignedByte)
    {
        // Unsigned bytes don't have direct support in Metal so we have to expand the
        // memory to a GLushort.
        const GLubyte *in     = static_cast<const GLubyte *>(sourcePointer);
        GLushort *expandedDst = reinterpret_cast<GLushort *>(dst);

        if (primitiveRestartEnabled)
        {
            for (size_t index = 0; index < indexCount; index++)
            {
                if (in[index] == 0xFF)
                {
                    expandedDst[index] = 0xFFFF;
                }
                else
                {
                    expandedDst[index] = static_cast<GLushort>(in[index]);
                }
            }
        }  // if (primitiveRestartEnabled)
        else
        {
            for (size_t index = 0; index < indexCount; index++)
            {
                expandedDst[index] = static_cast<GLushort>(in[index]);
            }
        }  // if (primitiveRestartEnabled)
    }
    else
    {
        memcpy(dst, sourcePointer, amount);
    }
    ANGLE_TRY(dynamicBuffer->commit(contextMtl));

    return angle::Result::Continue;
}

size_t GetVertexCount(BufferMtl *srcBuffer,
                      const gl::VertexBinding &binding,
                      uint32_t srcFormatSize)
{
    // Bytes usable for vertex data.
    GLint64 bytes = srcBuffer->size() - binding.getOffset();
    if (bytes < srcFormatSize)
        return 0;

    // Count the last vertex.  It may occupy less than a full stride.
    size_t numVertices = 1;
    bytes -= srcFormatSize;

    // Count how many strides fit remaining space.
    if (bytes > 0)
        numVertices += static_cast<size_t>(bytes) / binding.getStride();

    return numVertices;
}

inline size_t GetIndexCount(BufferMtl *srcBuffer, size_t offset, gl::DrawElementsType indexType)
{
    size_t elementSize = gl::GetDrawElementsTypeSize(indexType);
    return (srcBuffer->size() - offset) / elementSize;
}

inline void SetDefaultVertexBufferLayout(mtl::VertexBufferLayoutDesc *layout)
{
    layout->stepFunction = mtl::kVertexStepFunctionInvalid;
    layout->stepRate     = 0;
    layout->stride       = 0;
}

}  // namespace

// VertexArrayMtl implementation
VertexArrayMtl::VertexArrayMtl(const gl::VertexArrayState &state, ContextMtl *context)
    : VertexArrayImpl(state),
      mDefaultFloatVertexFormat(
          context->getVertexFormat(angle::FormatID::R32G32B32A32_FLOAT, false)),
      mDefaultIntVertexFormat(context->getVertexFormat(angle::FormatID::R32G32B32A32_SINT, false)),
      mDefaultUIntVertexFormat(context->getVertexFormat(angle::FormatID::R32G32B32A32_UINT, false))
{
    reset(context);

    mDynamicVertexData.initialize(context, 0, mtl::kVertexAttribBufferStrideAlignment);
}
VertexArrayMtl::~VertexArrayMtl() {}

void VertexArrayMtl::destroy(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);

    reset(contextMtl);

    mDynamicVertexData.destroy(contextMtl);
}

void VertexArrayMtl::reset(ContextMtl *context)
{
    for (BufferHolderMtl *&buffer : mCurrentArrayBuffers)
    {
        buffer = nullptr;
    }
    for (size_t &offset : mCurrentArrayBufferOffsets)
    {
        offset = 0;
    }
    for (GLuint &stride : mCurrentArrayBufferStrides)
    {
        stride = 0;
    }
    for (const mtl::VertexFormat *&format : mCurrentArrayBufferFormats)
    {
        format = &mDefaultFloatVertexFormat;
    }

    for (size_t &inlineDataSize : mCurrentArrayInlineDataSizes)
    {
        inlineDataSize = 0;
    }

    for (angle::MemoryBuffer &convertedClientArray : mConvertedClientSmallArrays)
    {
        convertedClientArray.resize(0);
    }

    for (const uint8_t *&clientPointer : mCurrentArrayInlineDataPointers)
    {
        clientPointer = nullptr;
    }

    if (context->getDisplay()->getFeatures().allowInlineConstVertexData.enabled)
    {
        mInlineDataMaxSize = mtl::kInlineConstDataMaxSize;
    }
    else
    {
        mInlineDataMaxSize = 0;
    }

    mVertexArrayDirty = true;
}

angle::Result VertexArrayMtl::syncState(const gl::Context *context,
                                        const gl::VertexArray::DirtyBits &dirtyBits,
                                        gl::VertexArray::DirtyAttribBitsArray *attribBits,
                                        gl::VertexArray::DirtyBindingBitsArray *bindingBits)
{
    const std::vector<gl::VertexAttribute> &attribs = mState.getVertexAttributes();
    const std::vector<gl::VertexBinding> &bindings  = mState.getVertexBindings();

    for (size_t dirtyBit : dirtyBits)
    {
        switch (dirtyBit)
        {
            case gl::VertexArray::DIRTY_BIT_ELEMENT_ARRAY_BUFFER:
            case gl::VertexArray::DIRTY_BIT_ELEMENT_ARRAY_BUFFER_DATA:
            {
                break;
            }

#define ANGLE_VERTEX_DIRTY_ATTRIB_FUNC(INDEX)                                                     \
    case gl::VertexArray::DIRTY_BIT_ATTRIB_0 + INDEX:                                             \
        ANGLE_TRY(syncDirtyAttrib(context, attribs[INDEX], bindings[attribs[INDEX].bindingIndex], \
                                  INDEX));                                                        \
        mVertexArrayDirty = true;                                                                 \
        (*attribBits)[INDEX].reset();                                                             \
        break;

                ANGLE_VERTEX_INDEX_CASES(ANGLE_VERTEX_DIRTY_ATTRIB_FUNC)

#define ANGLE_VERTEX_DIRTY_BINDING_FUNC(INDEX)                                                    \
    case gl::VertexArray::DIRTY_BIT_BINDING_0 + INDEX:                                            \
        ANGLE_TRY(syncDirtyAttrib(context, attribs[INDEX], bindings[attribs[INDEX].bindingIndex], \
                                  INDEX));                                                        \
        mVertexArrayDirty = true;                                                                 \
        (*bindingBits)[INDEX].reset();                                                            \
        break;

                ANGLE_VERTEX_INDEX_CASES(ANGLE_VERTEX_DIRTY_BINDING_FUNC)

#define ANGLE_VERTEX_DIRTY_BUFFER_DATA_FUNC(INDEX)                                                \
    case gl::VertexArray::DIRTY_BIT_BUFFER_DATA_0 + INDEX:                                        \
        ANGLE_TRY(syncDirtyAttrib(context, attribs[INDEX], bindings[attribs[INDEX].bindingIndex], \
                                  INDEX));                                                        \
        mVertexArrayDirty = true;                                                                 \
        break;

                ANGLE_VERTEX_INDEX_CASES(ANGLE_VERTEX_DIRTY_BUFFER_DATA_FUNC)

            default:
                UNREACHABLE();
                break;
        }
    }

    return angle::Result::Continue;
}

ANGLE_INLINE void VertexArrayMtl::getVertexAttribFormatAndArraySize(const sh::ShaderVariable &var,
                                                                    MTLVertexFormat *formatOut,
                                                                    uint32_t *arraySizeOut)
{
    uint32_t arraySize = var.getArraySizeProduct();

    MTLVertexFormat format;
    switch (var.type)
    {
        case GL_INT:
        case GL_INT_VEC2:
        case GL_INT_VEC3:
        case GL_INT_VEC4:
            format = mDefaultIntVertexFormat.metalFormat;
            break;
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_INT_VEC2:
        case GL_UNSIGNED_INT_VEC3:
        case GL_UNSIGNED_INT_VEC4:
            format = mDefaultUIntVertexFormat.metalFormat;
            break;
        case GL_FLOAT_MAT2:
        case GL_FLOAT_MAT2x3:
        case GL_FLOAT_MAT2x4:
            arraySize *= 2;
            format = mDefaultFloatVertexFormat.metalFormat;
            break;
        case GL_FLOAT_MAT3:
        case GL_FLOAT_MAT3x2:
        case GL_FLOAT_MAT3x4:
            arraySize *= 3;
            format = mDefaultFloatVertexFormat.metalFormat;
            break;
        case GL_FLOAT_MAT4:
        case GL_FLOAT_MAT4x2:
        case GL_FLOAT_MAT4x3:
            arraySize *= 4;
            format = mDefaultFloatVertexFormat.metalFormat;
            break;
        default:
            format = mDefaultFloatVertexFormat.metalFormat;
    }

    *arraySizeOut = arraySize;
    *formatOut    = format;
}

// vertexDescChanged is both input and output, the input value if is true, will force new
// mtl::VertexDesc to be returned via vertexDescOut. This typically happens when active shader
// program is changed.
// Otherwise, it is only returned when the vertex array is dirty.
angle::Result VertexArrayMtl::setupDraw(const gl::Context *glContext,
                                        mtl::RenderCommandEncoder *cmdEncoder,
                                        bool *vertexDescChanged,
                                        mtl::VertexDesc *vertexDescOut)
{
    // NOTE(hqle): consider only updating dirty attributes
    bool dirty = mVertexArrayDirty || *vertexDescChanged;

    if (dirty)
    {
        ContextMtl *contextMtl = mtl::GetImpl(glContext);

        mVertexArrayDirty = false;
        mEmulatedInstanceAttribs.clear();

        const gl::ProgramState &programState = glContext->getState().getProgram()->getState();
        const gl::AttributesMask &programActiveAttribsMask =
            programState.getActiveAttribLocationsMask();

        const std::vector<gl::VertexAttribute> &attribs = mState.getVertexAttributes();
        const std::vector<gl::VertexBinding> &bindings  = mState.getVertexBindings();

        mtl::VertexDesc &desc = *vertexDescOut;

        desc.numAttribs       = mtl::kMaxVertexAttribs;
        desc.numBufferLayouts = mtl::kMaxVertexAttribs;

        // Initialize the buffer layouts with constant step rate
        for (uint32_t b = 0; b < mtl::kMaxVertexAttribs; ++b)
        {
            SetDefaultVertexBufferLayout(&desc.layouts[b]);
        }

        for (uint32_t v = 0; v < mtl::kMaxVertexAttribs; ++v)
        {
            if (!programActiveAttribsMask.test(v))
            {
                desc.attributes[v].format      = MTLVertexFormatInvalid;
                desc.attributes[v].bufferIndex = 0;
                desc.attributes[v].offset      = 0;
                continue;
            }

            const auto &attrib               = attribs[v];
            const gl::VertexBinding &binding = bindings[attrib.bindingIndex];

            bool attribEnabled = attrib.enabled;
            if (attribEnabled && !mCurrentArrayBuffers[v] && !mCurrentArrayInlineDataPointers[v])
            {
                // Disable it to avoid crash.
                attribEnabled = false;
            }

            if (!attribEnabled)
            {
                // Use default attribute
                // Need to find the attribute having the exact binding location = v in the program
                // inputs list to retrieve its coresponding data type:
                const std::vector<sh::ShaderVariable> &programInputs =
                    programState.getProgramInputs();
                std::vector<sh::ShaderVariable>::const_iterator attribInfoIte = std::find_if(
                    begin(programInputs), end(programInputs), [v](const sh::ShaderVariable &sv) {
                        return static_cast<uint32_t>(sv.location) == v;
                    });

                if (attribInfoIte == end(programInputs))
                {
                    // Most likely this is array element with index > 0.
                    // Already handled when encounter first element.
                    continue;
                }

                uint32_t arraySize;
                MTLVertexFormat format;

                getVertexAttribFormatAndArraySize(*attribInfoIte, &format, &arraySize);

                for (uint32_t vaIdx = v; vaIdx < v + arraySize; ++vaIdx)
                {
                    desc.attributes[vaIdx].bufferIndex = mtl::kDefaultAttribsBindingIndex;
                    desc.attributes[vaIdx].offset      = vaIdx * mtl::kDefaultAttributeSize;
                    desc.attributes[vaIdx].format      = format;
                }
            }
            else
            {
                uint32_t bufferIdx    = mtl::kVboBindingIndexStart + v;
                uint32_t bufferOffset = static_cast<uint32_t>(mCurrentArrayBufferOffsets[v]);

                const angle::Format &angleFormat =
                    mCurrentArrayBufferFormats[v]->actualAngleFormat();
                desc.attributes[v].format = mCurrentArrayBufferFormats[v]->metalFormat;

                desc.attributes[v].bufferIndex = bufferIdx;
                desc.attributes[v].offset      = 0;
                ASSERT((bufferOffset % angleFormat.pixelBytes) == 0);

                ASSERT(bufferIdx < mtl::kMaxVertexAttribs);
                if (binding.getDivisor() == 0)
                {
                    desc.layouts[bufferIdx].stepFunction = MTLVertexStepFunctionPerVertex;
                    desc.layouts[bufferIdx].stepRate     = 1;
                }
                else if (contextMtl->getDisplay()->getFeatures().hasBaseVertexInstancedDraw.enabled)
                {
                    desc.layouts[bufferIdx].stepFunction = MTLVertexStepFunctionPerInstance;
                    desc.layouts[bufferIdx].stepRate     = binding.getDivisor();
                }
                else
                {
                    // Emulate instance attribute
                    mEmulatedInstanceAttribs.push_back(v);
                    desc.layouts[bufferIdx].stepFunction = MTLVertexStepFunctionConstant;
                    desc.layouts[bufferIdx].stepRate     = 0;
                }

                desc.layouts[bufferIdx].stride = mCurrentArrayBufferStrides[v];

                if (mCurrentArrayBuffers[v])
                {
                    cmdEncoder->setVertexBuffer(mCurrentArrayBuffers[v]->getCurrentBuffer(),
                                                bufferOffset, bufferIdx);
                }
                else
                {
                    // No buffer specified, use the client memory directly as inline constant data
                    ASSERT(mCurrentArrayInlineDataSizes[v] <= mInlineDataMaxSize);
                    cmdEncoder->setVertexBytes(mCurrentArrayInlineDataPointers[v],
                                               mCurrentArrayInlineDataSizes[v], bufferIdx);
                }
            }
        }  // for (v)
    }

    *vertexDescChanged = dirty;

    return angle::Result::Continue;
}

void VertexArrayMtl::emulateInstanceDrawStep(mtl::RenderCommandEncoder *cmdEncoder,
                                             uint32_t instanceId)
{

    const std::vector<gl::VertexAttribute> &attribs = mState.getVertexAttributes();
    const std::vector<gl::VertexBinding> &bindings  = mState.getVertexBindings();

    for (uint32_t instanceAttribIdx : mEmulatedInstanceAttribs)
    {
        uint32_t bufferIdx               = mtl::kVboBindingIndexStart + instanceAttribIdx;
        const auto &attrib               = attribs[instanceAttribIdx];
        const gl::VertexBinding &binding = bindings[attrib.bindingIndex];
        uint32_t offset =
            instanceId / binding.getDivisor() * mCurrentArrayBufferStrides[instanceAttribIdx];
        if (mCurrentArrayBuffers[instanceAttribIdx])
        {
            offset += static_cast<uint32_t>(mCurrentArrayBufferOffsets[instanceAttribIdx]);

            cmdEncoder->setVertexBuffer(mCurrentArrayBuffers[instanceAttribIdx]->getCurrentBuffer(),
                                        offset, bufferIdx);
        }
        else
        {
            // No buffer specified, use the client memory directly as inline constant data
            ASSERT(mCurrentArrayInlineDataSizes[instanceAttribIdx] <= mInlineDataMaxSize);
            if (offset > mCurrentArrayInlineDataSizes[instanceAttribIdx])
            {
                offset = static_cast<uint32_t>(mCurrentArrayInlineDataSizes[instanceAttribIdx]);
            }
            cmdEncoder->setVertexBytes(mCurrentArrayInlineDataPointers[instanceAttribIdx] + offset,
                                       mCurrentArrayInlineDataSizes[instanceAttribIdx] - offset,
                                       bufferIdx);
        }
    }
}

angle::Result VertexArrayMtl::updateClientAttribs(const gl::Context *context,
                                                  GLint firstVertex,
                                                  GLsizei vertexOrIndexCount,
                                                  GLsizei instanceCount,
                                                  gl::DrawElementsType indexTypeOrInvalid,
                                                  const void *indices)
{
    ContextMtl *contextMtl                  = mtl::GetImpl(context);
    const gl::AttributesMask &clientAttribs = context->getStateCache().getActiveClientAttribsMask();

    ASSERT(clientAttribs.any());

    GLint startVertex;
    size_t vertexCount;
    ANGLE_TRY(GetVertexRangeInfo(context, firstVertex, vertexOrIndexCount, indexTypeOrInvalid,
                                 indices, 0, &startVertex, &vertexCount));

    mDynamicVertexData.releaseInFlightBuffers(contextMtl);

    const std::vector<gl::VertexAttribute> &attribs = mState.getVertexAttributes();
    const std::vector<gl::VertexBinding> &bindings  = mState.getVertexBindings();

    for (size_t attribIndex : clientAttribs)
    {
        const gl::VertexAttribute &attrib = attribs[attribIndex];
        const gl::VertexBinding &binding  = bindings[attrib.bindingIndex];
        ASSERT(attrib.enabled && binding.getBuffer().get() == nullptr);

        // Source client memory pointer
        const uint8_t *src = static_cast<const uint8_t *>(attrib.pointer);
        ASSERT(src);

        GLint startElement;
        size_t elementCount;
        if (binding.getDivisor() == 0)
        {
            // Per vertex attribute
            startElement = startVertex;
            elementCount = vertexCount;
        }
        else
        {
            // Per instance attribute
            startElement = 0;
            elementCount = UnsignedCeilDivide(instanceCount, binding.getDivisor());
        }

        ASSERT(elementCount);
        const mtl::VertexFormat &format = contextMtl->getVertexFormat(attrib.format->id, false);

        // Actual bytes to use: the last element doesn't need to be full stride
        size_t bytesIntendedToUse = (startElement + elementCount - 1) * binding.getStride() +
                                    format.actualAngleFormat().pixelBytes;

        bool needStreaming = format.actualFormatId != format.intendedFormatId ||
                             (binding.getStride() % mtl::kVertexAttribBufferStrideAlignment) != 0 ||
                             (binding.getStride() < format.actualAngleFormat().pixelBytes) ||
                             bytesIntendedToUse > mInlineDataMaxSize;

        if (!needStreaming)
        {
            // Data will be uploaded directly as inline constant data
            mCurrentArrayBuffers[attribIndex]            = nullptr;
            mCurrentArrayInlineDataPointers[attribIndex] = src;
            mCurrentArrayInlineDataSizes[attribIndex]    = bytesIntendedToUse;
            mCurrentArrayBufferOffsets[attribIndex]      = 0;
            mCurrentArrayBufferFormats[attribIndex]      = &format;
            mCurrentArrayBufferStrides[attribIndex]      = binding.getStride();
        }
        else
        {
            GLuint convertedStride;
            // Need to stream the client vertex data to a buffer.
            const mtl::VertexFormat &streamFormat =
                GetVertexConversionFormat(contextMtl, attrib.format->id, &convertedStride);

            // Allocate space for startElement + elementCount so indexing will work.  If we don't
            // start at zero all the indices will be off.
            // Only elementCount vertices will be used by the upcoming draw so that is all we copy.
            size_t bytesToAllocate = (startElement + elementCount) * convertedStride;
            src += startElement * binding.getStride();
            size_t destOffset = startElement * convertedStride;

            mCurrentArrayBufferFormats[attribIndex] = &streamFormat;
            mCurrentArrayBufferStrides[attribIndex] = convertedStride;

            if (bytesToAllocate <= mInlineDataMaxSize)
            {
                // If the data is small enough, use host memory instead of creating GPU buffer. To
                // avoid synchronizing access to GPU buffer that is still in use.
                angle::MemoryBuffer &convertedClientArray =
                    mConvertedClientSmallArrays[attribIndex];
                if (bytesToAllocate > convertedClientArray.size())
                {
                    ANGLE_CHECK_GL_ALLOC(contextMtl, convertedClientArray.resize(bytesToAllocate));
                }

                ASSERT(streamFormat.vertexLoadFunction);
                streamFormat.vertexLoadFunction(src, binding.getStride(), elementCount,
                                                convertedClientArray.data() + destOffset);

                mCurrentArrayBuffers[attribIndex]            = nullptr;
                mCurrentArrayInlineDataPointers[attribIndex] = convertedClientArray.data();
                mCurrentArrayInlineDataSizes[attribIndex]    = bytesToAllocate;
                mCurrentArrayBufferOffsets[attribIndex]      = 0;
            }
            else
            {
                // Stream the client data to a GPU buffer. Synchronization might happen if buffer is
                // in use.
                mDynamicVertexData.updateAlignment(contextMtl,
                                                   streamFormat.actualAngleFormat().pixelBytes);
                ANGLE_TRY(StreamVertexData(contextMtl, &mDynamicVertexData, src, bytesToAllocate,
                                           destOffset, elementCount, binding.getStride(),
                                           streamFormat.vertexLoadFunction,
                                           &mConvertedArrayBufferHolders[attribIndex],
                                           &mCurrentArrayBufferOffsets[attribIndex]));

                mCurrentArrayBuffers[attribIndex] = &mConvertedArrayBufferHolders[attribIndex];
            }
        }  // if (needStreaming)
    }

    mVertexArrayDirty = true;

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::syncDirtyAttrib(const gl::Context *glContext,
                                              const gl::VertexAttribute &attrib,
                                              const gl::VertexBinding &binding,
                                              size_t attribIndex)
{
    ContextMtl *contextMtl = mtl::GetImpl(glContext);
    ASSERT(mtl::kMaxVertexAttribs > attribIndex);

    if (attrib.enabled)
    {
        gl::Buffer *bufferGL            = binding.getBuffer().get();
        const mtl::VertexFormat &format = contextMtl->getVertexFormat(attrib.format->id, false);

        if (bufferGL)
        {
            BufferMtl *bufferMtl = mtl::GetImpl(bufferGL);
            bool needConversion =
                format.actualFormatId != format.intendedFormatId ||
                (binding.getOffset() % format.actualAngleFormat().pixelBytes) != 0 ||
                (binding.getOffset() % 4) != 0 ||
                (binding.getStride() < format.actualAngleFormat().pixelBytes) ||
                (binding.getStride() % mtl::kVertexAttribBufferStrideAlignment) != 0;

            if (needConversion)
            {
                ANGLE_TRY(convertVertexBuffer(glContext, bufferMtl, binding, attribIndex, format));
            }
            else
            {
                mCurrentArrayBuffers[attribIndex]       = bufferMtl;
                mCurrentArrayBufferOffsets[attribIndex] = binding.getOffset();
                mCurrentArrayBufferStrides[attribIndex] = binding.getStride();

                mCurrentArrayBufferFormats[attribIndex] = &format;
            }
        }
        else
        {
            // ContextMtl must feed the client data using updateClientAttribs()
        }
    }
    else
    {
        // Use default attribute value. Handled in setupDraw().
        mCurrentArrayBuffers[attribIndex]       = nullptr;
        mCurrentArrayBufferOffsets[attribIndex] = 0;
        mCurrentArrayBufferStrides[attribIndex] = 0;
        mCurrentArrayBufferFormats[attribIndex] =
            &contextMtl->getVertexFormat(angle::FormatID::NONE, false);
    }

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::getIndexBuffer(const gl::Context *context,
                                             gl::DrawElementsType type,
                                             size_t count,
                                             const void *indices,
                                             mtl::BufferRef *idxBufferOut,
                                             size_t *idxBufferOffsetOut,
                                             gl::DrawElementsType *indexTypeOut)
{
    const gl::Buffer *glElementArrayBuffer = getState().getElementArrayBuffer();

    size_t convertedOffset = reinterpret_cast<size_t>(indices);
    if (!glElementArrayBuffer)
    {
        ANGLE_TRY(streamIndexBufferFromClient(context, type, count, indices, idxBufferOut,
                                              idxBufferOffsetOut));
    }
    else
    {
        bool needConversion = type == gl::DrawElementsType::UnsignedByte ||
                              (convertedOffset % mtl::kIndexBufferOffsetAlignment) != 0;
        if (needConversion)
        {
            ANGLE_TRY(convertIndexBuffer(context, type, convertedOffset, idxBufferOut,
                                         idxBufferOffsetOut));
        }
        else
        {
            // No conversion needed:
            BufferMtl *bufferMtl = mtl::GetImpl(glElementArrayBuffer);
            *idxBufferOut        = bufferMtl->getCurrentBuffer();
            *idxBufferOffsetOut  = convertedOffset;
        }
    }

    *indexTypeOut = type;
    if (type == gl::DrawElementsType::UnsignedByte)
    {
        // This buffer is already converted to ushort indices above
        *indexTypeOut = gl::DrawElementsType::UnsignedShort;
    }

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::convertIndexBuffer(const gl::Context *glContext,
                                                 gl::DrawElementsType indexType,
                                                 size_t offset,
                                                 mtl::BufferRef *idxBufferOut,
                                                 size_t *idxBufferOffsetOut)
{
    size_t offsetModulo = offset % mtl::kIndexBufferOffsetAlignment;
    ASSERT(offsetModulo != 0 || indexType == gl::DrawElementsType::UnsignedByte);

    size_t alignedOffset = offset - offsetModulo;
    if (indexType == gl::DrawElementsType::UnsignedByte)
    {
        // Unsigned byte index will be promoted to unsigned short, thus double its offset.
        alignedOffset = alignedOffset << 1;
    }

    ContextMtl *contextMtl   = mtl::GetImpl(glContext);
    const gl::State &glState = glContext->getState();
    BufferMtl *idxBuffer     = mtl::GetImpl(getState().getElementArrayBuffer());

    IndexConversionBufferMtl *conversion = idxBuffer->getIndexConversionBuffer(
        contextMtl, indexType, glState.isPrimitiveRestartEnabled(), offsetModulo);

    // Has the content of the buffer has changed since last conversion?
    if (!conversion->dirty)
    {
        // reuse the converted buffer
        *idxBufferOut       = conversion->convertedBuffer;
        *idxBufferOffsetOut = conversion->convertedOffset + alignedOffset;
        return angle::Result::Continue;
    }

    size_t indexCount = GetIndexCount(idxBuffer, offsetModulo, indexType);

    if (!contextMtl->getDisplay()->getFeatures().breakRenderPassIsCheap.enabled &&
        contextMtl->getRenderCommandEncoder())
    {
        // We shouldn't use GPU to convert when we are in a middle of a render pass.
        conversion->data.releaseInFlightBuffers(contextMtl);
        ANGLE_TRY(StreamIndexData(contextMtl, &conversion->data,
                                  idxBuffer->getClientShadowCopyData(contextMtl) + offsetModulo,
                                  indexType, indexCount, glState.isPrimitiveRestartEnabled(),
                                  &conversion->convertedBuffer, &conversion->convertedOffset));
    }
    else
    {
        ANGLE_TRY(convertIndexBufferGPU(glContext, indexType, idxBuffer, offsetModulo, indexCount,
                                        conversion));
    }

    *idxBufferOut       = conversion->convertedBuffer;
    *idxBufferOffsetOut = conversion->convertedOffset + alignedOffset;

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::convertIndexBufferGPU(const gl::Context *glContext,
                                                    gl::DrawElementsType indexType,
                                                    BufferMtl *idxBuffer,
                                                    size_t offset,
                                                    size_t indexCount,
                                                    IndexConversionBufferMtl *conversion)
{
    ContextMtl *contextMtl = mtl::GetImpl(glContext);
    DisplayMtl *display    = contextMtl->getDisplay();

    const size_t amount = GetIndexConvertedBufferSize(indexType, indexCount);

    // Allocate new buffer, save it in conversion struct so that we can reuse it when the content
    // of the original buffer is not dirty.
    conversion->data.releaseInFlightBuffers(contextMtl);
    ANGLE_TRY(conversion->data.allocate(contextMtl, amount, nullptr, &conversion->convertedBuffer,
                                        &conversion->convertedOffset));

    // Do the conversion on GPU.
    ANGLE_TRY(display->getUtils().convertIndexBufferGPU(
        contextMtl, {indexType, static_cast<uint32_t>(indexCount), idxBuffer->getCurrentBuffer(),
                     static_cast<uint32_t>(offset), conversion->convertedBuffer,
                     static_cast<uint32_t>(conversion->convertedOffset),
                     glContext->getState().isPrimitiveRestartEnabled()}));

    ANGLE_TRY(conversion->data.commit(contextMtl));

    ASSERT(conversion->dirty);
    conversion->dirty = false;

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::streamIndexBufferFromClient(const gl::Context *context,
                                                          gl::DrawElementsType indexType,
                                                          size_t indexCount,
                                                          const void *sourcePointer,
                                                          mtl::BufferRef *idxBufferOut,
                                                          size_t *idxBufferOffsetOut)
{
    ASSERT(getState().getElementArrayBuffer() == nullptr);
    ContextMtl *contextMtl = mtl::GetImpl(context);

    // Generate index buffer
    auto srcData = static_cast<const uint8_t *>(sourcePointer);
    ANGLE_TRY(StreamIndexData(
        contextMtl, &contextMtl->getClientIndexBufferPool(), srcData, indexType, indexCount,
        context->getState().isPrimitiveRestartEnabled(), idxBufferOut, idxBufferOffsetOut));

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::convertVertexBuffer(const gl::Context *glContext,
                                                  BufferMtl *srcBuffer,
                                                  const gl::VertexBinding &binding,
                                                  size_t attribIndex,
                                                  const mtl::VertexFormat &srcVertexFormat)
{
    unsigned srcFormatSize = srcVertexFormat.intendedAngleFormat().pixelBytes;

    size_t numVertices = GetVertexCount(srcBuffer, binding, srcFormatSize);
    if (numVertices == 0)
    {
        // Out of bound buffer access, can return any values.
        // See KHR_robust_buffer_access_behavior
        mCurrentArrayBuffers[attribIndex]       = srcBuffer;
        mCurrentArrayBufferFormats[attribIndex] = &srcVertexFormat;
        mCurrentArrayBufferOffsets[attribIndex] = 0;
        mCurrentArrayBufferStrides[attribIndex] = 16;
        return angle::Result::Continue;
    }

    ContextMtl *contextMtl = mtl::GetImpl(glContext);

    // Convert to tightly packed format
    GLuint stride;
    const mtl::VertexFormat &convertedFormat =
        GetVertexConversionFormat(contextMtl, srcVertexFormat.intendedFormatId, &stride);

    ConversionBufferMtl *conversion = srcBuffer->getVertexConversionBuffer(
        contextMtl, srcVertexFormat.intendedFormatId, binding.getStride(), binding.getOffset());

    // Has the content of the buffer has changed since last conversion?
    if (!conversion->dirty)
    {
        mConvertedArrayBufferHolders[attribIndex].set(conversion->convertedBuffer);
        mCurrentArrayBufferOffsets[attribIndex] = conversion->convertedOffset;

        mCurrentArrayBuffers[attribIndex]       = &mConvertedArrayBufferHolders[attribIndex];
        mCurrentArrayBufferFormats[attribIndex] = &convertedFormat;
        mCurrentArrayBufferStrides[attribIndex] = stride;
        return angle::Result::Continue;
    }

    const angle::Format &convertedAngleFormat = convertedFormat.actualAngleFormat();
    bool canConvertToFloatOnGPU =
        convertedAngleFormat.isFloat() && !convertedAngleFormat.isVertexTypeHalfFloat();

    bool canExpandComponentsOnGPU = convertedFormat.actualSameGLType;

    if (contextMtl->getRenderCommandEncoder() &&
        !contextMtl->getDisplay()->getFeatures().breakRenderPassIsCheap.enabled &&
        !contextMtl->getDisplay()->getFeatures().hasExplicitMemBarrier.enabled)
    {
        // Cannot use GPU to convert when we are in a middle of a render pass.
        canConvertToFloatOnGPU = canExpandComponentsOnGPU = false;
    }

    conversion->data.releaseInFlightBuffers(contextMtl);
    conversion->data.updateAlignment(contextMtl, convertedAngleFormat.pixelBytes);

    if (canConvertToFloatOnGPU || canExpandComponentsOnGPU)
    {
        ANGLE_TRY(convertVertexBufferGPU(glContext, srcBuffer, binding, attribIndex,
                                         convertedFormat, stride, numVertices,
                                         canExpandComponentsOnGPU, conversion));
    }
    else
    {
        ANGLE_TRY(convertVertexBufferCPU(contextMtl, srcBuffer, binding, attribIndex,
                                         convertedFormat, stride, numVertices, conversion));
    }

    mCurrentArrayBuffers[attribIndex]       = &mConvertedArrayBufferHolders[attribIndex];
    mCurrentArrayBufferFormats[attribIndex] = &convertedFormat;
    mCurrentArrayBufferStrides[attribIndex] = stride;

    // Cache the last converted results to be re-used later if the buffer's content won't ever be
    // changed.
    conversion->convertedBuffer = mConvertedArrayBufferHolders[attribIndex].getCurrentBuffer();
    conversion->convertedOffset = mCurrentArrayBufferOffsets[attribIndex];

    ASSERT(conversion->dirty);
    conversion->dirty = false;

#ifndef NDEBUG
    ANGLE_MTL_OBJC_SCOPE
    {
        mConvertedArrayBufferHolders[attribIndex].getCurrentBuffer()->get().label =
            [NSString stringWithFormat:@"Converted from %p offset=%zu stride=%u", srcBuffer,
                                       binding.getOffset(), binding.getStride()];
    }
#endif

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::convertVertexBufferCPU(ContextMtl *contextMtl,
                                                     BufferMtl *srcBuffer,
                                                     const gl::VertexBinding &binding,
                                                     size_t attribIndex,
                                                     const mtl::VertexFormat &convertedFormat,
                                                     GLuint targetStride,
                                                     size_t numVertices,
                                                     ConversionBufferMtl *conversion)
{

    const uint8_t *srcBytes = srcBuffer->getClientShadowCopyData(contextMtl);
    ANGLE_CHECK_GL_ALLOC(contextMtl, srcBytes);

    srcBytes += binding.getOffset();

    ANGLE_TRY(StreamVertexData(
        contextMtl, &conversion->data, srcBytes, numVertices * targetStride, 0, numVertices,
        binding.getStride(), convertedFormat.vertexLoadFunction,
        &mConvertedArrayBufferHolders[attribIndex], &mCurrentArrayBufferOffsets[attribIndex]));

    return angle::Result::Continue;
}

angle::Result VertexArrayMtl::convertVertexBufferGPU(const gl::Context *glContext,
                                                     BufferMtl *srcBuffer,
                                                     const gl::VertexBinding &binding,
                                                     size_t attribIndex,
                                                     const mtl::VertexFormat &convertedFormat,
                                                     GLuint targetStride,
                                                     size_t numVertices,
                                                     bool isExpandingComponents,
                                                     ConversionBufferMtl *conversion)
{
    ContextMtl *contextMtl = mtl::GetImpl(glContext);

    mtl::BufferRef newBuffer;
    size_t newBufferOffset;
    ANGLE_TRY(conversion->data.allocate(contextMtl, numVertices * targetStride, nullptr, &newBuffer,
                                        &newBufferOffset));

    ANGLE_CHECK_GL_MATH(contextMtl, binding.getOffset() <= std::numeric_limits<uint32_t>::max());
    ANGLE_CHECK_GL_MATH(contextMtl, newBufferOffset <= std::numeric_limits<uint32_t>::max());
    ANGLE_CHECK_GL_MATH(contextMtl, numVertices <= std::numeric_limits<uint32_t>::max());

    mtl::VertexFormatConvertParams params;

    params.srcBuffer            = srcBuffer->getCurrentBuffer();
    params.srcBufferStartOffset = static_cast<uint32_t>(binding.getOffset());
    params.srcStride            = binding.getStride();
    params.srcDefaultAlphaData  = convertedFormat.defaultAlpha;

    params.dstBuffer            = newBuffer;
    params.dstBufferStartOffset = static_cast<uint32_t>(newBufferOffset);
    params.dstStride            = targetStride;
    params.dstComponents        = convertedFormat.actualAngleFormat().channelCount;

    params.vertexCount = static_cast<uint32_t>(numVertices);

    mtl::RenderUtils &utils                  = contextMtl->getDisplay()->getUtils();
    mtl::RenderCommandEncoder *renderEncoder = contextMtl->getRenderCommandEncoder();
    if (renderEncoder && contextMtl->getDisplay()->getFeatures().hasExplicitMemBarrier.enabled)
    {
        // If we are in the middle of a render pass, use vertex shader based buffer conversion to
        // avoid breaking the render pass.
        if (!isExpandingComponents)
        {
            ANGLE_TRY(utils.convertVertexFormatToFloatVS(
                glContext, renderEncoder, convertedFormat.intendedAngleFormat(), params));
        }
        else
        {
            ANGLE_TRY(utils.expandVertexFormatComponentsVS(
                glContext, renderEncoder, convertedFormat.intendedAngleFormat(), params));
        }
    }
    else
    {
        // Compute based buffer conversion.
        if (!isExpandingComponents)
        {
            ANGLE_TRY(utils.convertVertexFormatToFloatCS(
                contextMtl, convertedFormat.intendedAngleFormat(), params));
        }
        else
        {
            ANGLE_TRY(utils.expandVertexFormatComponentsCS(
                contextMtl, convertedFormat.intendedAngleFormat(), params));
        }
    }

    ANGLE_TRY(conversion->data.commit(contextMtl));

    mConvertedArrayBufferHolders[attribIndex].set(newBuffer);
    mCurrentArrayBufferOffsets[attribIndex] = newBufferOffset;

    return angle::Result::Continue;
}
}
