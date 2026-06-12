#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Buffer;
    class IOFileHandle;
    class SharedEvent;
    class Texture;
}
namespace NS {
    class Error;
    class String;
}

namespace MTL
{

_MTL_ENUM(NS::Integer, IOStatus) {
    IOStatusPending = 0,
    IOStatusCancelled = 1,
    IOStatusError = 2,
    IOStatusComplete = 3,
};


class IOCommandBuffer : public NS::Referencing<IOCommandBuffer>
{
public:
    void          addBarrier();
    void          addCompletedHandler(MTL::IOCommandBufferHandler block);
    void          addCompletedHandler(const MTL::IOCommandBufferHandlerFunction& block);
    void          commit();
    void          copyStatusToBuffer(MTL::Buffer* buffer, NS::UInteger offset);
    void          enqueue();
    NS::Error*    error() const;
    NS::String*   label() const;
    void          loadBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger size, MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset);
    void          loadBytes(void * pointer, NS::UInteger size, MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset);
    void          loadTexture(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level, MTL::Size size, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Origin destinationOrigin, MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset);
    void          popDebugGroup();
    void          pushDebugGroup(NS::String* string);
    void          setLabel(NS::String* label);
    void          signalEvent(MTL::SharedEvent* event, uint64_t value);
    MTL::IOStatus status() const;
    void          tryCancel();
    void          waitForEvent(MTL::SharedEvent* event, uint64_t value);
    void          waitUntilCompleted();

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLIOCommandBuffer;

_MTL_INLINE NS::String* MTL::IOCommandBuffer::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::IOStatus MTL::IOCommandBuffer::status() const
{
    return _MTL_msg_MTL__IOStatus_status((const void*)this, nullptr);
}

_MTL_INLINE NS::Error* MTL::IOCommandBuffer::error() const
{
    return _MTL_msg_NS__Errorp_error((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::addCompletedHandler(MTL::IOCommandBufferHandler block)
{
    _MTL_msg_v_addCompletedHandler__MTL__IOCommandBufferHandler((const void*)this, nullptr, block);
}

_MTL_INLINE void MTL::IOCommandBuffer::addCompletedHandler(const MTL::IOCommandBufferHandlerFunction& block)
{
    __block MTL::IOCommandBufferHandlerFunction blockFunction = block;
    addCompletedHandler(^(MTL::IOCommandBuffer* x0) { blockFunction(x0); });
}

_MTL_INLINE void MTL::IOCommandBuffer::loadBytes(void * pointer, NS::UInteger size, MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset)
{
    _MTL_msg_v_loadBytes_size_sourceHandle_sourceHandleOffset__voidp_NS__UInteger_MTL__IOFileHandlep_NS__UInteger((const void*)this, nullptr, pointer, size, sourceHandle, sourceHandleOffset);
}

_MTL_INLINE void MTL::IOCommandBuffer::loadBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger size, MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset)
{
    _MTL_msg_v_loadBuffer_offset_size_sourceHandle_sourceHandleOffset__MTL__Bufferp_NS__UInteger_NS__UInteger_MTL__IOFileHandlep_NS__UInteger((const void*)this, nullptr, buffer, offset, size, sourceHandle, sourceHandleOffset);
}

_MTL_INLINE void MTL::IOCommandBuffer::loadTexture(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level, MTL::Size size, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Origin destinationOrigin, MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset)
{
    _MTL_msg_v_loadTexture_slice_level_size_sourceBytesPerRow_sourceBytesPerImage_destinationOrigin_sourceHandle_sourceHandleOffset__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Size_NS__UInteger_NS__UInteger_MTL__Origin_MTL__IOFileHandlep_NS__UInteger((const void*)this, nullptr, texture, slice, level, size, sourceBytesPerRow, sourceBytesPerImage, destinationOrigin, sourceHandle, sourceHandleOffset);
}

_MTL_INLINE void MTL::IOCommandBuffer::copyStatusToBuffer(MTL::Buffer* buffer, NS::UInteger offset)
{
    _MTL_msg_v_copyStatusToBuffer_offset__MTL__Bufferp_NS__UInteger((const void*)this, nullptr, buffer, offset);
}

_MTL_INLINE void MTL::IOCommandBuffer::commit()
{
    _MTL_msg_v_commit((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::waitUntilCompleted()
{
    _MTL_msg_v_waitUntilCompleted((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::tryCancel()
{
    _MTL_msg_v_tryCancel((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::addBarrier()
{
    _MTL_msg_v_addBarrier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::pushDebugGroup(NS::String* string)
{
    _MTL_msg_v_pushDebugGroup__NS__Stringp((const void*)this, nullptr, string);
}

_MTL_INLINE void MTL::IOCommandBuffer::popDebugGroup()
{
    _MTL_msg_v_popDebugGroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::enqueue()
{
    _MTL_msg_v_enqueue((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandBuffer::waitForEvent(MTL::SharedEvent* event, uint64_t value)
{
    _MTL_msg_v_waitForEvent_value__MTL__SharedEventp_uint64_t((const void*)this, nullptr, event, value);
}

_MTL_INLINE void MTL::IOCommandBuffer::signalEvent(MTL::SharedEvent* event, uint64_t value)
{
    _MTL_msg_v_signalEvent_value__MTL__SharedEventp_uint64_t((const void*)this, nullptr, event, value);
}
