#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL
{

extern NS::ErrorDomain const LogStateErrorDomain __asm__("_MTLLogStateErrorDomain");
_MTL_ENUM(NS::Integer, LogLevel) {
    LogLevelUndefined = 0,
    LogLevelDebug = 1,
    LogLevelInfo = 2,
    LogLevelNotice = 3,
    LogLevelError = 4,
    LogLevelFault = 5,
};

_MTL_ENUM(NS::UInteger, LogStateError) {
    LogStateErrorInvalidSize = 1,
    LogStateErrorInvalid = 2,
};


class LogState;
class LogStateDescriptor;

class LogState : public NS::Referencing<LogState>
{
public:
    void addLogHandler(MTL::LogHandlerBlock block);
    void addLogHandler(const MTL::LogHandlerFunction& block);

};

class LogStateDescriptor : public NS::Copying<LogStateDescriptor>
{
public:
    static LogStateDescriptor* alloc();
    LogStateDescriptor*        init() const;

    NS::Integer   bufferSize() const;
    MTL::LogLevel level() const;
    void          setBufferSize(NS::Integer bufferSize);
    void          setLevel(MTL::LogLevel level);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLLogState;
extern "C" void *OBJC_CLASS_$_MTLLogStateDescriptor;

_MTL_INLINE void MTL::LogState::addLogHandler(MTL::LogHandlerBlock block)
{
    _MTL_msg_v_addLogHandler__MTL__LogHandlerBlock((const void*)this, nullptr, block);
}

_MTL_INLINE void MTL::LogState::addLogHandler(const MTL::LogHandlerFunction& block)
{
    __block MTL::LogHandlerFunction blockFunction = block;
    addLogHandler(^(NS::String* x0, NS::String* x1, MTL::LogLevel x2, NS::String* x3) { blockFunction(x0, x1, x2, x3); });
}

_MTL_INLINE MTL::LogStateDescriptor* MTL::LogStateDescriptor::alloc()
{
    return _MTL_msg_MTL__LogStateDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLLogStateDescriptor, nullptr);
}

_MTL_INLINE MTL::LogStateDescriptor* MTL::LogStateDescriptor::init() const
{
    return _MTL_msg_MTL__LogStateDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::LogLevel MTL::LogStateDescriptor::level() const
{
    return _MTL_msg_MTL__LogLevel_level((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LogStateDescriptor::setLevel(MTL::LogLevel level)
{
    _MTL_msg_v_setLevel__MTL__LogLevel((const void*)this, nullptr, level);
}

_MTL_INLINE NS::Integer MTL::LogStateDescriptor::bufferSize() const
{
    return _MTL_msg_NS__Integer_bufferSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LogStateDescriptor::setBufferSize(NS::Integer bufferSize)
{
    _MTL_msg_v_setBufferSize__NS__Integer((const void*)this, nullptr, bufferSize);
}
