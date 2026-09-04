#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Function;
}
namespace NS {
    class String;
    class URL;
}

namespace MTL
{

_MTL_ENUM(NS::UInteger, FunctionLogType) {
    FunctionLogTypeValidation = 0,
};


class LogContainer;
class FunctionLogDebugLocation;
class FunctionLog;

class LogContainer : public NS::Referencing<LogContainer>
{
public:
};

class FunctionLogDebugLocation : public NS::Referencing<FunctionLogDebugLocation>
{
public:
    NS::URL*     URL() const;
    NS::UInteger column() const;
    NS::String*  functionName() const;
    NS::UInteger line() const;

};

class FunctionLog : public NS::Referencing<FunctionLog>
{
public:
    MTL::FunctionLogDebugLocation* debugLocation() const;
    NS::String*                    encoderLabel() const;
    MTL::Function*                 function() const;
    MTL::FunctionLogType           type() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLLogContainer;
extern "C" void *OBJC_CLASS_$_MTLFunctionLogDebugLocation;
extern "C" void *OBJC_CLASS_$_MTLFunctionLog;

_MTL_INLINE NS::String* MTL::FunctionLogDebugLocation::functionName() const
{
    return _MTL_msg_NS__Stringp_functionName((const void*)this, nullptr);
}

_MTL_INLINE NS::URL* MTL::FunctionLogDebugLocation::URL() const
{
    return _MTL_msg_NS__URLp_URL((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::FunctionLogDebugLocation::line() const
{
    return _MTL_msg_NS__UInteger_line((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::FunctionLogDebugLocation::column() const
{
    return _MTL_msg_NS__UInteger_column((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionLogType MTL::FunctionLog::type() const
{
    return _MTL_msg_MTL__FunctionLogType_type((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::FunctionLog::encoderLabel() const
{
    return _MTL_msg_NS__Stringp_encoderLabel((const void*)this, nullptr);
}

_MTL_INLINE MTL::Function* MTL::FunctionLog::function() const
{
    return _MTL_msg_MTL__Functionp_function((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionLogDebugLocation* MTL::FunctionLog::debugLocation() const
{
    return _MTL_msg_MTL__FunctionLogDebugLocationp_debugLocation((const void*)this, nullptr);
}
