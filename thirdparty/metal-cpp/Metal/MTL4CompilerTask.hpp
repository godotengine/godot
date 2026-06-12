#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL4 {
    class Compiler;
}

namespace MTL4
{

_MTL4_ENUM(NS::Integer, CompilerTaskStatus) {
    CompilerTaskStatusNone = 0,
    CompilerTaskStatusScheduled = 1,
    CompilerTaskStatusCompiling = 2,
    CompilerTaskStatusFinished = 3,
};


class CompilerTask : public NS::Referencing<CompilerTask>
{
public:
    MTL4::Compiler*          compiler() const;
    MTL4::CompilerTaskStatus status() const;
    void                     waitUntilCompleted();

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CompilerTask;

_MTL4_INLINE MTL4::Compiler* MTL4::CompilerTask::compiler() const
{
    return _MTL4_msg_MTL4__Compilerp_compiler((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::CompilerTaskStatus MTL4::CompilerTask::status() const
{
    return _MTL4_msg_MTL4__CompilerTaskStatus_status((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CompilerTask::waitUntilCompleted()
{
    _MTL4_msg_v_waitUntilCompleted((const void*)this, nullptr);
}
