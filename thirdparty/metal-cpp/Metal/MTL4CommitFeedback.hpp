#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace NS {
    class Error;
}

namespace MTL4
{

class CommitFeedback : public NS::Referencing<CommitFeedback>
{
public:
    CFTimeInterval GPUEndTime() const;
    CFTimeInterval GPUStartTime() const;
    NS::Error*     error() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CommitFeedback;

_MTL4_INLINE NS::Error* MTL4::CommitFeedback::error() const
{
    return _MTL4_msg_NS__Errorp_error((const void*)this, nullptr);
}

_MTL4_INLINE CFTimeInterval MTL4::CommitFeedback::GPUStartTime() const
{
    return _MTL4_msg_CFTimeInterval_GPUStartTime((const void*)this, nullptr);
}

_MTL4_INLINE CFTimeInterval MTL4::CommitFeedback::GPUEndTime() const
{
    return _MTL4_msg_CFTimeInterval_GPUEndTime((const void*)this, nullptr);
}
