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

class Allocation : public NS::Referencing<Allocation>
{
public:
    NS::UInteger allocatedSize() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLAllocation;

_MTL_INLINE NS::UInteger MTL::Allocation::allocatedSize() const
{
    return _MTL_msg_NS__UInteger_allocatedSize((const void*)this, nullptr);
}
