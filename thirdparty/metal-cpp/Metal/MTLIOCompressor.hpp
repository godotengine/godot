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

_MTL_ENUM(NS::Integer, IOCompressionStatus) {
    IOCompressionStatusComplete = 0,
    IOCompressionStatusError = 1,
};


} // namespace MTL
