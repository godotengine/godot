#pragma once

#include "MTL4Defines.hpp"
#include "../Foundation/NSTypes.hpp"

namespace MTL {
    enum SparseTextureMappingMode : NS::UInteger;
}

namespace MTL4 {

struct UpdateSparseTextureMappingOperation {
    MTL::SparseTextureMappingMode mode;
    MTL::Region textureRegion;
    NS::UInteger textureLevel;
    NS::UInteger textureSlice;
    NS::UInteger heapOffset;
} _MTL4_PACKED;

struct CopySparseTextureMappingOperation {
    MTL::Region sourceRegion;
    NS::UInteger sourceLevel;
    NS::UInteger sourceSlice;
    MTL::Origin destinationOrigin;
    NS::UInteger destinationLevel;
    NS::UInteger destinationSlice;
} _MTL4_PACKED;

struct UpdateSparseBufferMappingOperation {
    MTL::SparseTextureMappingMode mode;
    NS::Range bufferRange;
    NS::UInteger heapOffset;
} _MTL4_PACKED;

struct CopySparseBufferMappingOperation {
    NS::Range sourceRange;
    NS::UInteger destinationOffset;
} _MTL4_PACKED;

struct TimestampHeapEntry {
    uint64_t timestamp;
} _MTL4_PACKED;

} // MTL4
