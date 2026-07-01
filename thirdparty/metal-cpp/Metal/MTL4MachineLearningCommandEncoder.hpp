#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4CommandEncoder.hpp"

namespace MTL {
    class Heap;
}
namespace MTL4 {
    class ArgumentTable;
    class MachineLearningPipelineState;
}

namespace MTL4
{

class MachineLearningCommandEncoder : public NS::Referencing<MachineLearningCommandEncoder, MTL4::CommandEncoder>
{
public:
    void dispatchNetwork(MTL::Heap* heap);
    void setArgumentTable(MTL4::ArgumentTable* argumentTable);
    void setPipelineState(MTL4::MachineLearningPipelineState* pipelineState);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4MachineLearningCommandEncoder;

_MTL4_INLINE void MTL4::MachineLearningCommandEncoder::setPipelineState(MTL4::MachineLearningPipelineState* pipelineState)
{
    _MTL4_msg_v_setPipelineState__MTL4__MachineLearningPipelineStatep((const void*)this, nullptr, pipelineState);
}

_MTL4_INLINE void MTL4::MachineLearningCommandEncoder::setArgumentTable(MTL4::ArgumentTable* argumentTable)
{
    _MTL4_msg_v_setArgumentTable__MTL4__ArgumentTablep((const void*)this, nullptr, argumentTable);
}

_MTL4_INLINE void MTL4::MachineLearningCommandEncoder::dispatchNetwork(MTL::Heap* heap)
{
    _MTL4_msg_v_dispatchNetworkWithIntermediatesHeap__MTL__Heapp((const void*)this, nullptr, heap);
}
