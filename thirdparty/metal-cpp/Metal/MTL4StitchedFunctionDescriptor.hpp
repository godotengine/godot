#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4FunctionDescriptor.hpp"

namespace MTL {
    class FunctionStitchingGraph;
}
namespace NS {
    class Array;
}

namespace MTL4
{

class StitchedFunctionDescriptor : public NS::Referencing<StitchedFunctionDescriptor, MTL4::FunctionDescriptor>
{
public:
    static StitchedFunctionDescriptor* alloc();
    StitchedFunctionDescriptor*        init() const;

    NS::Array*                   functionDescriptors() const;
    MTL::FunctionStitchingGraph* functionGraph() const;
    void                         setFunctionDescriptors(NS::Array* functionDescriptors);
    void                         setFunctionGraph(MTL::FunctionStitchingGraph* functionGraph);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4StitchedFunctionDescriptor;

_MTL4_INLINE MTL4::StitchedFunctionDescriptor* MTL4::StitchedFunctionDescriptor::alloc()
{
    return _MTL4_msg_MTL4__StitchedFunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4StitchedFunctionDescriptor, nullptr);
}

_MTL4_INLINE MTL4::StitchedFunctionDescriptor* MTL4::StitchedFunctionDescriptor::init() const
{
    return _MTL4_msg_MTL4__StitchedFunctionDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL::FunctionStitchingGraph* MTL4::StitchedFunctionDescriptor::functionGraph() const
{
    return _MTL4_msg_MTL__FunctionStitchingGraphp_functionGraph((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::StitchedFunctionDescriptor::setFunctionGraph(MTL::FunctionStitchingGraph* functionGraph)
{
    _MTL4_msg_v_setFunctionGraph__MTL__FunctionStitchingGraphp((const void*)this, nullptr, functionGraph);
}

_MTL4_INLINE NS::Array* MTL4::StitchedFunctionDescriptor::functionDescriptors() const
{
    return _MTL4_msg_NS__Arrayp_functionDescriptors((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::StitchedFunctionDescriptor::setFunctionDescriptors(NS::Array* functionDescriptors)
{
    _MTL4_msg_v_setFunctionDescriptors__NS__Arrayp((const void*)this, nullptr, functionDescriptors);
}
