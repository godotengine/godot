#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResource.hpp"

namespace MTL {
    class FunctionHandle;
}

namespace MTL
{

class VisibleFunctionTableDescriptor;
class VisibleFunctionTable;

class VisibleFunctionTableDescriptor : public NS::Copying<VisibleFunctionTableDescriptor>
{
public:
    static VisibleFunctionTableDescriptor* alloc();
    VisibleFunctionTableDescriptor*        init() const;

    static MTL::VisibleFunctionTableDescriptor* visibleFunctionTableDescriptor();

    NS::UInteger functionCount() const;
    void         setFunctionCount(NS::UInteger functionCount);

};

class VisibleFunctionTable : public NS::Referencing<VisibleFunctionTable, MTL::Resource>
{
public:
    MTL::ResourceID gpuResourceID() const;
    void            setFunction(MTL::FunctionHandle* function, NS::UInteger index);
    void            setFunctions(const MTL::FunctionHandle* const * functions, NS::Range range);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLVisibleFunctionTableDescriptor;
extern "C" void *OBJC_CLASS_$_MTLVisibleFunctionTable;

_MTL_INLINE MTL::VisibleFunctionTableDescriptor* MTL::VisibleFunctionTableDescriptor::alloc()
{
    return _MTL_msg_MTL__VisibleFunctionTableDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLVisibleFunctionTableDescriptor, nullptr);
}

_MTL_INLINE MTL::VisibleFunctionTableDescriptor* MTL::VisibleFunctionTableDescriptor::init() const
{
    return _MTL_msg_MTL__VisibleFunctionTableDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::VisibleFunctionTableDescriptor* MTL::VisibleFunctionTableDescriptor::visibleFunctionTableDescriptor()
{
    return _MTL_msg_MTL__VisibleFunctionTableDescriptorp_visibleFunctionTableDescriptor((const void*)&OBJC_CLASS_$_MTLVisibleFunctionTableDescriptor, nullptr);
}

_MTL_INLINE NS::UInteger MTL::VisibleFunctionTableDescriptor::functionCount() const
{
    return _MTL_msg_NS__UInteger_functionCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VisibleFunctionTableDescriptor::setFunctionCount(NS::UInteger functionCount)
{
    _MTL_msg_v_setFunctionCount__NS__UInteger((const void*)this, nullptr, functionCount);
}

_MTL_INLINE MTL::ResourceID MTL::VisibleFunctionTable::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VisibleFunctionTable::setFunction(MTL::FunctionHandle* function, NS::UInteger index)
{
    _MTL_msg_v_setFunction_atIndex__MTL__FunctionHandlep_NS__UInteger((const void*)this, nullptr, function, index);
}

_MTL_INLINE void MTL::VisibleFunctionTable::setFunctions(const MTL::FunctionHandle* const * functions, NS::Range range)
{
    _MTL_msg_v_setFunctions_withRange__constMTL__FunctionHandlepconstp_NS__Range((const void*)this, nullptr, functions, range);
}
