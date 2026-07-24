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
    class Buffer;
    class FunctionHandle;
    class VisibleFunctionTable;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, IntersectionFunctionSignature) {
    IntersectionFunctionSignatureNone = 0,
    IntersectionFunctionSignatureInstancing = (1 << 0),
    IntersectionFunctionSignatureTriangleData = (1 << 1),
    IntersectionFunctionSignatureWorldSpaceData = (1 << 2),
    IntersectionFunctionSignatureInstanceMotion = (1 << 3),
    IntersectionFunctionSignaturePrimitiveMotion = (1 << 4),
    IntersectionFunctionSignatureExtendedLimits = (1 << 5),
    IntersectionFunctionSignatureMaxLevels = (1 << 6),
    IntersectionFunctionSignatureCurveData = (1 << 7),
    IntersectionFunctionSignatureIntersectionFunctionBuffer = (1 << 8),
    IntersectionFunctionSignatureUserData = (1 << 9),
};


class IntersectionFunctionTableDescriptor;
class IntersectionFunctionTable;

class IntersectionFunctionTableDescriptor : public NS::Copying<IntersectionFunctionTableDescriptor>
{
public:
    static IntersectionFunctionTableDescriptor* alloc();
    IntersectionFunctionTableDescriptor*        init() const;

    static MTL::IntersectionFunctionTableDescriptor* intersectionFunctionTableDescriptor();

    NS::UInteger functionCount() const;
    void         setFunctionCount(NS::UInteger functionCount);

};

class IntersectionFunctionTable : public NS::Referencing<IntersectionFunctionTable, MTL::Resource>
{
public:
    MTL::ResourceID gpuResourceID() const;
    void            setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void            setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void            setFunction(MTL::FunctionHandle* function, NS::UInteger index);
    void            setFunctions(const MTL::FunctionHandle* const * functions, NS::Range range);
    void            setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index);
    void            setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range);
    void            setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index);
    void            setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range);
    void            setVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void            setVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range bufferRange);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLIntersectionFunctionTableDescriptor;
extern "C" void *OBJC_CLASS_$_MTLIntersectionFunctionTable;

_MTL_INLINE MTL::IntersectionFunctionTableDescriptor* MTL::IntersectionFunctionTableDescriptor::alloc()
{
    return _MTL_msg_MTL__IntersectionFunctionTableDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLIntersectionFunctionTableDescriptor, nullptr);
}

_MTL_INLINE MTL::IntersectionFunctionTableDescriptor* MTL::IntersectionFunctionTableDescriptor::init() const
{
    return _MTL_msg_MTL__IntersectionFunctionTableDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::IntersectionFunctionTableDescriptor* MTL::IntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor()
{
    return _MTL_msg_MTL__IntersectionFunctionTableDescriptorp_intersectionFunctionTableDescriptor((const void*)&OBJC_CLASS_$_MTLIntersectionFunctionTableDescriptor, nullptr);
}

_MTL_INLINE NS::UInteger MTL::IntersectionFunctionTableDescriptor::functionCount() const
{
    return _MTL_msg_NS__UInteger_functionCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IntersectionFunctionTableDescriptor::setFunctionCount(NS::UInteger functionCount)
{
    _MTL_msg_v_setFunctionCount__NS__UInteger((const void*)this, nullptr, functionCount);
}

_MTL_INLINE MTL::ResourceID MTL::IntersectionFunctionTable::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setFunction(MTL::FunctionHandle* function, NS::UInteger index)
{
    _MTL_msg_v_setFunction_atIndex__MTL__FunctionHandlep_NS__UInteger((const void*)this, nullptr, function, index);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setFunctions(const MTL::FunctionHandle* const * functions, NS::Range range)
{
    _MTL_msg_v_setFunctions_withRange__constMTL__FunctionHandlepconstp_NS__Range((const void*)this, nullptr, functions, range);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index)
{
    _MTL_msg_v_setOpaqueTriangleIntersectionFunctionWithSignature_atIndex__MTL__IntersectionFunctionSignature_NS__UInteger((const void*)this, nullptr, signature, index);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range)
{
    _MTL_msg_v_setOpaqueTriangleIntersectionFunctionWithSignature_withRange__MTL__IntersectionFunctionSignature_NS__Range((const void*)this, nullptr, signature, range);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index)
{
    _MTL_msg_v_setOpaqueCurveIntersectionFunctionWithSignature_atIndex__MTL__IntersectionFunctionSignature_NS__UInteger((const void*)this, nullptr, signature, index);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range)
{
    _MTL_msg_v_setOpaqueCurveIntersectionFunctionWithSignature_withRange__MTL__IntersectionFunctionSignature_NS__Range((const void*)this, nullptr, signature, range);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setVisibleFunctionTable_atBufferIndex__MTL__VisibleFunctionTablep_NS__UInteger((const void*)this, nullptr, functionTable, bufferIndex);
}

_MTL_INLINE void MTL::IntersectionFunctionTable::setVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range bufferRange)
{
    _MTL_msg_v_setVisibleFunctionTables_withBufferRange__constMTL__VisibleFunctionTablepconstp_NS__Range((const void*)this, nullptr, functionTables, bufferRange);
}
