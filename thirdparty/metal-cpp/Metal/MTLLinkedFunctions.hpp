#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace NS {
    class Array;
    class Dictionary;
}

namespace MTL
{

class LinkedFunctions : public NS::Copying<LinkedFunctions>
{
public:
    static LinkedFunctions* alloc();
    LinkedFunctions*        init() const;

    static MTL::LinkedFunctions* linkedFunctions();

    NS::Array*      binaryFunctions() const;
    NS::Array*      functions() const;
    NS::Dictionary* groups() const;
    NS::Array*      privateFunctions() const;
    void            setBinaryFunctions(NS::Array* binaryFunctions);
    void            setFunctions(NS::Array* functions);
    void            setGroups(NS::Dictionary* groups);
    void            setPrivateFunctions(NS::Array* privateFunctions);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLLinkedFunctions;

_MTL_INLINE MTL::LinkedFunctions* MTL::LinkedFunctions::alloc()
{
    return _MTL_msg_MTL__LinkedFunctionsp_alloc((const void*)&OBJC_CLASS_$_MTLLinkedFunctions, nullptr);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::LinkedFunctions::init() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::LinkedFunctions::linkedFunctions()
{
    return _MTL_msg_MTL__LinkedFunctionsp_linkedFunctions((const void*)&OBJC_CLASS_$_MTLLinkedFunctions, nullptr);
}

_MTL_INLINE NS::Array* MTL::LinkedFunctions::functions() const
{
    return _MTL_msg_NS__Arrayp_functions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LinkedFunctions::setFunctions(NS::Array* functions)
{
    _MTL_msg_v_setFunctions__NS__Arrayp((const void*)this, nullptr, functions);
}

_MTL_INLINE NS::Array* MTL::LinkedFunctions::binaryFunctions() const
{
    return _MTL_msg_NS__Arrayp_binaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LinkedFunctions::setBinaryFunctions(NS::Array* binaryFunctions)
{
    _MTL_msg_v_setBinaryFunctions__NS__Arrayp((const void*)this, nullptr, binaryFunctions);
}

_MTL_INLINE NS::Dictionary* MTL::LinkedFunctions::groups() const
{
    return _MTL_msg_NS__Dictionaryp_groups((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LinkedFunctions::setGroups(NS::Dictionary* groups)
{
    _MTL_msg_v_setGroups__NS__Dictionaryp((const void*)this, nullptr, groups);
}

_MTL_INLINE NS::Array* MTL::LinkedFunctions::privateFunctions() const
{
    return _MTL_msg_NS__Arrayp_privateFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LinkedFunctions::setPrivateFunctions(NS::Array* privateFunctions)
{
    _MTL_msg_v_setPrivateFunctions__NS__Arrayp((const void*)this, nullptr, privateFunctions);
}
