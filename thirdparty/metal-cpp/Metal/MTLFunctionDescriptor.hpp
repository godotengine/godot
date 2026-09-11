#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class FunctionConstantValues;
}
namespace NS {
    class Array;
    class String;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, FunctionOptions) {
    FunctionOptionNone = 0,
    FunctionOptionCompileToBinary = 1 << 0,
    FunctionOptionStoreFunctionInMetalPipelinesScript = 1 << 1,
    FunctionOptionStoreFunctionInMetalScript = 1 << 1,
    FunctionOptionFailOnBinaryArchiveMiss = 1 << 2,
    FunctionOptionPipelineIndependent = 1 << 3,
};


class FunctionDescriptor;
class IntersectionFunctionDescriptor;

class FunctionDescriptor : public NS::Copying<FunctionDescriptor>
{
public:
    static FunctionDescriptor* alloc();
    FunctionDescriptor*        init() const;

    static MTL::FunctionDescriptor* functionDescriptor();

    NS::Array*                   binaryArchives() const;
    MTL::FunctionConstantValues* constantValues() const;
    NS::String*                  name() const;
    MTL::FunctionOptions         options() const;
    void                         setBinaryArchives(NS::Array* binaryArchives);
    void                         setConstantValues(MTL::FunctionConstantValues* constantValues);
    void                         setName(NS::String* name);
    void                         setOptions(MTL::FunctionOptions options);
    void                         setSpecializedName(NS::String* specializedName);
    NS::String*                  specializedName() const;

};

class IntersectionFunctionDescriptor : public NS::Copying<IntersectionFunctionDescriptor, MTL::FunctionDescriptor>
{
public:
    static IntersectionFunctionDescriptor* alloc();
    IntersectionFunctionDescriptor*        init() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFunctionDescriptor;
extern "C" void *OBJC_CLASS_$_MTLIntersectionFunctionDescriptor;

_MTL_INLINE MTL::FunctionDescriptor* MTL::FunctionDescriptor::alloc()
{
    return _MTL_msg_MTL__FunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLFunctionDescriptor, nullptr);
}

_MTL_INLINE MTL::FunctionDescriptor* MTL::FunctionDescriptor::init() const
{
    return _MTL_msg_MTL__FunctionDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionDescriptor* MTL::FunctionDescriptor::functionDescriptor()
{
    return _MTL_msg_MTL__FunctionDescriptorp_functionDescriptor((const void*)&OBJC_CLASS_$_MTLFunctionDescriptor, nullptr);
}

_MTL_INLINE NS::String* MTL::FunctionDescriptor::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionDescriptor::setName(NS::String* name)
{
    _MTL_msg_v_setName__NS__Stringp((const void*)this, nullptr, name);
}

_MTL_INLINE NS::String* MTL::FunctionDescriptor::specializedName() const
{
    return _MTL_msg_NS__Stringp_specializedName((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionDescriptor::setSpecializedName(NS::String* specializedName)
{
    _MTL_msg_v_setSpecializedName__NS__Stringp((const void*)this, nullptr, specializedName);
}

_MTL_INLINE MTL::FunctionConstantValues* MTL::FunctionDescriptor::constantValues() const
{
    return _MTL_msg_MTL__FunctionConstantValuesp_constantValues((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionDescriptor::setConstantValues(MTL::FunctionConstantValues* constantValues)
{
    _MTL_msg_v_setConstantValues__MTL__FunctionConstantValuesp((const void*)this, nullptr, constantValues);
}

_MTL_INLINE MTL::FunctionOptions MTL::FunctionDescriptor::options() const
{
    return _MTL_msg_MTL__FunctionOptions_options((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionDescriptor::setOptions(MTL::FunctionOptions options)
{
    _MTL_msg_v_setOptions__MTL__FunctionOptions((const void*)this, nullptr, options);
}

_MTL_INLINE NS::Array* MTL::FunctionDescriptor::binaryArchives() const
{
    return _MTL_msg_NS__Arrayp_binaryArchives((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionDescriptor::setBinaryArchives(NS::Array* binaryArchives)
{
    _MTL_msg_v_setBinaryArchives__NS__Arrayp((const void*)this, nullptr, binaryArchives);
}

_MTL_INLINE MTL::IntersectionFunctionDescriptor* MTL::IntersectionFunctionDescriptor::alloc()
{
    return _MTL_msg_MTL__IntersectionFunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLIntersectionFunctionDescriptor, nullptr);
}

_MTL_INLINE MTL::IntersectionFunctionDescriptor* MTL::IntersectionFunctionDescriptor::init() const
{
    return _MTL_msg_MTL__IntersectionFunctionDescriptorp_init((const void*)this, nullptr);
}
