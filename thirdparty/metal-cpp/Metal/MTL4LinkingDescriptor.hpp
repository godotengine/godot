#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace NS {
    class Array;
    class Dictionary;
}

namespace MTL4
{

class StaticLinkingDescriptor;
class PipelineStageDynamicLinkingDescriptor;
class RenderPipelineDynamicLinkingDescriptor;

class StaticLinkingDescriptor : public NS::Copying<StaticLinkingDescriptor>
{
public:
    static StaticLinkingDescriptor* alloc();
    StaticLinkingDescriptor*        init() const;

    NS::Array*      functionDescriptors() const;
    NS::Dictionary* groups() const;
    NS::Array*      privateFunctionDescriptors() const;
    void            setFunctionDescriptors(NS::Array* functionDescriptors);
    void            setGroups(NS::Dictionary* groups);
    void            setPrivateFunctionDescriptors(NS::Array* privateFunctionDescriptors);

};

class PipelineStageDynamicLinkingDescriptor : public NS::Copying<PipelineStageDynamicLinkingDescriptor>
{
public:
    static PipelineStageDynamicLinkingDescriptor* alloc();
    PipelineStageDynamicLinkingDescriptor*        init() const;

    NS::Array*   binaryLinkedFunctions() const;
    NS::UInteger maxCallStackDepth() const;
    NS::Array*   preloadedLibraries() const;
    void         setBinaryLinkedFunctions(NS::Array* binaryLinkedFunctions);
    void         setMaxCallStackDepth(NS::UInteger maxCallStackDepth);
    void         setPreloadedLibraries(NS::Array* preloadedLibraries);

};

class RenderPipelineDynamicLinkingDescriptor : public NS::Copying<RenderPipelineDynamicLinkingDescriptor>
{
public:
    static RenderPipelineDynamicLinkingDescriptor* alloc();
    RenderPipelineDynamicLinkingDescriptor*        init() const;

    MTL4::PipelineStageDynamicLinkingDescriptor* fragmentLinkingDescriptor() const;
    MTL4::PipelineStageDynamicLinkingDescriptor* meshLinkingDescriptor() const;
    MTL4::PipelineStageDynamicLinkingDescriptor* objectLinkingDescriptor() const;
    MTL4::PipelineStageDynamicLinkingDescriptor* tileLinkingDescriptor() const;
    MTL4::PipelineStageDynamicLinkingDescriptor* vertexLinkingDescriptor() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4StaticLinkingDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4PipelineStageDynamicLinkingDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4RenderPipelineDynamicLinkingDescriptor;

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::StaticLinkingDescriptor::alloc()
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4StaticLinkingDescriptor, nullptr);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::StaticLinkingDescriptor::init() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::Array* MTL4::StaticLinkingDescriptor::functionDescriptors() const
{
    return _MTL4_msg_NS__Arrayp_functionDescriptors((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::StaticLinkingDescriptor::setFunctionDescriptors(NS::Array* functionDescriptors)
{
    _MTL4_msg_v_setFunctionDescriptors__NS__Arrayp((const void*)this, nullptr, functionDescriptors);
}

_MTL4_INLINE NS::Array* MTL4::StaticLinkingDescriptor::privateFunctionDescriptors() const
{
    return _MTL4_msg_NS__Arrayp_privateFunctionDescriptors((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::StaticLinkingDescriptor::setPrivateFunctionDescriptors(NS::Array* privateFunctionDescriptors)
{
    _MTL4_msg_v_setPrivateFunctionDescriptors__NS__Arrayp((const void*)this, nullptr, privateFunctionDescriptors);
}

_MTL4_INLINE NS::Dictionary* MTL4::StaticLinkingDescriptor::groups() const
{
    return _MTL4_msg_NS__Dictionaryp_groups((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::StaticLinkingDescriptor::setGroups(NS::Dictionary* groups)
{
    _MTL4_msg_v_setGroups__NS__Dictionaryp((const void*)this, nullptr, groups);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::PipelineStageDynamicLinkingDescriptor::alloc()
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4PipelineStageDynamicLinkingDescriptor, nullptr);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::PipelineStageDynamicLinkingDescriptor::init() const
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::UInteger MTL4::PipelineStageDynamicLinkingDescriptor::maxCallStackDepth() const
{
    return _MTL4_msg_NS__UInteger_maxCallStackDepth((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineStageDynamicLinkingDescriptor::setMaxCallStackDepth(NS::UInteger maxCallStackDepth)
{
    _MTL4_msg_v_setMaxCallStackDepth__NS__UInteger((const void*)this, nullptr, maxCallStackDepth);
}

_MTL4_INLINE NS::Array* MTL4::PipelineStageDynamicLinkingDescriptor::binaryLinkedFunctions() const
{
    return _MTL4_msg_NS__Arrayp_binaryLinkedFunctions((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineStageDynamicLinkingDescriptor::setBinaryLinkedFunctions(NS::Array* binaryLinkedFunctions)
{
    _MTL4_msg_v_setBinaryLinkedFunctions__NS__Arrayp((const void*)this, nullptr, binaryLinkedFunctions);
}

_MTL4_INLINE NS::Array* MTL4::PipelineStageDynamicLinkingDescriptor::preloadedLibraries() const
{
    return _MTL4_msg_NS__Arrayp_preloadedLibraries((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineStageDynamicLinkingDescriptor::setPreloadedLibraries(NS::Array* preloadedLibraries)
{
    _MTL4_msg_v_setPreloadedLibraries__NS__Arrayp((const void*)this, nullptr, preloadedLibraries);
}

_MTL4_INLINE MTL4::RenderPipelineDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::alloc()
{
    return _MTL4_msg_MTL4__RenderPipelineDynamicLinkingDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4RenderPipelineDynamicLinkingDescriptor, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::init() const
{
    return _MTL4_msg_MTL4__RenderPipelineDynamicLinkingDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::vertexLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_vertexLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::fragmentLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_fragmentLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::tileLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_tileLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::objectLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_objectLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::meshLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__PipelineStageDynamicLinkingDescriptorp_meshLinkingDescriptor((const void*)this, nullptr);
}
