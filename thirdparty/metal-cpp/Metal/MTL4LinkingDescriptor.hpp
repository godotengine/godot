//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4LinkingDescriptor.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL4
{
class PipelineStageDynamicLinkingDescriptor;
class RenderPipelineDynamicLinkingDescriptor;
class StaticLinkingDescriptor;

class StaticLinkingDescriptor : public NS::Copying<StaticLinkingDescriptor>
{
public:
    static StaticLinkingDescriptor* alloc();

    NS::Array*                      functionDescriptors() const;

    NS::Dictionary*                 groups() const;

    StaticLinkingDescriptor*        init();

    NS::Array*                      privateFunctionDescriptors() const;

    void                            setFunctionDescriptors(const NS::Array* functionDescriptors);

    void                            setGroups(const NS::Dictionary* groups);

    void                            setPrivateFunctionDescriptors(const NS::Array* privateFunctionDescriptors);
};
class PipelineStageDynamicLinkingDescriptor : public NS::Copying<PipelineStageDynamicLinkingDescriptor>
{
public:
    static PipelineStageDynamicLinkingDescriptor* alloc();

    NS::Array*                                    binaryLinkedFunctions() const;

    PipelineStageDynamicLinkingDescriptor*        init();

    NS::UInteger                                  maxCallStackDepth() const;

    NS::Array*                                    preloadedLibraries() const;

    void                                          setBinaryLinkedFunctions(const NS::Array* binaryLinkedFunctions);

    void                                          setMaxCallStackDepth(NS::UInteger maxCallStackDepth);

    void                                          setPreloadedLibraries(const NS::Array* preloadedLibraries);
};
class RenderPipelineDynamicLinkingDescriptor : public NS::Copying<RenderPipelineDynamicLinkingDescriptor>
{
public:
    static RenderPipelineDynamicLinkingDescriptor* alloc();

    PipelineStageDynamicLinkingDescriptor*         fragmentLinkingDescriptor() const;

    RenderPipelineDynamicLinkingDescriptor*        init();

    PipelineStageDynamicLinkingDescriptor*         meshLinkingDescriptor() const;

    PipelineStageDynamicLinkingDescriptor*         objectLinkingDescriptor() const;

    PipelineStageDynamicLinkingDescriptor*         tileLinkingDescriptor() const;

    PipelineStageDynamicLinkingDescriptor*         vertexLinkingDescriptor() const;
};

}
_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::StaticLinkingDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::StaticLinkingDescriptor>(_MTL_PRIVATE_CLS(MTL4StaticLinkingDescriptor));
}

_MTL_INLINE NS::Array* MTL4::StaticLinkingDescriptor::functionDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(functionDescriptors));
}

_MTL_INLINE NS::Dictionary* MTL4::StaticLinkingDescriptor::groups() const
{
    return Object::sendMessage<NS::Dictionary*>(this, _MTL_PRIVATE_SEL(groups));
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::StaticLinkingDescriptor::init()
{
    return NS::Object::init<MTL4::StaticLinkingDescriptor>();
}

_MTL_INLINE NS::Array* MTL4::StaticLinkingDescriptor::privateFunctionDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(privateFunctionDescriptors));
}

_MTL_INLINE void MTL4::StaticLinkingDescriptor::setFunctionDescriptors(const NS::Array* functionDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctionDescriptors_), functionDescriptors);
}

_MTL_INLINE void MTL4::StaticLinkingDescriptor::setGroups(const NS::Dictionary* groups)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setGroups_), groups);
}

_MTL_INLINE void MTL4::StaticLinkingDescriptor::setPrivateFunctionDescriptors(const NS::Array* privateFunctionDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrivateFunctionDescriptors_), privateFunctionDescriptors);
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::PipelineStageDynamicLinkingDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::PipelineStageDynamicLinkingDescriptor>(_MTL_PRIVATE_CLS(MTL4PipelineStageDynamicLinkingDescriptor));
}

_MTL_INLINE NS::Array* MTL4::PipelineStageDynamicLinkingDescriptor::binaryLinkedFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryLinkedFunctions));
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::PipelineStageDynamicLinkingDescriptor::init()
{
    return NS::Object::init<MTL4::PipelineStageDynamicLinkingDescriptor>();
}

_MTL_INLINE NS::UInteger MTL4::PipelineStageDynamicLinkingDescriptor::maxCallStackDepth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxCallStackDepth));
}

_MTL_INLINE NS::Array* MTL4::PipelineStageDynamicLinkingDescriptor::preloadedLibraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(preloadedLibraries));
}

_MTL_INLINE void MTL4::PipelineStageDynamicLinkingDescriptor::setBinaryLinkedFunctions(const NS::Array* binaryLinkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryLinkedFunctions_), binaryLinkedFunctions);
}

_MTL_INLINE void MTL4::PipelineStageDynamicLinkingDescriptor::setMaxCallStackDepth(NS::UInteger maxCallStackDepth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCallStackDepth_), maxCallStackDepth);
}

_MTL_INLINE void MTL4::PipelineStageDynamicLinkingDescriptor::setPreloadedLibraries(const NS::Array* preloadedLibraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPreloadedLibraries_), preloadedLibraries);
}

_MTL_INLINE MTL4::RenderPipelineDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::RenderPipelineDynamicLinkingDescriptor>(_MTL_PRIVATE_CLS(MTL4RenderPipelineDynamicLinkingDescriptor));
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::fragmentLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::PipelineStageDynamicLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(fragmentLinkingDescriptor));
}

_MTL_INLINE MTL4::RenderPipelineDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::init()
{
    return NS::Object::init<MTL4::RenderPipelineDynamicLinkingDescriptor>();
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::meshLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::PipelineStageDynamicLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(meshLinkingDescriptor));
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::objectLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::PipelineStageDynamicLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(objectLinkingDescriptor));
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::tileLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::PipelineStageDynamicLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(tileLinkingDescriptor));
}

_MTL_INLINE MTL4::PipelineStageDynamicLinkingDescriptor* MTL4::RenderPipelineDynamicLinkingDescriptor::vertexLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::PipelineStageDynamicLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(vertexLinkingDescriptor));
}
