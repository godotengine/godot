//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4LibraryDescriptor.hpp
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
class LibraryDescriptor;
}

namespace MTL
{
class CompileOptions;
}

namespace MTL4
{
class LibraryDescriptor : public NS::Copying<LibraryDescriptor>
{
public:
    static LibraryDescriptor* alloc();

    LibraryDescriptor*        init();

    NS::String*               name() const;

    MTL::CompileOptions*      options() const;

    void                      setName(const NS::String* name);

    void                      setOptions(const MTL::CompileOptions* options);

    void                      setSource(const NS::String* source);
    NS::String*               source() const;
};

}
_MTL_INLINE MTL4::LibraryDescriptor* MTL4::LibraryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::LibraryDescriptor>(_MTL_PRIVATE_CLS(MTL4LibraryDescriptor));
}

_MTL_INLINE MTL4::LibraryDescriptor* MTL4::LibraryDescriptor::init()
{
    return NS::Object::init<MTL4::LibraryDescriptor>();
}

_MTL_INLINE NS::String* MTL4::LibraryDescriptor::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE MTL::CompileOptions* MTL4::LibraryDescriptor::options() const
{
    return Object::sendMessage<MTL::CompileOptions*>(this, _MTL_PRIVATE_SEL(options));
}

_MTL_INLINE void MTL4::LibraryDescriptor::setName(const NS::String* name)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setName_), name);
}

_MTL_INLINE void MTL4::LibraryDescriptor::setOptions(const MTL::CompileOptions* options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOptions_), options);
}

_MTL_INLINE void MTL4::LibraryDescriptor::setSource(const NS::String* source)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSource_), source);
}

_MTL_INLINE NS::String* MTL4::LibraryDescriptor::source() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(source));
}
