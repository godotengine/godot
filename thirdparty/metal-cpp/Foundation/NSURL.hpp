//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSURL.hpp
//
// Copyright 2020-2024 Apple Inc.
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "NSDefines.hpp"
#include "NSObject.hpp"
#include "NSPrivate.hpp"
#include "NSTypes.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
class URL : public Copying<URL>
{
public:
    static URL* fileURLWithPath(const class String* pPath);

    static URL* alloc();
    URL*        init();
    URL*        init(const class String* pString);
    URL*        initFileURLWithPath(const class String* pPath);

    const char* fileSystemRepresentation() const;
};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::URL::fileURLWithPath(const String* pPath)
{
    return Object::sendMessage<URL*>(_NS_PRIVATE_CLS(NSURL), _NS_PRIVATE_SEL(fileURLWithPath_), pPath);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::URL::alloc()
{
    return Object::alloc<URL>(_NS_PRIVATE_CLS(NSURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::URL::init()
{
    return Object::init<URL>();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::URL::init(const String* pString)
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(initWithString_), pString);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::URL::initFileURLWithPath(const String* pPath)
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(initFileURLWithPath_), pPath);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE const char* NS::URL::fileSystemRepresentation() const
{
    return Object::sendMessage<const char*>(this, _NS_PRIVATE_SEL(fileSystemRepresentation));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
