//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSEnumerator.hpp
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

#include "NSObject.hpp"
#include "NSTypes.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
struct FastEnumerationState
{
    unsigned long  state;
    Object**       itemsPtr;
    unsigned long* mutationsPtr;
    unsigned long  extra[5];
} _NS_PACKED;

class FastEnumeration : public Referencing<FastEnumeration>
{
public:
    NS::UInteger countByEnumerating(FastEnumerationState* pState, Object** pBuffer, NS::UInteger len);
};

template <class _ObjectType>
class Enumerator : public Referencing<Enumerator<_ObjectType>, FastEnumeration>
{
public:
    _ObjectType* nextObject();
    class Array* allObjects();
};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::FastEnumeration::countByEnumerating(FastEnumerationState* pState, Object** pBuffer, NS::UInteger len)
{
    return Object::sendMessage<UInteger>(this, _NS_PRIVATE_SEL(countByEnumeratingWithState_objects_count_), pState, pBuffer, len);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _ObjectType>
_NS_INLINE _ObjectType* NS::Enumerator<_ObjectType>::nextObject()
{
    return Object::sendMessage<_ObjectType*>(this, _NS_PRIVATE_SEL(nextObject));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _ObjectType>
_NS_INLINE NS::Array* NS::Enumerator<_ObjectType>::allObjects()
{
    return Object::sendMessage<Array*>(this, _NS_PRIVATE_SEL(allObjects));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
