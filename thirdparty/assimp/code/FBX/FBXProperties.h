/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file  FBXProperties.h
 *  @brief FBX dynamic properties
 */
#ifndef INCLUDED_AI_FBX_PROPERTIES_H
#define INCLUDED_AI_FBX_PROPERTIES_H

#include "FBXCompileConfig.h"
#include <memory>
#include <string>

namespace Assimp {
namespace FBX {

// Forward declarations
class Element;

/** Represents a dynamic property. Type info added by deriving classes,
 *  see #TypedProperty.
 Example:
 @verbatim
   P: "ShininessExponent", "double", "Number", "",0.5
 @endvebatim
*/
class Property {
protected:
    Property();

public:
    virtual ~Property();

public:
    template <typename T>
    const T* As() const {
        return dynamic_cast<const T*>(this);
    }
};

template<typename T>
class TypedProperty : public Property {
public:
    explicit TypedProperty(const T& value)
    : value(value) {
        // empty
    }

    const T& Value() const {
        return value;
    }

private:
    T value;
};


typedef std::fbx_unordered_map<std::string,std::shared_ptr<Property> > DirectPropertyMap;
typedef std::fbx_unordered_map<std::string,const Property*>            PropertyMap;
typedef std::fbx_unordered_map<std::string,const Element*>             LazyPropertyMap;

/** 
 *  Represents a property table as can be found in the newer FBX files (Properties60, Properties70)
 */
class PropertyTable {
public:
    // in-memory property table with no source element
    PropertyTable();
    PropertyTable(const Element& element, std::shared_ptr<const PropertyTable> templateProps);
    ~PropertyTable();

    const Property* Get(const std::string& name) const;

    // PropertyTable's need not be coupled with FBX elements so this can be NULL
    const Element* GetElement() const {
        return element;
    }

    const PropertyTable* TemplateProps() const {
        return templateProps.get();
    }

    DirectPropertyMap GetUnparsedProperties() const;

private:
    LazyPropertyMap lazyProps;
    mutable PropertyMap props;
    const std::shared_ptr<const PropertyTable> templateProps;
    const Element* const element;
};

// ------------------------------------------------------------------------------------------------
template <typename T>
inline 
T PropertyGet(const PropertyTable& in, const std::string& name, const T& defaultValue) {
    const Property* const prop = in.Get(name);
    if( nullptr == prop) {
        return defaultValue;
    }

    // strong typing, no need to be lenient
    const TypedProperty<T>* const tprop = prop->As< TypedProperty<T> >();
    if( nullptr == tprop) {
        return defaultValue;
    }

    return tprop->Value();
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline 
T PropertyGet(const PropertyTable& in, const std::string& name, bool& result, bool useTemplate=false ) {
    const Property* prop = in.Get(name);
    if( nullptr == prop) {
        if ( ! useTemplate ) {
            result = false;
            return T();
        }
        const PropertyTable* templ = in.TemplateProps();
        if ( nullptr == templ ) {
            result = false;
            return T();
        }
        prop = templ->Get(name);
        if ( nullptr == prop ) {
            result = false;
            return T();
        }
    }

    // strong typing, no need to be lenient
    const TypedProperty<T>* const tprop = prop->As< TypedProperty<T> >();
    if( nullptr == tprop) {
        result = false;
        return T();
    }

    result = true;
    return tprop->Value();
}

} //! FBX
} //! Assimp

#endif // INCLUDED_AI_FBX_PROPERTIES_H
