/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team
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

/** @file  FBXDocumentUtil.h
 *  @brief FBX internal utilities used by the DOM reading code
 */
#ifndef INCLUDED_AI_FBX_DOCUMENT_UTIL_H
#define INCLUDED_AI_FBX_DOCUMENT_UTIL_H

#include <assimp/defs.h>
#include <string>
#include <memory>
#include "FBXDocument.h"

struct Token;
struct Element;

namespace Assimp {
namespace FBX {
namespace Util {

/* DOM/Parse error reporting - does not return */
AI_WONT_RETURN void DOMError(const std::string& message, const Token& token) AI_WONT_RETURN_SUFFIX;
AI_WONT_RETURN void DOMError(const std::string& message, const Element* element = NULL) AI_WONT_RETURN_SUFFIX;

// does return
void DOMWarning(const std::string& message, const Token& token);
void DOMWarning(const std::string& message, const Element* element = NULL);


// fetch a property table and the corresponding property template
std::shared_ptr<const PropertyTable> GetPropertyTable(const Document& doc,
    const std::string& templateName,
    const Element &element,
    const Scope& sc,
    bool no_warn = false);

// ------------------------------------------------------------------------------------------------
template <typename T>
inline
const T* ProcessSimpleConnection(const Connection& con,
    bool is_object_property_conn,
    const char* name,
    const Element& element,
    const char** propNameOut = nullptr)
{
    if (is_object_property_conn && !con.PropertyName().length()) {
        DOMWarning("expected incoming " + std::string(name) +
            " link to be an object-object connection, ignoring",
            &element
            );
        return nullptr;
    }
    else if (!is_object_property_conn && con.PropertyName().length()) {
        DOMWarning("expected incoming " + std::string(name) +
            " link to be an object-property connection, ignoring",
            &element
            );
        return nullptr;
    }

    if(is_object_property_conn && propNameOut) {
        // note: this is ok, the return value of PropertyValue() is guaranteed to
        // remain valid and unchanged as long as the document exists.
        *propNameOut = con.PropertyName().c_str();
    }

    const Object* const ob = con.SourceObject();
    if(!ob) {
        DOMWarning("failed to read source object for incoming " + std::string(name) +
            " link, ignoring",
            &element);
        return nullptr;
    }

    return dynamic_cast<const T*>(ob);
}

} //!Util
} //!FBX
} //!Assimp

#endif
