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

/** @file  FBXDocumentUtil.cpp
 *  @brief Implementation of the FBX DOM utility functions declared in FBXDocumentUtil.h
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include "FBXParser.h"
#include "FBXDocument.h"
#include "FBXUtil.h"
#include "FBXDocumentUtil.h"
#include "FBXProperties.h"


namespace Assimp {
namespace FBX {
namespace Util {

// ------------------------------------------------------------------------------------------------
// signal DOM construction error, this is always unrecoverable. Throws DeadlyImportError.
void DOMError(const std::string& message, const Token& token)
{
    throw DeadlyImportError(Util::AddTokenText("FBX-DOM",message,&token));
}

// ------------------------------------------------------------------------------------------------
void DOMError(const std::string& message, const Element* element /*= NULL*/)
{
    if(element) {
        DOMError(message,element->KeyToken());
    }
    throw DeadlyImportError("FBX-DOM " + message);
}


// ------------------------------------------------------------------------------------------------
// print warning, do return
void DOMWarning(const std::string& message, const Token& token)
{
    if(DefaultLogger::get()) {
        ASSIMP_LOG_WARN(Util::AddTokenText("FBX-DOM",message,&token));
    }
}

// ------------------------------------------------------------------------------------------------
void DOMWarning(const std::string& message, const Element* element /*= NULL*/)
{
    if(element) {
        DOMWarning(message,element->KeyToken());
        return;
    }
    if(DefaultLogger::get()) {
        ASSIMP_LOG_WARN("FBX-DOM: " + message);
    }
}


// ------------------------------------------------------------------------------------------------
// fetch a property table and the corresponding property template
std::shared_ptr<const PropertyTable> GetPropertyTable(const Document& doc,
    const std::string& templateName,
    const Element &element,
    const Scope& sc,
    bool no_warn /*= false*/)
{
    const Element* const Properties70 = sc["Properties70"];
    std::shared_ptr<const PropertyTable> templateProps = std::shared_ptr<const PropertyTable>(
        static_cast<const PropertyTable*>(NULL));

    if(templateName.length()) {
        PropertyTemplateMap::const_iterator it = doc.Templates().find(templateName);
        if(it != doc.Templates().end()) {
            templateProps = (*it).second;
        }
    }

    if(!Properties70 || !Properties70->Compound()) {
        if(!no_warn) {
            DOMWarning("property table (Properties70) not found",&element);
        }
        if(templateProps) {
            return templateProps;
        }
        else {
            return std::make_shared<const PropertyTable>();
        }
    }
    return std::make_shared<const PropertyTable>(*Properties70,templateProps);
}
} // !Util
} // !FBX
} // !Assimp

#endif
