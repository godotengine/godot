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

/** @file  FBXNoteAttribute.cpp
 *  @brief Assimp::FBX::NodeAttribute (and subclasses) implementation
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include "FBXParser.h"
#include "FBXDocument.h"
#include "FBXImporter.h"
#include "FBXDocumentUtil.h"
#include <iostream>

namespace Assimp {
    namespace FBX {

        using namespace Util;

// ------------------------------------------------------------------------------------------------
        FbxPose::FbxPose(uint64_t id, const Element& element, const Document& doc, const std::string& name)
                : Object(id,element,name)
        {
            const Scope& sc = GetRequiredScope(element);
            std::cout << "pose name: " << name << std::endl;
            for( auto element : sc.Elements())
            {
                std::cout << "pose element: " << element.first << std::endl;
            }
            const std::string& classname = ParseTokenAsString(GetRequiredToken(element,2));

        }


// ------------------------------------------------------------------------------------------------
        FbxPose::~FbxPose()
        {
            // empty
        }


    }
}

#endif
