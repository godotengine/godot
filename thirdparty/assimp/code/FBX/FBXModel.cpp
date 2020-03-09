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

/** @file  FBXModel.cpp
 *  @brief Assimp::FBX::Model implementation
 */

#ifndef ASSIMP_BUILD_NO_FBX_IMPORTER

#include "FBXParser.h"
#include "FBXMeshGeometry.h"
#include "FBXDocument.h"
#include "FBXImporter.h"
#include "FBXDocumentUtil.h"

namespace Assimp {
namespace FBX {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Model::Model(uint64_t id, const Element& element, const Document& doc, const std::string& name)
    : Object(id,element,name)
    , shading("Y")
{
    const Scope& sc = GetRequiredScope(element);
    const Element* const Shading = sc["Shading"];
    const Element* const Culling = sc["Culling"];

    if(Shading) {
        shading = GetRequiredToken(*Shading,0).StringContents();
    }

    if (Culling) {
        culling = ParseTokenAsString(GetRequiredToken(*Culling,0));
    }

    props = GetPropertyTable(doc,"Model.FbxNode",element,sc);
    ResolveLinks(element,doc);
}

// ------------------------------------------------------------------------------------------------
Model::~Model()
{

}

// ------------------------------------------------------------------------------------------------
void Model::ResolveLinks(const Element& element, const Document& doc)
{
    const char* const arr[] = {"Geometry","Material","NodeAttribute"};

    // resolve material
    const std::vector<const Connection*>& conns = doc.GetConnectionsByDestinationSequenced(ID(),arr, 3);

    materials.reserve(conns.size());
    geometry.reserve(conns.size());
    attributes.reserve(conns.size());
    for(const Connection* con : conns) {

        // material and geometry links should be Object-Object connections
        if (con->PropertyName().length()) {
            continue;
        }

        const Object* const ob = con->SourceObject();
        if(!ob) {
            DOMWarning("failed to read source object for incoming Model link, ignoring",&element);
            continue;
        }

        const Material* const mat = dynamic_cast<const Material*>(ob);
        if(mat) {
            materials.push_back(mat);
            continue;
        }

        const Geometry* const geo = dynamic_cast<const Geometry*>(ob);
        if(geo) {
            geometry.push_back(geo);
            continue;
        }

        const NodeAttribute* const att = dynamic_cast<const NodeAttribute*>(ob);
        if(att) {
            attributes.push_back(att);
            continue;
        }

        DOMWarning("source object for model link is neither Material, NodeAttribute nor Geometry, ignoring",&element);
        continue;
    }
}

// ------------------------------------------------------------------------------------------------
bool Model::IsNull() const
{
    const std::vector<const NodeAttribute*>& attrs = GetAttributes();
    for(const NodeAttribute* att : attrs) {

        const Null* null_tag = dynamic_cast<const Null*>(att);
        if(null_tag) {
            return true;
        }
    }

    return false;
}


} //!FBX
} //!Assimp

#endif
