/**************************************************************************/
/*  FBXModel.cpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

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

#include "FBXDocument.h"
#include "FBXDocumentUtil.h"
#include "FBXMeshGeometry.h"
#include "FBXParser.h"

namespace FBXDocParser {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Model::Model(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Object(id, element, name), shading("Y") {
	const ScopePtr sc = GetRequiredScope(element);
	const ElementPtr Shading = sc->GetElement("Shading");
	const ElementPtr Culling = sc->GetElement("Culling");

	if (Shading) {
		shading = GetRequiredToken(Shading, 0)->StringContents();
	}

	if (Culling) {
		culling = ParseTokenAsString(GetRequiredToken(Culling, 0));
	}

	props = GetPropertyTable(doc, "Model.FbxNode", element, sc);
	ResolveLinks(element, doc);
}

// ------------------------------------------------------------------------------------------------
Model::~Model() {
	if (props != nullptr) {
		delete props;
		props = nullptr;
	}
}

ModelLimbNode::ModelLimbNode(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Model(id, element, doc, name){

		};

ModelLimbNode::~ModelLimbNode() {
}

// ------------------------------------------------------------------------------------------------
void Model::ResolveLinks(const ElementPtr element, const Document &doc) {
	const char *const arr[] = { "Geometry", "Material", "NodeAttribute" };

	// resolve material
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), arr, 3);

	materials.reserve(conns.size());
	geometry.reserve(conns.size());
	attributes.reserve(conns.size());
	for (const Connection *con : conns) {
		// material and geometry links should be Object-Object connections
		if (con->PropertyName().length()) {
			continue;
		}

		const Object *const ob = con->SourceObject();
		if (!ob) {
			//DOMWarning("failed to read source object for incoming Model link, ignoring",&element);
			continue;
		}

		const Material *const mat = dynamic_cast<const Material *>(ob);
		if (mat) {
			materials.push_back(mat);
			continue;
		}

		const Geometry *const geo = dynamic_cast<const Geometry *>(ob);
		if (geo) {
			geometry.push_back(geo);
			continue;
		}

		const NodeAttribute *const att = dynamic_cast<const NodeAttribute *>(ob);
		if (att) {
			attributes.push_back(att);
			continue;
		}

		DOMWarning("source object for model link is neither Material, NodeAttribute nor Geometry, ignoring", element);
		continue;
	}
}

// ------------------------------------------------------------------------------------------------
bool Model::IsNull() const {
	const std::vector<const NodeAttribute *> &attrs = GetAttributes();
	for (const NodeAttribute *att : attrs) {
		const Null *null_tag = dynamic_cast<const Null *>(att);
		if (null_tag) {
			return true;
		}
	}

	return false;
}

} // namespace FBXDocParser
