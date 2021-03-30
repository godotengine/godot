/*************************************************************************/
/*  FBXDeformer.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

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

#include "FBXDocument.h"
#include "FBXDocumentUtil.h"
#include "FBXMeshGeometry.h"
#include "FBXParser.h"
#include "core/math/math_funcs.h"
#include "core/math/transform.h"

#include <iostream>

namespace FBXDocParser {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Deformer::Deformer(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Object(id, element, name) {
}

// ------------------------------------------------------------------------------------------------
Deformer::~Deformer() {
}

Constraint::Constraint(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Object(id, element, name) {
}

Constraint::~Constraint() {
}

// ------------------------------------------------------------------------------------------------
Cluster::Cluster(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Deformer(id, element, doc, name), valid_transformAssociateModel(false) {
	const ScopePtr sc = GetRequiredScope(element);
	//    for( auto element : sc.Elements())
	//    {
	//        std::cout << "cluster element: " << element.first << std::endl;
	//    }
	//
	//    element: Indexes
	//    element: Transform
	//    element: TransformAssociateModel
	//    element: TransformLink
	//    element: UserData
	//    element: Version
	//    element: Weights

	const ElementPtr Indexes = sc->GetElement("Indexes");
	const ElementPtr Weights = sc->GetElement("Weights");

	const ElementPtr TransformAssociateModel = sc->GetElement("TransformAssociateModel");
	if (TransformAssociateModel != nullptr) {
		//Transform t = ReadMatrix(*TransformAssociateModel);
		link_mode = SkinLinkMode_Additive;
		valid_transformAssociateModel = true;
	} else {
		link_mode = SkinLinkMode_Normalized;
		valid_transformAssociateModel = false;
	}

	const ElementPtr Transform = GetRequiredElement(sc, "Transform", element);
	const ElementPtr TransformLink = GetRequiredElement(sc, "TransformLink", element);

	// todo: check if we need this
	//const Element& TransformAssociateModel = GetRequiredElement(sc, "TransformAssociateModel", &element);

	transform = ReadMatrix(Transform);
	transformLink = ReadMatrix(TransformLink);

	// it is actually possible that there be Deformer's with no weights
	if (!!Indexes != !!Weights) {
		DOMError("either Indexes or Weights are missing from Cluster", element);
	}

	if (Indexes) {
		ParseVectorDataArray(indices, Indexes);
		ParseVectorDataArray(weights, Weights);
	}

	if (indices.size() != weights.size()) {
		DOMError("sizes of index and weight array don't match up", element);
	}

	// read assigned node
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "Model");
	for (const Connection *con : conns) {
		const Model *mod = ProcessSimpleConnection<Model>(*con, false, "Model -> Cluster", element);
		if (mod) {
			node = mod;
			break;
		}
	}

	if (!node) {
		DOMError("failed to read target Node for Cluster", element);
		node = nullptr;
	}
}

// ------------------------------------------------------------------------------------------------
Cluster::~Cluster() {
}

// ------------------------------------------------------------------------------------------------
Skin::Skin(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Deformer(id, element, doc, name), accuracy(0.0f) {
	const ScopePtr sc = GetRequiredScope(element);

	// keep this it is used for debugging and any FBX format changes
	// for (auto element : sc.Elements()) {
	// 	std::cout << "skin element: " << element.first << std::endl;
	// }

	const ElementPtr Link_DeformAcuracy = sc->GetElement("Link_DeformAcuracy");
	if (Link_DeformAcuracy) {
		accuracy = ParseTokenAsFloat(GetRequiredToken(Link_DeformAcuracy, 0));
	}

	const ElementPtr SkinType = sc->GetElement("SkinningType");

	if (SkinType) {
		std::string skin_type = ParseTokenAsString(GetRequiredToken(SkinType, 0));

		if (skin_type == "Linear") {
			skinType = Skin_Linear;
		} else if (skin_type == "Rigid") {
			skinType = Skin_Rigid;
		} else if (skin_type == "DualQuaternion") {
			skinType = Skin_DualQuaternion;
		} else if (skin_type == "Blend") {
			skinType = Skin_Blend;
		} else {
			print_error("[doc:skin] could not find valid skin type: " + String(skin_type.c_str()));
		}
	}

	// resolve assigned clusters
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "Deformer");

	//

	clusters.reserve(conns.size());
	for (const Connection *con : conns) {
		const Cluster *cluster = ProcessSimpleConnection<Cluster>(*con, false, "Cluster -> Skin", element);
		if (cluster) {
			clusters.push_back(cluster);
			continue;
		}
	}
}

// ------------------------------------------------------------------------------------------------
Skin::~Skin() {
}
// ------------------------------------------------------------------------------------------------
BlendShape::BlendShape(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Deformer(id, element, doc, name) {
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "Deformer");
	blendShapeChannels.reserve(conns.size());
	for (const Connection *con : conns) {
		const BlendShapeChannel *bspc = ProcessSimpleConnection<BlendShapeChannel>(*con, false, "BlendShapeChannel -> BlendShape", element);
		if (bspc) {
			blendShapeChannels.push_back(bspc);
			continue;
		}
	}
}
// ------------------------------------------------------------------------------------------------
BlendShape::~BlendShape() {
}
// ------------------------------------------------------------------------------------------------
BlendShapeChannel::BlendShapeChannel(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Deformer(id, element, doc, name) {
	const ScopePtr sc = GetRequiredScope(element);
	const ElementPtr DeformPercent = sc->GetElement("DeformPercent");
	if (DeformPercent) {
		percent = ParseTokenAsFloat(GetRequiredToken(DeformPercent, 0));
	}
	const ElementPtr FullWeights = sc->GetElement("FullWeights");
	if (FullWeights) {
		ParseVectorDataArray(fullWeights, FullWeights);
	}
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "Geometry");
	shapeGeometries.reserve(conns.size());
	for (const Connection *con : conns) {
		const ShapeGeometry *const sg = ProcessSimpleConnection<ShapeGeometry>(*con, false, "Shape -> BlendShapeChannel", element);
		if (sg) {
			shapeGeometries.push_back(sg);
			continue;
		}
	}
}
// ------------------------------------------------------------------------------------------------
BlendShapeChannel::~BlendShapeChannel() {
}
// ------------------------------------------------------------------------------------------------
} // namespace FBXDocParser
