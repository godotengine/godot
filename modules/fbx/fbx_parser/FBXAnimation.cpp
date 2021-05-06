/*************************************************************************/
/*  FBXAnimation.cpp                                                     */
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

/** @file  FBXAnimation.cpp
 *  @brief Assimp::FBX::AnimationCurve, Assimp::FBX::AnimationCurveNode,
 *         Assimp::FBX::AnimationLayer, Assimp::FBX::AnimationStack
 */

#include "FBXCommon.h"
#include "FBXDocument.h"
#include "FBXDocumentUtil.h"
#include "FBXParser.h"

namespace FBXDocParser {

using namespace Util;

// ------------------------------------------------------------------------------------------------
AnimationCurve::AnimationCurve(uint64_t id, const ElementPtr element, const std::string &name, const Document & /*doc*/) :
		Object(id, element, name) {
	const ScopePtr sc = GetRequiredScope(element);
	const ElementPtr KeyTime = GetRequiredElement(sc, "KeyTime");
	const ElementPtr KeyValueFloat = GetRequiredElement(sc, "KeyValueFloat");

	// note preserved keys and values for legacy FBXConverter.cpp
	// we can remove this once the animation system is written
	// and clean up this code so we do not have copies everywhere.
	ParseVectorDataArray(keys, KeyTime);
	ParseVectorDataArray(values, KeyValueFloat);

	if (keys.size() != values.size()) {
		DOMError("the number of key times does not match the number of keyframe values", KeyTime);
	}

	// put the two lists into the map, underlying container is really just a dictionary
	// these will always match, if not an error will throw and the file will not import
	// this is useful because we then can report something and fix this later if it becomes an issue
	// at this point we do not need a different count of these elements so this makes the
	// most sense to do.
	for (size_t x = 0; x < keys.size(); x++) {
		keyvalues[keys[x]] = values[x];
	}

	const ElementPtr KeyAttrDataFloat = sc->GetElement("KeyAttrDataFloat");
	if (KeyAttrDataFloat) {
		ParseVectorDataArray(attributes, KeyAttrDataFloat);
	}

	const ElementPtr KeyAttrFlags = sc->GetElement("KeyAttrFlags");
	if (KeyAttrFlags) {
		ParseVectorDataArray(flags, KeyAttrFlags);
	}
}

// ------------------------------------------------------------------------------------------------
AnimationCurve::~AnimationCurve() {
	// empty
}

// ------------------------------------------------------------------------------------------------
AnimationCurveNode::AnimationCurveNode(uint64_t id, const ElementPtr element, const std::string &name,
		const Document &doc, const char *const *target_prop_whitelist /*= nullptr*/,
		size_t whitelist_size /*= 0*/) :
		Object(id, element, name), target(), doc(doc) {
	// find target node
	const char *whitelist[] = { "Model", "NodeAttribute", "Deformer" };
	const std::vector<const Connection *> &conns = doc.GetConnectionsBySourceSequenced(ID(), whitelist, 3);

	for (const Connection *con : conns) {
		// link should go for a property
		if (!con->PropertyName().length()) {
			continue;
		}

		Object *object = con->DestinationObject();

		if (!object) {
			DOMWarning("failed to read destination object for AnimationCurveNode->Model link, ignoring", element);
			continue;
		}

		target = object;
		prop = con->PropertyName();
		break;
	}
}

// ------------------------------------------------------------------------------------------------
AnimationCurveNode::~AnimationCurveNode() {
	curves.clear();
}

// ------------------------------------------------------------------------------------------------
const AnimationMap &AnimationCurveNode::Curves() const {
	/* Lazy loaded animation curves, will only load if required */
	if (curves.empty()) {
		// resolve attached animation curves
		const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "AnimationCurve");

		for (const Connection *con : conns) {
			// So the advantage of having this STL boilerplate is that it's dead simple once you get it.
			// The other advantage is casting is guaranteed to be safe and nullptr will be returned in the last step if it fails.
			Object *ob = con->SourceObject();
			AnimationCurve *anim_curve = dynamic_cast<AnimationCurve *>(ob);
			ERR_CONTINUE_MSG(!anim_curve, "Failed to convert animation curve from object");

			curves.insert(std::make_pair(con->PropertyName(), anim_curve));
		}
	}

	return curves;
}

// ------------------------------------------------------------------------------------------------
AnimationLayer::AnimationLayer(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc) :
		Object(id, element, name), doc(doc) {
}

// ------------------------------------------------------------------------------------------------
AnimationLayer::~AnimationLayer() {
	// empty
}

// ------------------------------------------------------------------------------------------------
const AnimationCurveNodeList AnimationLayer::Nodes(const char *const *target_prop_whitelist,
		size_t whitelist_size /*= 0*/) const {
	AnimationCurveNodeList nodes;

	// resolve attached animation nodes
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "AnimationCurveNode");
	nodes.reserve(conns.size());

	for (const Connection *con : conns) {
		// link should not go to a property
		if (con->PropertyName().length()) {
			continue;
		}

		Object *ob = con->SourceObject();

		if (!ob) {
			DOMWarning("failed to read source object for AnimationCurveNode->AnimationLayer link, ignoring", element);
			continue;
		}

		const AnimationCurveNode *anim = dynamic_cast<AnimationCurveNode *>(ob);
		if (!anim) {
			DOMWarning("source object for ->AnimationLayer link is not an AnimationCurveNode", element);
			continue;
		}

		if (target_prop_whitelist) {
			const char *s = anim->TargetProperty().c_str();
			bool ok = false;
			for (size_t i = 0; i < whitelist_size; ++i) {
				if (!strcmp(s, target_prop_whitelist[i])) {
					ok = true;
					break;
				}
			}
			if (!ok) {
				continue;
			}
		}
		nodes.push_back(anim);
	}

	return nodes;
}

// ------------------------------------------------------------------------------------------------
AnimationStack::AnimationStack(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc) :
		Object(id, element, name) {
	// resolve attached animation layers
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "AnimationLayer");
	layers.reserve(conns.size());

	for (const Connection *con : conns) {
		// link should not go to a property
		if (con->PropertyName().length()) {
			continue;
		}

		Object *ob = con->SourceObject();
		if (!ob) {
			DOMWarning("failed to read source object for AnimationLayer->AnimationStack link, ignoring", element);
			continue;
		}

		const AnimationLayer *anim = dynamic_cast<const AnimationLayer *>(ob);

		if (!anim) {
			DOMWarning("source object for ->AnimationStack link is not an AnimationLayer", element);
			continue;
		}

		layers.push_back(anim);
	}
}

// ------------------------------------------------------------------------------------------------
AnimationStack::~AnimationStack() {
}
} // namespace FBXDocParser
