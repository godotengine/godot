/*************************************************************************/
/*  FBXDocument.cpp                                                      */
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
  copyright notice, this list of conditions and the*
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

/** @file  FBXDocument.cpp
 *  @brief Implementation of the FBX DOM classes
 */

#include "FBXDocument.h"
#include "FBXDocumentUtil.h"
#include "FBXImportSettings.h"
#include "FBXMeshGeometry.h"
#include "FBXParser.h"
#include "FBXProperties.h"
#include "FBXUtil.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>

namespace FBXDocParser {

using namespace Util;

// ------------------------------------------------------------------------------------------------
LazyObject::LazyObject(uint64_t id, const ElementPtr element, const Document &doc) :
		doc(doc), element(element), id(id) {
	// empty
}

// ------------------------------------------------------------------------------------------------
LazyObject::~LazyObject() {
	object.reset();
}

ObjectPtr LazyObject::LoadObject() {
	if (IsBeingConstructed() || FailedToConstruct()) {
		return nullptr;
	}

	if (object) {
		return object.get();
	}

	TokenPtr key = element->KeyToken();
	ERR_FAIL_COND_V(!key, nullptr);
	const TokenList &tokens = element->Tokens();

	if (tokens.size() < 3) {
		//DOMError("expected at least 3 tokens: id, name and class tag",&element);
		return nullptr;
	}

	const char *err = nullptr;
	std::string name = ParseTokenAsString(tokens[1], err);
	if (err) {
		DOMError(err, element);
	}

	// small fix for binary reading: binary fbx files don't use
	// prefixes such as Model:: in front of their names. The
	// loading code expects this at many places, though!
	// so convert the binary representation (a 0x0001) to the
	// double colon notation.
	if (tokens[1]->IsBinary()) {
		for (size_t i = 0; i < name.length(); ++i) {
			if (name[i] == 0x0 && name[i + 1] == 0x1) {
				name = name.substr(i + 2) + "::" + name.substr(0, i);
			}
		}
	}

	const std::string classtag = ParseTokenAsString(tokens[2], err);
	if (err) {
		DOMError(err, element);
	}

	// prevent recursive calls
	flags |= BEING_CONSTRUCTED;

	// this needs to be relatively fast since it happens a lot,
	// so avoid constructing strings all the time.
	const char *obtype = key->begin();
	const size_t length = static_cast<size_t>(key->end() - key->begin());

	if (!strncmp(obtype, "Pose", length)) {
		object.reset(new FbxPose(id, element, doc, name));
	} else if (!strncmp(obtype, "Geometry", length)) {
		if (!strcmp(classtag.c_str(), "Mesh")) {
			object.reset(new MeshGeometry(id, element, name, doc));
		}
		if (!strcmp(classtag.c_str(), "Shape")) {
			object.reset(new ShapeGeometry(id, element, name, doc));
		}
		if (!strcmp(classtag.c_str(), "Line")) {
			object.reset(new LineGeometry(id, element, name, doc));
		}
	} else if (!strncmp(obtype, "NodeAttribute", length)) {
		if (!strcmp(classtag.c_str(), "Camera")) {
			object.reset(new Camera(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "CameraSwitcher")) {
			object.reset(new CameraSwitcher(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "Light")) {
			object.reset(new Light(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "Null")) {
			object.reset(new Null(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "LimbNode")) {
			// This is an older format for bones
			// this is what blender uses I believe
			object.reset(new LimbNode(id, element, doc, name));
		}
	} else if (!strncmp(obtype, "Constraint", length)) {
		object.reset(new Constraint(id, element, doc, name));
	} else if (!strncmp(obtype, "Deformer", length)) {
		if (!strcmp(classtag.c_str(), "Cluster")) {
			object.reset(new Cluster(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "Skin")) {
			object.reset(new Skin(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "BlendShape")) {
			object.reset(new BlendShape(id, element, doc, name));
		} else if (!strcmp(classtag.c_str(), "BlendShapeChannel")) {
			object.reset(new BlendShapeChannel(id, element, doc, name));
		}
	} else if (!strncmp(obtype, "Model", length)) {
		// Model is normal node

		// LimbNode model is a 'bone' node.
		if (!strcmp(classtag.c_str(), "LimbNode")) {
			object.reset(new ModelLimbNode(id, element, doc, name));

		} else if (strcmp(classtag.c_str(), "IKEffector") && strcmp(classtag.c_str(), "FKEffector")) {
			// FK and IK effectors are not supporte
			object.reset(new Model(id, element, doc, name));
		}
	} else if (!strncmp(obtype, "Material", length)) {
		object.reset(new Material(id, element, doc, name));
	} else if (!strncmp(obtype, "Texture", length)) {
		object.reset(new Texture(id, element, doc, name));
	} else if (!strncmp(obtype, "LayeredTexture", length)) {
		object.reset(new LayeredTexture(id, element, doc, name));
	} else if (!strncmp(obtype, "Video", length)) {
		object.reset(new Video(id, element, doc, name));
	} else if (!strncmp(obtype, "AnimationStack", length)) {
		object.reset(new AnimationStack(id, element, name, doc));
	} else if (!strncmp(obtype, "AnimationLayer", length)) {
		object.reset(new AnimationLayer(id, element, name, doc));
	} else if (!strncmp(obtype, "AnimationCurve", length)) {
		object.reset(new AnimationCurve(id, element, name, doc));
	} else if (!strncmp(obtype, "AnimationCurveNode", length)) {
		object.reset(new AnimationCurveNode(id, element, name, doc));
	} else {
		ERR_FAIL_V_MSG(nullptr, "FBX contains unsupported object: " + String(obtype));
	}

	flags &= ~BEING_CONSTRUCTED;

	return object.get();
}

// ------------------------------------------------------------------------------------------------
Object::Object(uint64_t id, const ElementPtr element, const std::string &name) :
		PropertyTable(element), element(element), name(name), id(id) {
}

// ------------------------------------------------------------------------------------------------
Object::~Object() {
	// empty
}

// ------------------------------------------------------------------------------------------------
FileGlobalSettings::FileGlobalSettings(const Document &doc) :
		PropertyTable(), doc(doc) {
	// empty
}

// ------------------------------------------------------------------------------------------------
FileGlobalSettings::~FileGlobalSettings() {
}

// ------------------------------------------------------------------------------------------------
Document::Document(const Parser &parser, const ImportSettings &settings) :
		settings(settings), parser(parser) {
	// Cannot use array default initialization syntax because vc8 fails on it
	for (unsigned int &timeStamp : creationTimeStamp) {
		timeStamp = 0;
	}

	// we must check if we can read the header version safely, if its outdated then drop it.
	if (ReadHeader()) {
		SafeToImport = true;
		ReadPropertyTemplates();

		ReadGlobalSettings();

		// This order is important, connections need parsed objects to check
		// whether connections are ok or not. Objects may not be evaluated yet,
		// though, since this may require valid connections.
		ReadObjects();
		ReadConnections();
	}
}

// ------------------------------------------------------------------------------------------------
Document::~Document() {
	for (PropertyTemplateMap::value_type v : templates) {
		delete v.second;
	}

	for (ObjectMap::value_type &v : objects) {
		delete v.second;
	}

	for (ConnectionMap::value_type &v : src_connections) {
		delete v.second;
	}

	// clear globals import pointer
	globals.reset();
}

// ------------------------------------------------------------------------------------------------
static const unsigned int LowerSupportedVersion = 7100;
static const unsigned int UpperSupportedVersion = 7700;

bool Document::ReadHeader() {
	// Read ID objects from "Objects" section
	ScopePtr sc = parser.GetRootScope();
	ElementPtr ehead = sc->GetElement("FBXHeaderExtension");
	if (!ehead || !ehead->Compound()) {
		DOMError("no FBXHeaderExtension dictionary found");
	}

	if (parser.IsCorrupt()) {
		DOMError("File is corrupt");
		return false;
	}

	const ScopePtr shead = ehead->Compound();
	fbxVersion = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(shead, "FBXVersion", ehead), 0));

	// While we may have some success with newer files, we don't support
	// the older 6.n fbx format
	if (fbxVersion < LowerSupportedVersion) {
		DOMWarning("unsupported, old format version, FBX 2015-2020, you must re-export in a more modern version of your original modelling application");
		return false;
	}
	if (fbxVersion > UpperSupportedVersion) {
		DOMWarning("unsupported, newer format version, supported are only FBX 2015, up to FBX 2020"
				   " trying to read it nevertheless");
	}

	const ElementPtr ecreator = shead->GetElement("Creator");
	if (ecreator) {
		creator = ParseTokenAsString(GetRequiredToken(ecreator, 0));
	}

	// Scene Info
	const ElementPtr scene_info = shead->GetElement("SceneInfo");

	if (scene_info) {
		metadata_properties.Setup(scene_info);
	}

	const ElementPtr etimestamp = shead->GetElement("CreationTimeStamp");
	if (etimestamp && etimestamp->Compound()) {
		const ScopePtr stimestamp = etimestamp->Compound();
		creationTimeStamp[0] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Year"), 0));
		creationTimeStamp[1] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Month"), 0));
		creationTimeStamp[2] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Day"), 0));
		creationTimeStamp[3] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Hour"), 0));
		creationTimeStamp[4] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Minute"), 0));
		creationTimeStamp[5] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Second"), 0));
		creationTimeStamp[6] = ParseTokenAsInt(GetRequiredToken(GetRequiredElement(stimestamp, "Millisecond"), 0));
	}

	return true;
}

// ------------------------------------------------------------------------------------------------
void Document::ReadGlobalSettings() {
	ERR_FAIL_COND_MSG(globals != nullptr, "Global settings is already setup this is a serious error and should be reported");

	globals = std::make_shared<FileGlobalSettings>(*this);
}

// ------------------------------------------------------------------------------------------------
void Document::ReadObjects() {
	// read ID objects from "Objects" section
	const ScopePtr sc = parser.GetRootScope();
	const ElementPtr eobjects = sc->GetElement("Objects");
	if (!eobjects || !eobjects->Compound()) {
		DOMError("no Objects dictionary found");
	}

	// add a dummy entry to represent the Model::RootNode object (id 0),
	// which is only indirectly defined in the input file
	objects[0] = new LazyObject(0L, eobjects, *this);

	const ScopePtr sobjects = eobjects->Compound();
	for (const ElementMap::value_type &iter : sobjects->Elements()) {
		// extract ID
		const TokenList &tok = iter.second->Tokens();

		if (tok.empty()) {
			DOMError("expected ID after object key", iter.second);
		}

		const char *err;
		const uint64_t id = ParseTokenAsID(tok[0], err);
		if (err) {
			DOMError(err, iter.second);
		}

		// id=0 is normally implicit
		if (id == 0L) {
			DOMError("encountered object with implicitly defined id 0", iter.second);
		}

		if (objects.find(id) != objects.end()) {
			DOMWarning("encountered duplicate object id, ignoring first occurrence", iter.second);
		}

		objects[id] = new LazyObject(id, iter.second, *this);

		// grab all animation stacks upfront since there is no listing of them
		if (!strcmp(iter.first.c_str(), "AnimationStack")) {
			animationStacks.push_back(id);
		} else if (!strcmp(iter.first.c_str(), "Constraint")) {
			constraints.push_back(id);
		} else if (!strcmp(iter.first.c_str(), "Pose")) {
			bind_poses.push_back(id);
		} else if (!strcmp(iter.first.c_str(), "Material")) {
			materials.push_back(id);
		} else if (!strcmp(iter.first.c_str(), "Deformer")) {
			TokenPtr key = iter.second->KeyToken();
			ERR_CONTINUE_MSG(!key, "[parser bug] invalid token key for deformer");
			const TokenList &tokens = iter.second->Tokens();
			const std::string class_tag = ParseTokenAsString(tokens[2], err);

			if (err) {
				DOMError(err, iter.second);
			}

			if (class_tag == "Skin") {
				//print_verbose("registered skin:" + itos(id));
				skins.push_back(id);
			}
		}
	}
}

// ------------------------------------------------------------------------------------------------
void Document::ReadPropertyTemplates() {
}

// ------------------------------------------------------------------------------------------------
void Document::ReadConnections() {
	const ScopePtr sc = parser.GetRootScope();

	// read property templates from "Definitions" section
	const ElementPtr econns = sc->GetElement("Connections");
	if (!econns || !econns->Compound()) {
		DOMError("no Connections dictionary found");
	}

	uint64_t insertionOrder = 0l;
	const ScopePtr sconns = econns->Compound();
	const ElementCollection conns = sconns->GetCollection("C");
	for (ElementMap::const_iterator it = conns.first; it != conns.second; ++it) {
		const ElementPtr el = (*it).second;
		const std::string &type = ParseTokenAsString(GetRequiredToken(el, 0));

		// PP = property-property connection, ignored for now
		// (tokens: "PP", ID1, "Property1", ID2, "Property2")
		if (type == "PP") {
			continue;
		}

		const uint64_t src = ParseTokenAsID(GetRequiredToken(el, 1));
		const uint64_t dest = ParseTokenAsID(GetRequiredToken(el, 2));

		// OO = object-object connection
		// OP = object-property connection, in which case the destination property follows the object ID
		const std::string &prop = (type == "OP" ? ParseTokenAsString(GetRequiredToken(el, 3)) : "");

		if (objects.find(src) == objects.end()) {
			DOMWarning("source object for connection does not exist", el);
			continue;
		}

		// dest may be 0 (root node) but we added a dummy object before
		if (objects.find(dest) == objects.end()) {
			DOMWarning("destination object for connection does not exist", el);
			continue;
		}

		// add new connection
		const Connection *const c = new Connection(insertionOrder++, src, dest, prop, *this);
		src_connections.insert(ConnectionMap::value_type(src, c));
		dest_connections.insert(ConnectionMap::value_type(dest, c));
	}
}

// ------------------------------------------------------------------------------------------------
const std::vector<const AnimationStack *> &Document::AnimationStacks() const {
	if (!animationStacksResolved.empty() || animationStacks.empty()) {
		return animationStacksResolved;
	}

	animationStacksResolved.reserve(animationStacks.size());
	for (uint64_t id : animationStacks) {
		LazyObject *lazy = GetObject(id);

		// Two things happen here:
		// We cast internally an Object PTR to an Animation Stack PTR
		// We return invalid weak_ptrs for objects which are invalid

		const AnimationStack *stack = lazy->Get<AnimationStack>();
		ERR_CONTINUE_MSG(!stack, "invalid ptr to AnimationStack - conversion failure");

		// We push back the weak reference :) to keep things simple, as ownership is on the parser side so it won't be cleaned up.
		animationStacksResolved.push_back(stack);
	}

	return animationStacksResolved;
}

// ------------------------------------------------------------------------------------------------
LazyObject *Document::GetObject(uint64_t id) const {
	ObjectMap::const_iterator it = objects.find(id);
	return it == objects.end() ? nullptr : (*it).second;
}

#define MAX_CLASSNAMES 6

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsSequenced(uint64_t id, const ConnectionMap &conns) const {
	std::vector<const Connection *> temp;

	const std::pair<ConnectionMap::const_iterator, ConnectionMap::const_iterator> range =
			conns.equal_range(id);

	temp.reserve(std::distance(range.first, range.second));
	for (ConnectionMap::const_iterator it = range.first; it != range.second; ++it) {
		temp.push_back((*it).second);
	}

	std::sort(temp.begin(), temp.end(), std::mem_fn(&Connection::Compare));

	return temp; // NRVO should handle this
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsSequenced(uint64_t id, bool is_src,
		const ConnectionMap &conns,
		const char *const *classnames,
		size_t count) const

{
	size_t lengths[MAX_CLASSNAMES];

	const size_t c = count;
	for (size_t i = 0; i < c; ++i) {
		lengths[i] = strlen(classnames[i]);
	}

	std::vector<const Connection *> temp;
	const std::pair<ConnectionMap::const_iterator, ConnectionMap::const_iterator> range =
			conns.equal_range(id);

	temp.reserve(std::distance(range.first, range.second));
	for (ConnectionMap::const_iterator it = range.first; it != range.second; ++it) {
		TokenPtr key = (is_src ? (*it).second->LazyDestinationObject() : (*it).second->LazySourceObject())->GetElement()->KeyToken();

		const char *obtype = key->begin();

		for (size_t i = 0; i < c; ++i) {
			//ai_assert(classnames[i]);
			if (static_cast<size_t>(std::distance(key->begin(), key->end())) == lengths[i] && !strncmp(classnames[i], obtype, lengths[i])) {
				obtype = nullptr;
				break;
			}
		}

		if (obtype) {
			continue;
		}

		temp.push_back((*it).second);
	}

	std::sort(temp.begin(), temp.end(), std::mem_fn(&Connection::Compare));
	return temp; // NRVO should handle this
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsBySourceSequenced(uint64_t source) const {
	return GetConnectionsSequenced(source, ConnectionsBySource());
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsBySourceSequenced(uint64_t src, const char *classname) const {
	const char *arr[] = { classname };
	return GetConnectionsBySourceSequenced(src, arr, 1);
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsBySourceSequenced(uint64_t source,
		const char *const *classnames, size_t count) const {
	return GetConnectionsSequenced(source, true, ConnectionsBySource(), classnames, count);
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsByDestinationSequenced(uint64_t dest,
		const char *classname) const {
	const char *arr[] = { classname };
	return GetConnectionsByDestinationSequenced(dest, arr, 1);
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsByDestinationSequenced(uint64_t dest) const {
	return GetConnectionsSequenced(dest, ConnectionsByDestination());
}

// ------------------------------------------------------------------------------------------------
std::vector<const Connection *> Document::GetConnectionsByDestinationSequenced(uint64_t dest,
		const char *const *classnames, size_t count) const {
	return GetConnectionsSequenced(dest, false, ConnectionsByDestination(), classnames, count);
}

// ------------------------------------------------------------------------------------------------
Connection::Connection(uint64_t insertionOrder, uint64_t src, uint64_t dest, const std::string &prop,
		const Document &doc) :
		insertionOrder(insertionOrder), prop(prop), src(src), dest(dest), doc(doc) {
}

// ------------------------------------------------------------------------------------------------
Connection::~Connection() {
	// empty
}

// ------------------------------------------------------------------------------------------------
LazyObject *Connection::LazySourceObject() const {
	LazyObject *const lazy = doc.GetObject(src);
	return lazy;
}

// ------------------------------------------------------------------------------------------------
LazyObject *Connection::LazyDestinationObject() const {
	LazyObject *const lazy = doc.GetObject(dest);
	return lazy;
}

// ------------------------------------------------------------------------------------------------
Object *Connection::SourceObject() const {
	LazyObject *lazy = doc.GetObject(src);
	//ai_assert(lazy);
	return lazy->LoadObject();
}

// ------------------------------------------------------------------------------------------------
Object *Connection::DestinationObject() const {
	LazyObject *lazy = doc.GetObject(dest);
	//ai_assert(lazy);
	return lazy->LoadObject();
}
} // namespace FBXDocParser
