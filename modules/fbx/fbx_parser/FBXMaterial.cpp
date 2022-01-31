/*************************************************************************/
/*  FBXMaterial.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

/** @file  FBXMaterial.cpp
 *  @brief Assimp::FBX::Material and Assimp::FBX::Texture implementation
 */

#include "ByteSwapper.h"
#include "FBXDocument.h"
#include "FBXDocumentUtil.h"
#include "FBXImportSettings.h"
#include "FBXParser.h"
#include "FBXProperties.h"

#include "FBXUtil.h"
#include <algorithm> // std::transform

namespace FBXDocParser {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Material::Material(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Object(id, element, name) {
	const ScopePtr sc = GetRequiredScope(element);

	const ElementPtr ShadingModel = sc->GetElement("ShadingModel");
	const ElementPtr MultiLayer = sc->GetElement("MultiLayer");

	if (MultiLayer) {
		multilayer = !!ParseTokenAsInt(GetRequiredToken(MultiLayer, 0));
	}

	if (ShadingModel) {
		shading = ParseTokenAsString(GetRequiredToken(ShadingModel, 0));
	} else {
		DOMWarning("shading mode not specified, assuming phong", element);
		shading = "phong";
	}

	std::string templateName;

	if (shading == "phong") {
		templateName = "Material.Phong";
	} else if (shading == "lambert") {
		templateName = "Material.Lambert";
	} else if (shading == "unknown") {
		templateName = "Material.StingRay";
	} else {
		DOMWarning("shading mode not recognized: " + shading, element);
	}

	// resolve texture links
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID());
	for (const Connection *con : conns) {
		// texture link to properties, not objects
		if (!con->PropertyName().length()) {
			continue;
		}

		Object *ob = con->SourceObject();
		if (!ob) {
			DOMWarning("failed to read source object for texture link, ignoring", element);
			continue;
		}

		const Texture *tex = dynamic_cast<const Texture *>(ob);
		if (!tex) {
			LayeredTexture *layeredTexture = dynamic_cast<LayeredTexture *>(ob);

			if (!layeredTexture) {
				DOMWarning("source object for texture link is not a texture or layered texture, ignoring", element);
				continue;
			}

			const std::string &prop = con->PropertyName();
			if (layeredTextures.find(prop) != layeredTextures.end()) {
				DOMWarning("duplicate layered texture link: " + prop, element);
			}

			layeredTextures[prop] = layeredTexture;
			layeredTexture->fillTexture(doc);
		} else {
			const std::string &prop = con->PropertyName();
			if (textures.find(prop) != textures.end()) {
				DOMWarning("duplicate texture link: " + prop, element);
			}

			textures[prop] = tex;
		}
	}
}

// ------------------------------------------------------------------------------------------------
Material::~Material() {
}

// ------------------------------------------------------------------------------------------------
Texture::Texture(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Object(id, element, name), uvScaling(1.0f, 1.0f) {
	const ScopePtr sc = GetRequiredScope(element);

	const ElementPtr Type = sc->GetElement("Type");
	const ElementPtr FileName = sc->GetElement("FileName");
	const ElementPtr RelativeFilename = sc->GetElement("RelativeFilename");
	const ElementPtr ModelUVTranslation = sc->GetElement("ModelUVTranslation");
	const ElementPtr ModelUVScaling = sc->GetElement("ModelUVScaling");
	const ElementPtr Texture_Alpha_Source = sc->GetElement("Texture_Alpha_Source");
	const ElementPtr Cropping = sc->GetElement("Cropping");

	if (Type) {
		type = ParseTokenAsString(GetRequiredToken(Type, 0));
	}

	if (FileName) {
		fileName = ParseTokenAsString(GetRequiredToken(FileName, 0));
	}

	if (RelativeFilename) {
		relativeFileName = ParseTokenAsString(GetRequiredToken(RelativeFilename, 0));
	}

	if (ModelUVTranslation) {
		uvTrans = Vector2(ParseTokenAsFloat(GetRequiredToken(ModelUVTranslation, 0)),
				ParseTokenAsFloat(GetRequiredToken(ModelUVTranslation, 1)));
	}

	if (ModelUVScaling) {
		uvScaling = Vector2(ParseTokenAsFloat(GetRequiredToken(ModelUVScaling, 0)),
				ParseTokenAsFloat(GetRequiredToken(ModelUVScaling, 1)));
	}

	if (Cropping) {
		crop[0] = ParseTokenAsInt(GetRequiredToken(Cropping, 0));
		crop[1] = ParseTokenAsInt(GetRequiredToken(Cropping, 1));
		crop[2] = ParseTokenAsInt(GetRequiredToken(Cropping, 2));
		crop[3] = ParseTokenAsInt(GetRequiredToken(Cropping, 3));
	} else {
		// vc8 doesn't support the crop() syntax in initialization lists
		// (and vc9 WARNS about the new (i.e. compliant) behaviour).
		crop[0] = crop[1] = crop[2] = crop[3] = 0;
	}

	if (Texture_Alpha_Source) {
		alphaSource = ParseTokenAsString(GetRequiredToken(Texture_Alpha_Source, 0));
	}

	// 3DS Max and FBX SDK use "Scaling" and "Translation" instead of "ModelUVScaling" and "ModelUVTranslation". Use these properties if available.
	bool ok = true;
	const Vector3 &scaling = PropertyGet<Vector3>(this, "Scaling", ok);
	if (ok) {
		uvScaling.x = scaling.x;
		uvScaling.y = scaling.y;
	}

	const Vector3 &trans = PropertyGet<Vector3>(this, "Translation", ok);
	if (ok) {
		uvTrans.x = trans.x;
		uvTrans.y = trans.y;
	}

	// resolve video links
	if (doc.Settings().readTextures) {
		const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID());
		for (const Connection *con : conns) {
			const Object *const ob = con->SourceObject();
			if (!ob) {
				DOMWarning("failed to read source object for texture link, ignoring", element);
				continue;
			}

			const Video *const video = dynamic_cast<const Video *>(ob);
			if (video) {
				media = video;
			}
		}
	}
}

Texture::~Texture() {
}

LayeredTexture::LayeredTexture(uint64_t id, const ElementPtr element, const Document & /*doc*/, const std::string &name) :
		Object(id, element, name), blendMode(BlendMode_Modulate), alpha(1) {
	const ScopePtr sc = GetRequiredScope(element);

	ElementPtr BlendModes = sc->GetElement("BlendModes");
	ElementPtr Alphas = sc->GetElement("Alphas");

	if (BlendModes != nullptr) {
		blendMode = (BlendMode)ParseTokenAsInt(GetRequiredToken(BlendModes, 0));
	}
	if (Alphas != nullptr) {
		alpha = ParseTokenAsFloat(GetRequiredToken(Alphas, 0));
	}
}

LayeredTexture::~LayeredTexture() {
}

void LayeredTexture::fillTexture(const Document &doc) {
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID());
	for (size_t i = 0; i < conns.size(); ++i) {
		const Connection *con = conns.at(i);

		const Object *const ob = con->SourceObject();
		if (!ob) {
			DOMWarning("failed to read source object for texture link, ignoring", element);
			continue;
		}

		const Texture *const tex = dynamic_cast<const Texture *>(ob);

		textures.push_back(tex);
	}
}

// ------------------------------------------------------------------------------------------------
Video::Video(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name) :
		Object(id, element, name) {
	const ScopePtr sc = GetRequiredScope(element);

	const ElementPtr Type = sc->GetElement("Type");
	// File Version 7500 Crashes if this is not checked fully.
	// As of writing this comment 7700 exists, in August 2020
	ElementPtr FileName = nullptr;
	if (HasElement(sc, "Filename")) {
		FileName = (ElementPtr)sc->GetElement("Filename");
	} else if (HasElement(sc, "FileName")) {
		FileName = (ElementPtr)sc->GetElement("FileName");
	} else {
		print_error("file has invalid video material returning...");
		return;
	}
	const ElementPtr RelativeFilename = sc->GetElement("RelativeFilename");
	const ElementPtr Content = sc->GetElement("Content");

	if (Type) {
		type = ParseTokenAsString(GetRequiredToken(Type, 0));
	}

	if (FileName) {
		fileName = ParseTokenAsString(GetRequiredToken(FileName, 0));
	}

	if (RelativeFilename) {
		relativeFileName = ParseTokenAsString(GetRequiredToken(RelativeFilename, 0));
	}

	if (Content && !Content->Tokens().empty()) {
		//this field is omitted when the embedded texture is already loaded, let's ignore if it's not found
		try {
			const Token *token = GetRequiredToken(Content, 0);
			const char *data = token->begin();
			if (!token->IsBinary()) {
				if (*data != '"') {
					DOMError("embedded content is not surrounded by quotation marks", element);
				} else {
					size_t targetLength = 0;
					const size_t numTokens = Content->Tokens().size();
					// First time compute size (it could be large like 64Gb and it is good to allocate it once)
					for (uint32_t tokenIdx = 0; tokenIdx < numTokens; ++tokenIdx) {
						const Token *dataToken = GetRequiredToken(Content, tokenIdx);
						size_t tokenLength = dataToken->end() - dataToken->begin() - 2; // ignore double quotes
						const char *base64data = dataToken->begin() + 1;
						const size_t outLength = Util::ComputeDecodedSizeBase64(base64data, tokenLength);
						if (outLength == 0) {
							DOMError("Corrupted embedded content found", element);
						}
						targetLength += outLength;
					}
					if (targetLength == 0) {
						DOMError("Corrupted embedded content found", element);
					} else {
						content = new uint8_t[targetLength];
						contentLength = static_cast<uint64_t>(targetLength);
						size_t dst_offset = 0;
						for (uint32_t tokenIdx = 0; tokenIdx < numTokens; ++tokenIdx) {
							const Token *dataToken = GetRequiredToken(Content, tokenIdx);
							ERR_FAIL_COND(!dataToken);
							size_t tokenLength = dataToken->end() - dataToken->begin() - 2; // ignore double quotes
							const char *base64data = dataToken->begin() + 1;
							dst_offset += Util::DecodeBase64(base64data, tokenLength, content + dst_offset, targetLength - dst_offset);
						}
						if (targetLength != dst_offset) {
							delete[] content;
							contentLength = 0;
							DOMError("Corrupted embedded content found", element);
						}
					}
				}
			} else if (static_cast<size_t>(token->end() - data) < 5) {
				DOMError("binary data array is too short, need five (5) bytes for type signature and element count", element);
			} else if (*data != 'R') {
				DOMWarning("video content is not raw binary data, ignoring", element);
			} else {
				// read number of elements
				uint32_t len = 0;
				::memcpy(&len, data + 1, sizeof(len));
				AI_SWAP4(len);

				contentLength = len;

				content = new uint8_t[len];
				::memcpy(content, data + 5, len);
			}
		} catch (...) {
			//			//we don't need the content data for contents that has already been loaded
			//			ASSIMP_LOG_VERBOSE_DEBUG_F("Caught exception in FBXMaterial (likely because content was already loaded): ",
			//									   runtimeError.what());
		}
	}
}

Video::~Video() {
	if (content) {
		delete[] content;
	}
}
} // namespace FBXDocParser
