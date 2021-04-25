/*************************************************************************/
/*  FBXDocumentUtil.h                                                    */
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

Copyright (c) 2006-2012, assimp team
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
#ifndef FBX_DOCUMENT_UTIL_H
#define FBX_DOCUMENT_UTIL_H

#include "FBXDocument.h"
#include <memory>
#include <string>

struct Token;
struct Element;

namespace FBXDocParser {
namespace Util {

// Parser errors
void DOMError(const std::string &message);
void DOMError(const std::string &message, const Token *token);
void DOMError(const std::string &message, const Element *element);
void DOMError(const std::string &message, const std::shared_ptr<Element> element);
void DOMError(const std::string &message, const std::shared_ptr<Token> token);

// Parser warnings
void DOMWarning(const std::string &message);
void DOMWarning(const std::string &message, const Token *token);
void DOMWarning(const std::string &message, const Element *element);
void DOMWarning(const std::string &message, const std::shared_ptr<Token> token);
void DOMWarning(const std::string &message, const std::shared_ptr<Element> element);

// ------------------------------------------------------------------------------------------------
template <typename T>
const T *ProcessSimpleConnection(const Connection &con,
		bool is_object_property_conn,
		const char *name,
		const ElementPtr element,
		const char **propNameOut = nullptr) {
	if (is_object_property_conn && !con.PropertyName().length()) {
		DOMWarning("expected incoming " + std::string(name) +
						   " link to be an object-object connection, ignoring",
				element);
		return nullptr;
	} else if (!is_object_property_conn && con.PropertyName().length()) {
		DOMWarning("expected incoming " + std::string(name) +
						   " link to be an object-property connection, ignoring",
				element);
		return nullptr;
	}

	if (is_object_property_conn && propNameOut) {
		// note: this is ok, the return value of PropertyValue() is guaranteed to
		// remain valid and unchanged as long as the document exists.
		*propNameOut = con.PropertyName().c_str();
	}

	// Cast Object to AnimationPlayer for example using safe functions, which return nullptr etc
	Object *ob = con.SourceObject();
	ERR_FAIL_COND_V_MSG(!ob, nullptr, "Failed to load object from SourceObject ptr");
	return dynamic_cast<const T *>(ob);
}
} // namespace Util
} // namespace FBXDocParser

#endif // FBX_DOCUMENT_UTIL_H
