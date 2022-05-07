/*************************************************************************/
/*  FBXDocumentUtil.cpp                                                  */
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

/** @file  FBXDocumentUtil.cpp
 *  @brief Implementation of the FBX DOM utility functions declared in FBXDocumentUtil.h
 */

#include "FBXDocumentUtil.h"
#include "FBXDocument.h"
#include "FBXParser.h"
#include "FBXProperties.h"
#include "FBXUtil.h"
#include "core/print_string.h"

namespace FBXDocParser {
namespace Util {

void DOMError(const std::string &message) {
	print_error("[FBX-DOM]" + String(message.c_str()));
}

void DOMError(const std::string &message, const Token *token) {
	print_error("[FBX-DOM]" + String(message.c_str()) + ";" + String(token->StringContents().c_str()));
}

void DOMError(const std::string &message, const std::shared_ptr<Token> token) {
	print_error("[FBX-DOM]" + String(message.c_str()) + ";" + String(token->StringContents().c_str()));
}

void DOMError(const std::string &message, const Element *element /*= NULL*/) {
	if (element) {
		DOMError(message, element->KeyToken());
	}
	print_error("[FBX-DOM] " + String(message.c_str()));
}

void DOMError(const std::string &message, const std::shared_ptr<Element> element /*= NULL*/) {
	if (element) {
		DOMError(message, element->KeyToken());
	}
	print_error("[FBX-DOM] " + String(message.c_str()));
}

void DOMWarning(const std::string &message) {
	print_verbose("[FBX-DOM] warning:" + String(message.c_str()));
}

void DOMWarning(const std::string &message, const Token *token) {
	print_verbose("[FBX-DOM] warning:" + String(message.c_str()) + ";" + String(token->StringContents().c_str()));
}

void DOMWarning(const std::string &message, const Element *element /*= NULL*/) {
	if (element) {
		DOMWarning(message, element->KeyToken());
		return;
	}
	print_verbose("[FBX-DOM] warning:" + String(message.c_str()));
}

void DOMWarning(const std::string &message, const std::shared_ptr<Token> token) {
	print_verbose("[FBX-DOM] warning:" + String(message.c_str()) + ";" + String(token->StringContents().c_str()));
}

void DOMWarning(const std::string &message, const std::shared_ptr<Element> element /*= NULL*/) {
	if (element) {
		DOMWarning(message, element->KeyToken());
		return;
	}
	print_verbose("[FBX-DOM] warning:" + String(message.c_str()));
}

// ------------------------------------------------------------------------------------------------
// fetch a property table and the corresponding property template
const PropertyTable *GetPropertyTable(const Document &doc,
		const std::string &templateName,
		const ElementPtr element,
		const ScopePtr sc,
		bool no_warn /*= false*/) {
	// todo: make this an abstraction
	const ElementPtr Properties70 = sc->GetElement("Properties70");
	const PropertyTable *templateProps = static_cast<const PropertyTable *>(nullptr);

	if (templateName.length()) {
		PropertyTemplateMap::const_iterator it = doc.Templates().find(templateName);
		if (it != doc.Templates().end()) {
			templateProps = (*it).second;
		}
	}

	if (!Properties70 || !Properties70->Compound()) {
		if (!no_warn) {
			DOMWarning("property table (Properties70) not found", element);
		}
		if (templateProps) {
			return new const PropertyTable(templateProps);
		} else {
			return new const PropertyTable();
		}
	}

	return new PropertyTable(Properties70, templateProps);
}
} // namespace Util
} // namespace FBXDocParser
