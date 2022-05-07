/*************************************************************************/
/*  FBXParser.h                                                          */
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

/** @file  FBXParser.h
 *  @brief FBX parsing code
 */
#ifndef FBX_PARSER_H
#define FBX_PARSER_H

#include <stdint.h>
#include <map>
#include <memory>

#include "core/color.h"
#include "core/math/transform.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"

#include "FBXTokenizer.h"

namespace FBXDocParser {

class Scope;
class Parser;
class Element;

typedef Element *ElementPtr;
typedef Scope *ScopePtr;

typedef std::vector<ScopePtr> ScopeList;
typedef std::multimap<std::string, ElementPtr> ElementMap;
typedef std::pair<ElementMap::const_iterator, ElementMap::const_iterator> ElementCollection;

#define new_Scope new Scope
#define new_Element new Element

/** FBX data entity that consists of a key:value tuple.
 *
 *  Example:
 *  @verbatim
 *    AnimationCurve: 23, "AnimCurve::", "" {
 *        [..]
 *    }
 *  @endverbatim
 *
 *  As can be seen in this sample, elements can contain nested #Scope
 *  as their trailing member.  **/
class Element {
public:
	Element(TokenPtr key_token, Parser &parser);
	~Element();

	ScopePtr Compound() const {
		return compound;
	}

	TokenPtr KeyToken() const {
		return key_token;
	}

	const TokenList &Tokens() const {
		return tokens;
	}

private:
	TokenList tokens;
	ScopePtr compound = nullptr;
	std::vector<ScopePtr> compound_scope;
	TokenPtr key_token = nullptr;
};

/** FBX data entity that consists of a 'scope', a collection
 *  of not necessarily unique #Element instances.
 *
 *  Example:
 *  @verbatim
 *    GlobalSettings:  {
 *        Version: 1000
 *        Properties70:
 *        [...]
 *    }
 *  @endverbatim  */
class Scope {
public:
	Scope(Parser &parser, bool topLevel = false);
	~Scope();

	ElementPtr GetElement(const std::string &index) const {
		ElementMap::const_iterator it = elements.find(index);
		return it == elements.end() ? nullptr : (*it).second;
	}

	ElementPtr FindElementCaseInsensitive(const std::string &elementName) const {
		for (auto element = elements.begin(); element != elements.end(); ++element) {
			if (element->first.compare(elementName)) {
				return element->second;
			}
		}

		// nothing to reference / expired.
		return nullptr;
	}

	ElementCollection GetCollection(const std::string &index) const {
		return elements.equal_range(index);
	}

	const ElementMap &Elements() const {
		return elements;
	}

private:
	ElementMap elements;
};

/** FBX parsing class, takes a list of input tokens and generates a hierarchy
 *  of nested #Scope instances, representing the fbx DOM.*/
class Parser {
public:
	/** Parse given a token list. Does not take ownership of the tokens -
	 *  the objects must persist during the entire parser lifetime */
	Parser(const TokenList &tokens, bool is_binary);
	~Parser();

	ScopePtr GetRootScope() const {
		return root;
	}

	bool IsBinary() const {
		return is_binary;
	}

private:
	friend class Scope;
	friend class Element;

	TokenPtr AdvanceToNextToken();
	TokenPtr LastToken() const;
	TokenPtr CurrentToken() const;

private:
	ScopeList scopes;
	const TokenList &tokens;

	TokenPtr last = nullptr, current = nullptr;
	TokenList::const_iterator cursor;
	ScopePtr root = nullptr;

	const bool is_binary;
};

/* token parsing - this happens when building the DOM out of the parse-tree*/
uint64_t ParseTokenAsID(const TokenPtr t, const char *&err_out);
size_t ParseTokenAsDim(const TokenPtr t, const char *&err_out);
float ParseTokenAsFloat(const TokenPtr t, const char *&err_out);
int ParseTokenAsInt(const TokenPtr t, const char *&err_out);
int64_t ParseTokenAsInt64(const TokenPtr t, const char *&err_out);
std::string ParseTokenAsString(const TokenPtr t, const char *&err_out);

/* wrapper around ParseTokenAsXXX() with DOMError handling */
uint64_t ParseTokenAsID(const TokenPtr t);
size_t ParseTokenAsDim(const TokenPtr t);
float ParseTokenAsFloat(const TokenPtr t);
int ParseTokenAsInt(const TokenPtr t);
int64_t ParseTokenAsInt64(const TokenPtr t);
std::string ParseTokenAsString(const TokenPtr t);

/* read data arrays */
void ParseVectorDataArray(std::vector<Vector3> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<Color> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<Vector2> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<int> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<float> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<float> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<unsigned int> &out, const ElementPtr el);
void ParseVectorDataArray(std::vector<uint64_t> &out, const ElementPtr ep);
void ParseVectorDataArray(std::vector<int64_t> &out, const ElementPtr el);
bool HasElement(const ScopePtr sc, const std::string &index);

// extract a required element from a scope, abort if the element cannot be found
ElementPtr GetRequiredElement(const ScopePtr sc, const std::string &index, const ElementPtr element = nullptr);
ScopePtr GetRequiredScope(const ElementPtr el); // New in 2020. (less likely to destroy application)
ElementPtr GetOptionalElement(const ScopePtr sc, const std::string &index, const ElementPtr element = nullptr);
// extract required compound scope
ScopePtr GetRequiredScope(const ElementPtr el);
// get token at a particular index
TokenPtr GetRequiredToken(const ElementPtr el, unsigned int index);

// ------------------------------------------------------------------------------------------------
// read a 4x4 matrix from an array of 16 floats
Transform ReadMatrix(const ElementPtr element);

} // namespace FBXDocParser

#endif // FBX_PARSER_H
