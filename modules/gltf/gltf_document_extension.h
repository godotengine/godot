/*************************************************************************/
/*  gltf_document_extension.h                                            */
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

#ifndef GLTF_DOCUMENT_EXTENSION_H
#define GLTF_DOCUMENT_EXTENSION_H

#include "gltf_accessor.h"
#include "gltf_node.h"
#include "gltf_state.h"

class GLTFDocumentExtension : public Resource {
	GDCLASS(GLTFDocumentExtension, Resource);

protected:
	static void _bind_methods();

public:
	virtual Error import_preflight(Ref<GLTFState> p_state);
	virtual Error import_post_parse(Ref<GLTFState> p_state);
	virtual Error export_post(Ref<GLTFState> p_state);
	virtual Error import_post(Ref<GLTFState> p_state, Node *p_node);
	virtual Error export_preflight(Node *p_state);
	virtual Error import_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_json, Node *p_node);
	virtual Error export_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_json, Node *p_node);
	GDVIRTUAL1R(int, _import_preflight, Ref<GLTFState>);
	GDVIRTUAL1R(int, _import_post_parse, Ref<GLTFState>);
	GDVIRTUAL4R(int, _import_node, Ref<GLTFState>, Ref<GLTFNode>, Dictionary, Node *);
	GDVIRTUAL2R(int, _import_post, Ref<GLTFState>, Node *);
	GDVIRTUAL1R(int, _export_preflight, Node *);
	GDVIRTUAL4R(int, _export_node, Ref<GLTFState>, Ref<GLTFNode>, Dictionary, Node *);
	GDVIRTUAL1R(int, _export_post, Ref<GLTFState>);
};

#endif // GLTF_DOCUMENT_EXTENSION_H
