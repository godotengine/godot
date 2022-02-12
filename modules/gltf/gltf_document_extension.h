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

#include "core/io/resource.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
class GLTFDocument;
class GLTFDocumentExtension : public Resource {
	GDCLASS(GLTFDocumentExtension, Resource);

	Dictionary import_settings;
	Dictionary export_settings;

protected:
	static void _bind_methods();

public:
	virtual Array get_import_setting_keys() const;
	virtual Variant get_import_setting(const StringName &p_key) const;
	virtual void set_import_setting(const StringName &p_key, Variant p_var);
	virtual Error import_preflight(Ref<GLTFDocument> p_document) { return OK; }
	virtual Error import_post(Ref<GLTFDocument> p_document, Node *p_node) { return OK; }

public:
	virtual Array get_export_setting_keys() const;
	virtual Variant get_export_setting(const StringName &p_key) const;
	virtual void set_export_setting(const StringName &p_key, Variant p_var);
	virtual Error export_preflight(Ref<GLTFDocument> p_document, Node *p_node) { return OK; }
	virtual Error export_post(Ref<GLTFDocument> p_document) { return OK; }
};

#endif // GLTF_DOCUMENT_EXTENSION_H
