/*************************************************************************/
/*  editor_preview_plugins.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITORPREVIEWPLUGINS_H
#define EDITORPREVIEWPLUGINS_H

#include "editor/editor_resource_preview.h"

class EditorTexturePreviewPlugin : public EditorResourcePreviewGenerator {
	GDCLASS(EditorTexturePreviewPlugin, EditorResourcePreviewGenerator)
public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);

	EditorTexturePreviewPlugin();
};

class EditorBitmapPreviewPlugin : public EditorResourcePreviewGenerator {
	GDCLASS(EditorBitmapPreviewPlugin, EditorResourcePreviewGenerator)
public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);

	EditorBitmapPreviewPlugin();
};

class EditorPackedScenePreviewPlugin : public EditorResourcePreviewGenerator {

public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);
	virtual Ref<Texture> generate_from_path(const String &p_path);

	EditorPackedScenePreviewPlugin();
};

class EditorMaterialPreviewPlugin : public EditorResourcePreviewGenerator {

	GDCLASS(EditorMaterialPreviewPlugin, EditorResourcePreviewGenerator)

	RID scenario;
	RID sphere;
	RID sphere_instance;
	RID viewport;
	RID viewport_texture;
	RID light;
	RID light_instance;
	RID light2;
	RID light_instance2;
	RID camera;
	volatile bool preview_done;

	void _preview_done(const Variant &p_udata);

protected:
	static void _bind_methods();

public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);

	EditorMaterialPreviewPlugin();
	~EditorMaterialPreviewPlugin();
};

class EditorScriptPreviewPlugin : public EditorResourcePreviewGenerator {
public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);

	EditorScriptPreviewPlugin();
};

// FIXME: Needs to be rewritten for AudioStream in Godot 3.0+
#if 0
class EditorSamplePreviewPlugin : public EditorResourcePreviewGenerator {
public:

	virtual bool handles(const String& p_type) const;
	virtual Ref<Texture> generate(const RES& p_from);

	EditorSamplePreviewPlugin();
};
#endif

class EditorMeshPreviewPlugin : public EditorResourcePreviewGenerator {

	GDCLASS(EditorMeshPreviewPlugin, EditorResourcePreviewGenerator)

	RID scenario;
	RID mesh_instance;
	RID viewport;
	RID viewport_texture;
	RID light;
	RID light_instance;
	RID light2;
	RID light_instance2;
	RID camera;
	volatile bool preview_done;

	void _preview_done(const Variant &p_udata);

protected:
	static void _bind_methods();

public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);

	EditorMeshPreviewPlugin();
	~EditorMeshPreviewPlugin();
};

#endif // EDITORPREVIEWPLUGINS_H
