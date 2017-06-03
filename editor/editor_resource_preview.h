/*************************************************************************/
/*  editor_resource_preview.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef EDITORRESOURCEPREVIEW_H
#define EDITORRESOURCEPREVIEW_H

#include "os/semaphore.h"
#include "os/thread.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"

/* make previews for:
*packdscene
*wav
*image
*mesh
-font
*script
*material
-shader
-shader graph?
-navigation mesh
-collision?
-occluder polygon
-navigation polygon
-tileset
-curve and curve2D
*/

class EditorResourcePreviewGenerator : public Reference {

	GDCLASS(EditorResourcePreviewGenerator, Reference);

protected:
	static void _bind_methods();

public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture> generate(const RES &p_from);
	virtual Ref<Texture> generate_from_path(const String &p_path);

	EditorResourcePreviewGenerator();
};

class EditorResourcePreview : public Node {

	GDCLASS(EditorResourcePreview, Node);

	static EditorResourcePreview *singleton;

	struct QueueItem {
		Ref<Resource> resource;
		String path;
		ObjectID id;
		StringName function;
		Variant userdata;
	};

	List<QueueItem> queue;

	Mutex *preview_mutex;
	Semaphore *preview_sem;
	Thread *thread;
	bool exit;

	struct Item {
		Ref<Texture> preview;
		int order;
		uint32_t last_hash;
		uint64_t modified_time;
	};

	int order;

	Map<String, Item> cache;

	void _preview_ready(const String &p_str, const Ref<Texture> &p_texture, ObjectID id, const StringName &p_func, const Variant &p_ud);
	Ref<Texture> _generate_preview(const QueueItem &p_item, const String &cache_base);

	static void _thread_func(void *ud);
	void _thread();

	Vector<Ref<EditorResourcePreviewGenerator> > preview_generators;

protected:
	static void _bind_methods();

public:
	static EditorResourcePreview *get_singleton();

	//callback function is callback(String p_path,Ref<Texture> preview,Variant udata) preview null if could not load
	void queue_resource_preview(const String &p_res, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata);
	void queue_edited_resource_preview(const Ref<Resource> &p_path, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata);

	void add_preview_generator(const Ref<EditorResourcePreviewGenerator> &p_generator);
	void remove_preview_generator(const Ref<EditorResourcePreviewGenerator> &p_generator);
	void check_for_invalidation(const String &p_path);

	EditorResourcePreview();
	~EditorResourcePreview();
};

#endif // EDITORRESOURCEPREVIEW_H
