/*************************************************************************/
/*  editor_resource_preview.h                                            */
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

#ifndef EDITORRESOURCEPREVIEW_H
#define EDITORRESOURCEPREVIEW_H

#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/safe_refcount.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"

class EditorResourcePreviewGenerator : public RefCounted {
	GDCLASS(EditorResourcePreviewGenerator, RefCounted);

protected:
	static void _bind_methods();

	GDVIRTUAL1RC(bool, _handles, String)
	GDVIRTUAL2RC(Ref<Texture2D>, _generate, RES, Vector2i)
	GDVIRTUAL2RC(Ref<Texture2D>, _generate_from_path, String, Vector2i)
	GDVIRTUAL0RC(bool, _generate_small_preview_automatically)
	GDVIRTUAL0RC(bool, _can_generate_small_preview)

public:
	virtual bool handles(const String &p_type) const;
	virtual Ref<Texture2D> generate(const RES &p_from, const Size2 &p_size) const;
	virtual Ref<Texture2D> generate_from_path(const String &p_path, const Size2 &p_size) const;

	virtual bool generate_small_preview_automatically() const;
	virtual bool can_generate_small_preview() const;

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

	Mutex preview_mutex;
	Semaphore preview_sem;
	Thread thread;
	SafeFlag exit;
	SafeFlag exited;

	// when running from GLES, we want to run the previews
	// in the main thread using an update, rather than create
	// a separate thread
	bool _mainthread_only = false;

	struct Item {
		Ref<Texture2D> preview;
		Ref<Texture2D> small_preview;
		int order = 0;
		uint32_t last_hash = 0;
		uint64_t modified_time = 0;
	};

	int order;

	Map<String, Item> cache;

	void _preview_ready(const String &p_str, const Ref<Texture2D> &p_texture, const Ref<Texture2D> &p_small_texture, ObjectID id, const StringName &p_func, const Variant &p_ud);
	void _generate_preview(Ref<ImageTexture> &r_texture, Ref<ImageTexture> &r_small_texture, const QueueItem &p_item, const String &cache_base);

	static void _thread_func(void *ud);
	void _thread();
	void _iterate();

	Vector<Ref<EditorResourcePreviewGenerator>> preview_generators;

protected:
	static void _bind_methods();

public:
	static EditorResourcePreview *get_singleton();

	// p_receiver_func callback has signature (String p_path, Ref<Texture2D> p_preview, Ref<Texture2D> p_preview_small, Variant p_userdata)
	// p_preview will be null if there was an error
	void queue_resource_preview(const String &p_path, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata);
	void queue_edited_resource_preview(const Ref<Resource> &p_res, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata);

	void add_preview_generator(const Ref<EditorResourcePreviewGenerator> &p_generator);
	void remove_preview_generator(const Ref<EditorResourcePreviewGenerator> &p_generator);
	void check_for_invalidation(const String &p_path);

	void start();
	void stop();

	// for single threaded mode
	void update();

	EditorResourcePreview();
	~EditorResourcePreview();
};

#endif // EDITORRESOURCEPREVIEW_H
