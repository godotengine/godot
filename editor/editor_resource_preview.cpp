/**************************************************************************/
/*  editor_resource_preview.cpp                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "editor_resource_preview.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/message_queue.h"
#include "core/variant/variant_utility.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "scene/resources/image_texture.h"

bool EditorResourcePreviewGenerator::handles(const String &p_type) const {
	bool success = false;
	if (GDVIRTUAL_CALL(_handles, p_type, success)) {
		return success;
	}
	ERR_FAIL_V_MSG(false, "EditorResourcePreviewGenerator::_handles needs to be overridden.");
}

Ref<Texture2D> EditorResourcePreviewGenerator::generate(const Ref<Resource> &p_from, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Texture2D> preview;
	if (GDVIRTUAL_CALL(_generate, p_from, p_size, p_metadata, preview)) {
		return preview;
	}
	ERR_FAIL_V_MSG(Ref<Texture2D>(), "EditorResourcePreviewGenerator::_generate needs to be overridden.");
}

Ref<Texture2D> EditorResourcePreviewGenerator::generate_from_path(const String &p_path, const Size2 &p_size, Dictionary &p_metadata) const {
	Ref<Texture2D> preview;
	if (GDVIRTUAL_CALL(_generate_from_path, p_path, p_size, p_metadata, preview)) {
		return preview;
	}

	Ref<Resource> res = ResourceLoader::load(p_path);
	if (!res.is_valid()) {
		return res;
	}
	return generate(res, p_size, p_metadata);
}

bool EditorResourcePreviewGenerator::generate_small_preview_automatically() const {
	bool success = false;
	GDVIRTUAL_CALL(_generate_small_preview_automatically, success);
	return success;
}

bool EditorResourcePreviewGenerator::can_generate_small_preview() const {
	bool success = false;
	GDVIRTUAL_CALL(_can_generate_small_preview, success);
	return success;
}

void EditorResourcePreviewGenerator::_bind_methods() {
	GDVIRTUAL_BIND(_handles, "type");
	GDVIRTUAL_BIND(_generate, "resource", "size", "metadata");
	GDVIRTUAL_BIND(_generate_from_path, "path", "size", "metadata");
	GDVIRTUAL_BIND(_generate_small_preview_automatically);
	GDVIRTUAL_BIND(_can_generate_small_preview);
}

EditorResourcePreviewGenerator::EditorResourcePreviewGenerator() {
}

EditorResourcePreview *EditorResourcePreview::singleton = nullptr;

void EditorResourcePreview::_thread_func(void *ud) {
	EditorResourcePreview *erp = (EditorResourcePreview *)ud;
	erp->_thread();
}

void EditorResourcePreview::_preview_ready(const String &p_path, int p_hash, const Ref<Texture2D> &p_texture, const Ref<Texture2D> &p_small_texture, ObjectID id, const StringName &p_func, const Variant &p_ud, const Dictionary &p_metadata) {
	{
		MutexLock lock(preview_mutex);

		uint64_t modified_time = 0;

		if (!p_path.begins_with("ID:")) {
			modified_time = FileAccess::get_modified_time(p_path);
		}

		Item item;
		item.order = order++;
		item.preview = p_texture;
		item.small_preview = p_small_texture;
		item.last_hash = p_hash;
		item.modified_time = modified_time;
		item.preview_metadata = p_metadata;

		cache[p_path] = item;
	}

	MessageQueue::get_singleton()->push_call(id, p_func, p_path, p_texture, p_small_texture, p_ud);
}

void EditorResourcePreview::_generate_preview(Ref<ImageTexture> &r_texture, Ref<ImageTexture> &r_small_texture, const QueueItem &p_item, const String &cache_base, Dictionary &p_metadata) {
	String type;

	if (p_item.resource.is_valid()) {
		type = p_item.resource->get_class();
	} else {
		type = ResourceLoader::get_resource_type(p_item.path);
	}

	if (type.is_empty()) {
		r_texture = Ref<ImageTexture>();
		r_small_texture = Ref<ImageTexture>();
		return; //could not guess type
	}

	int thumbnail_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;

	r_texture = Ref<ImageTexture>();
	r_small_texture = Ref<ImageTexture>();

	for (int i = 0; i < preview_generators.size(); i++) {
		if (!preview_generators[i]->handles(type)) {
			continue;
		}

		Ref<Texture2D> generated;
		if (p_item.resource.is_valid()) {
			generated = preview_generators.write[i]->generate(p_item.resource, Vector2(thumbnail_size, thumbnail_size), p_metadata);
		} else {
			generated = preview_generators.write[i]->generate_from_path(p_item.path, Vector2(thumbnail_size, thumbnail_size), p_metadata);
		}
		r_texture = generated;

		if (preview_generators[i]->can_generate_small_preview()) {
			Ref<Texture2D> generated_small;
			Dictionary d;
			if (p_item.resource.is_valid()) {
				generated_small = preview_generators.write[i]->generate(p_item.resource, Vector2(small_thumbnail_size, small_thumbnail_size), d);
			} else {
				generated_small = preview_generators.write[i]->generate_from_path(p_item.path, Vector2(small_thumbnail_size, small_thumbnail_size), d);
			}
			r_small_texture = generated_small;
		}

		if (!r_small_texture.is_valid() && r_texture.is_valid() && preview_generators[i]->generate_small_preview_automatically()) {
			Ref<Image> small_image = r_texture->get_image();
			small_image = small_image->duplicate();
			small_image->resize(small_thumbnail_size, small_thumbnail_size, Image::INTERPOLATE_CUBIC);
			r_small_texture.instantiate();
			r_small_texture->set_image(small_image);
		}

		break;
	}

	if (!p_item.resource.is_valid()) {
		// Cache the preview in case it's a resource on disk.
		if (r_texture.is_valid()) {
			// Wow it generated a preview... save cache.
			bool has_small_texture = r_small_texture.is_valid();
			ResourceSaver::save(r_texture, cache_base + ".png");
			if (has_small_texture) {
				ResourceSaver::save(r_small_texture, cache_base + "_small.png");
			}
			Ref<FileAccess> f = FileAccess::open(cache_base + ".txt", FileAccess::WRITE);
			ERR_FAIL_COND_MSG(f.is_null(), "Cannot create file '" + cache_base + ".txt'. Check user write permissions.");
			_write_preview_cache(f, thumbnail_size, has_small_texture, FileAccess::get_modified_time(p_item.path), FileAccess::get_md5(p_item.path), p_metadata);
		}
	}
}

const Dictionary EditorResourcePreview::get_preview_metadata(const String &p_path) const {
	ERR_FAIL_COND_V(!cache.has(p_path), Dictionary());
	return cache[p_path].preview_metadata;
}

void EditorResourcePreview::_iterate() {
	preview_mutex.lock();

	if (queue.size() == 0) {
		preview_mutex.unlock();
		return;
	}

	QueueItem item = queue.front()->get();
	queue.pop_front();

	if (cache.has(item.path)) {
		Item cached_item = cache[item.path];
		// Already has it because someone loaded it, just let it know it's ready.
		_preview_ready(item.path, cached_item.last_hash, cached_item.preview, cached_item.small_preview, item.id, item.function, item.userdata, cached_item.preview_metadata);
		preview_mutex.unlock();
		return;
	}
	preview_mutex.unlock();

	Ref<ImageTexture> texture;
	Ref<ImageTexture> small_texture;

	int thumbnail_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;

	if (item.resource.is_valid()) {
		Dictionary preview_metadata;
		_generate_preview(texture, small_texture, item, String(), preview_metadata);
		_preview_ready(item.path, item.resource->hash_edited_version(), texture, small_texture, item.id, item.function, item.userdata, preview_metadata);
		return;
	}

	Dictionary preview_metadata;
	String temp_path = EditorPaths::get_singleton()->get_cache_dir();
	String cache_base = ProjectSettings::get_singleton()->globalize_path(item.path).md5_text();
	cache_base = temp_path.path_join("resthumb-" + cache_base);

	// Does not have it, try to load a cached thumbnail.

	String file = cache_base + ".txt";
	Ref<FileAccess> f = FileAccess::open(file, FileAccess::READ);
	if (f.is_null()) {
		// No cache found, generate.
		_generate_preview(texture, small_texture, item, cache_base, preview_metadata);
	} else {
		uint64_t modtime = FileAccess::get_modified_time(item.path);
		int tsize;
		bool has_small_texture;
		uint64_t last_modtime;
		String hash;
		_read_preview_cache(f, &tsize, &has_small_texture, &last_modtime, &hash, &preview_metadata);

		bool cache_valid = true;

		if (tsize != thumbnail_size) {
			cache_valid = false;
			f.unref();
		} else if (last_modtime != modtime) {
			String last_md5 = f->get_line();
			String md5 = FileAccess::get_md5(item.path);
			f.unref();

			if (last_md5 != md5) {
				cache_valid = false;
			} else {
				// Update modified time.

				Ref<FileAccess> f2 = FileAccess::open(file, FileAccess::WRITE);
				if (f2.is_null()) {
					// Not returning as this would leave the thread hanging and would require
					// some proper cleanup/disabling of resource preview generation.
					ERR_PRINT("Cannot create file '" + file + "'. Check user write permissions.");
				} else {
					_write_preview_cache(f2, thumbnail_size, has_small_texture, modtime, md5, preview_metadata);
				}
			}
		} else {
			f.unref();
		}

		if (cache_valid) {
			Ref<Image> img;
			img.instantiate();
			Ref<Image> small_img;
			small_img.instantiate();

			if (img->load(cache_base + ".png") != OK) {
				cache_valid = false;
			} else {
				texture.instantiate();
				texture->set_image(img);

				if (has_small_texture) {
					if (small_img->load(cache_base + "_small.png") != OK) {
						cache_valid = false;
					} else {
						small_texture.instantiate();
						small_texture->set_image(small_img);
					}
				}
			}
		}

		if (!cache_valid) {
			_generate_preview(texture, small_texture, item, cache_base, preview_metadata);
		}
	}
	_preview_ready(item.path, 0, texture, small_texture, item.id, item.function, item.userdata, preview_metadata);
}

void EditorResourcePreview::_write_preview_cache(Ref<FileAccess> p_file, int p_thumbnail_size, bool p_has_small_texture, uint64_t p_modified_time, String p_hash, const Dictionary &p_metadata) {
	p_file->store_line(itos(p_thumbnail_size));
	p_file->store_line(itos(p_has_small_texture));
	p_file->store_line(itos(p_modified_time));
	p_file->store_line(p_hash);
	p_file->store_line(VariantUtilityFunctions::var_to_str(p_metadata).replace("\n", " "));
}

void EditorResourcePreview::_read_preview_cache(Ref<FileAccess> p_file, int *r_thumbnail_size, bool *r_has_small_texture, uint64_t *r_modified_time, String *r_hash, Dictionary *r_metadata) {
	*r_thumbnail_size = p_file->get_line().to_int();
	*r_has_small_texture = p_file->get_line().to_int();
	*r_modified_time = p_file->get_line().to_int();
	*r_hash = p_file->get_line();
	*r_metadata = VariantUtilityFunctions::str_to_var(p_file->get_line());
}

void EditorResourcePreview::_thread() {
	exited.clear();
	while (!exiting.is_set()) {
		preview_sem.wait();
		_iterate();
	}
	exited.set();
}

void EditorResourcePreview::_update_thumbnail_sizes() {
	if (small_thumbnail_size == -1) {
		// Kind of a workaround to retrieve the default icon size.
		small_thumbnail_size = EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Object"), EditorStringName(EditorIcons))->get_width();
	}
}

void EditorResourcePreview::queue_edited_resource_preview(const Ref<Resource> &p_res, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata) {
	ERR_FAIL_NULL(p_receiver);
	ERR_FAIL_COND(!p_res.is_valid());
	_update_thumbnail_sizes();

	{
		MutexLock lock(preview_mutex);

		String path_id = "ID:" + itos(p_res->get_instance_id());

		if (cache.has(path_id) && cache[path_id].last_hash == p_res->hash_edited_version()) {
			cache[path_id].order = order++;
			p_receiver->call(p_receiver_func, path_id, cache[path_id].preview, cache[path_id].small_preview, p_userdata);
			return;
		}

		cache.erase(path_id); //erase if exists, since it will be regen

		QueueItem item;
		item.function = p_receiver_func;
		item.id = p_receiver->get_instance_id();
		item.resource = p_res;
		item.path = path_id;
		item.userdata = p_userdata;

		queue.push_back(item);
	}
	preview_sem.post();
}

void EditorResourcePreview::queue_resource_preview(const String &p_path, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata) {
	ERR_FAIL_NULL(p_receiver);
	_update_thumbnail_sizes();

	{
		MutexLock lock(preview_mutex);

		if (cache.has(p_path)) {
			cache[p_path].order = order++;
			p_receiver->call(p_receiver_func, p_path, cache[p_path].preview, cache[p_path].small_preview, p_userdata);
			return;
		}

		QueueItem item;
		item.function = p_receiver_func;
		item.id = p_receiver->get_instance_id();
		item.path = p_path;
		item.userdata = p_userdata;

		queue.push_back(item);
	}
	preview_sem.post();
}

void EditorResourcePreview::add_preview_generator(const Ref<EditorResourcePreviewGenerator> &p_generator) {
	preview_generators.push_back(p_generator);
}

void EditorResourcePreview::remove_preview_generator(const Ref<EditorResourcePreviewGenerator> &p_generator) {
	preview_generators.erase(p_generator);
}

EditorResourcePreview *EditorResourcePreview::get_singleton() {
	return singleton;
}

void EditorResourcePreview::_bind_methods() {
	ClassDB::bind_method(D_METHOD("queue_resource_preview", "path", "receiver", "receiver_func", "userdata"), &EditorResourcePreview::queue_resource_preview);
	ClassDB::bind_method(D_METHOD("queue_edited_resource_preview", "resource", "receiver", "receiver_func", "userdata"), &EditorResourcePreview::queue_edited_resource_preview);
	ClassDB::bind_method(D_METHOD("add_preview_generator", "generator"), &EditorResourcePreview::add_preview_generator);
	ClassDB::bind_method(D_METHOD("remove_preview_generator", "generator"), &EditorResourcePreview::remove_preview_generator);
	ClassDB::bind_method(D_METHOD("check_for_invalidation", "path"), &EditorResourcePreview::check_for_invalidation);

	ADD_SIGNAL(MethodInfo("preview_invalidated", PropertyInfo(Variant::STRING, "path")));
}

void EditorResourcePreview::check_for_invalidation(const String &p_path) {
	bool call_invalidated = false;
	{
		MutexLock lock(preview_mutex);

		if (cache.has(p_path)) {
			uint64_t modified_time = FileAccess::get_modified_time(p_path);
			if (modified_time != cache[p_path].modified_time) {
				cache.erase(p_path);
				call_invalidated = true;
			}
		}
	}

	if (call_invalidated) { //do outside mutex
		call_deferred(SNAME("emit_signal"), "preview_invalidated", p_path);
	}
}

void EditorResourcePreview::start() {
	if (DisplayServer::get_singleton()->get_name() != "headless") {
		ERR_FAIL_COND_MSG(thread.is_started(), "Thread already started.");
		thread.start(_thread_func, this);
	}
}

void EditorResourcePreview::stop() {
	if (thread.is_started()) {
		exiting.set();
		preview_sem.post();

		for (int i = 0; i < preview_generators.size(); i++) {
			preview_generators.write[i]->abort();
		}

		while (!exited.is_set()) {
			OS::get_singleton()->delay_usec(10000);
			RenderingServer::get_singleton()->sync(); //sync pending stuff, as thread may be blocked on rendering server
		}

		thread.wait_to_finish();
	}
}

EditorResourcePreview::EditorResourcePreview() {
	singleton = this;
	order = 0;
}

EditorResourcePreview::~EditorResourcePreview() {
	stop();
}
