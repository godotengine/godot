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

#include "core/method_bind_ext.gen.inc"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/message_queue.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"

bool EditorResourcePreviewGenerator::handles(const String &p_type) const {
	if (get_script_instance() && get_script_instance()->has_method("handles")) {
		return get_script_instance()->call("handles", p_type);
	}
	ERR_FAIL_V_MSG(false, "EditorResourcePreviewGenerator::handles needs to be overridden.");
}

Ref<Texture> EditorResourcePreviewGenerator::generate(const RES &p_from, const Size2 &p_size) const {
	if (get_script_instance() && get_script_instance()->has_method("generate")) {
		return get_script_instance()->call("generate", p_from, p_size);
	}
	ERR_FAIL_V_MSG(Ref<Texture>(), "EditorResourcePreviewGenerator::generate needs to be overridden.");
}

Ref<Texture> EditorResourcePreviewGenerator::generate_from_path(const String &p_path, const Size2 &p_size) const {
	if (get_script_instance() && get_script_instance()->has_method("generate_from_path")) {
		return get_script_instance()->call("generate_from_path", p_path, p_size);
	}

	RES res = ResourceLoader::load(p_path);
	if (!res.is_valid()) {
		return res;
	}
	return generate(res, p_size);
}

bool EditorResourcePreviewGenerator::generate_small_preview_automatically() const {
	if (get_script_instance() && get_script_instance()->has_method("generate_small_preview_automatically")) {
		return get_script_instance()->call("generate_small_preview_automatically");
	}

	return false;
}

bool EditorResourcePreviewGenerator::can_generate_small_preview() const {
	if (get_script_instance() && get_script_instance()->has_method("can_generate_small_preview")) {
		return get_script_instance()->call("can_generate_small_preview");
	}

	return false;
}

void EditorResourcePreviewGenerator::_bind_methods() {
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "handles", PropertyInfo(Variant::STRING, "type")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(CLASS_INFO(Texture), "generate", PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_RESOURCE_TYPE, "Resource"), PropertyInfo(Variant::VECTOR2, "size")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(CLASS_INFO(Texture), "generate_from_path", PropertyInfo(Variant::STRING, "path", PROPERTY_HINT_FILE), PropertyInfo(Variant::VECTOR2, "size")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "generate_small_preview_automatically"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "can_generate_small_preview"));
}

EditorResourcePreviewGenerator::EditorResourcePreviewGenerator() {
}

EditorResourcePreview *EditorResourcePreview::singleton = nullptr;

void EditorResourcePreview::_thread_func(void *ud) {
	EditorResourcePreview *erp = (EditorResourcePreview *)ud;
	erp->_thread();
}

void EditorResourcePreview::_preview_ready(const String &p_str, const Ref<Texture> &p_texture, const Ref<Texture> &p_small_texture, ObjectID id, const StringName &p_func, const Variant &p_ud) {
	preview_mutex.lock();

	String path = p_str;
	uint32_t hash = 0;
	uint64_t modified_time = 0;

	if (p_str.begins_with("ID:")) {
		hash = uint32_t(p_str.get_slicec(':', 2).to_int64());
		path = "ID:" + p_str.get_slicec(':', 1);
	} else {
		modified_time = FileAccess::get_modified_time(path);
	}

	Item item;
	item.order = order++;
	item.preview = p_texture;
	item.small_preview = p_small_texture;
	item.last_hash = hash;
	item.modified_time = modified_time;

	cache[path] = item;

	preview_mutex.unlock();

	MessageQueue::get_singleton()->push_call(id, p_func, path, p_texture, p_small_texture, p_ud);
}

void EditorResourcePreview::_generate_preview(Ref<ImageTexture> &r_texture, Ref<ImageTexture> &r_small_texture, const QueueItem &p_item, const String &cache_base) {
	String type;

	if (p_item.resource.is_valid()) {
		type = p_item.resource->get_class();
	} else {
		type = ResourceLoader::get_resource_type(p_item.path);
	}

	if (type == "") {
		r_texture = Ref<ImageTexture>();
		r_small_texture = Ref<ImageTexture>();
		return; //could not guess type
	}

	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;

	r_texture = Ref<ImageTexture>();
	r_small_texture = Ref<ImageTexture>();

	for (int i = 0; i < preview_generators.size(); i++) {
		if (!preview_generators[i]->handles(type)) {
			continue;
		}

		Ref<Texture> generated;
		if (p_item.resource.is_valid()) {
			generated = preview_generators[i]->generate(p_item.resource, Vector2(thumbnail_size, thumbnail_size));
		} else {
			generated = preview_generators[i]->generate_from_path(p_item.path, Vector2(thumbnail_size, thumbnail_size));
		}
		r_texture = generated;

		if (!EditorNode::get_singleton()->get_theme_base()) {
			return;
		}

		int small_thumbnail_size = EditorNode::get_singleton()->get_theme_base()->get_icon("Object", "EditorIcons")->get_width(); // Kind of a workaround to retrieve the default icon size

		if (preview_generators[i]->can_generate_small_preview()) {
			Ref<Texture> generated_small;
			if (p_item.resource.is_valid()) {
				generated_small = preview_generators[i]->generate(p_item.resource, Vector2(small_thumbnail_size, small_thumbnail_size));
			} else {
				generated_small = preview_generators[i]->generate_from_path(p_item.path, Vector2(small_thumbnail_size, small_thumbnail_size));
			}
			r_small_texture = generated_small;
		}

		if (!r_small_texture.is_valid() && r_texture.is_valid() && preview_generators[i]->generate_small_preview_automatically()) {
			Ref<Image> small_image = r_texture->get_data();
			small_image = small_image->duplicate();
			small_image->resize(small_thumbnail_size, small_thumbnail_size, Image::INTERPOLATE_CUBIC);
			r_small_texture.instance();
			r_small_texture->create_from_image(small_image);
		}

		break;
	}

	if (!p_item.resource.is_valid()) {
		// cache the preview in case it's a resource on disk
		if (r_texture.is_valid()) {
			//wow it generated a preview... save cache
			bool has_small_texture = r_small_texture.is_valid();
			ResourceSaver::save(cache_base + ".png", r_texture);
			if (has_small_texture) {
				ResourceSaver::save(cache_base + "_small.png", r_small_texture);
			}
			FileAccess *f = FileAccess::open(cache_base + ".txt", FileAccess::WRITE);
			ERR_FAIL_COND_MSG(!f, "Cannot create file '" + cache_base + ".txt'. Check user write permissions.");
			f->store_line(itos(thumbnail_size));
			f->store_line(itos(has_small_texture));
			f->store_line(itos(FileAccess::get_modified_time(p_item.path)));
			f->store_line(FileAccess::get_md5(p_item.path));
			f->close();
			memdelete(f);
		}
	}
}

void EditorResourcePreview::_thread() {
	exited.clear();
	while (!exit.is_set()) {
		preview_sem.wait();
		preview_mutex.lock();

		if (queue.size()) {
			QueueItem item = queue.front()->get();
			queue.pop_front();

			if (cache.has(item.path)) {
				//already has it because someone loaded it, just let it know it's ready
				String path = item.path;
				if (item.resource.is_valid()) {
					path += ":" + itos(cache[item.path].last_hash); //keep last hash (see description of what this is in condition below)
				}

				_preview_ready(path, cache[item.path].preview, cache[item.path].small_preview, item.id, item.function, item.userdata);

				preview_mutex.unlock();
			} else {
				preview_mutex.unlock();

				Ref<ImageTexture> texture;
				Ref<ImageTexture> small_texture;

				int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
				thumbnail_size *= EDSCALE;

				if (item.resource.is_valid()) {
					_generate_preview(texture, small_texture, item, String());

					//adding hash to the end of path (should be ID:<objid>:<hash>) because of 5 argument limit to call_deferred
					_preview_ready(item.path + ":" + itos(item.resource->hash_edited_version()), texture, small_texture, item.id, item.function, item.userdata);

				} else {
					String temp_path = EditorSettings::get_singleton()->get_cache_dir();
					String cache_base = ProjectSettings::get_singleton()->globalize_path(item.path).md5_text();
					cache_base = temp_path.plus_file("resthumb-" + cache_base);

					//does not have it, try to load a cached thumbnail

					String file = cache_base + ".txt";
					FileAccess *f = FileAccess::open(file, FileAccess::READ);
					if (!f) {
						// No cache found, generate
						_generate_preview(texture, small_texture, item, cache_base);
					} else {
						uint64_t modtime = FileAccess::get_modified_time(item.path);
						int tsize = f->get_line().to_int64();
						bool has_small_texture = f->get_line().to_int();
						uint64_t last_modtime = f->get_line().to_int64();

						bool cache_valid = true;

						if (tsize != thumbnail_size) {
							cache_valid = false;
							memdelete(f);
						} else if (last_modtime != modtime) {
							String last_md5 = f->get_line();
							String md5 = FileAccess::get_md5(item.path);
							memdelete(f);

							if (last_md5 != md5) {
								cache_valid = false;

							} else {
								//update modified time

								f = FileAccess::open(file, FileAccess::WRITE);
								if (!f) {
									// Not returning as this would leave the thread hanging and would require
									// some proper cleanup/disabling of resource preview generation.
									ERR_PRINT("Cannot create file '" + file + "'. Check user write permissions.");
								} else {
									f->store_line(itos(thumbnail_size));
									f->store_line(itos(has_small_texture));
									f->store_line(itos(modtime));
									f->store_line(md5);
									memdelete(f);
								}
							}
						} else {
							memdelete(f);
						}

						if (cache_valid) {
							Ref<Image> img;
							img.instance();
							Ref<Image> small_img;
							small_img.instance();

							if (img->load(cache_base + ".png") != OK) {
								cache_valid = false;
							} else {
								texture.instance();
								texture->create_from_image(img, Texture::FLAG_FILTER);

								if (has_small_texture) {
									if (small_img->load(cache_base + "_small.png") != OK) {
										cache_valid = false;
									} else {
										small_texture.instance();
										small_texture->create_from_image(small_img, Texture::FLAG_FILTER);
									}
								}
							}
						}

						if (!cache_valid) {
							_generate_preview(texture, small_texture, item, cache_base);
						}
					}
					_preview_ready(item.path, texture, small_texture, item.id, item.function, item.userdata);
				}
			}

		} else {
			preview_mutex.unlock();
		}
	}
	exited.set();
}

void EditorResourcePreview::queue_edited_resource_preview(const Ref<Resource> &p_res, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata) {
	ERR_FAIL_NULL(p_receiver);
	ERR_FAIL_COND(!p_res.is_valid());

	preview_mutex.lock();

	String path_id = "ID:" + itos(p_res->get_instance_id());

	if (cache.has(path_id) && cache[path_id].last_hash == p_res->hash_edited_version()) {
		cache[path_id].order = order++;
		p_receiver->call(p_receiver_func, path_id, cache[path_id].preview, cache[path_id].small_preview, p_userdata);
		preview_mutex.unlock();
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
	preview_mutex.unlock();
	preview_sem.post();
}

void EditorResourcePreview::queue_resource_preview(const String &p_path, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata) {
	ERR_FAIL_NULL(p_receiver);
	preview_mutex.lock();
	if (cache.has(p_path)) {
		cache[p_path].order = order++;
		p_receiver->call(p_receiver_func, p_path, cache[p_path].preview, cache[p_path].small_preview, p_userdata);
		preview_mutex.unlock();
		return;
	}

	QueueItem item;
	item.function = p_receiver_func;
	item.id = p_receiver->get_instance_id();
	item.path = p_path;
	item.userdata = p_userdata;

	queue.push_back(item);
	preview_mutex.unlock();
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
	ClassDB::bind_method("_preview_ready", &EditorResourcePreview::_preview_ready);

	ClassDB::bind_method(D_METHOD("queue_resource_preview", "path", "receiver", "receiver_func", "userdata"), &EditorResourcePreview::queue_resource_preview);
	ClassDB::bind_method(D_METHOD("queue_edited_resource_preview", "resource", "receiver", "receiver_func", "userdata"), &EditorResourcePreview::queue_edited_resource_preview);
	ClassDB::bind_method(D_METHOD("add_preview_generator", "generator"), &EditorResourcePreview::add_preview_generator);
	ClassDB::bind_method(D_METHOD("remove_preview_generator", "generator"), &EditorResourcePreview::remove_preview_generator);
	ClassDB::bind_method(D_METHOD("check_for_invalidation", "path"), &EditorResourcePreview::check_for_invalidation);

	ADD_SIGNAL(MethodInfo("preview_invalidated", PropertyInfo(Variant::STRING, "path")));
}

void EditorResourcePreview::check_for_invalidation(const String &p_path) {
	preview_mutex.lock();

	bool call_invalidated = false;
	if (cache.has(p_path)) {
		uint64_t modified_time = FileAccess::get_modified_time(p_path);
		if (modified_time != cache[p_path].modified_time) {
			cache.erase(p_path);
			call_invalidated = true;
		}
	}

	preview_mutex.unlock();

	if (call_invalidated) { //do outside mutex
		call_deferred("emit_signal", "preview_invalidated", p_path);
	}
}

void EditorResourcePreview::start() {
	ERR_FAIL_COND_MSG(thread.is_started(), "Thread already started.");
	thread.start(_thread_func, this);
}

void EditorResourcePreview::stop() {
	if (thread.is_started()) {
		exit.set();
		preview_sem.post();
		while (!exited.is_set()) {
			OS::get_singleton()->delay_usec(10000);
			VisualServer::get_singleton()->sync(); //sync pending stuff, as thread may be blocked on visual server
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
