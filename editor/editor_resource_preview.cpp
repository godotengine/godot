/*************************************************************************/
/*  editor_resource_preview.cpp                                          */
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
#include "editor_resource_preview.h"

#include "editor_scale.h"
#include "editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "message_queue.h"
#include "os/file_access.h"

bool EditorResourcePreviewGenerator::handles(const String &p_type) const {

	if (get_script_instance() && get_script_instance()->has_method("handles")) {
		return get_script_instance()->call("handles", p_type);
	}
	ERR_EXPLAIN("EditorResourcePreviewGenerator::handles needs to be overridden");
	ERR_FAIL_V(false);
}
Ref<Texture> EditorResourcePreviewGenerator::generate(const RES &p_from) {

	if (get_script_instance() && get_script_instance()->has_method("generate")) {
		return get_script_instance()->call("generate", p_from);
	}
	ERR_EXPLAIN("EditorResourcePreviewGenerator::generate needs to be overridden");
	ERR_FAIL_V(Ref<Texture>());
}

Ref<Texture> EditorResourcePreviewGenerator::generate_from_path(const String &p_path) {

	if (get_script_instance() && get_script_instance()->has_method("generate_from_path")) {
		return get_script_instance()->call("generate_from_path", p_path);
	}

	RES res = ResourceLoader::load(p_path);
	if (!res.is_valid())
		return res;
	return generate(res);
}

void EditorResourcePreviewGenerator::_bind_methods() {

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "handles", PropertyInfo(Variant::STRING, "type")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::OBJECT, "generate:Texture", PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_RESOURCE_TYPE, "Resource")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::OBJECT, "generate_from_path:Texture", PropertyInfo(Variant::STRING, "path", PROPERTY_HINT_FILE)));
}

EditorResourcePreviewGenerator::EditorResourcePreviewGenerator() {
}

EditorResourcePreview *EditorResourcePreview::singleton = NULL;

void EditorResourcePreview::_thread_func(void *ud) {

	EditorResourcePreview *erp = (EditorResourcePreview *)ud;
	erp->_thread();
}

void EditorResourcePreview::_preview_ready(const String &p_str, const Ref<Texture> &p_texture, ObjectID id, const StringName &p_func, const Variant &p_ud) {

	//print_line("preview is ready");
	preview_mutex->lock();

	String path = p_str;
	uint32_t hash = 0;
	uint64_t modified_time = 0;

	if (p_str.begins_with("ID:")) {
		hash = p_str.get_slicec(':', 2).to_int();
		path = "ID:" + p_str.get_slicec(':', 1);
	} else {
		modified_time = FileAccess::get_modified_time(path);
	}

	Item item;
	item.order = order++;
	item.preview = p_texture;
	item.last_hash = hash;
	item.modified_time = modified_time;

	cache[path] = item;

	preview_mutex->unlock();

	MessageQueue::get_singleton()->push_call(id, p_func, path, p_texture, p_ud);
}

Ref<Texture> EditorResourcePreview::_generate_preview(const QueueItem &p_item, const String &cache_base) {

	String type;

	if (p_item.resource.is_valid())
		type = p_item.resource->get_class();
	else
		type = ResourceLoader::get_resource_type(p_item.path);
	//print_line("resource type is: "+type);

	if (type == "")
		return Ref<Texture>(); //could not guess type

	Ref<Texture> generated;

	for (int i = 0; i < preview_generators.size(); i++) {

		if (!preview_generators[i]->handles(type))
			continue;
		if (p_item.resource.is_valid()) {
			generated = preview_generators[i]->generate(p_item.resource);
		} else {
			generated = preview_generators[i]->generate_from_path(p_item.path);
		}

		break;
	}

	if (!p_item.resource.is_valid()) {
		// cache the preview in case it's a resource on disk
		if (generated.is_valid()) {
			//print_line("was generated");
			int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
			thumbnail_size *= EDSCALE;
			//wow it generated a preview... save cache
			ResourceSaver::save(cache_base + ".png", generated);
			FileAccess *f = FileAccess::open(cache_base + ".txt", FileAccess::WRITE);
			f->store_line(itos(thumbnail_size));
			f->store_line(itos(FileAccess::get_modified_time(p_item.path)));
			f->store_line(FileAccess::get_md5(p_item.path));
			memdelete(f);
		} else {
			//print_line("was not generated");
		}
	}

	return generated;
}

void EditorResourcePreview::_thread() {

	//print_line("begin thread");
	while (!exit) {

		//print_line("wait for semaphore");
		preview_sem->wait();
		preview_mutex->lock();

		//print_line("blue team go");

		if (queue.size()) {

			QueueItem item = queue.front()->get();
			queue.pop_front();

			if (cache.has(item.path)) {
				//already has it because someone loaded it, just let it know it's ready
				if (item.resource.is_valid()) {
					item.path += ":" + itos(cache[item.path].last_hash); //keep last hash (see description of what this is in condition below)
				}

				_preview_ready(item.path, cache[item.path].preview, item.id, item.function, item.userdata);

				preview_mutex->unlock();
			} else {
				preview_mutex->unlock();

				Ref<Texture> texture;

				//print_line("pop from queue "+item.path);

				int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
				thumbnail_size *= EDSCALE;

				if (item.resource.is_valid()) {

					texture = _generate_preview(item, String());
					//adding hash to the end of path (should be ID:<objid>:<hash>) because of 5 argument limit to call_deferred
					_preview_ready(item.path + ":" + itos(item.resource->hash_edited_version()), texture, item.id, item.function, item.userdata);

				} else {

					String temp_path = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp");
					String cache_base = GlobalConfig::get_singleton()->globalize_path(item.path).md5_text();
					cache_base = temp_path.plus_file("resthumb-" + cache_base);

					//does not have it, try to load a cached thumbnail

					String file = cache_base + ".txt";
					//print_line("cachetxt at "+file);
					FileAccess *f = FileAccess::open(file, FileAccess::READ);
					if (!f) {

						//print_line("generate because not cached");

						//generate
						texture = _generate_preview(item, cache_base);
					} else {

						uint64_t modtime = FileAccess::get_modified_time(item.path);
						int tsize = f->get_line().to_int64();
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
								f->store_line(itos(modtime));
								f->store_line(md5);
								memdelete(f);
							}
						} else {
							memdelete(f);
						}

						cache_valid = false;

						if (cache_valid) {

							texture = ResourceLoader::load(cache_base + ".png", "ImageTexture", true);
							if (!texture.is_valid()) {
								//well fuck
								cache_valid = false;
							}
						}

						if (!cache_valid) {

							texture = _generate_preview(item, cache_base);
						}
					}

					//print_line("notify of preview ready");
					_preview_ready(item.path, texture, item.id, item.function, item.userdata);
				}
			}

		} else {
			preview_mutex->unlock();
		}
	}
}

void EditorResourcePreview::queue_edited_resource_preview(const Ref<Resource> &p_res, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata) {

	ERR_FAIL_NULL(p_receiver);
	ERR_FAIL_COND(!p_res.is_valid());

	preview_mutex->lock();

	String path_id = "ID:" + itos(p_res->get_instance_ID());

	if (cache.has(path_id) && cache[path_id].last_hash == p_res->hash_edited_version()) {

		cache[path_id].order = order++;
		p_receiver->call_deferred(p_receiver_func, path_id, cache[path_id].preview, p_userdata);
		preview_mutex->unlock();
		return;
	}

	cache.erase(path_id); //erase if exists, since it will be regen

	//print_line("send to thread "+p_path);
	QueueItem item;
	item.function = p_receiver_func;
	item.id = p_receiver->get_instance_ID();
	item.resource = p_res;
	item.path = path_id;
	item.userdata = p_userdata;

	queue.push_back(item);
	preview_mutex->unlock();
	preview_sem->post();
}

void EditorResourcePreview::queue_resource_preview(const String &p_path, Object *p_receiver, const StringName &p_receiver_func, const Variant &p_userdata) {

	ERR_FAIL_NULL(p_receiver);
	preview_mutex->lock();
	if (cache.has(p_path)) {
		cache[p_path].order = order++;
		p_receiver->call_deferred(p_receiver_func, p_path, cache[p_path].preview, p_userdata);
		preview_mutex->unlock();
		return;
	}

	//print_line("send to thread "+p_path);
	QueueItem item;
	item.function = p_receiver_func;
	item.id = p_receiver->get_instance_ID();
	item.path = p_path;
	item.userdata = p_userdata;

	queue.push_back(item);
	preview_mutex->unlock();
	preview_sem->post();
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

	ClassDB::bind_method(D_METHOD("queue_resource_preview", "path", "receiver", "receiver_func", "userdata:Variant"), &EditorResourcePreview::queue_resource_preview);
	ClassDB::bind_method(D_METHOD("queue_edited_resource_preview", "resource:Resource", "receiver", "receiver_func", "userdata:Variant"), &EditorResourcePreview::queue_edited_resource_preview);
	ClassDB::bind_method(D_METHOD("add_preview_generator", "generator:EditorResourcePreviewGenerator"), &EditorResourcePreview::add_preview_generator);
	ClassDB::bind_method(D_METHOD("remove_preview_generator", "generator:EditorResourcePreviewGenerator"), &EditorResourcePreview::remove_preview_generator);
	ClassDB::bind_method(D_METHOD("check_for_invalidation", "path"), &EditorResourcePreview::check_for_invalidation);

	ADD_SIGNAL(MethodInfo("preview_invalidated", PropertyInfo(Variant::STRING, "path")));
}

void EditorResourcePreview::check_for_invalidation(const String &p_path) {

	preview_mutex->lock();

	bool call_invalidated = false;
	if (cache.has(p_path)) {

		uint64_t modified_time = FileAccess::get_modified_time(p_path);
		if (modified_time != cache[p_path].modified_time) {
			cache.erase(p_path);
			call_invalidated = true;
		}
	}

	preview_mutex->unlock();

	if (call_invalidated) { //do outside mutex
		call_deferred("emit_signal", "preview_invalidated", p_path);
	}
}

EditorResourcePreview::EditorResourcePreview() {
	singleton = this;
	preview_mutex = Mutex::create();
	preview_sem = Semaphore::create();
	order = 0;
	exit = false;

	thread = Thread::create(_thread_func, this);
}

EditorResourcePreview::~EditorResourcePreview() {

	exit = true;
	preview_sem->post();
	Thread::wait_to_finish(thread);
	memdelete(thread);
	memdelete(preview_mutex);
	memdelete(preview_sem);
}
