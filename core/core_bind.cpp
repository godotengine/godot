/*************************************************************************/
/*  core_bind.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core_bind.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/file_access_compressed.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/marshalls.h"
#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"

namespace core_bind {

////// ResourceLoader //////

ResourceLoader *ResourceLoader::singleton = nullptr;

Error ResourceLoader::load_threaded_request(const String &p_path, const String &p_type_hint, bool p_use_sub_threads) {
	return ::ResourceLoader::load_threaded_request(p_path, p_type_hint, p_use_sub_threads);
}

ResourceLoader::ThreadLoadStatus ResourceLoader::load_threaded_get_status(const String &p_path, Array r_progress) {
	float progress = 0;
	::ResourceLoader::ThreadLoadStatus tls = ::ResourceLoader::load_threaded_get_status(p_path, &progress);
	r_progress.resize(1);
	r_progress[0] = progress;
	return (ThreadLoadStatus)tls;
}

RES ResourceLoader::load_threaded_get(const String &p_path) {
	Error error;
	RES res = ::ResourceLoader::load_threaded_get(p_path, &error);
	return res;
}

RES ResourceLoader::load(const String &p_path, const String &p_type_hint, CacheMode p_cache_mode) {
	Error err = OK;
	RES ret = ::ResourceLoader::load(p_path, p_type_hint, ResourceFormatLoader::CacheMode(p_cache_mode), &err);

	ERR_FAIL_COND_V_MSG(err != OK, ret, "Error loading resource: '" + p_path + "'.");
	return ret;
}

Vector<String> ResourceLoader::get_recognized_extensions_for_type(const String &p_type) {
	List<String> exts;
	::ResourceLoader::get_recognized_extensions_for_type(p_type, &exts);
	Vector<String> ret;
	for (const String &E : exts) {
		ret.push_back(E);
	}

	return ret;
}

void ResourceLoader::set_abort_on_missing_resources(bool p_abort) {
	::ResourceLoader::set_abort_on_missing_resources(p_abort);
}

PackedStringArray ResourceLoader::get_dependencies(const String &p_path) {
	List<String> deps;
	::ResourceLoader::get_dependencies(p_path, &deps);

	PackedStringArray ret;
	for (const String &E : deps) {
		ret.push_back(E);
	}

	return ret;
}

bool ResourceLoader::has_cached(const String &p_path) {
	String local_path = ProjectSettings::get_singleton()->localize_path(p_path);
	return ResourceCache::has(local_path);
}

bool ResourceLoader::exists(const String &p_path, const String &p_type_hint) {
	return ::ResourceLoader::exists(p_path, p_type_hint);
}

ResourceUID::ID ResourceLoader::get_resource_uid(const String &p_path) {
	return ::ResourceLoader::get_resource_uid(p_path);
}

void ResourceLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_threaded_request", "path", "type_hint", "use_sub_threads"), &ResourceLoader::load_threaded_request, DEFVAL(""), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("load_threaded_get_status", "path", "progress"), &ResourceLoader::load_threaded_get_status, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("load_threaded_get", "path"), &ResourceLoader::load_threaded_get);

	ClassDB::bind_method(D_METHOD("load", "path", "type_hint", "cache_mode"), &ResourceLoader::load, DEFVAL(""), DEFVAL(CACHE_MODE_REUSE));
	ClassDB::bind_method(D_METHOD("get_recognized_extensions_for_type", "type"), &ResourceLoader::get_recognized_extensions_for_type);
	ClassDB::bind_method(D_METHOD("set_abort_on_missing_resources", "abort"), &ResourceLoader::set_abort_on_missing_resources);
	ClassDB::bind_method(D_METHOD("get_dependencies", "path"), &ResourceLoader::get_dependencies);
	ClassDB::bind_method(D_METHOD("has_cached", "path"), &ResourceLoader::has_cached);
	ClassDB::bind_method(D_METHOD("exists", "path", "type_hint"), &ResourceLoader::exists, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_resource_uid", "path"), &ResourceLoader::get_resource_uid);

	BIND_ENUM_CONSTANT(THREAD_LOAD_INVALID_RESOURCE);
	BIND_ENUM_CONSTANT(THREAD_LOAD_IN_PROGRESS);
	BIND_ENUM_CONSTANT(THREAD_LOAD_FAILED);
	BIND_ENUM_CONSTANT(THREAD_LOAD_LOADED);

	BIND_ENUM_CONSTANT(CACHE_MODE_IGNORE);
	BIND_ENUM_CONSTANT(CACHE_MODE_REUSE);
	BIND_ENUM_CONSTANT(CACHE_MODE_REPLACE);
}

////// ResourceSaver //////

Error ResourceSaver::save(const String &p_path, const RES &p_resource, SaverFlags p_flags) {
	ERR_FAIL_COND_V_MSG(p_resource.is_null(), ERR_INVALID_PARAMETER, "Can't save empty resource to path '" + String(p_path) + "'.");
	return ::ResourceSaver::save(p_path, p_resource, p_flags);
}

Vector<String> ResourceSaver::get_recognized_extensions(const RES &p_resource) {
	ERR_FAIL_COND_V_MSG(p_resource.is_null(), Vector<String>(), "It's not a reference to a valid Resource object.");
	List<String> exts;
	::ResourceSaver::get_recognized_extensions(p_resource, &exts);
	Vector<String> ret;
	for (const String &E : exts) {
		ret.push_back(E);
	}
	return ret;
}

ResourceSaver *ResourceSaver::singleton = nullptr;

void ResourceSaver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("save", "path", "resource", "flags"), &ResourceSaver::save, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_recognized_extensions", "type"), &ResourceSaver::get_recognized_extensions);

	BIND_ENUM_CONSTANT(FLAG_RELATIVE_PATHS);
	BIND_ENUM_CONSTANT(FLAG_BUNDLE_RESOURCES);
	BIND_ENUM_CONSTANT(FLAG_CHANGE_PATH);
	BIND_ENUM_CONSTANT(FLAG_OMIT_EDITOR_PROPERTIES);
	BIND_ENUM_CONSTANT(FLAG_SAVE_BIG_ENDIAN);
	BIND_ENUM_CONSTANT(FLAG_COMPRESS);
	BIND_ENUM_CONSTANT(FLAG_REPLACE_SUBRESOURCE_PATHS);
}

////// OS //////

PackedStringArray OS::get_connected_midi_inputs() {
	return ::OS::get_singleton()->get_connected_midi_inputs();
}

void OS::open_midi_inputs() {
	::OS::get_singleton()->open_midi_inputs();
}

void OS::close_midi_inputs() {
	::OS::get_singleton()->close_midi_inputs();
}

void OS::set_use_file_access_save_and_swap(bool p_enable) {
	FileAccess::set_backup_save(p_enable);
}

void OS::set_low_processor_usage_mode(bool p_enabled) {
	::OS::get_singleton()->set_low_processor_usage_mode(p_enabled);
}

bool OS::is_in_low_processor_usage_mode() const {
	return ::OS::get_singleton()->is_in_low_processor_usage_mode();
}

void OS::set_low_processor_usage_mode_sleep_usec(int p_usec) {
	::OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(p_usec);
}

int OS::get_low_processor_usage_mode_sleep_usec() const {
	return ::OS::get_singleton()->get_low_processor_usage_mode_sleep_usec();
}

void OS::alert(const String &p_alert, const String &p_title) {
	::OS::get_singleton()->alert(p_alert, p_title);
}

String OS::get_executable_path() const {
	return ::OS::get_singleton()->get_executable_path();
}

Error OS::shell_open(String p_uri) {
	if (p_uri.begins_with("res://")) {
		WARN_PRINT("Attempting to open an URL with the \"res://\" protocol. Use `ProjectSettings.globalize_path()` to convert a Godot-specific path to a system path before opening it with `OS.shell_open()`.");
	} else if (p_uri.begins_with("user://")) {
		WARN_PRINT("Attempting to open an URL with the \"user://\" protocol. Use `ProjectSettings.globalize_path()` to convert a Godot-specific path to a system path before opening it with `OS.shell_open()`.");
	}
	return ::OS::get_singleton()->shell_open(p_uri);
}

int OS::execute(const String &p_path, const Vector<String> &p_arguments, Array r_output, bool p_read_stderr) {
	List<String> args;
	for (int i = 0; i < p_arguments.size(); i++) {
		args.push_back(p_arguments[i]);
	}
	String pipe;
	int exitcode = 0;
	Error err = ::OS::get_singleton()->execute(p_path, args, &pipe, &exitcode, p_read_stderr);
	r_output.push_back(pipe);
	if (err != OK) {
		return -1;
	}
	return exitcode;
}

int OS::create_instance(const Vector<String> &p_arguments) {
	List<String> args;
	for (int i = 0; i < p_arguments.size(); i++) {
		args.push_back(p_arguments[i]);
	}
	::OS::ProcessID pid = 0;
	Error err = ::OS::get_singleton()->create_instance(args, &pid);
	if (err != OK) {
		return -1;
	}
	return pid;
}

int OS::create_process(const String &p_path, const Vector<String> &p_arguments) {
	List<String> args;
	for (int i = 0; i < p_arguments.size(); i++) {
		args.push_back(p_arguments[i]);
	}
	::OS::ProcessID pid = 0;
	Error err = ::OS::get_singleton()->create_process(p_path, args, &pid);
	if (err != OK) {
		return -1;
	}
	return pid;
}

Error OS::kill(int p_pid) {
	return ::OS::get_singleton()->kill(p_pid);
}

int OS::get_process_id() const {
	return ::OS::get_singleton()->get_process_id();
}

bool OS::has_environment(const String &p_var) const {
	return ::OS::get_singleton()->has_environment(p_var);
}

String OS::get_environment(const String &p_var) const {
	return ::OS::get_singleton()->get_environment(p_var);
}

bool OS::set_environment(const String &p_var, const String &p_value) const {
	return ::OS::get_singleton()->set_environment(p_var, p_value);
}

String OS::get_name() const {
	return ::OS::get_singleton()->get_name();
}

Vector<String> OS::get_cmdline_args() {
	List<String> cmdline = ::OS::get_singleton()->get_cmdline_args();
	Vector<String> cmdlinev;
	for (const String &E : cmdline) {
		cmdlinev.push_back(E);
	}

	return cmdlinev;
}

String OS::get_locale() const {
	return ::OS::get_singleton()->get_locale();
}

String OS::get_locale_language() const {
	return ::OS::get_singleton()->get_locale_language();
}

String OS::get_model_name() const {
	return ::OS::get_singleton()->get_model_name();
}

Error OS::set_thread_name(const String &p_name) {
	return ::Thread::set_name(p_name);
}

::Thread::ID OS::get_thread_caller_id() const {
	return ::Thread::get_caller_id();
};

bool OS::has_feature(const String &p_feature) const {
	return ::OS::get_singleton()->has_feature(p_feature);
}

uint64_t OS::get_static_memory_usage() const {
	return ::OS::get_singleton()->get_static_memory_usage();
}

uint64_t OS::get_static_memory_peak_usage() const {
	return ::OS::get_singleton()->get_static_memory_peak_usage();
}

/** This method uses a signed argument for better error reporting as it's used from the scripting API. */
void OS::delay_usec(int p_usec) const {
	ERR_FAIL_COND_MSG(
			p_usec < 0,
			vformat("Can't sleep for %d microseconds. The delay provided must be greater than or equal to 0 microseconds.", p_usec));
	::OS::get_singleton()->delay_usec(p_usec);
}

/** This method uses a signed argument for better error reporting as it's used from the scripting API. */
void OS::delay_msec(int p_msec) const {
	ERR_FAIL_COND_MSG(
			p_msec < 0,
			vformat("Can't sleep for %d milliseconds. The delay provided must be greater than or equal to 0 milliseconds.", p_msec));
	::OS::get_singleton()->delay_usec(int64_t(p_msec) * 1000);
}

bool OS::can_use_threads() const {
	return ::OS::get_singleton()->can_use_threads();
}

bool OS::is_userfs_persistent() const {
	return ::OS::get_singleton()->is_userfs_persistent();
}

int OS::get_processor_count() const {
	return ::OS::get_singleton()->get_processor_count();
}

bool OS::is_stdout_verbose() const {
	return ::OS::get_singleton()->is_stdout_verbose();
}

void OS::dump_memory_to_file(const String &p_file) {
	::OS::get_singleton()->dump_memory_to_file(p_file.utf8().get_data());
}

struct OSCoreBindImg {
	String path;
	Size2 size;
	int fmt = 0;
	ObjectID id;
	int vram = 0;
	bool operator<(const OSCoreBindImg &p_img) const { return vram == p_img.vram ? id < p_img.id : vram > p_img.vram; }
};

void OS::print_all_textures_by_size() {
	List<OSCoreBindImg> imgs;
	uint64_t total = 0;
	{
		List<Ref<Resource>> rsrc;
		ResourceCache::get_cached_resources(&rsrc);

		for (Ref<Resource> &res : rsrc) {
			if (!res->is_class("Texture")) {
				continue;
			}

			Size2 size = res->call("get_size");
			int fmt = res->call("get_format");

			OSCoreBindImg img;
			img.size = size;
			img.fmt = fmt;
			img.path = res->get_path();
			img.vram = Image::get_image_data_size(img.size.width, img.size.height, Image::Format(img.fmt));
			img.id = res->get_instance_id();
			total += img.vram;
			imgs.push_back(img);
		}
	}

	imgs.sort();

	if (imgs.size() == 0) {
		print_line("No textures seem used in this project.");
	} else {
		print_line("Textures currently in use, sorted by VRAM usage:\n"
				   "Path - VRAM usage (Dimensions)");
	}

	for (const OSCoreBindImg &img : imgs) {
		print_line(vformat("%s - %s %s",
				img.path,
				String::humanize_size(img.vram),
				img.size));
	}

	print_line(vformat("Total VRAM usage: %s.", String::humanize_size(total)));
}

void OS::print_resources_by_type(const Vector<String> &p_types) {
	ERR_FAIL_COND_MSG(p_types.size() == 0,
			"At least one type should be provided to print resources by type.");

	print_line(vformat("Resources currently in use for the following types: %s", p_types));

	Map<String, int> type_count;
	List<Ref<Resource>> resources;
	ResourceCache::get_cached_resources(&resources);

	for (const Ref<Resource> &r : resources) {
		bool found = false;

		for (int i = 0; i < p_types.size(); i++) {
			if (r->is_class(p_types[i])) {
				found = true;
			}
		}
		if (!found) {
			continue;
		}

		if (!type_count.has(r->get_class())) {
			type_count[r->get_class()] = 0;
		}

		type_count[r->get_class()]++;

		print_line(vformat("%s: %s", r->get_class(), r->get_path()));

		List<StringName> metas;
		r->get_meta_list(&metas);
		for (const StringName &meta : metas) {
			print_line(vformat("  %s: %s", meta, r->get_meta(meta)));
		}
	}

	for (const KeyValue<String, int> &E : type_count) {
		print_line(vformat("%s count: %d", E.key, E.value));
	}
}

void OS::print_all_resources(const String &p_to_file) {
	::OS::get_singleton()->print_all_resources(p_to_file);
}

void OS::print_resources_in_use(bool p_short) {
	::OS::get_singleton()->print_resources_in_use(p_short);
}

void OS::dump_resources_to_file(const String &p_file) {
	::OS::get_singleton()->dump_resources_to_file(p_file.utf8().get_data());
}

String OS::get_user_data_dir() const {
	return ::OS::get_singleton()->get_user_data_dir();
}

String OS::get_config_dir() const {
	// Exposed as `get_config_dir()` instead of `get_config_path()` for consistency with other exposed OS methods.
	return ::OS::get_singleton()->get_config_path();
}

String OS::get_data_dir() const {
	// Exposed as `get_data_dir()` instead of `get_data_path()` for consistency with other exposed OS methods.
	return ::OS::get_singleton()->get_data_path();
}

String OS::get_cache_dir() const {
	// Exposed as `get_cache_dir()` instead of `get_cache_path()` for consistency with other exposed OS methods.
	return ::OS::get_singleton()->get_cache_path();
}

bool OS::is_debug_build() const {
#ifdef DEBUG_ENABLED
	return true;
#else
	return false;
#endif
}

String OS::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	return ::OS::get_singleton()->get_system_dir(::OS::SystemDir(p_dir), p_shared_storage);
}

String OS::get_keycode_string(uint32_t p_code) const {
	return ::keycode_get_string(p_code);
}

bool OS::is_keycode_unicode(uint32_t p_unicode) const {
	return ::keycode_has_unicode(p_unicode);
}

int OS::find_keycode_from_string(const String &p_code) const {
	return find_keycode(p_code);
}

bool OS::request_permission(const String &p_name) {
	return ::OS::get_singleton()->request_permission(p_name);
}

bool OS::request_permissions() {
	return ::OS::get_singleton()->request_permissions();
}

Vector<String> OS::get_granted_permissions() const {
	return ::OS::get_singleton()->get_granted_permissions();
}

String OS::get_unique_id() const {
	return ::OS::get_singleton()->get_unique_id();
}

OS *OS::singleton = nullptr;

void OS::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_connected_midi_inputs"), &OS::get_connected_midi_inputs);
	ClassDB::bind_method(D_METHOD("open_midi_inputs"), &OS::open_midi_inputs);
	ClassDB::bind_method(D_METHOD("close_midi_inputs"), &OS::close_midi_inputs);

	ClassDB::bind_method(D_METHOD("alert", "text", "title"), &OS::alert, DEFVAL("Alert!"));

	ClassDB::bind_method(D_METHOD("set_low_processor_usage_mode", "enable"), &OS::set_low_processor_usage_mode);
	ClassDB::bind_method(D_METHOD("is_in_low_processor_usage_mode"), &OS::is_in_low_processor_usage_mode);

	ClassDB::bind_method(D_METHOD("set_low_processor_usage_mode_sleep_usec", "usec"), &OS::set_low_processor_usage_mode_sleep_usec);
	ClassDB::bind_method(D_METHOD("get_low_processor_usage_mode_sleep_usec"), &OS::get_low_processor_usage_mode_sleep_usec);

	ClassDB::bind_method(D_METHOD("get_processor_count"), &OS::get_processor_count);

	ClassDB::bind_method(D_METHOD("get_executable_path"), &OS::get_executable_path);
	ClassDB::bind_method(D_METHOD("execute", "path", "arguments", "output", "read_stderr"), &OS::execute, DEFVAL(Array()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("create_process", "path", "arguments"), &OS::create_process);
	ClassDB::bind_method(D_METHOD("create_instance", "arguments"), &OS::create_instance);
	ClassDB::bind_method(D_METHOD("kill", "pid"), &OS::kill);
	ClassDB::bind_method(D_METHOD("shell_open", "uri"), &OS::shell_open);
	ClassDB::bind_method(D_METHOD("get_process_id"), &OS::get_process_id);

	ClassDB::bind_method(D_METHOD("get_environment", "variable"), &OS::get_environment);
	ClassDB::bind_method(D_METHOD("set_environment", "variable", "value"), &OS::set_environment);
	ClassDB::bind_method(D_METHOD("has_environment", "variable"), &OS::has_environment);

	ClassDB::bind_method(D_METHOD("get_name"), &OS::get_name);
	ClassDB::bind_method(D_METHOD("get_cmdline_args"), &OS::get_cmdline_args);

	ClassDB::bind_method(D_METHOD("delay_usec", "usec"), &OS::delay_usec);
	ClassDB::bind_method(D_METHOD("delay_msec", "msec"), &OS::delay_msec);
	ClassDB::bind_method(D_METHOD("get_locale"), &OS::get_locale);
	ClassDB::bind_method(D_METHOD("get_locale_language"), &OS::get_locale_language);
	ClassDB::bind_method(D_METHOD("get_model_name"), &OS::get_model_name);

	ClassDB::bind_method(D_METHOD("is_userfs_persistent"), &OS::is_userfs_persistent);
	ClassDB::bind_method(D_METHOD("is_stdout_verbose"), &OS::is_stdout_verbose);

	ClassDB::bind_method(D_METHOD("can_use_threads"), &OS::can_use_threads);

	ClassDB::bind_method(D_METHOD("is_debug_build"), &OS::is_debug_build);

	ClassDB::bind_method(D_METHOD("dump_memory_to_file", "file"), &OS::dump_memory_to_file);
	ClassDB::bind_method(D_METHOD("dump_resources_to_file", "file"), &OS::dump_resources_to_file);
	ClassDB::bind_method(D_METHOD("print_resources_in_use", "short"), &OS::print_resources_in_use, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("print_all_resources", "tofile"), &OS::print_all_resources, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("get_static_memory_usage"), &OS::get_static_memory_usage);
	ClassDB::bind_method(D_METHOD("get_static_memory_peak_usage"), &OS::get_static_memory_peak_usage);

	ClassDB::bind_method(D_METHOD("get_user_data_dir"), &OS::get_user_data_dir);
	ClassDB::bind_method(D_METHOD("get_system_dir", "dir", "shared_storage"), &OS::get_system_dir, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_config_dir"), &OS::get_config_dir);
	ClassDB::bind_method(D_METHOD("get_data_dir"), &OS::get_data_dir);
	ClassDB::bind_method(D_METHOD("get_cache_dir"), &OS::get_cache_dir);
	ClassDB::bind_method(D_METHOD("get_unique_id"), &OS::get_unique_id);

	ClassDB::bind_method(D_METHOD("print_all_textures_by_size"), &OS::print_all_textures_by_size);
	ClassDB::bind_method(D_METHOD("print_resources_by_type", "types"), &OS::print_resources_by_type);

	ClassDB::bind_method(D_METHOD("get_keycode_string", "code"), &OS::get_keycode_string);
	ClassDB::bind_method(D_METHOD("is_keycode_unicode", "code"), &OS::is_keycode_unicode);
	ClassDB::bind_method(D_METHOD("find_keycode_from_string", "string"), &OS::find_keycode_from_string);

	ClassDB::bind_method(D_METHOD("set_use_file_access_save_and_swap", "enabled"), &OS::set_use_file_access_save_and_swap);

	ClassDB::bind_method(D_METHOD("set_thread_name", "name"), &OS::set_thread_name);
	ClassDB::bind_method(D_METHOD("get_thread_caller_id"), &OS::get_thread_caller_id);

	ClassDB::bind_method(D_METHOD("has_feature", "tag_name"), &OS::has_feature);

	ClassDB::bind_method(D_METHOD("request_permission", "name"), &OS::request_permission);
	ClassDB::bind_method(D_METHOD("request_permissions"), &OS::request_permissions);
	ClassDB::bind_method(D_METHOD("get_granted_permissions"), &OS::get_granted_permissions);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "low_processor_usage_mode"), "set_low_processor_usage_mode", "is_in_low_processor_usage_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "low_processor_usage_mode_sleep_usec"), "set_low_processor_usage_mode_sleep_usec", "get_low_processor_usage_mode_sleep_usec");

	// Those default values need to be specified for the docs generator,
	// to avoid using values from the documentation writer's own OS instance.
	ADD_PROPERTY_DEFAULT("exit_code", 0);
	ADD_PROPERTY_DEFAULT("low_processor_usage_mode", false);
	ADD_PROPERTY_DEFAULT("low_processor_usage_mode_sleep_usec", 6900);

	BIND_ENUM_CONSTANT(VIDEO_DRIVER_VULKAN);
	BIND_ENUM_CONSTANT(VIDEO_DRIVER_OPENGL_3);

	BIND_ENUM_CONSTANT(DAY_SUNDAY);
	BIND_ENUM_CONSTANT(DAY_MONDAY);
	BIND_ENUM_CONSTANT(DAY_TUESDAY);
	BIND_ENUM_CONSTANT(DAY_WEDNESDAY);
	BIND_ENUM_CONSTANT(DAY_THURSDAY);
	BIND_ENUM_CONSTANT(DAY_FRIDAY);
	BIND_ENUM_CONSTANT(DAY_SATURDAY);

	BIND_ENUM_CONSTANT(MONTH_JANUARY);
	BIND_ENUM_CONSTANT(MONTH_FEBRUARY);
	BIND_ENUM_CONSTANT(MONTH_MARCH);
	BIND_ENUM_CONSTANT(MONTH_APRIL);
	BIND_ENUM_CONSTANT(MONTH_MAY);
	BIND_ENUM_CONSTANT(MONTH_JUNE);
	BIND_ENUM_CONSTANT(MONTH_JULY);
	BIND_ENUM_CONSTANT(MONTH_AUGUST);
	BIND_ENUM_CONSTANT(MONTH_SEPTEMBER);
	BIND_ENUM_CONSTANT(MONTH_OCTOBER);
	BIND_ENUM_CONSTANT(MONTH_NOVEMBER);
	BIND_ENUM_CONSTANT(MONTH_DECEMBER);

	BIND_ENUM_CONSTANT(SYSTEM_DIR_DESKTOP);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_DCIM);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_DOCUMENTS);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_DOWNLOADS);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_MOVIES);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_MUSIC);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_PICTURES);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_RINGTONES);
}

////// Geometry2D //////

Geometry2D *Geometry2D::singleton = nullptr;

Geometry2D *Geometry2D::get_singleton() {
	return singleton;
}

bool Geometry2D::is_point_in_circle(const Vector2 &p_point, const Vector2 &p_circle_pos, real_t p_circle_radius) {
	return ::Geometry2D::is_point_in_circle(p_point, p_circle_pos, p_circle_radius);
}

real_t Geometry2D::segment_intersects_circle(const Vector2 &p_from, const Vector2 &p_to, const Vector2 &p_circle_pos, real_t p_circle_radius) {
	return ::Geometry2D::segment_intersects_circle(p_from, p_to, p_circle_pos, p_circle_radius);
}

Variant Geometry2D::segment_intersects_segment(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b) {
	Vector2 result;
	if (::Geometry2D::segment_intersects_segment(p_from_a, p_to_a, p_from_b, p_to_b, &result)) {
		return result;
	} else {
		return Variant();
	}
}

Variant Geometry2D::line_intersects_line(const Vector2 &p_from_a, const Vector2 &p_dir_a, const Vector2 &p_from_b, const Vector2 &p_dir_b) {
	Vector2 result;
	if (::Geometry2D::line_intersects_line(p_from_a, p_dir_a, p_from_b, p_dir_b, result)) {
		return result;
	} else {
		return Variant();
	}
}

Vector<Vector2> Geometry2D::get_closest_points_between_segments(const Vector2 &p1, const Vector2 &q1, const Vector2 &p2, const Vector2 &q2) {
	Vector2 r1, r2;
	::Geometry2D::get_closest_points_between_segments(p1, q1, p2, q2, r1, r2);
	Vector<Vector2> r;
	r.resize(2);
	r.set(0, r1);
	r.set(1, r2);
	return r;
}

Vector2 Geometry2D::get_closest_point_to_segment(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b) {
	Vector2 s[2] = { p_a, p_b };
	return ::Geometry2D::get_closest_point_to_segment(p_point, s);
}

Vector2 Geometry2D::get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b) {
	Vector2 s[2] = { p_a, p_b };
	return ::Geometry2D::get_closest_point_to_segment_uncapped(p_point, s);
}

bool Geometry2D::point_is_inside_triangle(const Vector2 &s, const Vector2 &a, const Vector2 &b, const Vector2 &c) const {
	return ::Geometry2D::is_point_in_triangle(s, a, b, c);
}

bool Geometry2D::is_polygon_clockwise(const Vector<Vector2> &p_polygon) {
	return ::Geometry2D::is_polygon_clockwise(p_polygon);
}

bool Geometry2D::is_point_in_polygon(const Point2 &p_point, const Vector<Vector2> &p_polygon) {
	return ::Geometry2D::is_point_in_polygon(p_point, p_polygon);
}

Vector<int> Geometry2D::triangulate_polygon(const Vector<Vector2> &p_polygon) {
	return ::Geometry2D::triangulate_polygon(p_polygon);
}

Vector<int> Geometry2D::triangulate_delaunay(const Vector<Vector2> &p_points) {
	return ::Geometry2D::triangulate_delaunay(p_points);
}

Vector<Point2> Geometry2D::convex_hull(const Vector<Point2> &p_points) {
	return ::Geometry2D::convex_hull(p_points);
}

Array Geometry2D::merge_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = ::Geometry2D::merge_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::clip_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = ::Geometry2D::clip_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::intersect_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = ::Geometry2D::intersect_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::exclude_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = ::Geometry2D::exclude_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::clip_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon) {
	Vector<Vector<Point2>> polys = ::Geometry2D::clip_polyline_with_polygon(p_polyline, p_polygon);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::intersect_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon) {
	Vector<Vector<Point2>> polys = ::Geometry2D::intersect_polyline_with_polygon(p_polyline, p_polygon);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::offset_polygon(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type) {
	Vector<Vector<Point2>> polys = ::Geometry2D::offset_polygon(p_polygon, p_delta, ::Geometry2D::PolyJoinType(p_join_type));

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array Geometry2D::offset_polyline(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type) {
	Vector<Vector<Point2>> polys = ::Geometry2D::offset_polyline(p_polygon, p_delta, ::Geometry2D::PolyJoinType(p_join_type), ::Geometry2D::PolyEndType(p_end_type));

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Dictionary Geometry2D::make_atlas(const Vector<Size2> &p_rects) {
	Dictionary ret;

	Vector<Size2i> rects;
	for (int i = 0; i < p_rects.size(); i++) {
		rects.push_back(p_rects[i]);
	}

	Vector<Point2i> result;
	Size2i size;

	::Geometry2D::make_atlas(rects, result, size);

	Size2 r_size = size;
	Vector<Point2> r_result;
	for (int i = 0; i < result.size(); i++) {
		r_result.push_back(result[i]);
	}

	ret["points"] = r_result;
	ret["size"] = r_size;

	return ret;
}

void Geometry2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_point_in_circle", "point", "circle_position", "circle_radius"), &Geometry2D::is_point_in_circle);
	ClassDB::bind_method(D_METHOD("segment_intersects_segment", "from_a", "to_a", "from_b", "to_b"), &Geometry2D::segment_intersects_segment);
	ClassDB::bind_method(D_METHOD("line_intersects_line", "from_a", "dir_a", "from_b", "dir_b"), &Geometry2D::line_intersects_line);

	ClassDB::bind_method(D_METHOD("get_closest_points_between_segments", "p1", "q1", "p2", "q2"), &Geometry2D::get_closest_points_between_segments);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "point", "s1", "s2"), &Geometry2D::get_closest_point_to_segment);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment_uncapped", "point", "s1", "s2"), &Geometry2D::get_closest_point_to_segment_uncapped);

	ClassDB::bind_method(D_METHOD("point_is_inside_triangle", "point", "a", "b", "c"), &Geometry2D::point_is_inside_triangle);

	ClassDB::bind_method(D_METHOD("is_polygon_clockwise", "polygon"), &Geometry2D::is_polygon_clockwise);
	ClassDB::bind_method(D_METHOD("is_point_in_polygon", "point", "polygon"), &Geometry2D::is_point_in_polygon);
	ClassDB::bind_method(D_METHOD("triangulate_polygon", "polygon"), &Geometry2D::triangulate_polygon);
	ClassDB::bind_method(D_METHOD("triangulate_delaunay", "points"), &Geometry2D::triangulate_delaunay);
	ClassDB::bind_method(D_METHOD("convex_hull", "points"), &Geometry2D::convex_hull);

	ClassDB::bind_method(D_METHOD("merge_polygons", "polygon_a", "polygon_b"), &Geometry2D::merge_polygons);
	ClassDB::bind_method(D_METHOD("clip_polygons", "polygon_a", "polygon_b"), &Geometry2D::clip_polygons);
	ClassDB::bind_method(D_METHOD("intersect_polygons", "polygon_a", "polygon_b"), &Geometry2D::intersect_polygons);
	ClassDB::bind_method(D_METHOD("exclude_polygons", "polygon_a", "polygon_b"), &Geometry2D::exclude_polygons);

	ClassDB::bind_method(D_METHOD("clip_polyline_with_polygon", "polyline", "polygon"), &Geometry2D::clip_polyline_with_polygon);
	ClassDB::bind_method(D_METHOD("intersect_polyline_with_polygon", "polyline", "polygon"), &Geometry2D::intersect_polyline_with_polygon);

	ClassDB::bind_method(D_METHOD("offset_polygon", "polygon", "delta", "join_type"), &Geometry2D::offset_polygon, DEFVAL(JOIN_SQUARE));
	ClassDB::bind_method(D_METHOD("offset_polyline", "polyline", "delta", "join_type", "end_type"), &Geometry2D::offset_polyline, DEFVAL(JOIN_SQUARE), DEFVAL(END_SQUARE));

	ClassDB::bind_method(D_METHOD("make_atlas", "sizes"), &Geometry2D::make_atlas);

	BIND_ENUM_CONSTANT(OPERATION_UNION);
	BIND_ENUM_CONSTANT(OPERATION_DIFFERENCE);
	BIND_ENUM_CONSTANT(OPERATION_INTERSECTION);
	BIND_ENUM_CONSTANT(OPERATION_XOR);

	BIND_ENUM_CONSTANT(JOIN_SQUARE);
	BIND_ENUM_CONSTANT(JOIN_ROUND);
	BIND_ENUM_CONSTANT(JOIN_MITER);

	BIND_ENUM_CONSTANT(END_POLYGON);
	BIND_ENUM_CONSTANT(END_JOINED);
	BIND_ENUM_CONSTANT(END_BUTT);
	BIND_ENUM_CONSTANT(END_SQUARE);
	BIND_ENUM_CONSTANT(END_ROUND);
}

////// Geometry3D //////

Geometry3D *Geometry3D::singleton = nullptr;

Geometry3D *Geometry3D::get_singleton() {
	return singleton;
}

Vector<Plane> Geometry3D::build_box_planes(const Vector3 &p_extents) {
	return ::Geometry3D::build_box_planes(p_extents);
}

Vector<Plane> Geometry3D::build_cylinder_planes(float p_radius, float p_height, int p_sides, Vector3::Axis p_axis) {
	return ::Geometry3D::build_cylinder_planes(p_radius, p_height, p_sides, p_axis);
}

Vector<Plane> Geometry3D::build_capsule_planes(float p_radius, float p_height, int p_sides, int p_lats, Vector3::Axis p_axis) {
	return ::Geometry3D::build_capsule_planes(p_radius, p_height, p_sides, p_lats, p_axis);
}

Vector<Vector3> Geometry3D::get_closest_points_between_segments(const Vector3 &p1, const Vector3 &p2, const Vector3 &q1, const Vector3 &q2) {
	Vector3 r1, r2;
	::Geometry3D::get_closest_points_between_segments(p1, p2, q1, q2, r1, r2);
	Vector<Vector3> r;
	r.resize(2);
	r.set(0, r1);
	r.set(1, r2);
	return r;
}

Vector3 Geometry3D::get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b) {
	Vector3 s[2] = { p_a, p_b };
	return ::Geometry3D::get_closest_point_to_segment(p_point, s);
}

Vector3 Geometry3D::get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b) {
	Vector3 s[2] = { p_a, p_b };
	return ::Geometry3D::get_closest_point_to_segment_uncapped(p_point, s);
}

Variant Geometry3D::ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2) {
	Vector3 res;
	if (::Geometry3D::ray_intersects_triangle(p_from, p_dir, p_v0, p_v1, p_v2, &res)) {
		return res;
	} else {
		return Variant();
	}
}

Variant Geometry3D::segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2) {
	Vector3 res;
	if (::Geometry3D::segment_intersects_triangle(p_from, p_to, p_v0, p_v1, p_v2, &res)) {
		return res;
	} else {
		return Variant();
	}
}

Vector<Vector3> Geometry3D::segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius) {
	Vector<Vector3> r;
	Vector3 res, norm;
	if (!::Geometry3D::segment_intersects_sphere(p_from, p_to, p_sphere_pos, p_sphere_radius, &res, &norm)) {
		return r;
	}

	r.resize(2);
	r.set(0, res);
	r.set(1, norm);
	return r;
}

Vector<Vector3> Geometry3D::segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, float p_height, float p_radius) {
	Vector<Vector3> r;
	Vector3 res, norm;
	if (!::Geometry3D::segment_intersects_cylinder(p_from, p_to, p_height, p_radius, &res, &norm)) {
		return r;
	}

	r.resize(2);
	r.set(0, res);
	r.set(1, norm);
	return r;
}

Vector<Vector3> Geometry3D::segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Vector<Plane> &p_planes) {
	Vector<Vector3> r;
	Vector3 res, norm;
	if (!::Geometry3D::segment_intersects_convex(p_from, p_to, p_planes.ptr(), p_planes.size(), &res, &norm)) {
		return r;
	}

	r.resize(2);
	r.set(0, res);
	r.set(1, norm);
	return r;
}

Vector<Vector3> Geometry3D::clip_polygon(const Vector<Vector3> &p_points, const Plane &p_plane) {
	return ::Geometry3D::clip_polygon(p_points, p_plane);
}

void Geometry3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("build_box_planes", "extents"), &Geometry3D::build_box_planes);
	ClassDB::bind_method(D_METHOD("build_cylinder_planes", "radius", "height", "sides", "axis"), &Geometry3D::build_cylinder_planes, DEFVAL(Vector3::AXIS_Z));
	ClassDB::bind_method(D_METHOD("build_capsule_planes", "radius", "height", "sides", "lats", "axis"), &Geometry3D::build_capsule_planes, DEFVAL(Vector3::AXIS_Z));

	ClassDB::bind_method(D_METHOD("get_closest_points_between_segments", "p1", "p2", "q1", "q2"), &Geometry3D::get_closest_points_between_segments);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "point", "s1", "s2"), &Geometry3D::get_closest_point_to_segment);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment_uncapped", "point", "s1", "s2"), &Geometry3D::get_closest_point_to_segment_uncapped);

	ClassDB::bind_method(D_METHOD("ray_intersects_triangle", "from", "dir", "a", "b", "c"), &Geometry3D::ray_intersects_triangle);
	ClassDB::bind_method(D_METHOD("segment_intersects_triangle", "from", "to", "a", "b", "c"), &Geometry3D::segment_intersects_triangle);
	ClassDB::bind_method(D_METHOD("segment_intersects_sphere", "from", "to", "sphere_position", "sphere_radius"), &Geometry3D::segment_intersects_sphere);
	ClassDB::bind_method(D_METHOD("segment_intersects_cylinder", "from", "to", "height", "radius"), &Geometry3D::segment_intersects_cylinder);
	ClassDB::bind_method(D_METHOD("segment_intersects_convex", "from", "to", "planes"), &Geometry3D::segment_intersects_convex);

	ClassDB::bind_method(D_METHOD("clip_polygon", "points", "plane"), &Geometry3D::clip_polygon);
}

////// File //////

Error File::open_encrypted(const String &p_path, ModeFlags p_mode_flags, const Vector<uint8_t> &p_key) {
	Error err = open(p_path, p_mode_flags);
	if (err) {
		return err;
	}

	FileAccessEncrypted *fae = memnew(FileAccessEncrypted);
	err = fae->open_and_parse(f, p_key, (p_mode_flags == WRITE) ? FileAccessEncrypted::MODE_WRITE_AES256 : FileAccessEncrypted::MODE_READ);
	if (err) {
		memdelete(fae);
		close();
		return err;
	}
	f = fae;
	return OK;
}

Error File::open_encrypted_pass(const String &p_path, ModeFlags p_mode_flags, const String &p_pass) {
	Error err = open(p_path, p_mode_flags);
	if (err) {
		return err;
	}

	FileAccessEncrypted *fae = memnew(FileAccessEncrypted);
	err = fae->open_and_parse_password(f, p_pass, (p_mode_flags == WRITE) ? FileAccessEncrypted::MODE_WRITE_AES256 : FileAccessEncrypted::MODE_READ);
	if (err) {
		memdelete(fae);
		close();
		return err;
	}

	f = fae;
	return OK;
}

Error File::open_compressed(const String &p_path, ModeFlags p_mode_flags, CompressionMode p_compress_mode) {
	FileAccessCompressed *fac = memnew(FileAccessCompressed);

	fac->configure("GCPF", (Compression::Mode)p_compress_mode);

	Error err = fac->_open(p_path, p_mode_flags);

	if (err) {
		memdelete(fac);
		return err;
	}

	f = fac;
	return OK;
}

Error File::open(const String &p_path, ModeFlags p_mode_flags) {
	close();
	Error err;
	f = FileAccess::open(p_path, p_mode_flags, &err);
	if (f) {
		f->set_big_endian(big_endian);
	}
	return err;
}

void File::flush() {
	ERR_FAIL_COND_MSG(!f, "File must be opened before flushing.");
	f->flush();
}

void File::close() {
	if (f) {
		memdelete(f);
	}
	f = nullptr;
}

bool File::is_open() const {
	return f != nullptr;
}

String File::get_path() const {
	ERR_FAIL_COND_V_MSG(!f, "", "File must be opened before use, or is lacking read-write permission.");
	return f->get_path();
}

String File::get_path_absolute() const {
	ERR_FAIL_COND_V_MSG(!f, "", "File must be opened before use, or is lacking read-write permission.");
	return f->get_path_absolute();
}

void File::seek(int64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");
	ERR_FAIL_COND_MSG(p_position < 0, "Seek position must be a positive integer.");
	f->seek(p_position);
}

void File::seek_end(int64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");
	f->seek_end(p_position);
}

uint64_t File::get_position() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_position();
}

uint64_t File::get_length() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_length();
}

bool File::eof_reached() const {
	ERR_FAIL_COND_V_MSG(!f, false, "File must be opened before use, or is lacking read-write permission.");
	return f->eof_reached();
}

uint8_t File::get_8() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_8();
}

uint16_t File::get_16() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_16();
}

uint32_t File::get_32() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_32();
}

uint64_t File::get_64() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_64();
}

float File::get_float() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_float();
}

double File::get_double() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_double();
}

real_t File::get_real() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use, or is lacking read-write permission.");
	return f->get_real();
}

Vector<uint8_t> File::get_buffer(int64_t p_length) const {
	Vector<uint8_t> data;
	ERR_FAIL_COND_V_MSG(!f, data, "File must be opened before use, or is lacking read-write permission.");

	ERR_FAIL_COND_V_MSG(p_length < 0, data, "Length of buffer cannot be smaller than 0.");
	if (p_length == 0) {
		return data;
	}

	Error err = data.resize(p_length);
	ERR_FAIL_COND_V_MSG(err != OK, data, "Can't resize data to " + itos(p_length) + " elements.");

	uint8_t *w = data.ptrw();
	int64_t len = f->get_buffer(&w[0], p_length);

	if (len < p_length) {
		data.resize(len);
	}

	return data;
}

String File::get_as_text() const {
	ERR_FAIL_COND_V_MSG(!f, String(), "File must be opened before use, or is lacking read-write permission.");

	String text;
	uint64_t original_pos = f->get_position();
	f->seek(0);

	String l = get_line();
	while (!eof_reached()) {
		text += l + "\n";
		l = get_line();
	}
	text += l;

	f->seek(original_pos);

	return text;
}

String File::get_md5(const String &p_path) const {
	return FileAccess::get_md5(p_path);
}

String File::get_sha256(const String &p_path) const {
	return FileAccess::get_sha256(p_path);
}

String File::get_line() const {
	ERR_FAIL_COND_V_MSG(!f, String(), "File must be opened before use, or is lacking read-write permission.");
	return f->get_line();
}

Vector<String> File::get_csv_line(const String &p_delim) const {
	ERR_FAIL_COND_V_MSG(!f, Vector<String>(), "File must be opened before use, or is lacking read-write permission.");
	return f->get_csv_line(p_delim);
}

/**< use this for files WRITTEN in _big_ endian machines (i.e. amiga/mac)
 * It's not about the current CPU type but file formats.
 * These flags get reset to false (little endian) on each open
 */

void File::set_big_endian(bool p_big_endian) {
	big_endian = p_big_endian;
	if (f) {
		f->set_big_endian(p_big_endian);
	}
}

bool File::is_big_endian() {
	return big_endian;
}

Error File::get_error() const {
	if (!f) {
		return ERR_UNCONFIGURED;
	}
	return f->get_error();
}

void File::store_8(uint8_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_8(p_dest);
}

void File::store_16(uint16_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_16(p_dest);
}

void File::store_32(uint32_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_32(p_dest);
}

void File::store_64(uint64_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_64(p_dest);
}

void File::store_float(float p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_float(p_dest);
}

void File::store_double(double p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_double(p_dest);
}

void File::store_real(real_t p_real) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_real(p_real);
}

void File::store_string(const String &p_string) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_string(p_string);
}

void File::store_pascal_string(const String &p_string) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	f->store_pascal_string(p_string);
}

String File::get_pascal_string() {
	ERR_FAIL_COND_V_MSG(!f, "", "File must be opened before use, or is lacking read-write permission.");

	return f->get_pascal_string();
}

void File::store_line(const String &p_string) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");
	f->store_line(p_string);
}

void File::store_csv_line(const Vector<String> &p_values, const String &p_delim) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");
	f->store_csv_line(p_values, p_delim);
}

void File::store_buffer(const Vector<uint8_t> &p_buffer) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");

	uint64_t len = p_buffer.size();
	if (len == 0) {
		return;
	}

	const uint8_t *r = p_buffer.ptr();

	f->store_buffer(&r[0], len);
}

bool File::file_exists(const String &p_name) const {
	return FileAccess::exists(p_name);
}

void File::store_var(const Variant &p_var, bool p_full_objects) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use, or is lacking read-write permission.");
	int len;
	Error err = encode_variant(p_var, nullptr, len, p_full_objects);
	ERR_FAIL_COND_MSG(err != OK, "Error when trying to encode Variant.");

	Vector<uint8_t> buff;
	buff.resize(len);

	uint8_t *w = buff.ptrw();
	err = encode_variant(p_var, &w[0], len, p_full_objects);
	ERR_FAIL_COND_MSG(err != OK, "Error when trying to encode Variant.");

	store_32(len);
	store_buffer(buff);
}

Variant File::get_var(bool p_allow_objects) const {
	ERR_FAIL_COND_V_MSG(!f, Variant(), "File must be opened before use, or is lacking read-write permission.");
	uint32_t len = get_32();
	Vector<uint8_t> buff = get_buffer(len);
	ERR_FAIL_COND_V((uint32_t)buff.size() != len, Variant());

	const uint8_t *r = buff.ptr();

	Variant v;
	Error err = decode_variant(v, &r[0], len, nullptr, p_allow_objects);
	ERR_FAIL_COND_V_MSG(err != OK, Variant(), "Error when trying to encode Variant.");

	return v;
}

uint64_t File::get_modified_time(const String &p_file) const {
	return FileAccess::get_modified_time(p_file);
}

void File::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open_encrypted", "path", "mode_flags", "key"), &File::open_encrypted);
	ClassDB::bind_method(D_METHOD("open_encrypted_with_pass", "path", "mode_flags", "pass"), &File::open_encrypted_pass);
	ClassDB::bind_method(D_METHOD("open_compressed", "path", "mode_flags", "compression_mode"), &File::open_compressed, DEFVAL(0));

	ClassDB::bind_method(D_METHOD("open", "path", "flags"), &File::open);
	ClassDB::bind_method(D_METHOD("flush"), &File::flush);
	ClassDB::bind_method(D_METHOD("close"), &File::close);
	ClassDB::bind_method(D_METHOD("get_path"), &File::get_path);
	ClassDB::bind_method(D_METHOD("get_path_absolute"), &File::get_path_absolute);
	ClassDB::bind_method(D_METHOD("is_open"), &File::is_open);
	ClassDB::bind_method(D_METHOD("seek", "position"), &File::seek);
	ClassDB::bind_method(D_METHOD("seek_end", "position"), &File::seek_end, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_position"), &File::get_position);
	ClassDB::bind_method(D_METHOD("get_length"), &File::get_length);
	ClassDB::bind_method(D_METHOD("eof_reached"), &File::eof_reached);
	ClassDB::bind_method(D_METHOD("get_8"), &File::get_8);
	ClassDB::bind_method(D_METHOD("get_16"), &File::get_16);
	ClassDB::bind_method(D_METHOD("get_32"), &File::get_32);
	ClassDB::bind_method(D_METHOD("get_64"), &File::get_64);
	ClassDB::bind_method(D_METHOD("get_float"), &File::get_float);
	ClassDB::bind_method(D_METHOD("get_double"), &File::get_double);
	ClassDB::bind_method(D_METHOD("get_real"), &File::get_real);
	ClassDB::bind_method(D_METHOD("get_buffer", "length"), &File::get_buffer);
	ClassDB::bind_method(D_METHOD("get_line"), &File::get_line);
	ClassDB::bind_method(D_METHOD("get_csv_line", "delim"), &File::get_csv_line, DEFVAL(","));
	ClassDB::bind_method(D_METHOD("get_as_text"), &File::get_as_text);
	ClassDB::bind_method(D_METHOD("get_md5", "path"), &File::get_md5);
	ClassDB::bind_method(D_METHOD("get_sha256", "path"), &File::get_sha256);
	ClassDB::bind_method(D_METHOD("is_big_endian"), &File::is_big_endian);
	ClassDB::bind_method(D_METHOD("set_big_endian", "big_endian"), &File::set_big_endian);
	ClassDB::bind_method(D_METHOD("get_error"), &File::get_error);
	ClassDB::bind_method(D_METHOD("get_var", "allow_objects"), &File::get_var, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("store_8", "value"), &File::store_8);
	ClassDB::bind_method(D_METHOD("store_16", "value"), &File::store_16);
	ClassDB::bind_method(D_METHOD("store_32", "value"), &File::store_32);
	ClassDB::bind_method(D_METHOD("store_64", "value"), &File::store_64);
	ClassDB::bind_method(D_METHOD("store_float", "value"), &File::store_float);
	ClassDB::bind_method(D_METHOD("store_double", "value"), &File::store_double);
	ClassDB::bind_method(D_METHOD("store_real", "value"), &File::store_real);
	ClassDB::bind_method(D_METHOD("store_buffer", "buffer"), &File::store_buffer);
	ClassDB::bind_method(D_METHOD("store_line", "line"), &File::store_line);
	ClassDB::bind_method(D_METHOD("store_csv_line", "values", "delim"), &File::store_csv_line, DEFVAL(","));
	ClassDB::bind_method(D_METHOD("store_string", "string"), &File::store_string);
	ClassDB::bind_method(D_METHOD("store_var", "value", "full_objects"), &File::store_var, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("store_pascal_string", "string"), &File::store_pascal_string);
	ClassDB::bind_method(D_METHOD("get_pascal_string"), &File::get_pascal_string);

	ClassDB::bind_method(D_METHOD("file_exists", "path"), &File::file_exists);
	ClassDB::bind_method(D_METHOD("get_modified_time", "file"), &File::get_modified_time);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "big_endian"), "set_big_endian", "is_big_endian");

	BIND_ENUM_CONSTANT(READ);
	BIND_ENUM_CONSTANT(WRITE);
	BIND_ENUM_CONSTANT(READ_WRITE);
	BIND_ENUM_CONSTANT(WRITE_READ);

	BIND_ENUM_CONSTANT(COMPRESSION_FASTLZ);
	BIND_ENUM_CONSTANT(COMPRESSION_DEFLATE);
	BIND_ENUM_CONSTANT(COMPRESSION_ZSTD);
	BIND_ENUM_CONSTANT(COMPRESSION_GZIP);
}

File::~File() {
	if (f) {
		memdelete(f);
	}
}

////// Directory //////

Error Directory::open(const String &p_path) {
	Error err;
	DirAccess *alt = DirAccess::open(p_path, &err);

	if (!alt) {
		return err;
	}
	if (d) {
		memdelete(d);
	}
	d = alt;
	dir_open = true;

	return OK;
}

bool Directory::is_open() const {
	return d && dir_open;
}

Error Directory::list_dir_begin(bool p_show_navigational, bool p_show_hidden) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");

	_list_skip_navigational = !p_show_navigational;
	_list_skip_hidden = !p_show_hidden;

	return d->list_dir_begin();
}

String Directory::get_next() {
	ERR_FAIL_COND_V_MSG(!is_open(), "", "Directory must be opened before use.");

	String next = d->get_next();
	while (next != "" && ((_list_skip_navigational && (next == "." || next == "..")) || (_list_skip_hidden && d->current_is_hidden()))) {
		next = d->get_next();
	}
	return next;
}

bool Directory::current_is_dir() const {
	ERR_FAIL_COND_V_MSG(!is_open(), false, "Directory must be opened before use.");
	return d->current_is_dir();
}

void Directory::list_dir_end() {
	ERR_FAIL_COND_MSG(!is_open(), "Directory must be opened before use.");
	d->list_dir_end();
}

int Directory::get_drive_count() {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "Directory must be opened before use.");
	return d->get_drive_count();
}

String Directory::get_drive(int p_drive) {
	ERR_FAIL_COND_V_MSG(!is_open(), "", "Directory must be opened before use.");
	return d->get_drive(p_drive);
}

int Directory::get_current_drive() {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "Directory must be opened before use.");
	return d->get_current_drive();
}

Error Directory::change_dir(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, ERR_UNCONFIGURED, "Directory is not configured properly.");
	Error err = d->change_dir(p_dir);

	if (err != OK) {
		return err;
	}
	dir_open = true;

	return OK;
}

String Directory::get_current_dir() {
	ERR_FAIL_COND_V_MSG(!is_open(), "", "Directory must be opened before use.");
	return d->get_current_dir();
}

Error Directory::make_dir(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, ERR_UNCONFIGURED, "Directory is not configured properly.");
	if (!p_dir.is_relative_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		Error err = d->make_dir(p_dir);
		memdelete(d);
		return err;
	}
	return d->make_dir(p_dir);
}

Error Directory::make_dir_recursive(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, ERR_UNCONFIGURED, "Directory is not configured properly.");
	if (!p_dir.is_relative_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		Error err = d->make_dir_recursive(p_dir);
		memdelete(d);
		return err;
	}
	return d->make_dir_recursive(p_dir);
}

bool Directory::file_exists(String p_file) {
	ERR_FAIL_COND_V_MSG(!d, false, "Directory is not configured properly.");
	if (!p_file.is_relative_path()) {
		return FileAccess::exists(p_file);
	}

	return d->file_exists(p_file);
}

bool Directory::dir_exists(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, false, "Directory is not configured properly.");
	if (!p_dir.is_relative_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		bool exists = d->dir_exists(p_dir);
		memdelete(d);
		return exists;
	}

	return d->dir_exists(p_dir);
}

uint64_t Directory::get_space_left() {
	ERR_FAIL_COND_V_MSG(!d, 0, "Directory must be opened before use.");
	return d->get_space_left() / 1024 * 1024; // Truncate to closest MiB.
}

Error Directory::copy(String p_from, String p_to) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");
	return d->copy(p_from, p_to);
}

Error Directory::rename(String p_from, String p_to) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");
	ERR_FAIL_COND_V_MSG(p_from.is_empty() || p_from == "." || p_from == "..", ERR_INVALID_PARAMETER, "Invalid path to rename.");

	if (!p_from.is_relative_path()) {
		DirAccess *d = DirAccess::create_for_path(p_from);
		ERR_FAIL_COND_V_MSG(!d->file_exists(p_from) && !d->dir_exists(p_from), ERR_DOES_NOT_EXIST, "File or directory does not exist.");
		Error err = d->rename(p_from, p_to);
		memdelete(d);
		return err;
	}

	ERR_FAIL_COND_V_MSG(!d->file_exists(p_from) && !d->dir_exists(p_from), ERR_DOES_NOT_EXIST, "File or directory does not exist.");
	return d->rename(p_from, p_to);
}

Error Directory::remove(String p_name) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");
	if (!p_name.is_relative_path()) {
		DirAccess *d = DirAccess::create_for_path(p_name);
		Error err = d->remove(p_name);
		memdelete(d);
		return err;
	}

	return d->remove(p_name);
}

void Directory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path"), &Directory::open);
	ClassDB::bind_method(D_METHOD("list_dir_begin", "show_navigational", "show_hidden"), &Directory::list_dir_begin, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_next"), &Directory::get_next);
	ClassDB::bind_method(D_METHOD("current_is_dir"), &Directory::current_is_dir);
	ClassDB::bind_method(D_METHOD("list_dir_end"), &Directory::list_dir_end);
	ClassDB::bind_method(D_METHOD("get_drive_count"), &Directory::get_drive_count);
	ClassDB::bind_method(D_METHOD("get_drive", "idx"), &Directory::get_drive);
	ClassDB::bind_method(D_METHOD("get_current_drive"), &Directory::get_current_drive);
	ClassDB::bind_method(D_METHOD("change_dir", "todir"), &Directory::change_dir);
	ClassDB::bind_method(D_METHOD("get_current_dir"), &Directory::get_current_dir);
	ClassDB::bind_method(D_METHOD("make_dir", "path"), &Directory::make_dir);
	ClassDB::bind_method(D_METHOD("make_dir_recursive", "path"), &Directory::make_dir_recursive);
	ClassDB::bind_method(D_METHOD("file_exists", "path"), &Directory::file_exists);
	ClassDB::bind_method(D_METHOD("dir_exists", "path"), &Directory::dir_exists);
	//ClassDB::bind_method(D_METHOD("get_modified_time","file"),&Directory::get_modified_time);
	ClassDB::bind_method(D_METHOD("get_space_left"), &Directory::get_space_left);
	ClassDB::bind_method(D_METHOD("copy", "from", "to"), &Directory::copy);
	ClassDB::bind_method(D_METHOD("rename", "from", "to"), &Directory::rename);
	ClassDB::bind_method(D_METHOD("remove", "path"), &Directory::remove);
}

Directory::Directory() {
	d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
}

Directory::~Directory() {
	if (d) {
		memdelete(d);
	}
}

////// Marshalls //////

Marshalls *Marshalls::singleton = nullptr;

Marshalls *Marshalls::get_singleton() {
	return singleton;
}

String Marshalls::variant_to_base64(const Variant &p_var, bool p_full_objects) {
	int len;
	Error err = encode_variant(p_var, nullptr, len, p_full_objects);
	ERR_FAIL_COND_V_MSG(err != OK, "", "Error when trying to encode Variant.");

	Vector<uint8_t> buff;
	buff.resize(len);
	uint8_t *w = buff.ptrw();

	err = encode_variant(p_var, &w[0], len, p_full_objects);
	ERR_FAIL_COND_V_MSG(err != OK, "", "Error when trying to encode Variant.");

	String ret = CryptoCore::b64_encode_str(&w[0], len);
	ERR_FAIL_COND_V(ret == "", ret);

	return ret;
}

Variant Marshalls::base64_to_variant(const String &p_str, bool p_allow_objects) {
	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1);
	uint8_t *w = buf.ptrw();

	size_t len = 0;
	ERR_FAIL_COND_V(CryptoCore::b64_decode(&w[0], buf.size(), &len, (unsigned char *)cstr.get_data(), strlen) != OK, Variant());

	Variant v;
	Error err = decode_variant(v, &w[0], len, nullptr, p_allow_objects);
	ERR_FAIL_COND_V_MSG(err != OK, Variant(), "Error when trying to decode Variant.");

	return v;
}

String Marshalls::raw_to_base64(const Vector<uint8_t> &p_arr) {
	String ret = CryptoCore::b64_encode_str(p_arr.ptr(), p_arr.size());
	ERR_FAIL_COND_V(ret == "", ret);
	return ret;
}

Vector<uint8_t> Marshalls::base64_to_raw(const String &p_str) {
	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	size_t arr_len = 0;
	Vector<uint8_t> buf;
	{
		buf.resize(strlen / 4 * 3 + 1);
		uint8_t *w = buf.ptrw();

		ERR_FAIL_COND_V(CryptoCore::b64_decode(&w[0], buf.size(), &arr_len, (unsigned char *)cstr.get_data(), strlen) != OK, Vector<uint8_t>());
	}
	buf.resize(arr_len);

	return buf;
}

String Marshalls::utf8_to_base64(const String &p_str) {
	CharString cstr = p_str.utf8();
	String ret = CryptoCore::b64_encode_str((unsigned char *)cstr.get_data(), cstr.length());
	ERR_FAIL_COND_V(ret == "", ret);
	return ret;
}

String Marshalls::base64_to_utf8(const String &p_str) {
	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1 + 1);
	uint8_t *w = buf.ptrw();

	size_t len = 0;
	ERR_FAIL_COND_V(CryptoCore::b64_decode(&w[0], buf.size(), &len, (unsigned char *)cstr.get_data(), strlen) != OK, String());

	w[len] = 0;
	String ret = String::utf8((char *)&w[0]);

	return ret;
}

void Marshalls::_bind_methods() {
	ClassDB::bind_method(D_METHOD("variant_to_base64", "variant", "full_objects"), &Marshalls::variant_to_base64, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("base64_to_variant", "base64_str", "allow_objects"), &Marshalls::base64_to_variant, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("raw_to_base64", "array"), &Marshalls::raw_to_base64);
	ClassDB::bind_method(D_METHOD("base64_to_raw", "base64_str"), &Marshalls::base64_to_raw);

	ClassDB::bind_method(D_METHOD("utf8_to_base64", "utf8_str"), &Marshalls::utf8_to_base64);
	ClassDB::bind_method(D_METHOD("base64_to_utf8", "base64_str"), &Marshalls::base64_to_utf8);
}

////// Semaphore //////

void Semaphore::wait() {
	semaphore.wait();
}

Error Semaphore::try_wait() {
	return semaphore.try_wait() ? OK : ERR_BUSY;
}

void Semaphore::post() {
	semaphore.post();
}

void Semaphore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("wait"), &Semaphore::wait);
	ClassDB::bind_method(D_METHOD("try_wait"), &Semaphore::try_wait);
	ClassDB::bind_method(D_METHOD("post"), &Semaphore::post);
}

////// Mutex //////

void Mutex::lock() {
	mutex.lock();
}

Error Mutex::try_lock() {
	return mutex.try_lock();
}

void Mutex::unlock() {
	mutex.unlock();
}

void Mutex::_bind_methods() {
	ClassDB::bind_method(D_METHOD("lock"), &Mutex::lock);
	ClassDB::bind_method(D_METHOD("try_lock"), &Mutex::try_lock);
	ClassDB::bind_method(D_METHOD("unlock"), &Mutex::unlock);
}

////// Thread //////

void Thread::_start_func(void *ud) {
	Ref<Thread> *tud = (Ref<Thread> *)ud;
	Ref<Thread> t = *tud;
	memdelete(tud);

	Object *target_instance = t->target_callable.get_object();
	if (!target_instance) {
		t->running.clear();
		ERR_FAIL_MSG(vformat("Could not call function '%s' on previously freed instance to start thread %s.", t->target_callable.get_method(), t->get_id()));
	}

	Callable::CallError ce;
	const Variant *arg[1] = { &t->userdata };
	int argc = 0;
	if (arg[0]->get_type() != Variant::NIL) {
		// Just pass to the target function whatever came as user data
		argc = 1;
	} else {
		// There are two cases of null user data:
		// a) The target function has zero parameters and the caller is just honoring that.
		// b) The target function has at least one parameter with no default and the caller is
		//    leveraging the fact that user data defaults to null in Thread.start().
		//    We care about the case of more than one parameter because, even if a thread
		//    function can have one at most, out mindset here is to do our best with the
		//    only/first one and let the call handle any other error conditions, like too
		//    much arguments.
		// We must check if we are in case b).
		int target_param_count = 0;
		int target_default_arg_count = 0;
		Ref<Script> script = target_instance->get_script();
		if (script.is_valid()) {
			MethodInfo mi = script->get_method_info(t->target_callable.get_method());
			target_param_count = mi.arguments.size();
			target_default_arg_count = mi.default_arguments.size();
		} else {
			MethodBind *method = ClassDB::get_method(target_instance->get_class_name(), t->target_callable.get_method());
			if (method) {
				target_param_count = method->get_argument_count();
				target_default_arg_count = method->get_default_argument_count();
			}
		}
		if (target_param_count >= 1 && target_default_arg_count < target_param_count) {
			argc = 1;
		}
	}

	::Thread::set_name(t->target_callable.get_method());

	t->target_callable.call(arg, argc, t->ret, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		t->running.clear();
		ERR_FAIL_MSG("Could not call function '" + t->target_callable.get_method().operator String() + "' to start thread " + t->get_id() + ": " + Variant::get_callable_error_text(t->target_callable, arg, argc, ce) + ".");
	}

	t->running.clear();
}

Error Thread::start(const Callable &p_callable, const Variant &p_userdata, Priority p_priority) {
	ERR_FAIL_COND_V_MSG(is_started(), ERR_ALREADY_IN_USE, "Thread already started.");
	ERR_FAIL_COND_V(p_callable.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_priority, PRIORITY_MAX, ERR_INVALID_PARAMETER);

	ret = Variant();
	target_callable = p_callable;
	userdata = p_userdata;
	running.set();

	Ref<Thread> *ud = memnew(Ref<Thread>(this));

	::Thread::Settings s;
	s.priority = (::Thread::Priority)p_priority;
	thread.start(_start_func, ud, s);

	return OK;
}

String Thread::get_id() const {
	return itos(thread.get_id());
}

bool Thread::is_started() const {
	return thread.is_started();
}

bool Thread::is_alive() const {
	return running.is_set();
}

Variant Thread::wait_to_finish() {
	ERR_FAIL_COND_V_MSG(!is_started(), Variant(), "Thread must have been started to wait for its completion.");
	thread.wait_to_finish();
	Variant r = ret;
	target_callable = Callable();
	userdata = Variant();

	return r;
}

void Thread::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start", "callable", "userdata", "priority"), &Thread::start, DEFVAL(Variant()), DEFVAL(PRIORITY_NORMAL));
	ClassDB::bind_method(D_METHOD("get_id"), &Thread::get_id);
	ClassDB::bind_method(D_METHOD("is_started"), &Thread::is_started);
	ClassDB::bind_method(D_METHOD("is_alive"), &Thread::is_alive);
	ClassDB::bind_method(D_METHOD("wait_to_finish"), &Thread::wait_to_finish);

	BIND_ENUM_CONSTANT(PRIORITY_LOW);
	BIND_ENUM_CONSTANT(PRIORITY_NORMAL);
	BIND_ENUM_CONSTANT(PRIORITY_HIGH);
}

namespace special {

////// ClassDB //////

PackedStringArray ClassDB::get_class_list() const {
	List<StringName> classes;
	::ClassDB::get_class_list(&classes);

	PackedStringArray ret;
	ret.resize(classes.size());
	int idx = 0;
	for (const StringName &E : classes) {
		ret.set(idx++, E);
	}

	return ret;
}

PackedStringArray ClassDB::get_inheriters_from_class(const StringName &p_class) const {
	List<StringName> classes;
	::ClassDB::get_inheriters_from_class(p_class, &classes);

	PackedStringArray ret;
	ret.resize(classes.size());
	int idx = 0;
	for (const StringName &E : classes) {
		ret.set(idx++, E);
	}

	return ret;
}

StringName ClassDB::get_parent_class(const StringName &p_class) const {
	return ::ClassDB::get_parent_class(p_class);
}

bool ClassDB::class_exists(const StringName &p_class) const {
	return ::ClassDB::class_exists(p_class);
}

bool ClassDB::is_parent_class(const StringName &p_class, const StringName &p_inherits) const {
	return ::ClassDB::is_parent_class(p_class, p_inherits);
}

bool ClassDB::can_instantiate(const StringName &p_class) const {
	return ::ClassDB::can_instantiate(p_class);
}

Variant ClassDB::instantiate(const StringName &p_class) const {
	Object *obj = ::ClassDB::instantiate(p_class);
	if (!obj) {
		return Variant();
	}

	RefCounted *r = Object::cast_to<RefCounted>(obj);
	if (r) {
		return REF(r);
	} else {
		return obj;
	}
}

bool ClassDB::has_signal(StringName p_class, StringName p_signal) const {
	return ::ClassDB::has_signal(p_class, p_signal);
}

Dictionary ClassDB::get_signal(StringName p_class, StringName p_signal) const {
	MethodInfo signal;
	if (::ClassDB::get_signal(p_class, p_signal, &signal)) {
		return signal.operator Dictionary();
	} else {
		return Dictionary();
	}
}

Array ClassDB::get_signal_list(StringName p_class, bool p_no_inheritance) const {
	List<MethodInfo> signals;
	::ClassDB::get_signal_list(p_class, &signals, p_no_inheritance);
	Array ret;

	for (const MethodInfo &E : signals) {
		ret.push_back(E.operator Dictionary());
	}

	return ret;
}

Array ClassDB::get_property_list(StringName p_class, bool p_no_inheritance) const {
	List<PropertyInfo> plist;
	::ClassDB::get_property_list(p_class, &plist, p_no_inheritance);
	Array ret;
	for (const PropertyInfo &E : plist) {
		ret.push_back(E.operator Dictionary());
	}

	return ret;
}

Variant ClassDB::get_property(Object *p_object, const StringName &p_property) const {
	Variant ret;
	::ClassDB::get_property(p_object, p_property, ret);
	return ret;
}

Error ClassDB::set_property(Object *p_object, const StringName &p_property, const Variant &p_value) const {
	Variant ret;
	bool valid;
	if (!::ClassDB::set_property(p_object, p_property, p_value, &valid)) {
		return ERR_UNAVAILABLE;
	} else if (!valid) {
		return ERR_INVALID_DATA;
	}
	return OK;
}

bool ClassDB::has_method(StringName p_class, StringName p_method, bool p_no_inheritance) const {
	return ::ClassDB::has_method(p_class, p_method, p_no_inheritance);
}

Array ClassDB::get_method_list(StringName p_class, bool p_no_inheritance) const {
	List<MethodInfo> methods;
	::ClassDB::get_method_list(p_class, &methods, p_no_inheritance);
	Array ret;

	for (const MethodInfo &E : methods) {
#ifdef DEBUG_METHODS_ENABLED
		ret.push_back(E.operator Dictionary());
#else
		Dictionary dict;
		dict["name"] = E.name;
		ret.push_back(dict);
#endif
	}

	return ret;
}

PackedStringArray ClassDB::get_integer_constant_list(const StringName &p_class, bool p_no_inheritance) const {
	List<String> constants;
	::ClassDB::get_integer_constant_list(p_class, &constants, p_no_inheritance);

	PackedStringArray ret;
	ret.resize(constants.size());
	int idx = 0;
	for (const String &E : constants) {
		ret.set(idx++, E);
	}

	return ret;
}

bool ClassDB::has_integer_constant(const StringName &p_class, const StringName &p_name) const {
	bool success;
	::ClassDB::get_integer_constant(p_class, p_name, &success);
	return success;
}

int ClassDB::get_integer_constant(const StringName &p_class, const StringName &p_name) const {
	bool found;
	int c = ::ClassDB::get_integer_constant(p_class, p_name, &found);
	ERR_FAIL_COND_V(!found, 0);
	return c;
}

StringName ClassDB::get_category(const StringName &p_node) const {
	return ::ClassDB::get_category(p_node);
}

bool ClassDB::has_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance) const {
	return ::ClassDB::has_enum(p_class, p_name, p_no_inheritance);
}

PackedStringArray ClassDB::get_enum_list(const StringName &p_class, bool p_no_inheritance) const {
	List<StringName> enums;
	::ClassDB::get_enum_list(p_class, &enums, p_no_inheritance);

	PackedStringArray ret;
	ret.resize(enums.size());
	int idx = 0;
	for (const StringName &E : enums) {
		ret.set(idx++, E);
	}

	return ret;
}

PackedStringArray ClassDB::get_enum_constants(const StringName &p_class, const StringName &p_enum, bool p_no_inheritance) const {
	List<StringName> constants;
	::ClassDB::get_enum_constants(p_class, p_enum, &constants, p_no_inheritance);

	PackedStringArray ret;
	ret.resize(constants.size());
	int idx = 0;
	for (const StringName &E : constants) {
		ret.set(idx++, E);
	}

	return ret;
}

StringName ClassDB::get_integer_constant_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance) const {
	return ::ClassDB::get_integer_constant_enum(p_class, p_name, p_no_inheritance);
}

bool ClassDB::is_class_enabled(StringName p_class) const {
	return ::ClassDB::is_class_enabled(p_class);
}

void ClassDB::_bind_methods() {
	::ClassDB::bind_method(D_METHOD("get_class_list"), &ClassDB::get_class_list);
	::ClassDB::bind_method(D_METHOD("get_inheriters_from_class", "class"), &ClassDB::get_inheriters_from_class);
	::ClassDB::bind_method(D_METHOD("get_parent_class", "class"), &ClassDB::get_parent_class);
	::ClassDB::bind_method(D_METHOD("class_exists", "class"), &ClassDB::class_exists);
	::ClassDB::bind_method(D_METHOD("is_parent_class", "class", "inherits"), &ClassDB::is_parent_class);
	::ClassDB::bind_method(D_METHOD("can_instantiate", "class"), &ClassDB::can_instantiate);
	::ClassDB::bind_method(D_METHOD("instantiate", "class"), &ClassDB::instantiate);

	::ClassDB::bind_method(D_METHOD("class_has_signal", "class", "signal"), &ClassDB::has_signal);
	::ClassDB::bind_method(D_METHOD("class_get_signal", "class", "signal"), &ClassDB::get_signal);
	::ClassDB::bind_method(D_METHOD("class_get_signal_list", "class", "no_inheritance"), &ClassDB::get_signal_list, DEFVAL(false));

	::ClassDB::bind_method(D_METHOD("class_get_property_list", "class", "no_inheritance"), &ClassDB::get_property_list, DEFVAL(false));
	::ClassDB::bind_method(D_METHOD("class_get_property", "object", "property"), &ClassDB::get_property);
	::ClassDB::bind_method(D_METHOD("class_set_property", "object", "property", "value"), &ClassDB::set_property);

	::ClassDB::bind_method(D_METHOD("class_has_method", "class", "method", "no_inheritance"), &ClassDB::has_method, DEFVAL(false));

	::ClassDB::bind_method(D_METHOD("class_get_method_list", "class", "no_inheritance"), &ClassDB::get_method_list, DEFVAL(false));

	::ClassDB::bind_method(D_METHOD("class_get_integer_constant_list", "class", "no_inheritance"), &ClassDB::get_integer_constant_list, DEFVAL(false));

	::ClassDB::bind_method(D_METHOD("class_has_integer_constant", "class", "name"), &ClassDB::has_integer_constant);
	::ClassDB::bind_method(D_METHOD("class_get_integer_constant", "class", "name"), &ClassDB::get_integer_constant);

	::ClassDB::bind_method(D_METHOD("class_has_enum", "class", "name", "no_inheritance"), &ClassDB::has_enum, DEFVAL(false));
	::ClassDB::bind_method(D_METHOD("class_get_enum_list", "class", "no_inheritance"), &ClassDB::get_enum_list, DEFVAL(false));
	::ClassDB::bind_method(D_METHOD("class_get_enum_constants", "class", "enum", "no_inheritance"), &ClassDB::get_enum_constants, DEFVAL(false));
	::ClassDB::bind_method(D_METHOD("class_get_integer_constant_enum", "class", "name", "no_inheritance"), &ClassDB::get_integer_constant_enum, DEFVAL(false));

	::ClassDB::bind_method(D_METHOD("class_get_category", "class"), &ClassDB::get_category);
	::ClassDB::bind_method(D_METHOD("is_class_enabled", "class"), &ClassDB::is_class_enabled);
}

} // namespace special

////// Engine //////

void Engine::set_physics_ticks_per_second(int p_ips) {
	::Engine::get_singleton()->set_physics_ticks_per_second(p_ips);
}

int Engine::get_physics_ticks_per_second() const {
	return ::Engine::get_singleton()->get_physics_ticks_per_second();
}

void Engine::set_physics_jitter_fix(double p_threshold) {
	::Engine::get_singleton()->set_physics_jitter_fix(p_threshold);
}

double Engine::get_physics_jitter_fix() const {
	return ::Engine::get_singleton()->get_physics_jitter_fix();
}

double Engine::get_physics_interpolation_fraction() const {
	return ::Engine::get_singleton()->get_physics_interpolation_fraction();
}

void Engine::set_target_fps(int p_fps) {
	::Engine::get_singleton()->set_target_fps(p_fps);
}

int Engine::get_target_fps() const {
	return ::Engine::get_singleton()->get_target_fps();
}

double Engine::get_frames_per_second() const {
	return ::Engine::get_singleton()->get_frames_per_second();
}

uint64_t Engine::get_physics_frames() const {
	return ::Engine::get_singleton()->get_physics_frames();
}

uint64_t Engine::get_process_frames() const {
	return ::Engine::get_singleton()->get_process_frames();
}

void Engine::set_time_scale(double p_scale) {
	::Engine::get_singleton()->set_time_scale(p_scale);
}

double Engine::get_time_scale() {
	return ::Engine::get_singleton()->get_time_scale();
}

int Engine::get_frames_drawn() {
	return ::Engine::get_singleton()->get_frames_drawn();
}

MainLoop *Engine::get_main_loop() const {
	// Needs to remain in OS, since it's actually OS that interacts with it, but it's better exposed here
	return ::OS::get_singleton()->get_main_loop();
}

Dictionary Engine::get_version_info() const {
	return ::Engine::get_singleton()->get_version_info();
}

Dictionary Engine::get_author_info() const {
	return ::Engine::get_singleton()->get_author_info();
}

Array Engine::get_copyright_info() const {
	return ::Engine::get_singleton()->get_copyright_info();
}

Dictionary Engine::get_donor_info() const {
	return ::Engine::get_singleton()->get_donor_info();
}

Dictionary Engine::get_license_info() const {
	return ::Engine::get_singleton()->get_license_info();
}

String Engine::get_license_text() const {
	return ::Engine::get_singleton()->get_license_text();
}

bool Engine::is_in_physics_frame() const {
	return ::Engine::get_singleton()->is_in_physics_frame();
}

bool Engine::has_singleton(const StringName &p_name) const {
	return ::Engine::get_singleton()->has_singleton(p_name);
}

Object *Engine::get_singleton_object(const StringName &p_name) const {
	return ::Engine::get_singleton()->get_singleton_object(p_name);
}

void Engine::register_singleton(const StringName &p_name, Object *p_object) {
	ERR_FAIL_COND_MSG(has_singleton(p_name), "Singleton already registered: " + String(p_name));
	ERR_FAIL_COND_MSG(!String(p_name).is_valid_identifier(), "Singleton name is not a valid identifier: " + p_name);
	::Engine::Singleton s;
	s.class_name = p_name;
	s.name = p_name;
	s.ptr = p_object;
	s.user_created = true;
	::Engine::get_singleton()->add_singleton(s);
	;
}
void Engine::unregister_singleton(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!has_singleton(p_name), "Attempt to remove unregisteres singleton: " + String(p_name));
	ERR_FAIL_COND_MSG(!::Engine::get_singleton()->is_singleton_user_created(p_name), "Attempt to remove non-user created singleton: " + String(p_name));
	::Engine::get_singleton()->remove_singleton(p_name);
}

Vector<String> Engine::get_singleton_list() const {
	List<::Engine::Singleton> singletons;
	::Engine::get_singleton()->get_singletons(&singletons);
	Vector<String> ret;
	for (List<::Engine::Singleton>::Element *E = singletons.front(); E; E = E->next()) {
		ret.push_back(E->get().name);
	}
	return ret;
}

void Engine::set_editor_hint(bool p_enabled) {
	::Engine::get_singleton()->set_editor_hint(p_enabled);
}

bool Engine::is_editor_hint() const {
	return ::Engine::get_singleton()->is_editor_hint();
}

void Engine::set_print_error_messages(bool p_enabled) {
	::Engine::get_singleton()->set_print_error_messages(p_enabled);
}

bool Engine::is_printing_error_messages() const {
	return ::Engine::get_singleton()->is_printing_error_messages();
}

void Engine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_physics_ticks_per_second", "physics_ticks_per_second"), &Engine::set_physics_ticks_per_second);
	ClassDB::bind_method(D_METHOD("get_physics_ticks_per_second"), &Engine::get_physics_ticks_per_second);
	ClassDB::bind_method(D_METHOD("set_physics_jitter_fix", "physics_jitter_fix"), &Engine::set_physics_jitter_fix);
	ClassDB::bind_method(D_METHOD("get_physics_jitter_fix"), &Engine::get_physics_jitter_fix);
	ClassDB::bind_method(D_METHOD("get_physics_interpolation_fraction"), &Engine::get_physics_interpolation_fraction);
	ClassDB::bind_method(D_METHOD("set_target_fps", "target_fps"), &Engine::set_target_fps);
	ClassDB::bind_method(D_METHOD("get_target_fps"), &Engine::get_target_fps);

	ClassDB::bind_method(D_METHOD("set_time_scale", "time_scale"), &Engine::set_time_scale);
	ClassDB::bind_method(D_METHOD("get_time_scale"), &Engine::get_time_scale);

	ClassDB::bind_method(D_METHOD("get_frames_drawn"), &Engine::get_frames_drawn);
	ClassDB::bind_method(D_METHOD("get_frames_per_second"), &Engine::get_frames_per_second);
	ClassDB::bind_method(D_METHOD("get_physics_frames"), &Engine::get_physics_frames);
	ClassDB::bind_method(D_METHOD("get_process_frames"), &Engine::get_process_frames);

	ClassDB::bind_method(D_METHOD("get_main_loop"), &Engine::get_main_loop);

	ClassDB::bind_method(D_METHOD("get_version_info"), &Engine::get_version_info);
	ClassDB::bind_method(D_METHOD("get_author_info"), &Engine::get_author_info);
	ClassDB::bind_method(D_METHOD("get_copyright_info"), &Engine::get_copyright_info);
	ClassDB::bind_method(D_METHOD("get_donor_info"), &Engine::get_donor_info);
	ClassDB::bind_method(D_METHOD("get_license_info"), &Engine::get_license_info);
	ClassDB::bind_method(D_METHOD("get_license_text"), &Engine::get_license_text);

	ClassDB::bind_method(D_METHOD("is_in_physics_frame"), &Engine::is_in_physics_frame);

	ClassDB::bind_method(D_METHOD("has_singleton", "name"), &Engine::has_singleton);
	ClassDB::bind_method(D_METHOD("get_singleton", "name"), &Engine::get_singleton_object);

	ClassDB::bind_method(D_METHOD("register_singleton", "name", "instance"), &Engine::register_singleton);
	ClassDB::bind_method(D_METHOD("unregister_singleton", "name"), &Engine::unregister_singleton);
	ClassDB::bind_method(D_METHOD("get_singleton_list"), &Engine::get_singleton_list);

	ClassDB::bind_method(D_METHOD("is_editor_hint"), &Engine::is_editor_hint);

	ClassDB::bind_method(D_METHOD("set_print_error_messages", "enabled"), &Engine::set_print_error_messages);
	ClassDB::bind_method(D_METHOD("is_printing_error_messages"), &Engine::is_printing_error_messages);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "print_error_messages"), "set_print_error_messages", "is_printing_error_messages");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "physics_ticks_per_second"), "set_physics_ticks_per_second", "get_physics_ticks_per_second");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "target_fps"), "set_target_fps", "get_target_fps");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_scale"), "set_time_scale", "get_time_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "physics_jitter_fix"), "set_physics_jitter_fix", "get_physics_jitter_fix");
}

Engine *Engine::singleton = nullptr;

////// EngineDebugger //////

bool EngineDebugger::is_active() {
	return ::EngineDebugger::is_active();
}

void EngineDebugger::register_profiler(const StringName &p_name, const Callable &p_toggle, const Callable &p_add, const Callable &p_tick) {
	ERR_FAIL_COND_MSG(profilers.has(p_name) || has_profiler(p_name), "Profiler already registered: " + p_name);
	profilers.insert(p_name, ProfilerCallable(p_toggle, p_add, p_tick));
	ProfilerCallable &p = profilers[p_name];
	::EngineDebugger::Profiler profiler(
			&p,
			&EngineDebugger::call_toggle,
			&EngineDebugger::call_add,
			&EngineDebugger::call_tick);
	::EngineDebugger::register_profiler(p_name, profiler);
}

void EngineDebugger::unregister_profiler(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!profilers.has(p_name), "Profiler not registered: " + p_name);
	::EngineDebugger::unregister_profiler(p_name);
	profilers.erase(p_name);
}

bool EngineDebugger::is_profiling(const StringName &p_name) {
	return ::EngineDebugger::is_profiling(p_name);
}

bool EngineDebugger::has_profiler(const StringName &p_name) {
	return ::EngineDebugger::has_profiler(p_name);
}

void EngineDebugger::profiler_add_frame_data(const StringName &p_name, const Array &p_data) {
	::EngineDebugger::profiler_add_frame_data(p_name, p_data);
}

void EngineDebugger::profiler_enable(const StringName &p_name, bool p_enabled, const Array &p_opts) {
	if (::EngineDebugger::get_singleton()) {
		::EngineDebugger::get_singleton()->profiler_enable(p_name, p_enabled, p_opts);
	}
}

void EngineDebugger::register_message_capture(const StringName &p_name, const Callable &p_callable) {
	ERR_FAIL_COND_MSG(captures.has(p_name) || has_capture(p_name), "Capture already registered: " + p_name);
	captures.insert(p_name, p_callable);
	Callable &c = captures[p_name];
	::EngineDebugger::Capture capture(&c, &EngineDebugger::call_capture);
	::EngineDebugger::register_message_capture(p_name, capture);
}

void EngineDebugger::unregister_message_capture(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!captures.has(p_name), "Capture not registered: " + p_name);
	::EngineDebugger::unregister_message_capture(p_name);
	captures.erase(p_name);
}

bool EngineDebugger::has_capture(const StringName &p_name) {
	return ::EngineDebugger::has_capture(p_name);
}

void EngineDebugger::send_message(const String &p_msg, const Array &p_data) {
	ERR_FAIL_COND_MSG(!::EngineDebugger::is_active(), "Can't send message. No active debugger");
	::EngineDebugger::get_singleton()->send_message(p_msg, p_data);
}

void EngineDebugger::call_toggle(void *p_user, bool p_enable, const Array &p_opts) {
	Callable &toggle = ((ProfilerCallable *)p_user)->callable_toggle;
	if (toggle.is_null()) {
		return;
	}
	Variant enable = p_enable, opts = p_opts;
	const Variant *args[2] = { &enable, &opts };
	Variant retval;
	Callable::CallError err;
	toggle.call(args, 2, retval, err);
	ERR_FAIL_COND_MSG(err.error != Callable::CallError::CALL_OK, "Error calling 'toggle' to callable: " + Variant::get_callable_error_text(toggle, args, 2, err));
}

void EngineDebugger::call_add(void *p_user, const Array &p_data) {
	Callable &add = ((ProfilerCallable *)p_user)->callable_add;
	if (add.is_null()) {
		return;
	}
	Variant data = p_data;
	const Variant *args[1] = { &data };
	Variant retval;
	Callable::CallError err;
	add.call(args, 1, retval, err);
	ERR_FAIL_COND_MSG(err.error != Callable::CallError::CALL_OK, "Error calling 'add' to callable: " + Variant::get_callable_error_text(add, args, 1, err));
}

void EngineDebugger::call_tick(void *p_user, double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time) {
	Callable &tick = ((ProfilerCallable *)p_user)->callable_tick;
	if (tick.is_null()) {
		return;
	}
	Variant frame_time = p_frame_time, idle_time = p_idle_time, physics_time = p_physics_time, physics_frame_time = p_physics_frame_time;
	const Variant *args[4] = { &frame_time, &idle_time, &physics_time, &physics_frame_time };
	Variant retval;
	Callable::CallError err;
	tick.call(args, 4, retval, err);
	ERR_FAIL_COND_MSG(err.error != Callable::CallError::CALL_OK, "Error calling 'tick' to callable: " + Variant::get_callable_error_text(tick, args, 4, err));
}

Error EngineDebugger::call_capture(void *p_user, const String &p_cmd, const Array &p_data, bool &r_captured) {
	Callable &capture = *(Callable *)p_user;
	if (capture.is_null()) {
		return FAILED;
	}
	Variant cmd = p_cmd, data = p_data;
	const Variant *args[2] = { &cmd, &data };
	Variant retval;
	Callable::CallError err;
	capture.call(args, 2, retval, err);
	ERR_FAIL_COND_V_MSG(err.error != Callable::CallError::CALL_OK, FAILED, "Error calling 'capture' to callable: " + Variant::get_callable_error_text(capture, args, 2, err));
	ERR_FAIL_COND_V_MSG(retval.get_type() != Variant::BOOL, FAILED, "Error calling 'capture' to callable: " + String(capture) + ". Return type is not bool.");
	r_captured = retval;
	return OK;
}

EngineDebugger::~EngineDebugger() {
	for (const KeyValue<StringName, Callable> &E : captures) {
		::EngineDebugger::unregister_message_capture(E.key);
	}
	captures.clear();
	for (const KeyValue<StringName, ProfilerCallable> &E : profilers) {
		::EngineDebugger::unregister_profiler(E.key);
	}
	profilers.clear();
}

EngineDebugger *EngineDebugger::singleton = nullptr;

void EngineDebugger::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_active"), &EngineDebugger::is_active);

	ClassDB::bind_method(D_METHOD("register_profiler", "name", "toggle", "add", "tick"), &EngineDebugger::register_profiler);
	ClassDB::bind_method(D_METHOD("unregister_profiler", "name"), &EngineDebugger::unregister_profiler);
	ClassDB::bind_method(D_METHOD("is_profiling", "name"), &EngineDebugger::is_profiling);
	ClassDB::bind_method(D_METHOD("has_profiler", "name"), &EngineDebugger::has_profiler);

	ClassDB::bind_method(D_METHOD("profiler_add_frame_data", "name", "data"), &EngineDebugger::profiler_add_frame_data);
	ClassDB::bind_method(D_METHOD("profiler_enable", "name", "enable", "arguments"), &EngineDebugger::profiler_enable, DEFVAL(Array()));

	ClassDB::bind_method(D_METHOD("register_message_capture", "name", "callable"), &EngineDebugger::register_message_capture);
	ClassDB::bind_method(D_METHOD("unregister_message_capture", "name"), &EngineDebugger::unregister_message_capture);
	ClassDB::bind_method(D_METHOD("has_capture", "name"), &EngineDebugger::has_capture);

	ClassDB::bind_method(D_METHOD("send_message", "message", "data"), &EngineDebugger::send_message);
}

} // namespace core_bind
