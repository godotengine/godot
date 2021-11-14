/*************************************************************************/
/*  core_bind.h                                                          */
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

#ifndef CORE_BIND_H
#define CORE_BIND_H

#include "core/io/compression.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/safe_refcount.h"

class MainLoop;

namespace core_bind {

class ResourceLoader : public Object {
	GDCLASS(ResourceLoader, Object);

protected:
	static void _bind_methods();
	static ResourceLoader *singleton;

public:
	enum ThreadLoadStatus {
		THREAD_LOAD_INVALID_RESOURCE,
		THREAD_LOAD_IN_PROGRESS,
		THREAD_LOAD_FAILED,
		THREAD_LOAD_LOADED
	};

	enum CacheMode {
		CACHE_MODE_IGNORE, // Resource and subresources do not use path cache, no path is set into resource.
		CACHE_MODE_REUSE, // Resource and subresources use patch cache, reuse existing loaded resources instead of loading from disk when available.
		CACHE_MODE_REPLACE, // Resource and subresource use path cache, but replace existing loaded resources when available with information from disk.
	};

	static ResourceLoader *get_singleton() { return singleton; }

	Error load_threaded_request(const String &p_path, const String &p_type_hint = "", bool p_use_sub_threads = false);
	ThreadLoadStatus load_threaded_get_status(const String &p_path, Array r_progress = Array());
	RES load_threaded_get(const String &p_path);

	RES load(const String &p_path, const String &p_type_hint = "", CacheMode p_cache_mode = CACHE_MODE_REUSE);
	Vector<String> get_recognized_extensions_for_type(const String &p_type);
	void set_abort_on_missing_resources(bool p_abort);
	PackedStringArray get_dependencies(const String &p_path);
	bool has_cached(const String &p_path);
	bool exists(const String &p_path, const String &p_type_hint = "");
	ResourceUID::ID get_resource_uid(const String &p_path);

	ResourceLoader() { singleton = this; }
};

class ResourceSaver : public Object {
	GDCLASS(ResourceSaver, Object);

protected:
	static void _bind_methods();
	static ResourceSaver *singleton;

public:
	enum SaverFlags {
		FLAG_RELATIVE_PATHS = 1,
		FLAG_BUNDLE_RESOURCES = 2,
		FLAG_CHANGE_PATH = 4,
		FLAG_OMIT_EDITOR_PROPERTIES = 8,
		FLAG_SAVE_BIG_ENDIAN = 16,
		FLAG_COMPRESS = 32,
		FLAG_REPLACE_SUBRESOURCE_PATHS = 64,
	};

	static ResourceSaver *get_singleton() { return singleton; }

	Error save(const String &p_path, const RES &p_resource, SaverFlags p_flags);
	Vector<String> get_recognized_extensions(const RES &p_resource);

	ResourceSaver() { singleton = this; }
};

class OS : public Object {
	GDCLASS(OS, Object);

protected:
	static void _bind_methods();
	static OS *singleton;

public:
	enum VideoDriver {
		VIDEO_DRIVER_VULKAN,
		VIDEO_DRIVER_OPENGL_3,
	};

	enum Weekday {
		DAY_SUNDAY,
		DAY_MONDAY,
		DAY_TUESDAY,
		DAY_WEDNESDAY,
		DAY_THURSDAY,
		DAY_FRIDAY,
		DAY_SATURDAY
	};

	enum Month {
		// Start at 1 to follow Windows SYSTEMTIME structure
		// https://msdn.microsoft.com/en-us/library/windows/desktop/ms724950(v=vs.85).aspx
		MONTH_JANUARY = 1,
		MONTH_FEBRUARY,
		MONTH_MARCH,
		MONTH_APRIL,
		MONTH_MAY,
		MONTH_JUNE,
		MONTH_JULY,
		MONTH_AUGUST,
		MONTH_SEPTEMBER,
		MONTH_OCTOBER,
		MONTH_NOVEMBER,
		MONTH_DECEMBER
	};

	virtual PackedStringArray get_connected_midi_inputs();
	virtual void open_midi_inputs();
	virtual void close_midi_inputs();

	void set_low_processor_usage_mode(bool p_enabled);
	bool is_in_low_processor_usage_mode() const;

	void set_low_processor_usage_mode_sleep_usec(int p_usec);
	int get_low_processor_usage_mode_sleep_usec() const;

	void alert(const String &p_alert, const String &p_title = "ALERT!");

	String get_executable_path() const;
	int execute(const String &p_path, const Vector<String> &p_arguments, Array r_output = Array(), bool p_read_stderr = false);
	int create_process(const String &p_path, const Vector<String> &p_arguments);
	int create_instance(const Vector<String> &p_arguments);
	Error kill(int p_pid);
	Error shell_open(String p_uri);

	int get_process_id() const;

	bool has_environment(const String &p_var) const;
	String get_environment(const String &p_var) const;
	bool set_environment(const String &p_var, const String &p_value) const;

	String get_name() const;
	Vector<String> get_cmdline_args();

	String get_locale() const;
	String get_locale_language() const;

	String get_model_name() const;

	void dump_memory_to_file(const String &p_file);
	void dump_resources_to_file(const String &p_file);

	void print_resources_in_use(bool p_short = false);
	void print_all_resources(const String &p_to_file);
	void print_all_textures_by_size();
	void print_resources_by_type(const Vector<String> &p_types);

	bool is_debug_build() const;

	String get_unique_id() const;

	String get_keycode_string(Key p_code) const;
	bool is_keycode_unicode(char32_t p_unicode) const;
	Key find_keycode_from_string(const String &p_code) const;

	void set_use_file_access_save_and_swap(bool p_enable);

	uint64_t get_static_memory_usage() const;
	uint64_t get_static_memory_peak_usage() const;

	void delay_usec(int p_usec) const;
	void delay_msec(int p_msec) const;
	uint64_t get_ticks_msec() const;
	uint64_t get_ticks_usec() const;

	bool can_use_threads() const;

	bool is_userfs_persistent() const;

	bool is_stdout_verbose() const;

	int get_processor_count() const;

	enum SystemDir {
		SYSTEM_DIR_DESKTOP,
		SYSTEM_DIR_DCIM,
		SYSTEM_DIR_DOCUMENTS,
		SYSTEM_DIR_DOWNLOADS,
		SYSTEM_DIR_MOVIES,
		SYSTEM_DIR_MUSIC,
		SYSTEM_DIR_PICTURES,
		SYSTEM_DIR_RINGTONES,
	};

	String get_system_dir(SystemDir p_dir, bool p_shared_storage = true) const;

	String get_user_data_dir() const;
	String get_config_dir() const;
	String get_data_dir() const;
	String get_cache_dir() const;

	Error set_thread_name(const String &p_name);
	Thread::ID get_thread_caller_id() const;

	bool has_feature(const String &p_feature) const;

	bool request_permission(const String &p_name);
	bool request_permissions();
	Vector<String> get_granted_permissions() const;

	static OS *get_singleton() { return singleton; }

	OS() { singleton = this; }
};

class Geometry2D : public Object {
	GDCLASS(Geometry2D, Object);

	static Geometry2D *singleton;

protected:
	static void _bind_methods();

public:
	static Geometry2D *get_singleton();
	Variant segment_intersects_segment(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b);
	Variant line_intersects_line(const Vector2 &p_from_a, const Vector2 &p_dir_a, const Vector2 &p_from_b, const Vector2 &p_dir_b);
	Vector<Vector2> get_closest_points_between_segments(const Vector2 &p1, const Vector2 &q1, const Vector2 &p2, const Vector2 &q2);
	Vector2 get_closest_point_to_segment(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b);
	Vector2 get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b);
	bool point_is_inside_triangle(const Vector2 &s, const Vector2 &a, const Vector2 &b, const Vector2 &c) const;

	bool is_point_in_circle(const Vector2 &p_point, const Vector2 &p_circle_pos, real_t p_circle_radius);
	real_t segment_intersects_circle(const Vector2 &p_from, const Vector2 &p_to, const Vector2 &p_circle_pos, real_t p_circle_radius);

	bool is_polygon_clockwise(const Vector<Vector2> &p_polygon);
	bool is_point_in_polygon(const Point2 &p_point, const Vector<Vector2> &p_polygon);
	Vector<int> triangulate_polygon(const Vector<Vector2> &p_polygon);
	Vector<int> triangulate_delaunay(const Vector<Vector2> &p_points);
	Vector<Point2> convex_hull(const Vector<Point2> &p_points);

	enum PolyBooleanOperation {
		OPERATION_UNION,
		OPERATION_DIFFERENCE,
		OPERATION_INTERSECTION,
		OPERATION_XOR
	};
	// 2D polygon boolean operations.
	Array merge_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b); // Union (add).
	Array clip_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b); // Difference (subtract).
	Array intersect_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b); // Common area (multiply).
	Array exclude_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b); // All but common area (xor).

	// 2D polyline vs polygon operations.
	Array clip_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon); // Cut.
	Array intersect_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon); // Chop.

	// 2D offset polygons/polylines.
	enum PolyJoinType {
		JOIN_SQUARE,
		JOIN_ROUND,
		JOIN_MITER
	};
	enum PolyEndType {
		END_POLYGON,
		END_JOINED,
		END_BUTT,
		END_SQUARE,
		END_ROUND
	};
	Array offset_polygon(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type = JOIN_SQUARE);
	Array offset_polyline(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type = JOIN_SQUARE, PolyEndType p_end_type = END_SQUARE);

	Dictionary make_atlas(const Vector<Size2> &p_rects);

	Geometry2D() { singleton = this; }
};

class Geometry3D : public Object {
	GDCLASS(Geometry3D, Object);

	static Geometry3D *singleton;

protected:
	static void _bind_methods();

public:
	static Geometry3D *get_singleton();
	Vector<Plane> build_box_planes(const Vector3 &p_extents);
	Vector<Plane> build_cylinder_planes(float p_radius, float p_height, int p_sides, Vector3::Axis p_axis = Vector3::AXIS_Z);
	Vector<Plane> build_capsule_planes(float p_radius, float p_height, int p_sides, int p_lats, Vector3::Axis p_axis = Vector3::AXIS_Z);
	Vector<Vector3> get_closest_points_between_segments(const Vector3 &p1, const Vector3 &p2, const Vector3 &q1, const Vector3 &q2);
	Vector3 get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b);
	Vector3 get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b);
	Variant ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2);
	Variant segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2);

	Vector<Vector3> segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius);
	Vector<Vector3> segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, float p_height, float p_radius);
	Vector<Vector3> segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Vector<Plane> &p_planes);

	Vector<Vector3> clip_polygon(const Vector<Vector3> &p_points, const Plane &p_plane);

	Geometry3D() { singleton = this; }
};

class File : public RefCounted {
	GDCLASS(File, RefCounted);

	FileAccess *f = nullptr;
	bool big_endian = false;

protected:
	static void _bind_methods();

public:
	enum ModeFlags {
		READ = 1,
		WRITE = 2,
		READ_WRITE = 3,
		WRITE_READ = 7,
	};

	enum CompressionMode {
		COMPRESSION_FASTLZ = Compression::MODE_FASTLZ,
		COMPRESSION_DEFLATE = Compression::MODE_DEFLATE,
		COMPRESSION_ZSTD = Compression::MODE_ZSTD,
		COMPRESSION_GZIP = Compression::MODE_GZIP
	};

	Error open_encrypted(const String &p_path, ModeFlags p_mode_flags, const Vector<uint8_t> &p_key);
	Error open_encrypted_pass(const String &p_path, ModeFlags p_mode_flags, const String &p_pass);
	Error open_compressed(const String &p_path, ModeFlags p_mode_flags, CompressionMode p_compress_mode = COMPRESSION_FASTLZ);

	Error open(const String &p_path, ModeFlags p_mode_flags); // open a file.
	void flush(); // Flush a file (write its buffer to disk).
	void close(); // Close a file.
	bool is_open() const; // True when file is open.

	String get_path() const; // Returns the path for the current open file.
	String get_path_absolute() const; // Returns the absolute path for the current open file.

	void seek(int64_t p_position); // Seek to a given position.
	void seek_end(int64_t p_position = 0); // Seek from the end of file.
	uint64_t get_position() const; // Get position in the file.
	uint64_t get_length() const; // Get size of the file.

	bool eof_reached() const; // Reading passed EOF.

	uint8_t get_8() const; // Get a byte.
	uint16_t get_16() const; // Get 16 bits uint.
	uint32_t get_32() const; // Get 32 bits uint.
	uint64_t get_64() const; // Get 64 bits uint.

	float get_float() const;
	double get_double() const;
	real_t get_real() const;

	Variant get_var(bool p_allow_objects = false) const;

	Vector<uint8_t> get_buffer(int64_t p_length) const; // Get an array of bytes.
	String get_line() const;
	Vector<String> get_csv_line(const String &p_delim = ",") const;
	String get_as_text() const;
	String get_md5(const String &p_path) const;
	String get_sha256(const String &p_path) const;

	/*
	 * Use this for files WRITTEN in _big_ endian machines (ie, amiga/mac).
	 * It's not about the current CPU type but file formats.
	 * This flag gets reset to `false` (little endian) on each open.
	 */
	void set_big_endian(bool p_big_endian);
	bool is_big_endian();

	Error get_error() const; // Get last error.

	void store_8(uint8_t p_dest); // Store a byte.
	void store_16(uint16_t p_dest); // Store 16 bits uint.
	void store_32(uint32_t p_dest); // Store 32 bits uint.
	void store_64(uint64_t p_dest); // Store 64 bits uint.

	void store_float(float p_dest);
	void store_double(double p_dest);
	void store_real(real_t p_real);

	void store_string(const String &p_string);
	void store_line(const String &p_string);
	void store_csv_line(const Vector<String> &p_values, const String &p_delim = ",");

	virtual void store_pascal_string(const String &p_string);
	virtual String get_pascal_string();

	void store_buffer(const Vector<uint8_t> &p_buffer); // Store an array of bytes.

	void store_var(const Variant &p_var, bool p_full_objects = false);

	bool file_exists(const String &p_name) const; // Return true if a file exists.

	uint64_t get_modified_time(const String &p_file) const;

	File() {}
	virtual ~File();
};

class Directory : public RefCounted {
	GDCLASS(Directory, RefCounted);
	DirAccess *d;
	bool dir_open = false;

protected:
	static void _bind_methods();

public:
	Error open(const String &p_path);

	bool is_open() const;

	Error list_dir_begin(bool p_show_navigational = false, bool p_show_hidden = false); // This starts dir listing.
	String get_next();
	bool current_is_dir() const;

	void list_dir_end();

	int get_drive_count();
	String get_drive(int p_drive);
	int get_current_drive();

	Error change_dir(String p_dir); // Can be relative or absolute, return false on success.
	String get_current_dir(); // Return current dir location.

	Error make_dir(String p_dir);
	Error make_dir_recursive(String p_dir);

	bool file_exists(String p_file);
	bool dir_exists(String p_dir);

	uint64_t get_space_left();

	Error copy(String p_from, String p_to);
	Error rename(String p_from, String p_to);
	Error remove(String p_name);

	Directory();
	virtual ~Directory();

private:
	bool _list_skip_navigational = false;
	bool _list_skip_hidden = false;
};

class Marshalls : public Object {
	GDCLASS(Marshalls, Object);

	static Marshalls *singleton;

protected:
	static void _bind_methods();

public:
	static Marshalls *get_singleton();

	String variant_to_base64(const Variant &p_var, bool p_full_objects = false);
	Variant base64_to_variant(const String &p_str, bool p_allow_objects = false);

	String raw_to_base64(const Vector<uint8_t> &p_arr);
	Vector<uint8_t> base64_to_raw(const String &p_str);

	String utf8_to_base64(const String &p_str);
	String base64_to_utf8(const String &p_str);

	Marshalls() { singleton = this; }
	~Marshalls() { singleton = nullptr; }
};

class Mutex : public RefCounted {
	GDCLASS(Mutex, RefCounted);
	::Mutex mutex;

	static void _bind_methods();

public:
	void lock();
	Error try_lock();
	void unlock();
};

class Semaphore : public RefCounted {
	GDCLASS(Semaphore, RefCounted);
	::Semaphore semaphore;

	static void _bind_methods();

public:
	void wait();
	Error try_wait();
	void post();
};

class Thread : public RefCounted {
	GDCLASS(Thread, RefCounted);

protected:
	Variant ret;
	Variant userdata;
	SafeFlag running;
	Callable target_callable;
	::Thread thread;
	static void _bind_methods();
	static void _start_func(void *ud);

public:
	enum Priority {
		PRIORITY_LOW,
		PRIORITY_NORMAL,
		PRIORITY_HIGH,
		PRIORITY_MAX
	};

	Error start(const Callable &p_callable, const Variant &p_userdata = Variant(), Priority p_priority = PRIORITY_NORMAL);
	String get_id() const;
	bool is_started() const;
	bool is_alive() const;
	Variant wait_to_finish();
};

namespace special {

class ClassDB : public Object {
	GDCLASS(ClassDB, Object);

protected:
	static void _bind_methods();

public:
	PackedStringArray get_class_list() const;
	PackedStringArray get_inheriters_from_class(const StringName &p_class) const;
	StringName get_parent_class(const StringName &p_class) const;
	bool class_exists(const StringName &p_class) const;
	bool is_parent_class(const StringName &p_class, const StringName &p_inherits) const;
	bool can_instantiate(const StringName &p_class) const;
	Variant instantiate(const StringName &p_class) const;

	bool has_signal(StringName p_class, StringName p_signal) const;
	Dictionary get_signal(StringName p_class, StringName p_signal) const;
	Array get_signal_list(StringName p_class, bool p_no_inheritance = false) const;

	Array get_property_list(StringName p_class, bool p_no_inheritance = false) const;
	Variant get_property(Object *p_object, const StringName &p_property) const;
	Error set_property(Object *p_object, const StringName &p_property, const Variant &p_value) const;

	bool has_method(StringName p_class, StringName p_method, bool p_no_inheritance = false) const;

	Array get_method_list(StringName p_class, bool p_no_inheritance = false) const;

	PackedStringArray get_integer_constant_list(const StringName &p_class, bool p_no_inheritance = false) const;
	bool has_integer_constant(const StringName &p_class, const StringName &p_name) const;
	int get_integer_constant(const StringName &p_class, const StringName &p_name) const;
	StringName get_category(const StringName &p_node) const;

	bool has_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false) const;
	PackedStringArray get_enum_list(const StringName &p_class, bool p_no_inheritance = false) const;
	PackedStringArray get_enum_constants(const StringName &p_class, const StringName &p_enum, bool p_no_inheritance = false) const;
	StringName get_integer_constant_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false) const;

	bool is_class_enabled(StringName p_class) const;

	ClassDB() {}
	~ClassDB() {}
};

} // namespace special

class Engine : public Object {
	GDCLASS(Engine, Object);

protected:
	static void _bind_methods();
	static Engine *singleton;

public:
	static Engine *get_singleton() { return singleton; }
	void set_physics_ticks_per_second(int p_ips);
	int get_physics_ticks_per_second() const;

	void set_physics_jitter_fix(double p_threshold);
	double get_physics_jitter_fix() const;
	double get_physics_interpolation_fraction() const;

	void set_target_fps(int p_fps);
	int get_target_fps() const;

	double get_frames_per_second() const;
	uint64_t get_physics_frames() const;
	uint64_t get_process_frames() const;

	int get_frames_drawn();

	void set_time_scale(double p_scale);
	double get_time_scale();

	MainLoop *get_main_loop() const;

	Dictionary get_version_info() const;
	Dictionary get_author_info() const;
	Array get_copyright_info() const;
	Dictionary get_donor_info() const;
	Dictionary get_license_info() const;
	String get_license_text() const;

	bool is_in_physics_frame() const;

	bool has_singleton(const StringName &p_name) const;
	Object *get_singleton_object(const StringName &p_name) const;
	void register_singleton(const StringName &p_name, Object *p_object);
	void unregister_singleton(const StringName &p_name);
	Vector<String> get_singleton_list() const;

	void set_editor_hint(bool p_enabled);
	bool is_editor_hint() const;

	void set_print_error_messages(bool p_enabled);
	bool is_printing_error_messages() const;

	Engine() { singleton = this; }
};

class EngineDebugger : public Object {
	GDCLASS(EngineDebugger, Object);

	class ProfilerCallable {
		friend class EngineDebugger;

		Callable callable_toggle;
		Callable callable_add;
		Callable callable_tick;

	public:
		ProfilerCallable() {}

		ProfilerCallable(const Callable &p_toggle, const Callable &p_add, const Callable &p_tick) {
			callable_toggle = p_toggle;
			callable_add = p_add;
			callable_tick = p_tick;
		}
	};

	Map<StringName, Callable> captures;
	Map<StringName, ProfilerCallable> profilers;

protected:
	static void _bind_methods();
	static EngineDebugger *singleton;

public:
	static EngineDebugger *get_singleton() { return singleton; }

	bool is_active();

	void register_profiler(const StringName &p_name, const Callable &p_toggle, const Callable &p_add, const Callable &p_tick);
	void unregister_profiler(const StringName &p_name);
	bool is_profiling(const StringName &p_name);
	bool has_profiler(const StringName &p_name);
	void profiler_add_frame_data(const StringName &p_name, const Array &p_data);
	void profiler_enable(const StringName &p_name, bool p_enabled, const Array &p_opts = Array());

	void register_message_capture(const StringName &p_name, const Callable &p_callable);
	void unregister_message_capture(const StringName &p_name);
	bool has_capture(const StringName &p_name);

	void send_message(const String &p_msg, const Array &p_data);

	static void call_toggle(void *p_user, bool p_enable, const Array &p_opts);
	static void call_add(void *p_user, const Array &p_data);
	static void call_tick(void *p_user, double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time);
	static Error call_capture(void *p_user, const String &p_cmd, const Array &p_data, bool &r_captured);

	EngineDebugger() { singleton = this; }
	~EngineDebugger();
};

} // namespace core_bind

VARIANT_ENUM_CAST(core_bind::ResourceLoader::ThreadLoadStatus);
VARIANT_ENUM_CAST(core_bind::ResourceLoader::CacheMode);

VARIANT_ENUM_CAST(core_bind::ResourceSaver::SaverFlags);

VARIANT_ENUM_CAST(core_bind::OS::VideoDriver);
VARIANT_ENUM_CAST(core_bind::OS::Weekday);
VARIANT_ENUM_CAST(core_bind::OS::Month);
VARIANT_ENUM_CAST(core_bind::OS::SystemDir);

VARIANT_ENUM_CAST(core_bind::Geometry2D::PolyBooleanOperation);
VARIANT_ENUM_CAST(core_bind::Geometry2D::PolyJoinType);
VARIANT_ENUM_CAST(core_bind::Geometry2D::PolyEndType);

VARIANT_ENUM_CAST(core_bind::File::ModeFlags);
VARIANT_ENUM_CAST(core_bind::File::CompressionMode);

VARIANT_ENUM_CAST(core_bind::Thread::Priority);

#endif // CORE_BIND_H
