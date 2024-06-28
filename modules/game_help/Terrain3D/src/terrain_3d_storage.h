// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_STORAGE_CLASS_H
#define TERRAIN3D_STORAGE_CLASS_H

//#include <godot_cpp/classes/resource_loader.hpp>
//#include <godot_cpp/classes/shader.hpp>
#include "core/io/resource_loader.h"
#include "scene/resources/shader.h"

#include "constants.h"
#include "generated_texture.h"
#include "terrain_3d_util.h"

class Terrain3D;

using namespace godot;

class Terrain3DStorage : public Resource {
	GDCLASS(Terrain3DStorage, Resource);
	CLASS_NAME();

public: // Constants
	static inline const real_t CURRENT_VERSION = 0.915f; // Dev version
	static inline const int REGION_MAP_SIZE = 16;
	static inline const Vector2i REGION_MAP_VSIZE = Vector2i(REGION_MAP_SIZE, REGION_MAP_SIZE);

	enum MapType {
		TYPE_HEIGHT,
		TYPE_CONTROL,
		TYPE_COLOR,
		TYPE_MAX,
	};

	static inline const Image::Format FORMAT[] = {
		Image::FORMAT_RF, // TYPE_HEIGHT
		Image::FORMAT_RF, // TYPE_CONTROL
		Image::FORMAT_RGBA8, // TYPE_COLOR
		Image::Format(TYPE_MAX), // Proper size of array instead of FORMAT_MAX
	};

	static inline const char *TYPESTR[] = {
		"TYPE_HEIGHT",
		"TYPE_CONTROL",
		"TYPE_COLOR",
		"TYPE_MAX",
	};

	static inline const Color COLOR[] = {
		COLOR_BLACK, // TYPE_HEIGHT
		COLOR_CONTROL, // TYPE_CONTROL
		COLOR_ROUGHNESS, // TYPE_COLOR
		COLOR_NAN, // TYPE_MAX, unused just in case someone indexes the array
	};

	enum RegionSize {
		//SIZE_64 = 64,
		//SIZE_128 = 128,
		//SIZE_256 = 256,
		//SIZE_512 = 512,
		SIZE_1024 = 1024,
		//SIZE_2048 = 2048,
	};

	enum HeightFilter {
		HEIGHT_FILTER_NEAREST,
		HEIGHT_FILTER_MINIMUM
	};

private:
	Terrain3D *_terrain = nullptr;

	// Work data
	bool _modified = false;
	bool _region_map_dirty = true;
	PackedInt32Array _region_map; // 16x16 Region grid with index into region_offsets (1 based array)
	// Generated Texture RIDs
	// These contain the TextureLayered RID from the RenderingServer, no Image
	GeneratedTexture _generated_height_maps;
	GeneratedTexture _generated_control_maps;
	GeneratedTexture _generated_color_maps;

	AABB _edited_area;
	uint64_t _last_region_bounds_error = 0;

	// Stored Data
	real_t _version = 0.8f; // Set to ensure Godot always saves this
	RegionSize _region_size = SIZE_1024;
	Vector2i _region_sizev = Vector2i(_region_size, _region_size);
	bool _save_16_bit = false;
	Vector2 _height_range = Vector2(0.f, 0.f);

	/**
	 * These arrays house all of the map data.
	 * The Image arrays are region_sized slices of all heightmap data. Their world
	 * location is tracked by region_offsets. The region data are combined into one large
	 * texture in generated_*_maps.
	 */
	TypedArray<Vector2i> _region_offsets; // Array of active region coordinates
	TypedArray<Image> _height_maps;
	TypedArray<Image> _control_maps;
	TypedArray<Image> _color_maps;

	// Foliage Instancer contains MultiMeshes saved to disk
	// Dictionary[region_offset:Vector2i] -> Dictionary[mesh_id:int] -> MultiMesh
	Dictionary _multimeshes;

	// Functions
	void _clear();

public:
	Terrain3DStorage() {}
	void initialize(Terrain3D *p_terrain);
	~Terrain3DStorage();

	void set_version(real_t p_version);
	real_t get_version() const { return _version; }
	void set_save_16_bit(bool p_enabled);
	bool get_save_16_bit() const { return _save_16_bit; }

	void set_height_range(Vector2 p_range);
	Vector2 get_height_range() const { return _height_range; }
	void update_heights(real_t p_height);
	void update_heights(Vector2 p_heights);
	void update_height_range();

	void clear_edited_area();
	void add_edited_area(AABB p_area);
	AABB get_edited_area() const { return _edited_area; }

	// Regions
	void set_region_size(RegionSize p_size);
	RegionSize get_region_size() const { return _region_size; }
	Vector2i get_region_sizev() const { return _region_sizev; }
	void set_region_offsets(const TypedArray<Vector2i> &p_offsets);
	TypedArray<Vector2i> get_region_offsets() const { return _region_offsets; }
	PackedInt32Array get_region_map() const { return _region_map; }
	int get_region_count() const { return _region_offsets.size(); }
	Vector2i get_region_offset(Vector3 p_global_position);
	Vector2i get_region_offset_from_index(int p_index);
	int get_region_index(Vector3 p_global_position);
	int get_region_index_from_offset(Vector2i p_region_offset);
	bool has_region(Vector3 p_global_position) { return get_region_index(p_global_position) != -1; }
	Error add_region(Vector3 p_global_position, const TypedArray<Image> &p_images = TypedArray<Image>(), bool p_update = true);
	void remove_region(Vector3 p_global_position, bool p_update = true);
	void update_regions(bool force_emit = false);

	// Maps
	void set_map_region(MapType p_map_type, int p_region_index, const Ref<Image> p_image);
	Ref<Image> get_map_region(MapType p_map_type, int p_region_index) const;
	void set_maps(MapType p_map_type, const TypedArray<Image> &p_maps);
	TypedArray<Image> get_maps(MapType p_map_type) const;
	TypedArray<Image> get_maps_copy(MapType p_map_type) const;
	void set_height_maps(const TypedArray<Image> &p_maps) { set_maps(TYPE_HEIGHT, p_maps); }
	TypedArray<Image> get_height_maps() const { return _height_maps; }
	RID get_height_rid() { return _generated_height_maps.get_rid(); }
	void set_control_maps(const TypedArray<Image> &p_maps) { set_maps(TYPE_CONTROL, p_maps); }
	TypedArray<Image> get_control_maps() const { return _control_maps; }
	RID get_control_rid() { return _generated_control_maps.get_rid(); }
	void set_color_maps(const TypedArray<Image> &p_maps) { set_maps(TYPE_COLOR, p_maps); }
	TypedArray<Image> get_color_maps() const { return _color_maps; }
	RID get_color_rid() { return _generated_color_maps.get_rid(); }
	void set_pixel(MapType p_map_type, Vector3 p_global_position, Color p_pixel);
	Color get_pixel(MapType p_map_type, Vector3 p_global_position);
	void set_height(Vector3 p_global_position, real_t p_height);
	real_t get_height(Vector3 p_global_position);
	void set_color(Vector3 p_global_position, Color p_color);
	Color get_color(Vector3 p_global_position);
	void set_control(Vector3 p_global_position, uint32_t p_control);
	uint32_t get_control(Vector3 p_global_position);
	void set_roughness(Vector3 p_global_position, real_t p_roughness);
	real_t get_roughness(Vector3 p_global_position);
	Vector3 get_texture_id(Vector3 p_global_position);
	real_t get_angle(Vector3 p_global_position);
	real_t get_scale(Vector3 p_global_position);
	TypedArray<Image> sanitize_maps(MapType p_map_type, const TypedArray<Image> &p_maps);
	void force_update_maps(MapType p_map = TYPE_MAX);

	// Instancer
	void set_multimeshes(Dictionary p_multimeshes);
	Dictionary get_multimeshes() { return _multimeshes; }

	// File I/O
	void save();
	void clear_modified() { _modified = false; }
	void set_modified() { _modified = true; }
	void import_images(const TypedArray<Image> &p_images, Vector3 p_global_position = Vector3(0.f, 0.f, 0.f),
			real_t p_offset = 0.f, real_t p_scale = 1.f);
	Error export_image(String p_file_name, MapType p_map_type = TYPE_HEIGHT);
	Ref<Image> layered_to_image(MapType p_map_type);

	// Utility
	Vector3 get_mesh_vertex(int32_t p_lod, HeightFilter p_filter, Vector3 p_global_position);
	Vector3 get_normal(Vector3 global_position);
	void print_audit_data();

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(Terrain3DStorage::MapType);
VARIANT_ENUM_CAST(Terrain3DStorage::RegionSize);
VARIANT_ENUM_CAST(Terrain3DStorage::HeightFilter);

// Inline Functions

inline void Terrain3DStorage::set_height(Vector3 p_global_position, real_t p_height) {
	set_pixel(TYPE_HEIGHT, p_global_position, Color(p_height, 0.f, 0.f, 1.f));
}

inline void Terrain3DStorage::set_color(Vector3 p_global_position, Color p_color) {
	p_color.a = get_roughness(p_global_position);
	set_pixel(TYPE_COLOR, p_global_position, p_color);
}

inline Color Terrain3DStorage::get_color(Vector3 p_global_position) {
	Color clr = get_pixel(TYPE_COLOR, p_global_position);
	clr.a = 1.0f;
	return clr;
}

inline void Terrain3DStorage::set_control(Vector3 p_global_position, uint32_t p_control) {
	set_pixel(TYPE_CONTROL, p_global_position, Color(as_float(p_control), 0.f, 0.f, 1.f));
}

inline uint32_t Terrain3DStorage::get_control(Vector3 p_global_position) {
	return as_uint(get_pixel(TYPE_CONTROL, p_global_position).r);
}

inline void Terrain3DStorage::set_roughness(Vector3 p_global_position, real_t p_roughness) {
	Color clr = get_pixel(TYPE_COLOR, p_global_position);
	clr.a = p_roughness;
	set_pixel(TYPE_COLOR, p_global_position, clr);
}

inline real_t Terrain3DStorage::get_roughness(Vector3 p_global_position) {
	return get_pixel(TYPE_COLOR, p_global_position).a;
}

#endif // TERRAIN3D_STORAGE_CLASS_H
