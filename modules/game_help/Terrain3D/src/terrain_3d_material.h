// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_MATERIAL_CLASS_H
#define TERRAIN3D_MATERIAL_CLASS_H

//#include <godot_cpp/classes/shader.hpp>
#include "scene/resources/shader.h"

#include "generated_tex.h"

using namespace godot;

class Terrain3DMaterial : public Resource {
	GDCLASS(Terrain3DMaterial, Resource);

public:
	// Constants
	static inline const char *__class__ = "Terrain3DMaterial";

	enum WorldBackground {
		NONE,
		FLAT,
		NOISE,
	};

	enum TextureFiltering {
		LINEAR,
		NEAREST,
	};

private:
	bool _initialized = false;
	RID _material;
	RID _shader;
	bool _shader_override_enabled = false;
	Ref<Shader> _shader_override;
	Ref<Shader> _shader_tmp;
	Dictionary _shader_code;
	mutable TypedArray<StringName> _active_params; // All shader params in the current shader
	mutable Dictionary _shader_params; // Public shader params saved to disk

	// Material Features
	WorldBackground _world_background = FLAT;
	TextureFiltering _texture_filtering = LINEAR;
	bool _auto_shader = false;
	bool _dual_scaling = false;

	// Editor Functions / Debug views
	bool _show_navigation = false;
	bool _debug_view_checkered = false;
	bool _debug_view_grey = false;
	bool _debug_view_heightmap = false;
	bool _debug_view_colormap = false;
	bool _debug_view_roughmap = false;
	bool _debug_view_control_texture = false;
	bool _debug_view_control_blend = false;
	bool _debug_view_autoshader = false;
	bool _debug_view_holes = false;
	bool _debug_view_tex_height = false;
	bool _debug_view_tex_normal = false;
	bool _debug_view_tex_rough = false;
	bool _debug_view_vertex_grid = false;

	// Cached data from Storage
	int _texture_count = 0;
	int _region_size = 1024;
	real_t _mesh_vertex_spacing = 1.0f;
	Vector2i _region_sizev = Vector2i(_region_size, _region_size);
	PackedInt32Array _region_map;
	GeneratedTex _generated_region_blend_map; // 512x512 blurred image of region_map

	// Functions
	void _preload_shaders();
	void _parse_shader(String p_shader, String p_name);
	String _apply_inserts(String p_shader, Array p_excludes = Array());
	String _generate_shader_code();
	String _inject_editor_code(String p_shader);
	void _update_shader();
	void _update_regions(const Array &p_args);
	void _generate_region_blend_map();
	void _update_texture_arrays(const Array &p_args);
	void _set_region_size(int p_size);
	void _set_shader_parameters(const Dictionary &p_dict);
	Dictionary _get_shader_parameters() const { return _shader_params; }

public:
	Terrain3DMaterial(){};
	void initialize(int p_region_size);
	~Terrain3DMaterial();

	RID get_material_rid() const { return _material; }
	RID get_shader_rid() const;
	RID get_region_blend_map() { return _generated_region_blend_map.get_rid(); }

	// Material settings
	void set_world_background(WorldBackground p_background);
	WorldBackground get_world_background() const { return _world_background; }
	void set_texture_filtering(TextureFiltering p_filtering);
	TextureFiltering get_texture_filtering() const { return _texture_filtering; }
	void set_auto_shader(bool p_enabled);
	bool get_auto_shader() const { return _auto_shader; }
	void set_dual_scaling(bool p_enabled);
	bool get_dual_scaling() const { return _dual_scaling; }

	void enable_shader_override(bool p_enabled);
	bool is_shader_override_enabled() const { return _shader_override_enabled; }
	void set_shader_override(const Ref<Shader> &p_shader);
	Ref<Shader> get_shader_override() const { return _shader_override; }

	void set_shader_param(const StringName &p_name, const Variant &p_value);
	Variant get_shader_param(const StringName &p_name) const;

	void set_mesh_vertex_spacing(real_t p_spacing);

	// Editor functions / Debug views
	void set_show_checkered(bool p_enabled);
	bool get_show_checkered() const { return _debug_view_checkered; }
	void set_show_grey(bool p_enabled);
	bool get_show_grey() const { return _debug_view_grey; }
	void set_show_heightmap(bool p_enabled);
	bool get_show_heightmap() const { return _debug_view_heightmap; }
	void set_show_colormap(bool p_enabled);
	bool get_show_colormap() const { return _debug_view_colormap; }
	void set_show_roughmap(bool p_enabled);
	bool get_show_roughmap() const { return _debug_view_roughmap; }
	void set_show_control_texture(bool p_enabled);
	bool get_show_control_texture() const { return _debug_view_control_texture; }
	void set_show_control_blend(bool p_enabled);
	bool get_show_control_blend() const { return _debug_view_control_blend; }
	void set_show_autoshader(bool p_enabled);
	bool get_show_autoshader() const { return _debug_view_autoshader; }
	void set_show_navigation(bool p_enabled);
	bool get_show_navigation() const { return _show_navigation; }
	void set_show_texture_height(bool p_enabled);
	bool get_show_texture_height() const { return _debug_view_tex_height; }
	void set_show_texture_normal(bool p_enabled);
	bool get_show_texture_normal() const { return _debug_view_tex_normal; }
	void set_show_texture_rough(bool p_enabled);
	bool get_show_texture_rough() const { return _debug_view_tex_rough; }
	void set_show_vertex_grid(bool p_enabled);
	bool get_show_vertex_grid() const { return _debug_view_vertex_grid; }

	void save();

protected:
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _property_can_revert(const StringName &p_name) const;
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const;
	bool _set(const StringName &p_name, const Variant &p_property);
	bool _get(const StringName &p_name, Variant &r_property) const;

	static void _bind_methods();
};

VARIANT_ENUM_CAST(Terrain3DMaterial::WorldBackground);
VARIANT_ENUM_CAST(Terrain3DMaterial::TextureFiltering);

#endif // TERRAIN3D_MATERIAL_CLASS_H