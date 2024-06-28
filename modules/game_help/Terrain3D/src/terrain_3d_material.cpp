// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.


#include "logger.h"
#include "terrain_3d_material.h"
#include "terrain_3d_util.h"
#include "modules/noise/fastnoise_lite.h"
#include "modules/noise/noise_texture_2d.h"

///////////////////////////
// Private Functions
///////////////////////////

void Terrain3DMaterial::_preload_shaders() {
	// Preprocessor loading of external shader inserts

	_parse_shader(
#include "shaders/uniforms.glsl"
			, "uniforms");
	_parse_shader(
#include "shaders/world_noise.glsl"
			, "world_noise");
	_parse_shader(
#include "shaders/auto_shader.glsl"
			, "auto_shader");
	_parse_shader(
#include "shaders/dual_scaling.glsl"
			, "dual_scaling");
	_parse_shader(
#include "shaders/debug_views.glsl"
			, "debug_views");
	_parse_shader(
#include "shaders/editor_functions.glsl"
			, "editor_functions");

	// Load main code
	_shader_code["main"] = String(
#include "shaders/main.glsl"
	);

	if (Terrain3D::debug_level >= DEBUG) {
		Array keys = _shader_code.keys();
		for (int i = 0; i < keys.size(); i++) {
			LOG(DEBUG, "Loaded shader insert: ", keys[i]);
		}
	}
}

/**
 *	All `//INSERT: ID` blocks in p_shader are loaded into the DB _shader_code
 */
void Terrain3DMaterial::_parse_shader(String p_shader, String p_name) {
	if (p_name.is_empty()) {
		LOG(ERROR, "No dictionary key for saving shader snippets specified");
		return;
	}
	PackedStringArray parsed = p_shader.split("//INSERT:");
	for (int i = 0; i < parsed.size(); i++) {
		// First section of the file before any //INSERT:
		if (i == 0) {
			_shader_code[p_name] = parsed[0];
		} else {
			// There is at least one //INSERT:
			// Get the first ID on the first line
			PackedStringArray segment = parsed[i].split("\n", true, 1);
			// If there isn't an ID AND body, skip this insert
			if (segment.size() < 2) {
				continue;
			}
			String id = segment[0].strip_edges();
			// Process the insert
			if (!id.is_empty() && !segment[1].is_empty()) {
				_shader_code[id] = segment[1];
			}
		}
	}
	return;
}

/**
 *	`//INSERT: ID` blocks in p_shader are replaced by the entry in the DB
 *	returns a shader string with inserts applied
 *  Skips `EDITOR_*` and `DEBUG_*` inserts
 */
String Terrain3DMaterial::_apply_inserts(String p_shader, Array p_excludes) {
	PackedStringArray parsed = p_shader.split("//INSERT:");
	String shader;
	for (int i = 0; i < parsed.size(); i++) {
		// First section of the file before any //INSERT:
		if (i == 0) {
			shader = parsed[0];
		} else {
			// There is at least one //INSERT:
			// Get the first ID on the first line
			PackedStringArray segment = parsed[i].split("\n", true, 1);
			// If there isn't an ID AND body, skip this insert
			if (segment.size() < 2) {
				continue;
			}
			String id = segment[0].strip_edges();

			// Process the insert
			if (!id.is_empty() && !p_excludes.has(id) && _shader_code.has(id)) {
				if (!id.begins_with("DEBUG_") && !id.begins_with("EDITOR_")) {
					String str = _shader_code[id];
					shader += str;
				}
			}
			shader += segment[1];
		}
	}
	return shader;
}

String Terrain3DMaterial::_generate_shader_code() {
	LOG(INFO, "Generating default shader code");
	Array excludes;
	if (_world_background != NOISE) {
		excludes.push_back("WORLD_NOISE1");
		excludes.push_back("WORLD_NOISE2");
	}
	if (_texture_filtering == LINEAR) {
		excludes.push_back("TEXTURE_SAMPLERS_NEAREST");
	} else {
		excludes.push_back("TEXTURE_SAMPLERS_LINEAR");
	}
	if (_auto_shader) {
		excludes.push_back("TEXTURE_ID");
	} else {
		excludes.push_back("AUTO_SHADER_UNIFORMS");
		excludes.push_back("AUTO_SHADER_TEXTURE_ID");
	}
	if (_dual_scaling) {
		excludes.push_back("UNI_SCALING_BASE");
	} else {
		excludes.push_back("DUAL_SCALING_UNIFORMS");
		excludes.push_back("DUAL_SCALING_VERTEX");
		excludes.push_back("DUAL_SCALING_BASE");
		excludes.push_back("DUAL_SCALING_OVERLAY");
	}
	String shader = _apply_inserts(_shader_code["main"], excludes);
	return shader;
}

String Terrain3DMaterial::_inject_editor_code(String p_shader) {
	String shader = p_shader;
	int idx = p_shader.rfind("}");
	if (idx < 0) {
		return shader;
	}
	Array insert_names;
	if (_debug_view_checkered) {
		insert_names.push_back("DEBUG_CHECKERED");
	}
	if (_debug_view_grey) {
		insert_names.push_back("DEBUG_GREY");
	}
	if (_debug_view_heightmap) {
		insert_names.push_back("DEBUG_HEIGHTMAP");
	}
	if (_debug_view_colormap) {
		insert_names.push_back("DEBUG_COLORMAP");
	}
	if (_debug_view_roughmap) {
		insert_names.push_back("DEBUG_ROUGHMAP");
	}
	if (_debug_view_control_texture) {
		insert_names.push_back("DEBUG_CONTROL_TEXTURE");
	}
	if (_debug_view_control_angle) {
		insert_names.push_back("DEBUG_CONTROL_ANGLE");
	}
	if (_debug_view_control_scale) {
		insert_names.push_back("DEBUG_CONTROL_SCALE");
	}
	if (_debug_view_control_blend) {
		insert_names.push_back("DEBUG_CONTROL_BLEND");
	}
	if (_debug_view_autoshader) {
		insert_names.push_back("DEBUG_AUTOSHADER");
	}
	if (_debug_view_tex_height) {
		insert_names.push_back("DEBUG_TEXTURE_HEIGHT");
	}
	if (_debug_view_tex_normal) {
		insert_names.push_back("DEBUG_TEXTURE_NORMAL");
	}
	if (_debug_view_tex_rough) {
		insert_names.push_back("DEBUG_TEXTURE_ROUGHNESS");
	}
	if (_debug_view_vertex_grid) {
		insert_names.push_back("DEBUG_VERTEX_GRID");
	}
	if (_show_navigation) {
		insert_names.push_back("EDITOR_NAVIGATION");
	}
	for (int i = 0; i < insert_names.size(); i++) {
		String insert = _shader_code[insert_names[i]];
		shader = shader.insert(idx - 1, "\n" + insert);
		idx += insert.length();
	}
	return shader;
}

void Terrain3DMaterial::_update_shader() {
	IS_INIT(VOID);
	LOG(INFO, "Updating shader");
	RID shader_rid;
	if (_shader_override_enabled && _shader_override.is_valid()) {
		if (_shader_override->get_code().is_empty()) {
			String code = _generate_shader_code();
			_shader_override->set_code(code);
		}
		if (!_shader_override->is_connected("changed", callable_mp(this, &Terrain3DMaterial::_update_shader))) {
			LOG(DEBUG, "Connecting changed signal to _update_shader()");
			_shader_override->connect("changed", callable_mp(this, &Terrain3DMaterial::_update_shader));
		}
		String code = _shader_override->get_code();
		_shader_tmp->set_code(_inject_editor_code(code));
		shader_rid = _shader_tmp->get_rid();
	} else {
		String code = _generate_shader_code();
		RS::get_singleton()->shader_set_code(_shader, _inject_editor_code(code));
		shader_rid = _shader;
	}
	RS::get_singleton()->material_set_shader(_material, shader_rid);
	LOG(DEBUG, "Material rid: ", _material, ", shader rid: ", shader_rid);

	// Update custom shader params in RenderingServer
	{
		// Populate _active_params
		List<PropertyInfo> pi;
		_get_property_list(&pi);
		LOG(DEBUG_CONT, "_active_params: ", _active_params);
		Util::print_dict("_shader_params", _shader_params, DEBUG_CONT);
	}

	// Fetch saved shader parameters, converting textures to RIDs
	for (int i = 0; i < _active_params.size(); i++) {
		StringName param = _active_params[i];
		Variant value = _shader_params[param];
		if (value.get_type() == Variant::OBJECT) {
			Ref<Texture> tex = value;
			if (tex.is_valid()) {
				RS::get_singleton()->material_set_param(_material, param, tex->get_rid());
			} else {
				RS::get_singleton()->material_set_param(_material, param, Variant());
			}
		} else {
			RS::get_singleton()->material_set_param(_material, param, value);
		}
	}

	// Set specific shader parameters
	RS::get_singleton()->material_set_param(_material, "_background_mode", _world_background);

	// If no noise texture, generate one
	if (_active_params.has("noise_texture") && RS::get_singleton()->material_get_param(_material, "noise_texture").get_type() == Variant::NIL) {
		LOG(INFO, "Generating default noise_texture for shader");
		Ref<FastNoiseLite> fnoise;
		fnoise.instantiate();
		fnoise->set_noise_type(FastNoiseLite::TYPE_CELLULAR);
		fnoise->set_frequency(0.03f);
		fnoise->set_cellular_jitter(3.0f);
		fnoise->set_cellular_return_type(FastNoiseLite::RETURN_CELL_VALUE);
		fnoise->set_domain_warp_enabled(true);
		fnoise->set_domain_warp_type(FastNoiseLite::DOMAIN_WARP_SIMPLEX_REDUCED);
		fnoise->set_domain_warp_amplitude(50.f);
		fnoise->set_domain_warp_fractal_type(FastNoiseLite::DOMAIN_WARP_FRACTAL_INDEPENDENT);
		fnoise->set_domain_warp_fractal_lacunarity(1.5f);
		fnoise->set_domain_warp_fractal_gain(1.f);

		Ref<Gradient> curve;
		curve.instantiate();
		PackedFloat32Array pfa;
		pfa.push_back(0.2f);
		pfa.push_back(1.0f);
		curve->set_offsets(pfa);
		PackedColorArray pca;
		pca.push_back(Color(1.f, 1.f, 1.f, 1.f));
		pca.push_back(Color(0.f, 0.f, 0.f, 1.f));
		curve->set_colors(pca);

		Ref<NoiseTexture2D> noise_tex;
		noise_tex.instantiate();
		noise_tex->set_seamless(true);
		noise_tex->set_generate_mipmaps(true);
		noise_tex->set_noise(fnoise);
		noise_tex->set_color_ramp(curve);
		_set("noise_texture", noise_tex);
	}

	notify_property_list_changed();
}

void Terrain3DMaterial::_update_regions() {
	IS_STORAGE_INIT(VOID);
	LOG(DEBUG_CONT, "Updating region maps in shader");

	Ref<Terrain3DStorage> storage = _terrain->get_storage();
	RS::get_singleton()->material_set_param(_material, "_height_maps", storage->get_height_rid());
	RS::get_singleton()->material_set_param(_material, "_control_maps", storage->get_control_rid());
	RS::get_singleton()->material_set_param(_material, "_color_maps", storage->get_color_rid());
	LOG(DEBUG_CONT, "Height map RID: ", storage->get_height_rid());
	LOG(DEBUG_CONT, "Control map RID: ", storage->get_control_rid());
	LOG(DEBUG_CONT, "Color map RID: ", storage->get_color_rid());

	PackedInt32Array region_map = storage->get_region_map();
	LOG(DEBUG_CONT, "region_map.size(): ", region_map.size());
	if (region_map.size() != Terrain3DStorage::REGION_MAP_SIZE * Terrain3DStorage::REGION_MAP_SIZE) {
		LOG(ERROR, "Expected region_map.size() of ", Terrain3DStorage::REGION_MAP_SIZE * Terrain3DStorage::REGION_MAP_SIZE);
	}
	RS::get_singleton()->material_set_param(_material, "_region_map", region_map);
	RS::get_singleton()->material_set_param(_material, "_region_map_size", Terrain3DStorage::REGION_MAP_SIZE);
	if (Terrain3D::debug_level >= DEBUG_CONT) {
		LOG(DEBUG_CONT, "Region map");
		for (int i = 0; i < region_map.size(); i++) {
			if (region_map[i]) {
				LOG(DEBUG_CONT, "Region id: ", region_map[i], " array index: ", i);
			}
		}
	}

	TypedArray<Vector2i> region_offsets = storage->get_region_offsets();
	LOG(DEBUG_CONT, "Region_offsets size: ", region_offsets.size(), " ", region_offsets);
	RS::get_singleton()->material_set_param(_material, "_region_offsets", region_offsets);

	real_t region_size = real_t(storage->get_region_size());
	LOG(DEBUG_CONT, "Setting region size in material: ", region_size);
	RS::get_singleton()->material_set_param(_material, "_region_size", region_size);
	RS::get_singleton()->material_set_param(_material, "_region_pixel_size", 1.0f / region_size);

	real_t spacing = _terrain->get_mesh_vertex_spacing();
	LOG(DEBUG_CONT, "Setting mesh vertex spacing in material: ", spacing);
	RS::get_singleton()->material_set_param(_material, "_mesh_vertex_spacing", spacing);
	RS::get_singleton()->material_set_param(_material, "_mesh_vertex_density", 1.0f / spacing);

	_generate_region_blend_map();
}

void Terrain3DMaterial::_generate_region_blend_map() {
	IS_STORAGE_INIT_MESG("Material not initialized", VOID);
	PackedInt32Array region_map = _terrain->get_storage()->get_region_map();
	int rsize = Terrain3DStorage::REGION_MAP_SIZE;
	if (region_map.size() == rsize * rsize) {
		LOG(DEBUG_CONT, "Regenerating ", Vector2i(512, 512), " region blend map");
		Ref<Image> region_blend_img = Image::create_empty(rsize, rsize, false, Image::FORMAT_RH);
		for (int y = 0; y < rsize; y++) {
			for (int x = 0; x < rsize; x++) {
				if (region_map[y * rsize + x] > 0) {
					region_blend_img->set_pixel(x, y, COLOR_WHITE);
				}
			}
		}
		region_blend_img->resize(512, 512, Image::INTERPOLATE_TRILINEAR);
		_generated_region_blend_map.clear();
		_generated_region_blend_map.create(region_blend_img);
		RS::get_singleton()->material_set_param(_material, "_region_blend_map", _generated_region_blend_map.get_rid());
		Util::dump_gen(_generated_region_blend_map, "blend_map", DEBUG_CONT);
	}
}

// Called from signal connected in Terrain3D, emitted by texture_list
void Terrain3DMaterial::_update_texture_arrays() {
	IS_STORAGE_INIT_MESG("Material not initialized", VOID);
	Ref<Terrain3DAssets> asset_list = _terrain->get_assets();
	LOG(INFO, "Updating texture arrays in shader");
	if (asset_list.is_null()) {
		LOG(ERROR, "Asset list is null");
		return;
	}

	RS::get_singleton()->material_set_param(_material, "_texture_array_albedo", asset_list->get_albedo_array_rid());
	RS::get_singleton()->material_set_param(_material, "_texture_array_normal", asset_list->get_normal_array_rid());
	RS::get_singleton()->material_set_param(_material, "_texture_color_array", asset_list->get_texture_colors());
	RS::get_singleton()->material_set_param(_material, "_texture_uv_scale_array", asset_list->get_texture_uv_scales());
	RS::get_singleton()->material_set_param(_material, "_texture_detile_array", asset_list->get_texture_detiles());

	// Enable checkered view if texture_count is 0, disable if not
	if (asset_list->get_texture_count() == 0) {
		if (_debug_view_checkered == false) {
			set_show_checkered(true);
			LOG(DEBUG, "No textures, enabling checkered view");
		}
	} else {
		set_show_checkered(false);
		LOG(DEBUG, "Texture count >0: ", asset_list->get_texture_count(), ", disabling checkered view");
	}
}

void Terrain3DMaterial::_set_shader_parameters(const Dictionary &p_dict) {
	LOG(INFO, "Setting shader params dictionary: ", p_dict.size());
	_shader_params = p_dict;
}

///////////////////////////
// Public Functions
///////////////////////////

// This function serves as the constructor which is initialized by the class Terrain3D.
// Godot likes to create resource objects at startup, so this prevents it from creating
// uninitialized materials.
void Terrain3DMaterial::initialize(Terrain3D *p_terrain) {
	if (p_terrain != nullptr) {
		_terrain = p_terrain;
	} else {
		LOG(ERROR, "Initialization failed, p_terrain is null");
		return;
	}
	LOG(INFO, "Initializing material");
	_preload_shaders();
	_material = RS::get_singleton()->material_create();
	_shader = RS::get_singleton()->shader_create();
	_shader_tmp.instantiate();
	LOG(DEBUG, "Mat RID: ", _material, ", _shader RID: ", _shader);
	_update_shader();
	_update_regions();
}

Terrain3DMaterial::~Terrain3DMaterial() {
	IS_INIT(VOID);
	LOG(INFO, "Destroying material");
	RS::get_singleton()->free(_material);
	RS::get_singleton()->free(_shader);
	_generated_region_blend_map.clear();
}

RID Terrain3DMaterial::get_shader_rid() const {
	if (_shader_override_enabled) {
		return _shader_tmp->get_rid();
	} else {
		return _shader;
	}
}

void Terrain3DMaterial::set_world_background(WorldBackground p_background) {
	LOG(INFO, "Enable world background: ", p_background);
	_world_background = p_background;
	_update_shader();
}

void Terrain3DMaterial::set_texture_filtering(TextureFiltering p_filtering) {
	LOG(INFO, "Setting texture filtering: ", p_filtering);
	_texture_filtering = p_filtering;
	_update_shader();
}

void Terrain3DMaterial::set_auto_shader(bool p_enabled) {
	LOG(INFO, "Enable auto shader: ", p_enabled);
	_auto_shader = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_dual_scaling(bool p_enabled) {
	LOG(INFO, "Enable dual scaling: ", p_enabled);
	_dual_scaling = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::enable_shader_override(bool p_enabled) {
	LOG(INFO, "Enable shader override: ", p_enabled);
	_shader_override_enabled = p_enabled;
	if (_shader_override_enabled && _shader_override.is_null()) {
		_shader_override.instantiate();
		LOG(DEBUG, "_shader_override RID: ", _shader_override->get_rid());
	}
	_update_shader();
}

void Terrain3DMaterial::set_shader_override(const Ref<Shader> &p_shader) {
	LOG(INFO, "Setting override shader");
	_shader_override = p_shader;
	_update_shader();
}

void Terrain3DMaterial::set_shader_param(const StringName &p_name, const Variant &p_value) {
	LOG(INFO, "Setting shader parameter: ", p_name);
	_set(p_name, p_value);
}

Variant Terrain3DMaterial::get_shader_param(const StringName &p_name) const {
	LOG(INFO, "Getting shader parameter: ", p_name);
	Variant value;
	_get(p_name, value);
	return value;
}

void Terrain3DMaterial::set_show_checkered(bool p_enabled) {
	LOG(INFO, "Enable set_show_checkered: ", p_enabled);
	_debug_view_checkered = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_grey(bool p_enabled) {
	LOG(INFO, "Enable show_grey: ", p_enabled);
	_debug_view_grey = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_heightmap(bool p_enabled) {
	LOG(INFO, "Enable show_heightmap: ", p_enabled);
	_debug_view_heightmap = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_colormap(bool p_enabled) {
	LOG(INFO, "Enable show_colormap: ", p_enabled);
	_debug_view_colormap = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_roughmap(bool p_enabled) {
	LOG(INFO, "Enable show_roughmap: ", p_enabled);
	_debug_view_roughmap = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_control_texture(bool p_enabled) {
	LOG(INFO, "Enable show_control_texture: ", p_enabled);
	_debug_view_control_texture = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_control_angle(bool p_enabled) {
	LOG(INFO, "Enable show_control_angle: ", p_enabled);
	_debug_view_control_angle = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_control_scale(bool p_enabled) {
	LOG(INFO, "Enable show_control_scale: ", p_enabled);
	_debug_view_control_scale = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_control_blend(bool p_enabled) {
	LOG(INFO, "Enable show_control_blend: ", p_enabled);
	_debug_view_control_blend = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_autoshader(bool p_enabled) {
	LOG(INFO, "Enable show_autoshader: ", p_enabled);
	_debug_view_autoshader = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_navigation(bool p_enabled) {
	LOG(INFO, "Enable show_navigation: ", p_enabled);
	_show_navigation = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_texture_height(bool p_enabled) {
	LOG(INFO, "Enable show_texture_height: ", p_enabled);
	_debug_view_tex_height = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_texture_normal(bool p_enabled) {
	LOG(INFO, "Enable show_texture_normal: ", p_enabled);
	_debug_view_tex_normal = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_texture_rough(bool p_enabled) {
	LOG(INFO, "Enable show_texture_rough: ", p_enabled);
	_debug_view_tex_rough = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::set_show_vertex_grid(bool p_enabled) {
	LOG(INFO, "Enable show_vertex_grid: ", p_enabled);
	_debug_view_vertex_grid = p_enabled;
	_update_shader();
}

void Terrain3DMaterial::save() {
	LOG(DEBUG, "Generating parameter list from shaders");
	// Get shader parameters from default shader (eg world_noise)
	List<PropertyInfo>  param_list;
	RS::get_singleton()->get_shader_parameter_list(_shader,&param_list);
	// Get shader parameters from custom shader if present
	if (_shader_override.is_valid()) {
		List<PropertyInfo>  override_param_list;
		_shader_override->get_shader_uniform_list(&override_param_list, true);
		param_list.append(override_param_list);
	}

	// Remove saved shader params that don't exist in either shader
	Array keys = _shader_params.keys();
	for (int i = 0; i < keys.size(); i++) {
		bool has = false;
		StringName _name = keys[i];
		//for (int j = 0; j < param_list.size(); j++) {
		for(auto& j : param_list) {
			//Dictionary dict = j;
			StringName dname;
			//if (j < param_list.size()) {
			//	dict = param_list[j];
				//dname = dict["name"];
				if (_name == j.name) {
					has = true;
					break;
				}
			//}
		}
		if (!has) {
			LOG(DEBUG, "'", _name, "' not found in shader parameters. Removing from dictionary.");
			_shader_params.erase(_name);
		}
	}

	// Save to external resource file if used
	String path = get_path();
	if (path.get_extension() == "tres" || path.get_extension() == "res") {
		LOG(DEBUG, "Attempting to save material to external file: " + path);
		Error err;
		err = ResourceSaver::save(this, path, ResourceSaver::FLAG_COMPRESS);
		ERR_FAIL_COND(err);
		LOG(DEBUG, "ResourceSaver return error (0 is OK): ", err);
		LOG(INFO, "Finished saving material");
	}
}

///////////////////////////
// Protected Functions
///////////////////////////

// Add shader uniforms to properties. Hides uniforms that begin with _
void Terrain3DMaterial::_get_property_list(List<PropertyInfo> *p_list) const {
	Resource::_get_property_list(p_list);
	IS_INIT(VOID);
	List<PropertyInfo>  param_list;
	if (_shader_override_enabled && _shader_override.is_valid()) {
		// Get shader parameters from custom shader
		_shader_override->get_shader_uniform_list(&param_list, true);
	} else {
		// Get shader parameters from default shader (eg world_noise)
		RS::get_singleton()->get_shader_parameter_list(_shader,&param_list);
	}

	_active_params.clear();
	//for (int i = 0; i < param_list.size(); i++) {
	for(auto& j : param_list) {
		//Dictionary dict = param_list[i];
		StringName name = j.name;
		// Filter out private uniforms that start with _
		if (!name.begins_with("_")) {
			// Populate Godot's property list
			PropertyInfo pi;
			pi.name = name;
			pi.class_name = j.class_name;
			pi.type = Variant::Type(int(j.type));
			pi.hint = j.hint;
			pi.hint_string = j.hint_string;
			pi.usage = PROPERTY_USAGE_EDITOR;
			p_list->push_back(pi);

			// Populate list of public parameters for current shader
			_active_params.push_back(name);

			// Store this param in a dictionary that is saved in the resource file
			// Initially set with default value
			// Also acts as a cache for _get
			// Property usage above set to EDITOR so it won't be redundantly saved,
			// which won't get loaded since there is no bound property.
			if (!_shader_params.has(name)) {
				_property_get_revert(name, _shader_params[name]);
			}
		}
	}
	return;
}

// Flag uniforms with non-default values
// This is called 10x more than the others, so be efficient
bool Terrain3DMaterial::_property_can_revert(const StringName &p_name) const {
	IS_INIT_COND(!_active_params.has(p_name), Resource::_property_can_revert(p_name));
	RID shader;
	if (_shader_override_enabled && _shader_override.is_valid()) {
		shader = _shader_override->get_rid();
	} else {
		shader = _shader;
	}
	if (shader.is_valid()) {
		Variant default_value = RS::get_singleton()->shader_get_parameter_default(shader, p_name);
		Variant current_value = RS::get_singleton()->material_get_param(_material, p_name);
		return default_value != current_value;
	}
	return false;
}

// Provide uniform default values
bool Terrain3DMaterial::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	IS_INIT_COND(!_active_params.has(p_name), Resource::_property_get_revert(p_name, r_property));
	RID shader;
	if (_shader_override_enabled && _shader_override.is_valid()) {
		shader = _shader_override->get_rid();
	} else {
		shader = _shader;
	}
	if (shader.is_valid()) {
		r_property = RS::get_singleton()->shader_get_parameter_default(shader, p_name);
		return true;
	}
	return false;
}

bool Terrain3DMaterial::_set(const StringName &p_name, const Variant &p_property) {
	IS_INIT_COND(!_active_params.has(p_name), Resource::_set(p_name, p_property));
	if (p_property.get_type() == Variant::NIL) {
		RS::get_singleton()->material_set_param(_material, p_name, Variant());
		_shader_params.erase(p_name);
		return true;
	}

	// If value is an object, assume a Texture. RS only wants RIDs, but
	// Inspector wants the object, so set the RID and save the latter for _get
	if (p_property.get_type() == Variant::OBJECT) {
		Ref<Texture> tex = p_property;
		if (tex.is_valid()) {
			_shader_params[p_name] = tex;
			RS::get_singleton()->material_set_param(_material, p_name, tex->get_rid());
		} else {
			RS::get_singleton()->material_set_param(_material, p_name, Variant());
		}
	} else {
		_shader_params[p_name] = p_property;
		RS::get_singleton()->material_set_param(_material, p_name, p_property);
	}
	return true;
}

// This is called 200x more than the others, every second the material is open in the
// inspector, so be efficient
bool Terrain3DMaterial::_get(const StringName &p_name, Variant &r_property) const {
	IS_INIT_COND(!_active_params.has(p_name), Resource::_get(p_name, r_property));

	r_property = RS::get_singleton()->material_get_param(_material, p_name);
	// Material server only has RIDs, but inspector needs objects for things like Textures
	// So if its an RID, return the object
	if (r_property.get_type() == Variant::RID && _shader_params.has(p_name)) {
		r_property = _shader_params[p_name];
	}
	return true;
}

void Terrain3DMaterial::_bind_methods() {
	BIND_ENUM_CONSTANT(NONE);
	BIND_ENUM_CONSTANT(FLAT);
	BIND_ENUM_CONSTANT(NOISE);
	BIND_ENUM_CONSTANT(LINEAR);
	BIND_ENUM_CONSTANT(NEAREST);

	// Private
	ClassDB::bind_method(D_METHOD("_set_shader_parameters", "dict"), &Terrain3DMaterial::_set_shader_parameters);
	ClassDB::bind_method(D_METHOD("_get_shader_parameters"), &Terrain3DMaterial::_get_shader_parameters);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_shader_parameters", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "_set_shader_parameters", "_get_shader_parameters");

	// Public
	ClassDB::bind_method(D_METHOD("get_material_rid"), &Terrain3DMaterial::get_material_rid);
	ClassDB::bind_method(D_METHOD("get_shader_rid"), &Terrain3DMaterial::get_shader_rid);
	ClassDB::bind_method(D_METHOD("get_region_blend_map"), &Terrain3DMaterial::get_region_blend_map);

	ClassDB::bind_method(D_METHOD("set_world_background", "background"), &Terrain3DMaterial::set_world_background);
	ClassDB::bind_method(D_METHOD("get_world_background"), &Terrain3DMaterial::get_world_background);
	ClassDB::bind_method(D_METHOD("set_texture_filtering", "filtering"), &Terrain3DMaterial::set_texture_filtering);
	ClassDB::bind_method(D_METHOD("get_texture_filtering"), &Terrain3DMaterial::get_texture_filtering);
	ClassDB::bind_method(D_METHOD("set_auto_shader", "enabled"), &Terrain3DMaterial::set_auto_shader);
	ClassDB::bind_method(D_METHOD("get_auto_shader"), &Terrain3DMaterial::get_auto_shader);
	ClassDB::bind_method(D_METHOD("set_dual_scaling", "enabled"), &Terrain3DMaterial::set_dual_scaling);
	ClassDB::bind_method(D_METHOD("get_dual_scaling"), &Terrain3DMaterial::get_dual_scaling);

	ClassDB::bind_method(D_METHOD("enable_shader_override", "enabled"), &Terrain3DMaterial::enable_shader_override);
	ClassDB::bind_method(D_METHOD("is_shader_override_enabled"), &Terrain3DMaterial::is_shader_override_enabled);
	ClassDB::bind_method(D_METHOD("set_shader_override", "shader"), &Terrain3DMaterial::set_shader_override);
	ClassDB::bind_method(D_METHOD("get_shader_override"), &Terrain3DMaterial::get_shader_override);

	ClassDB::bind_method(D_METHOD("set_shader_param", "name", "value"), &Terrain3DMaterial::set_shader_param);
	ClassDB::bind_method(D_METHOD("get_shader_param", "name"), &Terrain3DMaterial::get_shader_param);

	ClassDB::bind_method(D_METHOD("set_show_checkered", "enabled"), &Terrain3DMaterial::set_show_checkered);
	ClassDB::bind_method(D_METHOD("get_show_checkered"), &Terrain3DMaterial::get_show_checkered);
	ClassDB::bind_method(D_METHOD("set_show_grey", "enabled"), &Terrain3DMaterial::set_show_grey);
	ClassDB::bind_method(D_METHOD("get_show_grey"), &Terrain3DMaterial::get_show_grey);
	ClassDB::bind_method(D_METHOD("set_show_heightmap", "enabled"), &Terrain3DMaterial::set_show_heightmap);
	ClassDB::bind_method(D_METHOD("get_show_heightmap"), &Terrain3DMaterial::get_show_heightmap);
	ClassDB::bind_method(D_METHOD("set_show_colormap", "enabled"), &Terrain3DMaterial::set_show_colormap);
	ClassDB::bind_method(D_METHOD("get_show_colormap"), &Terrain3DMaterial::get_show_colormap);
	ClassDB::bind_method(D_METHOD("set_show_roughmap", "enabled"), &Terrain3DMaterial::set_show_roughmap);
	ClassDB::bind_method(D_METHOD("get_show_roughmap"), &Terrain3DMaterial::get_show_roughmap);
	ClassDB::bind_method(D_METHOD("set_show_control_texture", "enabled"), &Terrain3DMaterial::set_show_control_texture);
	ClassDB::bind_method(D_METHOD("get_show_control_texture"), &Terrain3DMaterial::get_show_control_texture);
	ClassDB::bind_method(D_METHOD("set_show_control_angle", "enabled"), &Terrain3DMaterial::set_show_control_angle);
	ClassDB::bind_method(D_METHOD("get_show_control_angle"), &Terrain3DMaterial::get_show_control_angle);
	ClassDB::bind_method(D_METHOD("set_show_control_scale", "enabled"), &Terrain3DMaterial::set_show_control_scale);
	ClassDB::bind_method(D_METHOD("get_show_control_scale"), &Terrain3DMaterial::get_show_control_scale);
	ClassDB::bind_method(D_METHOD("set_show_control_blend", "enabled"), &Terrain3DMaterial::set_show_control_blend);
	ClassDB::bind_method(D_METHOD("get_show_control_blend"), &Terrain3DMaterial::get_show_control_blend);
	ClassDB::bind_method(D_METHOD("set_show_autoshader", "enabled"), &Terrain3DMaterial::set_show_autoshader);
	ClassDB::bind_method(D_METHOD("get_show_autoshader"), &Terrain3DMaterial::get_show_autoshader);
	ClassDB::bind_method(D_METHOD("set_show_navigation", "enabled"), &Terrain3DMaterial::set_show_navigation);
	ClassDB::bind_method(D_METHOD("get_show_navigation"), &Terrain3DMaterial::get_show_navigation);
	ClassDB::bind_method(D_METHOD("set_show_texture_height", "enabled"), &Terrain3DMaterial::set_show_texture_height);
	ClassDB::bind_method(D_METHOD("get_show_texture_height"), &Terrain3DMaterial::get_show_texture_height);
	ClassDB::bind_method(D_METHOD("set_show_texture_normal", "enabled"), &Terrain3DMaterial::set_show_texture_normal);
	ClassDB::bind_method(D_METHOD("get_show_texture_normal"), &Terrain3DMaterial::get_show_texture_normal);
	ClassDB::bind_method(D_METHOD("set_show_texture_rough", "enabled"), &Terrain3DMaterial::set_show_texture_rough);
	ClassDB::bind_method(D_METHOD("get_show_texture_rough"), &Terrain3DMaterial::get_show_texture_rough);
	ClassDB::bind_method(D_METHOD("set_show_vertex_grid", "enabled"), &Terrain3DMaterial::set_show_vertex_grid);
	ClassDB::bind_method(D_METHOD("get_show_vertex_grid"), &Terrain3DMaterial::get_show_vertex_grid);

	ClassDB::bind_method(D_METHOD("save"), &Terrain3DMaterial::save);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "world_background", PROPERTY_HINT_ENUM, "None,Flat,Noise"), "set_world_background", "get_world_background");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_filtering", PROPERTY_HINT_ENUM, "Linear,Nearest"), "set_texture_filtering", "get_texture_filtering");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_shader", PROPERTY_HINT_NONE), "set_auto_shader", "get_auto_shader");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dual_scaling", PROPERTY_HINT_NONE), "set_dual_scaling", "get_dual_scaling");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shader_override_enabled", PROPERTY_HINT_NONE), "enable_shader_override", "is_shader_override_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shader_override", PROPERTY_HINT_RESOURCE_TYPE, "Shader"), "set_shader_override", "get_shader_override");

	ADD_GROUP("Debug Views", "show_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_checkered", PROPERTY_HINT_NONE), "set_show_checkered", "get_show_checkered");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_grey", PROPERTY_HINT_NONE), "set_show_grey", "get_show_grey");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_heightmap", PROPERTY_HINT_NONE), "set_show_heightmap", "get_show_heightmap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_colormap", PROPERTY_HINT_NONE), "set_show_colormap", "get_show_colormap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_roughmap", PROPERTY_HINT_NONE), "set_show_roughmap", "get_show_roughmap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_control_texture", PROPERTY_HINT_NONE), "set_show_control_texture", "get_show_control_texture");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_control_angle", PROPERTY_HINT_NONE), "set_show_control_angle", "get_show_control_angle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_control_scale", PROPERTY_HINT_NONE), "set_show_control_scale", "get_show_control_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_control_blend", PROPERTY_HINT_NONE), "set_show_control_blend", "get_show_control_blend");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_autoshader", PROPERTY_HINT_NONE), "set_show_autoshader", "get_show_autoshader");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_navigation", PROPERTY_HINT_NONE), "set_show_navigation", "get_show_navigation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_texture_height", PROPERTY_HINT_NONE), "set_show_texture_height", "get_show_texture_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_texture_normal", PROPERTY_HINT_NONE), "set_show_texture_normal", "get_show_texture_normal");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_texture_rough", PROPERTY_HINT_NONE), "set_show_texture_rough", "get_show_texture_rough");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_vertex_grid", PROPERTY_HINT_NONE), "set_show_vertex_grid", "get_show_vertex_grid");
}
