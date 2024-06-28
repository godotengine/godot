// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#include "core/io/resource_saver.h"
#include <map>

#include "logger.h"
#include "terrain_3d_instancer.h"
#include "terrain_3d_util.h"

///////////////////////////
// Private Functions
///////////////////////////

void Terrain3DInstancer::_rebuild_mmis() {
	destroy();
	_update_mmis();
}

// Creates MMIs based on stored Multimeshes
void Terrain3DInstancer::_update_mmis() {
	IS_STORAGE_INIT(VOID);
	LOG(INFO, "Updating MMIs");
	// For all region_offsets
	Dictionary region_dict = _terrain->get_storage()->get_multimeshes();
	LOG(DEBUG, "Updating MMIs from regions: ", region_dict);
	Array keys = region_dict.keys();
	for (int r = 0; r < keys.size(); r++) {
		Vector2i region_offset = keys[r];
		Dictionary mesh_dict = region_dict.get(region_offset, Dictionary());
		LOG(DEBUG, "Updating MMIs from meshes: ", mesh_dict);

		// For all mesh ids
		for (int m = 0; m < mesh_dict.keys().size(); m++) {
			int mesh_id = mesh_dict.keys()[m];
			bool fail = false;
			Ref<MultiMesh> mm = mesh_dict.get(mesh_id, Ref<MultiMesh>());
			if (mm.is_null()) {
				LOG(DEBUG, "Dictionary for mesh id ", mesh_id, " is null, skipping");
				fail = true;
			}
			Ref<Terrain3DMeshAsset> ma = _terrain->get_assets()->get_mesh_asset(mesh_id);
			Ref<Mesh> mesh;
			if (ma.is_valid()) {
				mesh = ma->get_mesh();
				if (mesh.is_null()) {
					LOG(WARN, "MeshAsset ", mesh_id, " valid but mesh is null, skipping");
					fail = true;
				}
			} else {
				LOG(WARN, "MeshAsset ", mesh_id, " is null, skipping");
				fail = true;
			}
			if (fail) {
				clear_by_offset(region_offset, mesh_id);
				continue;
			}

			// Update meshes on all in case IDs changed.
			mm->set_mesh(mesh);
			// Set MMs in MMIs, creating any missing MMIs
			Vector3i mmi_key = Vector3i(region_offset.x, region_offset.y, mesh_id);
			MultiMeshInstance3D *mmi;
			if (!_mmis.has(mmi_key)) {
				LOG(DEBUG, "No MMI found, creating new MultiMeshInstance3D, attaching to tree");
				mmi = memnew(MultiMeshInstance3D);
				_terrain->add_child(mmi);
				_mmis[mmi_key] = mmi;
				LOG(DEBUG, _mmis);
			}
			mmi = cast_to<MultiMeshInstance3D>(_mmis[mmi_key]);
			mmi->set_multimesh(mm);
			mmi->set_cast_shadows_setting(ma->get_cast_shadows());
		}
	}
}

void Terrain3DInstancer::_destroy_mmi_by_region_id(int p_region_id, int p_mesh_id) {
	Vector2i region_offset = _terrain->get_storage()->get_region_offset_from_index(p_region_id);
	_destroy_mmi_by_offset(region_offset, p_mesh_id);
}

void Terrain3DInstancer::_destroy_mmi_by_offset(Vector2i p_region_offset, int p_mesh_id) {
	Vector3i mmi_key = Vector3i(p_region_offset.x, p_region_offset.y, p_mesh_id);
	LOG(DEBUG, "Deleting MMI at: ", p_region_offset, " mesh_id: ", p_mesh_id);
	MultiMeshInstance3D *mmi = cast_to<MultiMeshInstance3D>(_mmis[mmi_key]);
	bool result = _mmis.erase(mmi_key);
	LOG(DEBUG, "Removing mmi from dictionary: ", mmi, ", success: ", result);
	result = remove_from_tree(mmi);
	LOG(DEBUG, "Removing from tree, success: ", result);
	result = memdelete_safely(mmi);
	LOG(DEBUG, "Deleting MMI, success: ", result);
}

///////////////////////////
// Public Functions
///////////////////////////

Terrain3DInstancer::~Terrain3DInstancer() {
	destroy();
}

void Terrain3DInstancer::initialize(Terrain3D *p_terrain) {
	if (p_terrain) {
		_terrain = p_terrain;
	}
	IS_STORAGE_INIT_MESG("Terrain or storage not ready yet", VOID);
	LOG(INFO, "Initializing Instancer");
	_update_mmis();
}

void Terrain3DInstancer::destroy() {
	IS_STORAGE_INIT(VOID);
	LOG(INFO, "Destroying all MMIs");
	while (_mmis.size() > 0) {
		Vector3i key = _mmis.keys()[0];
		_destroy_mmi_by_offset(Vector2i(key.x, key.y), key.z);
	}
	_mmis.clear();
}

void Terrain3DInstancer::update_multimesh(Vector2i p_region_offset, int p_mesh_id, TypedArray<Transform3D> p_xforms, TypedArray<Color> p_colors, bool p_clear) {
	IS_STORAGE_INIT(VOID);

	// Merge with old instances
	if (!p_clear) {
		Ref<MultiMesh> multimesh = get_multimesh(p_region_offset, p_mesh_id);
		if (multimesh.is_valid()) {
			uint32_t old_count = multimesh->get_instance_count();
			LOG(DEBUG_CONT, "Merging w/ old instances: ", old_count, ": ", multimesh);
			for (int i = 0; i < old_count; i++) {
				p_xforms.push_back(multimesh->get_instance_transform(i));
				p_colors.push_back(multimesh->get_instance_color(i));
			}
		}
	}

	// Erase empties if no transforms
	if (p_xforms.size() == 0) {
		clear_by_offset(p_region_offset, p_mesh_id);
		return;
	}

	// Create a new Multimesh
	Ref<MultiMesh> mm;
	mm.instantiate();
	mm->set_transform_format(MultiMesh::TRANSFORM_3D);
	mm->set_use_colors(true);
	mm->set_instance_count(p_xforms.size());
	Ref<Terrain3DMeshAsset> mesh_asset = _terrain->get_assets()->get_mesh_asset(p_mesh_id);
	Ref<Mesh> mesh = mesh_asset->get_mesh();
	mm->set_mesh(mesh);
	for (int i = 0; i < p_xforms.size(); i++) {
		mm->set_instance_transform(i, p_xforms[i]);
		mm->set_instance_color(i, p_colors[i]);
	}
	LOG(DEBUG_CONT, "Setting multimesh in region: ", p_region_offset, ", mesh_id: ", p_mesh_id, " instance count: ", p_xforms.size(), " mm: ", mm);

	Dictionary region_dict = _terrain->get_storage()->get_multimeshes();
	Dictionary mesh_dict = region_dict.get(p_region_offset, Dictionary());
	mesh_dict[p_mesh_id] = mm;
	region_dict[p_region_offset] = mesh_dict;

	// Set MultiMeshInstance3D
	MultiMeshInstance3D *mmi = get_multimesh_instance(p_region_offset, p_mesh_id);
	if (!mmi) {
		LOG(DEBUG, "No MMI found, creating new MultiMeshInstance3D, attaching to tree");
		mmi = memnew(MultiMeshInstance3D);
		mmi->set_multimesh(mm);
		mmi->set_cast_shadows_setting(mesh_asset->get_cast_shadows());
		_terrain->add_child(mmi);
		Vector3i key = Vector3i(p_region_offset.x, p_region_offset.y, p_mesh_id);
		_mmis[key] = mmi;
	}
	mmi->set_multimesh(mm);
}

Ref<MultiMesh> Terrain3DInstancer::get_multimesh(Vector3 p_global_position, int p_mesh_id) {
	Vector2i region_offset = _terrain->get_storage()->get_region_offset(p_global_position);
	return get_multimesh(region_offset, p_mesh_id);
}

Ref<MultiMesh> Terrain3DInstancer::get_multimesh(Vector2i p_region_offset, int p_mesh_id) {
	IS_STORAGE_INIT(Ref<MultiMesh>());
	Dictionary mesh_dict = _terrain->get_storage()->get_multimeshes().get(p_region_offset, Dictionary());
	Ref<MultiMesh> mm = mesh_dict.get(p_mesh_id, Ref<MultiMesh>());
	LOG(DEBUG_CONT, "Retrieving MultiMesh at region: ", p_region_offset, " mesh_id: ", p_mesh_id, " : ", mm);
	return mm;
}

MultiMeshInstance3D *Terrain3DInstancer::get_multimesh_instance(Vector3 p_global_position, int p_mesh_id) {
	Vector2i region_offset = _terrain->get_storage()->get_region_offset(p_global_position);
	return get_multimesh_instance(region_offset, p_mesh_id);
}

MultiMeshInstance3D *Terrain3DInstancer::get_multimesh_instance(Vector2i p_region_offset, int p_mesh_id) {
	Vector3i key = Vector3i(p_region_offset.x, p_region_offset.y, p_mesh_id);
	MultiMeshInstance3D *mmi = cast_to<MultiMeshInstance3D>(_mmis[key]);
	LOG(DEBUG_CONT, "Retrieving MultiMeshInstance3D at region: ", p_region_offset, " mesh_id: ", p_mesh_id, " : ", mmi);
	return mmi;
}

void Terrain3DInstancer::add_instances(Vector3 p_global_position, Dictionary p_params) {
	IS_STORAGE_INIT_MESG("Instancer isn't initialized.", VOID);

	int mesh_id = p_params.get("asset_id", 0);
	if (mesh_id < 0 || mesh_id >= _terrain->get_assets()->get_mesh_count()) {
		LOG(ERROR, "Mesh ID out of range: ", mesh_id, ", valid: 0 to ", _terrain->get_assets()->get_mesh_count() - 1);
		return;
	}
	Ref<Terrain3DMeshAsset> mesh_asset = _terrain->get_assets()->get_mesh_asset(mesh_id);

	real_t brush_size = CLAMP(real_t(p_params.get("size", 10.f)), 2.f, 4096); // Meters
	real_t radius = brush_size * .4f; // Ring1's inner radius
	real_t strength = CLAMP(real_t(p_params.get("strength", .1f)), .01f, 100.f); // (premul) 1-10k%
	real_t fixed_scale = CLAMP(real_t(p_params.get("fixed_scale", 100.f)) * .01f, .01f, 100.f); // 1-10k%
	real_t random_scale = CLAMP(real_t(p_params.get("random_scale", 0.f)) * .01f, 0.f, 10.f); // +/- 1000%
	real_t density = CLAMP(.1f * brush_size * strength * mesh_asset->get_relative_density() /
					MAX(0.01f, fixed_scale + .5f * random_scale),
			.001f, 1000.f);

	// Density based on strength, mesh AABB and input scale determines how many to place, even fractional
	uint32_t count = _get_count(density);
	if (count <= 0) {
		return;
	}
	LOG(DEBUG_CONT, "Adding ", count, " instances at ", p_global_position);

	real_t fixed_spin = CLAMP(real_t(p_params.get("fixed_spin", 0.f)), .0f, 360.f); // degrees
	real_t random_spin = CLAMP(real_t(p_params.get("random_spin", 360.f)), 0.f, 360.f); // degrees
	real_t fixed_angle = CLAMP(real_t(p_params.get("fixed_angle", 0.f)), -180.f, 180.f); // degrees
	real_t random_angle = CLAMP(real_t(p_params.get("random_angle", 10.f)), 0.f, 180.f); // degrees
	bool align_to_normal = bool(p_params.get("align_to_normal", false));

	real_t height_offset = CLAMP(real_t(p_params.get("height_offset", 0.f)), -100.0f, 100.f); // meters
	real_t random_height = CLAMP(real_t(p_params.get("random_height", 0.f)), 0.f, 100.f); // meters

	Color vertex_color = Color(p_params.get("vertex_color", COLOR_WHITE));
	real_t random_hue = CLAMP(real_t(p_params.get("random_hue", 0.f)) / 360.f, 0.f, 1.f); // degrees -> 0-1
	real_t random_darken = CLAMP(real_t(p_params.get("random_darken", 0.f)) * .01f, 0.f, 1.f); // 0-100%

	TypedArray<Transform3D> xforms;
	TypedArray<Color> colors;
	for (int i = 0; i < count; i++) {
		Transform3D t;

		// Get random XZ position and height in a circle
		real_t r_radius = radius * sqrt(Math::randf());
		real_t r_theta = Math::randf() * Math_TAU;
		Vector3 rand_vec = Vector3(r_radius * cos(r_theta), 0.f, r_radius * sin(r_theta));
		Vector3 position = p_global_position + rand_vec;
		// Get height, but skip holes
		real_t height = _terrain->get_storage()->get_height(position);
		if (std::isnan(height)) {
			continue;
		} else {
			position.y = height;
		}

		// Orientation
		Vector3 normal = Vector3(0.f, 1.f, 0.f);
		if (align_to_normal) {
			normal = _terrain->get_storage()->get_normal(position);
			if (std::isnan(normal.x)) {
				normal = Vector3(0.f, 1.f, 0.f);
			} else {
				normal = normal.normalized();
				Vector3 z_axis = Vector3(0.f, 0.f, 1.f);
				Vector3 x_axis = -z_axis.cross(normal);
				t.basis = Basis(x_axis, normal, z_axis).orthonormalized();
			}
		}
		real_t spin = (fixed_spin + random_spin * Math::randf()) * Math_PI / 180.f;
		if (abs(spin) > 0.001f) {
			t.basis = t.basis.rotated(normal, spin);
		}
		real_t angle = (fixed_angle + random_angle * (2.f * Math::randf() - 1.f)) * Math_PI / 180.f;
		if (abs(angle) > 0.001f) {
			t.basis = t.basis.rotated(t.basis.get_column(0), angle); // Rotate pitch, X-axis
		}

		// Scale
		real_t t_scale = CLAMP(fixed_scale + random_scale * (2.f * Math::randf() - 1.f), 0.01f, 10.f);
		t = t.scaled(Vector3(t_scale, t_scale, t_scale));

		// Position
		real_t offset = height_offset + mesh_asset->get_height_offset() +
				random_height * (2.f * Math::randf() - 1.f);
		position += t.basis.get_column(1) * offset; // Offset along UP axis
		t = t.translated(position);

		// Color
		Color col = vertex_color;
		col.set_v(CLAMP(col.get_v() - random_darken * Math::randf(), 0.f, 1.f));
		col.set_h(fmod(col.get_h() + random_hue * (2.f * Math::randf() - 1.f), 1.f));

		xforms.push_back(t);
		colors.push_back(col);
	}

	if (xforms.size() > 0) {
		Vector2i region_offset = _terrain->get_storage()->get_region_offset(p_global_position);
		update_multimesh(region_offset, mesh_id, xforms, colors);
	}
	_terrain->get_storage()->set_modified();
}

void Terrain3DInstancer::remove_instances(Vector3 p_global_position, Dictionary p_params) {
	IS_STORAGE_INIT_MESG("Instancer isn't initialized.", VOID);

	int mesh_id = p_params.get("asset_id", 0);
	if (mesh_id < 0 || mesh_id >= _terrain->get_assets()->get_mesh_count()) {
		LOG(ERROR, "Mesh ID out of range: ", mesh_id, ", valid: 0 to ", _terrain->get_assets()->get_mesh_count() - 1);
		return;
	}
	Ref<Terrain3DMeshAsset> mesh_asset = _terrain->get_assets()->get_mesh_asset(mesh_id);

	real_t brush_size = CLAMP(real_t(p_params.get("size", 10.f)), 2.f, 4096); // Meters
	real_t radius = brush_size * .4f; // Ring1's inner radius
	real_t strength = CLAMP(real_t(p_params.get("strength", .1f)), .01f, 100.f); // (premul) 1-10k%
	real_t fixed_scale = CLAMP(real_t(p_params.get("fixed_scale", 100.f)) * .01f, .01f, 100.f); // 1-10k%
	real_t random_scale = CLAMP(real_t(p_params.get("random_scale", 0.f)) * .01f, 0.f, 10.f); // +/- 1000%
	real_t density = CLAMP(.1f * brush_size * strength * mesh_asset->get_relative_density() /
					MAX(0.01f, fixed_scale + .5f * random_scale),
			.001f, 1000.f);

	// Density based on strength, mesh AABB and input scale determines how many to place, even fractional
	uint32_t count = _get_count(density);
	if (count <= 0) {
		return;
	}

	LOG(DEBUG_CONT, "Removing ", count, " instances from ", p_global_position);
	Ref<MultiMesh> multimesh = get_multimesh(p_global_position, mesh_id);
	if (multimesh.is_null()) {
		LOG(DEBUG_CONT, "Multimesh is already null. doing nothing");
		return;
	}

	TypedArray<Transform3D> xforms;
	TypedArray<Color> colors;
	for (int i = 0; i < multimesh->get_instance_count(); i++) {
		Transform3D t = multimesh->get_instance_transform(i);
		// If quota not yet met and instance within a cylinder radius, remove it
		Vector2 origin2d = Vector2(t.origin.x, t.origin.z);
		Vector2 mouse2d = Vector2(p_global_position.x, p_global_position.z);
		if (count > 0 && (origin2d - mouse2d).length() < radius) {
			count--;
			continue;
		} else {
			xforms.push_back(t);
			colors.push_back(multimesh->get_instance_color(i));
		}
	}

	Vector2i region_offset = _terrain->get_storage()->get_region_offset(p_global_position);
	if (xforms.size() == 0) {
		LOG(DEBUG, "Removed all instances, erasing multimesh in region");
		clear_by_offset(region_offset, mesh_id);
	} else {
		update_multimesh(region_offset, mesh_id, xforms, colors, true);
	}
	_terrain->get_storage()->set_modified();
}

void Terrain3DInstancer::add_transforms(int p_mesh_id, TypedArray<Transform3D> p_xforms, TypedArray<Color> p_colors) {
	IS_STORAGE_INIT_MESG("Instancer isn't initialized.", VOID);
	if (p_xforms.size() == 0) {
		return;
	}
	if (p_mesh_id < 0 || p_mesh_id >= _terrain->get_assets()->get_mesh_count()) {
		LOG(ERROR, "Mesh ID out of range: ", p_mesh_id, ", valid: 0 to ", _terrain->get_assets()->get_mesh_count() - 1);
		return;
	}

	LOG(INFO, "Separating ", p_xforms.size(), " transforms and ", p_colors.size(), " colors into regions");
	Ref<Terrain3DMeshAsset> mesh_asset = _terrain->get_assets()->get_mesh_asset(p_mesh_id);

	// Separate incoming transforms/colors into regions
	// DEPRECATED 0.9.2 - Replace in Godot 4.3
	// Use std::map because Godot Dictionaries copy TypedArrays
	// See https://github.com/godotengine/godot-cpp/pull/1483
	// TODO Check all Dictionaries w/ TypedArrays
	// Dictionary xforms_dict;
	// Dictionary colors_dict;
	std::map<Vector2i, TypedArray<Transform3D>> xforms_dict;
	std::map<Vector2i, TypedArray<Color>> colors_dict;

	for (int i = 0; i < p_xforms.size(); i++) {
		// Get xform/color
		Transform3D trns = p_xforms[i];
		trns.origin += trns.basis.get_column(1) * mesh_asset->get_height_offset(); // Offset along UP axis
		Color col = COLOR_WHITE;
		if (p_colors.size() > i) {
			col = p_colors[i];
		}

		// Get region
		Vector2i region_offset = _terrain->get_storage()->get_region_offset(trns.origin);

		// DEPRECATED 0.9.2 - Replace in Godot 4.3
		//if (!xforms_dict.has(region_offset)) {
		if (xforms_dict.count(region_offset) == 0) {
			xforms_dict[region_offset] = TypedArray<Transform3D>();
			colors_dict[region_offset] = TypedArray<Color>();
		}

		TypedArray<Transform3D> xforms = xforms_dict[region_offset];
		TypedArray<Color> colors = colors_dict[region_offset];
		xforms.push_back(trns);
		colors.push_back(col);
	}

	// Merge incoming transforms with existing transforms

	// DEPRECATED 0.9.2 - Replace in Godot 4.3
	//for (int i = 0; i < xforms_dict.keys().size(); i++) {
	//	Vector2i region_offset = xforms_dict.keys()[i];
	//	TypedArray<Transform3D> xforms = xforms_dict[region_offset];
	for (auto const &[region_offset, xforms] : xforms_dict) {
		TypedArray<Color> colors = colors_dict[region_offset];
		LOG(DEBUG, "Adding ", xforms.size(), " transforms to region offset: ", region_offset);
		update_multimesh(region_offset, p_mesh_id, xforms, colors);
	}

	_terrain->get_storage()->set_modified();
}

void Terrain3DInstancer::add_multimesh(int p_mesh_id, Ref<MultiMesh> p_multimesh, Transform3D p_xform) {
	LOG(INFO, "Extracting ", p_multimesh->get_instance_count(), " transforms from multimesh");
	TypedArray<Transform3D> xforms;
	TypedArray<Color> colors;
	for (int i = 0; i < p_multimesh->get_instance_count(); i++) {
		Transform3D t = p_xform;
		xforms.push_back(t * p_multimesh->get_instance_transform(i));
		Color c = COLOR_WHITE;
		if (p_multimesh->is_using_colors()) {
			c = p_multimesh->get_instance_color(i);
		}
		colors.push_back(c);
	}
	add_transforms(p_mesh_id, xforms, colors);
}

void Terrain3DInstancer::set_cast_shadows(int p_mesh_id, GeometryInstance3D::ShadowCastingSetting p_cast_shadows) {
	LOG(INFO, "Setting shadow casting on MMIS with mesh: ", p_mesh_id, " to mode: ", p_cast_shadows);
	Array keys = _mmis.keys();
	for (int i = 0; i < keys.size(); i++) {
		Vector3i key = keys[i];
		if (key.z == p_mesh_id) {
			MultiMeshInstance3D *mmi = cast_to<MultiMeshInstance3D>(_mmis[key]);
			if (mmi) {
				mmi->set_cast_shadows_setting(p_cast_shadows);
			}
		}
	}
}

void Terrain3DInstancer::clear_by_mesh(int p_mesh_id) {
	LOG(INFO, "Deleting Multimeshes in all regions with mesh_id: ", p_mesh_id);
	Dictionary region_dict = _terrain->get_storage()->get_multimeshes();
	Array offsets = region_dict.keys();
	for (int i = 0; i < offsets.size(); i++) {
		clear_by_offset(offsets[i], p_mesh_id);
	}
}

void Terrain3DInstancer::clear_by_region_id(int p_region_id, int p_mesh_id) {
	Vector2i region_offset = _terrain->get_storage()->get_region_offset_from_index(p_region_id);
	clear_by_offset(region_offset, p_mesh_id);
}

void Terrain3DInstancer::clear_by_offset(Vector2i p_region_offset, int p_mesh_id) {
	LOG(INFO, "Deleting Multimeshes w/ mesh_id: ", p_mesh_id, " in region: ", p_region_offset);
	Dictionary region_dict = _terrain->get_storage()->get_multimeshes();
	LOG(DEBUG, "Original region_dict: ", region_dict);
	Dictionary mesh_dict = region_dict[p_region_offset];
	mesh_dict.erase(p_mesh_id);
	if (mesh_dict.is_empty()) {
		LOG(DEBUG, "No more multimeshes in region, removing region dictionary");
		region_dict.erase(p_region_offset);
	}
	LOG(DEBUG, "Final region_dict: ", region_dict);
	_destroy_mmi_by_offset(p_region_offset, p_mesh_id);
}

void Terrain3DInstancer::print_multimesh_buffer(MultiMeshInstance3D *p_mmi) {
	if (p_mmi == nullptr) {
		return;
	}
	Ref<MultiMesh> mm = p_mmi->get_multimesh();
	PackedRealArray b = mm->get_buffer();
	VariantUtilityFunctions::_print("MM instance count: ", mm->get_instance_count());
	int mmsize = b.size();
	if (mmsize <= 12 || mmsize % 12 != 0) {
		VariantUtilityFunctions::_print("MM buffer size not a multiple of 12: ", mmsize);
		return;
	}
	for (int i = 0; i < mmsize; i += 12) {
		Transform3D tfm;
		tfm.set(b[i + 0], b[i + 1], b[i + 2], // basis x
				b[i + 4], b[i + 5], b[i + 6], // basis y
				b[i + 8], b[i + 9], b[i + 10], // basis z
				b[i + 3], b[i + 7], b[i + 11]); // origin
		VariantUtilityFunctions::_print(i / 12, ": ", tfm);
	}
}

///////////////////////////
// Protected Functions
///////////////////////////

void Terrain3DInstancer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("clear_by_mesh", "mesh_id"), &Terrain3DInstancer::clear_by_mesh);
	ClassDB::bind_method(D_METHOD("clear_by_region_id", "region_id", "mesh_id"), &Terrain3DInstancer::clear_by_region_id);
	ClassDB::bind_method(D_METHOD("clear_by_offset", "region_offset", "mesh_id"), &Terrain3DInstancer::clear_by_offset);
	ClassDB::bind_method(D_METHOD("add_instances", "global_position", "params"), &Terrain3DInstancer::add_instances);
	ClassDB::bind_method(D_METHOD("remove_instances", "global_position", "params"), &Terrain3DInstancer::remove_instances);
	ClassDB::bind_method(D_METHOD("add_transforms", "mesh_id", "transforms", "colors"), &Terrain3DInstancer::add_transforms, DEFVAL(TypedArray<Color>()));
	ClassDB::bind_method(D_METHOD("add_multimesh", "mesh_id", "multimesh", "transform"), &Terrain3DInstancer::add_multimesh, DEFVAL(Transform3D()));
	ClassDB::bind_method(D_METHOD("set_cast_shadows", "mesh_id", "mode"), &Terrain3DInstancer::set_cast_shadows);
}
