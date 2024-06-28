// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_MESH_ASSET_CLASS_H
#define TERRAIN3D_MESH_ASSET_CLASS_H

#include "scene/resources/packed_scene.h"
#include "constants.h"
#include "terrain_3d_asset_resource.h"

using namespace godot;

class Terrain3DMeshAsset : public Terrain3DAssetResource {
	GDCLASS(Terrain3DMeshAsset, Terrain3DAssetResource);
	CLASS_NAME();
	friend class Terrain3DAssets;

public:
	enum GenType {
		TYPE_NONE,
		TYPE_TEXTURE_CARD,
		TYPE_MAX,
	};

private:
	// Saved data
	float _height_offset = 0.f;
	GeometryInstance3D::ShadowCastingSetting _cast_shadows = GeometryInstance3D::SHADOW_CASTING_SETTING_ON;
	GenType _generated_type = TYPE_NONE;
	int _generated_faces = 1;
	Vector2 _generated_size = Vector2(1.f, 1.f);
	Ref<PackedScene> _packed_scene;
	Ref<Material> _material_override;

	// Working data
	float _relative_density = 1.f;
	TypedArray<Mesh> _meshes;
	Ref<Texture2D> _thumbnail;

	// No signal versions
	void _set_generated_type(GenType p_type);
	void _set_material_override(const Ref<Material> p_material);
	Ref<ArrayMesh> _build_generated_mesh();
	Ref<Material> _get_material();

public:
	Terrain3DMeshAsset();
	~Terrain3DMeshAsset() {}

	void clear();

	void set_name(String p_name);
	String get_name() const { return _name; }

	void set_id(int p_new_id);
	int get_id() const { return _id; }

	void set_height_offset(real_t p_offset);
	real_t get_height_offset() const { return _height_offset; }

	void set_cast_shadows(GeometryInstance3D::ShadowCastingSetting p_cast_shadows);
	GeometryInstance3D::ShadowCastingSetting get_cast_shadows() const { return _cast_shadows; };

	void set_scene_file(const Ref<PackedScene> p_scene_file);
	Ref<PackedScene> get_scene_file() const { return _packed_scene; }

	void set_material_override(const Ref<Material> p_material);
	Ref<Material> get_material_override() const { return _material_override; }

	void set_generated_type(GenType p_type);
	GenType get_generated_type() const { return _generated_type; }
	void set_generated_faces(int p_count);
	int get_generated_faces() const { return _generated_faces; }
	void set_generated_size(Vector2 p_size);
	Vector2 get_generated_size() const { return _generated_size; }

	Ref<Mesh> get_mesh(int p_index = 0);
	int get_mesh_count() { return _meshes.size(); }
	real_t get_relative_density() const { return _relative_density; }
	Ref<Texture2D> get_thumbnail() const { return _thumbnail; }

protected:
	void _validate_property(PropertyInfo &p_property) const;
	static void _bind_methods();
};

VARIANT_ENUM_CAST(Terrain3DMeshAsset::GenType);

#endif // TERRAIN3D_MESH_ASSET_CLASS_H