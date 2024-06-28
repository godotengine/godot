// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_TEXTURE_CLASS_H
#define TERRAIN3D_TEXTURE_CLASS_H

#include "scene/resources/texture.h"

#include "constants.h"
#include "terrain_3d_asset_resource.h"

using namespace godot;

class Terrain3DTextureAsset : public Terrain3DAssetResource {
	GDCLASS(Terrain3DTextureAsset, Terrain3DAssetResource);
	CLASS_NAME();
	friend class Terrain3DAssets;

	Color _albedo_color = Color(1.f, 1.f, 1.f, 1.f);
	Ref<Texture2D> _albedo_texture;
	Ref<Texture2D> _normal_texture;
	real_t _uv_scale = 0.1f;
	real_t _detiling = 0.0f;

	bool _is_valid_format(const Ref<Texture2D> &p_texture) const;

public:
	Terrain3DTextureAsset();
	~Terrain3DTextureAsset();

	void clear();

	void set_name(String p_name);
	String get_name() const { return _name; }

	void set_id(int p_new_id);
	int get_id() const { return _id; }

	void set_albedo_color(Color p_color);
	Color get_albedo_color() const { return _albedo_color; }

	void set_albedo_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_albedo_texture() const { return _albedo_texture; }

	void set_normal_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_normal_texture() const { return _normal_texture; }

	void set_uv_scale(real_t p_scale);
	real_t get_uv_scale() const { return _uv_scale; }

	void set_detiling(real_t p_detiling);
	real_t get_detiling() const { return _detiling; }

protected:
	static void _bind_methods();
};

#endif // TERRAIN3D_TEXTURE_CLASS_H