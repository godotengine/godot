// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

#ifndef TERRAIN3D_EDITOR_CLASS_H
#define TERRAIN3D_EDITOR_CLASS_H

// #include <godot_cpp/classes/image.hpp>
// #include <godot_cpp/classes/image_texture.hpp>
#include "core/io/image.h"
#include "scene/resources/image_texture.h"
#include "modules/register_module_types.h"


#include "terrain_3d.h"

using namespace godot;

class Terrain3DEditor : public Object {
	GDCLASS(Terrain3DEditor, Object);

public:
	// Constants
	static inline const char *__class__ = "Terrain3DEditor";

	enum Operation {
		ADD,
		SUBTRACT,
		MULTIPLY,
		DIVIDE,
		REPLACE,
		AVERAGE,
		GRADIENT,
		OP_MAX,
	};

	static inline const char *OPNAME[] = {
		"Add",
		"Subtract",
		"Multiply",
		"Divide",
		"Replace",
		"Average",
		"Gradient",
		"OP_MAX",
	};

	enum Tool {
		HEIGHT,
		TEXTURE,
		COLOR,
		ROUGHNESS,
		AUTOSHADER,
		HOLES,
		NAVIGATION,
		REGION,
		TOOL_MAX,
	};

	static inline const char *TOOLNAME[] = {
		"Height",
		"Texture",
		"Color",
		"Roughness",
		"Auto Shader",
		"Holes",
		"Navigation",
		"Region",
		"TOOL_MAX",
	};

	class Brush {
	private:
		Ref<Image> _image;
		Vector2 _img_size;
		Ref<ImageTexture> _texture;

		int _size = 0;
		real_t _opacity = 0.0f;
		real_t _height = 0.0f;
		int _texture_index = 0;
		Color _color = COLOR_ROUGHNESS;
		real_t _roughness = 0.5f;
		PackedVector3Array _gradient_points;
		bool _enable = false;

		bool _auto_regions = false;
		bool _align_to_view = false;
		real_t _gamma = 1.0f;
		real_t _jitter = 0.0f;

	public:
		void set_data(Dictionary p_data);
		real_t get_alpha(Vector2i p_position) { return _image->get_pixelv(p_position).r; }

		Ref<ImageTexture> get_texture() const { return _texture; }
		Ref<Image> get_image() const { return _image; }
		Vector2 get_image_size() const { return _img_size; }

		int get_size() const { return _size; }
		real_t get_opacity() const { return _opacity; }
		real_t get_height() const { return _height; }
		int get_texture_index() const { return _texture_index; }
		Color get_color() const { return _color; }
		real_t get_roughness() const { return _roughness; }
		PackedVector3Array get_gradient_points() const { return _gradient_points; }
		real_t get_enable() const { return _enable; }

		bool auto_regions_enabled() const { return _auto_regions; }
		bool is_aligned_to_view() const { return _align_to_view; }
		real_t get_gamma() const { return _gamma; }
		real_t get_jitter() const { return _jitter; }
	};

private:
	// Object references
	Terrain3D *_terrain = nullptr;

	// Painter settings & variables
	Tool _tool = REGION;
	Operation _operation = ADD;
	Brush _brush;
	Vector3 _operation_position = Vector3();
	Vector3 _operation_movement = Vector3();
	bool _pending_undo = false;
	bool _modified = false;
	AABB _modified_area;
	Array _undo_set; // 0-2: map 0,1,2, 3: Region offsets, 4: height range, 5: edited AABB

	void _region_modified(Vector3 p_global_position, Vector2 p_height_range = Vector2());
	void _operate_region(Vector3 p_global_position);
	void _operate_map(Vector3 p_global_position, real_t p_camera_direction);
	bool _is_in_bounds(Vector2i p_position, Vector2i p_max_position);
	Vector2 _get_uv_position(Vector3 p_global_position, int p_region_size);
	Vector2 _rotate_uv(Vector2 p_uv, real_t p_angle);

	void _setup_undo();
	void _store_undo();
	void _apply_undo(const Array &p_set);

public:
	Terrain3DEditor();
	~Terrain3DEditor();

	void set_terrain(Terrain3D *p_terrain) { _terrain = p_terrain; }
	Terrain3D *get_terrain() const { return _terrain; }

	void set_brush_data(Dictionary data);
	void set_tool(Tool p_tool);
	Tool get_tool() const { return _tool; }
	void set_operation(Operation p_operation) { _operation = p_operation; }
	Operation get_operation() const { return _operation; }

	void start_operation(Vector3 p_global_position);
	void operate(Vector3 p_global_position, real_t p_camera_direction);
	void stop_operation();
	bool is_operating() const { return _pending_undo; }

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(Terrain3DEditor::Operation);
VARIANT_ENUM_CAST(Terrain3DEditor::Tool);

#endif // TERRAIN3D_EDITOR_CLASS_H
