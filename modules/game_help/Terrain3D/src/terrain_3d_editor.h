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
	CLASS_NAME();

public: // Constants
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
		ANGLE,
		SCALE,
		AUTOSHADER,
		HOLES,
		NAVIGATION,
		INSTANCER,
		REGION,
		TOOL_MAX,
	};

	static inline const char *TOOLNAME[] = {
		"Height",
		"Texture",
		"Color",
		"Roughness",
		"Angle",
		"Scale",
		"Auto Shader",
		"Holes",
		"Navigation",
		"Instancer",
		"Region",
		"TOOL_MAX",
	};

private:
	Terrain3D *_terrain = nullptr;

	// Painter settings & variables
	Tool _tool = REGION;
	Operation _operation = ADD;
	Ref<Image> _brush_image;
	Dictionary _brush_data;
	Vector3 _operation_position = Vector3();
	Vector3 _operation_movement = Vector3();
	Array _operation_movement_history;
	bool _pending_undo = false;
	bool _modified = false;
	AABB _modified_area;
	Dictionary _undo_set; // See _collect_undo_data for definition

	real_t _get_brush_alpha(Vector2i p_position);
	void _region_modified(Vector3 p_global_position, Vector2 p_height_range = Vector2());
	void _operate_region(Vector3 p_global_position);
	void _operate_map(Vector3 p_global_position, real_t p_camera_direction);
	bool _is_in_bounds(Vector2i p_position, Vector2i p_max_position);
	Vector2 _get_uv_position(Vector3 p_global_position, int p_region_size);
	Vector2 _rotate_uv(Vector2 p_uv, real_t p_angle);

	Dictionary _collect_undo_data();
	void _store_undo();
	void _apply_undo(Dictionary p_set);

public:
	Terrain3DEditor() {}
	~Terrain3DEditor() {}

	void set_terrain(Terrain3D *p_terrain) { _terrain = p_terrain; }
	Terrain3D *get_terrain() const { return _terrain; }

	void set_brush_data(Dictionary p_data);
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
