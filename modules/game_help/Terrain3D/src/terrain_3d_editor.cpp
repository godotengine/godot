// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

// #include <godot_cpp/classes/editor_undo_redo_manager.hpp>
// #include <godot_cpp/core/class_db.hpp>

#include "editor/editor_undo_redo_manager.h"

#include "logger.h"
#include "terrain_3d_editor.h"
#include "util.h"

///////////////////////////
// Subclass Functions
///////////////////////////

void Terrain3DEditor::Brush::set_data(Dictionary p_data) {
	LOG(DEBUG, "Setting brush data: ");
	Array ks = p_data.keys();
	for (int i = 0; i < ks.size(); i++) {
		LOG(DEBUG, ks[i], ": ", p_data[ks[i]]);
	}
	_image = p_data["image"];
	if (_image.is_valid()) {
		_img_size = Vector2(_image->get_size());
	} else {
		_img_size = Vector2(0, 0);
	}
	_texture = p_data["texture"];

	_size = p_data["size"];
	_opacity = p_data["opacity"];
	_height = p_data["height"];
	_texture_index = p_data["texture_index"];
	_color = p_data["color"];
	_roughness = p_data["roughness"];
	_gradient_points = p_data["gradient_points"];
	_enable = p_data["enable"];

	_auto_regions = p_data["automatic_regions"];
	_align_to_view = p_data["align_to_view"];
	_gamma = p_data["gamma"];
	_jitter = p_data["jitter"];
}

///////////////////////////
// Private Functions
///////////////////////////

void Terrain3DEditor::_region_modified(Vector3 p_global_position, Vector2 p_height_range) {
	Vector2i region_offset = _terrain->get_storage()->get_region_offset(p_global_position);
	Terrain3DStorage::RegionSize region_size = _terrain->get_storage()->get_region_size();

	AABB edited_area;
	edited_area.position = Vector3(region_offset.x * region_size, p_height_range.x, region_offset.y * region_size);
	edited_area.size = Vector3(region_size, p_height_range.y - p_height_range.x, region_size);

	_modified = true;
	_terrain->get_storage()->add_edited_area(edited_area);
}

void Terrain3DEditor::_operate_region(Vector3 p_global_position) {
	bool has_region = _terrain->get_storage()->has_region(p_global_position);
	bool modified = false;
	Vector2 height_range;

	if (_operation == ADD) {
		if (!has_region) {
			_terrain->get_storage()->add_region(p_global_position);
			modified = true;
		}
	}
	if (_operation == SUBTRACT) {
		if (has_region) {
			int region_index = _terrain->get_storage()->get_region_index(p_global_position);
			Ref<Image> height_map = _terrain->get_storage()->get_map_region(Terrain3DStorage::TYPE_HEIGHT, region_index);
			height_range = Util::get_min_max(height_map);

			_terrain->get_storage()->remove_region(p_global_position);
			modified = true;
		}
	}

	if (modified) {
		_region_modified(p_global_position, height_range);
	}
}

void Terrain3DEditor::_operate_map(Vector3 p_global_position, real_t p_camera_direction) {
	Ref<Terrain3DStorage> storage = _terrain->get_storage();
	int region_size = storage->get_region_size();
	Vector2i region_vsize = Vector2i(region_size, region_size);

	int region_index = storage->get_region_index(p_global_position);
	if (region_index == -1) {
		if (!_brush.auto_regions_enabled()) {
			return;
		} else {
			LOG(DEBUG, "No region to operate on, attempting to add");
			storage->add_region(p_global_position);
			region_size = storage->get_region_size();
			region_index = storage->get_region_index(p_global_position);
			if (region_index == -1) {
				LOG(ERROR, "Failed to add region, no region to operate on");
				return;
			}
			_region_modified(p_global_position);
		}
	}
	if (_tool < 0 || _tool >= REGION) {
		LOG(ERROR, "Invalid tool selected");
		return;
	}

	Terrain3DStorage::MapType map_type;
	switch (_tool) {
		case HEIGHT:
			map_type = Terrain3DStorage::TYPE_HEIGHT;
			break;
		case TEXTURE:
		case AUTOSHADER:
		case HOLES:
		case NAVIGATION:
			map_type = Terrain3DStorage::TYPE_CONTROL;
			break;
		case COLOR:
		case ROUGHNESS:
			map_type = Terrain3DStorage::TYPE_COLOR;
			break;
		default:
			return;
	}

	Ref<Image> map = storage->get_map_region(map_type, region_index);
	int brush_size = _brush.get_size();
	int texture_id = _brush.get_texture_index();
	Vector2 img_size = _brush.get_image_size();
	real_t opacity = _brush.get_opacity();
	real_t height = _brush.get_height();
	Color color = _brush.get_color();
	real_t roughness = _brush.get_roughness();
	PackedVector3Array gradient_points = _brush.get_gradient_points();
	bool enable = _brush.get_enable();
	real_t gamma = _brush.get_gamma();

	real_t randf = UtilityFunctions::randf();
	real_t rot = randf * Math_PI * _brush.get_jitter();
	if (_brush.is_aligned_to_view()) {
		rot += p_camera_direction;
	}
	Object::cast_to<Node>(_terrain->get_plugin()->get("ui"))->call("set_decal_rotation", rot);

	AABB edited_area;
	edited_area.position = p_global_position - Vector3(brush_size, 0.f, brush_size) / 2.f;
	edited_area.size = Vector3(brush_size, 0.f, brush_size);

	for (int x = 0; x < brush_size; x++) {
		for (int y = 0; y < brush_size; y++) {
			Vector2i brush_offset = Vector2i(x, y) - (Vector2i(brush_size, brush_size) / 2);
			Vector3 brush_global_position = Vector3(0.5f, 0.f, 0.5f) +
					Vector3(p_global_position.x + real_t(brush_offset.x), p_global_position.y,
							p_global_position.z + real_t(brush_offset.y));

			// If we're brushing across a region boundary, possibly add a region, and get the other map
			int new_region_index = storage->get_region_index(brush_global_position);
			if (new_region_index == -1) {
				if (!_brush.auto_regions_enabled()) {
					continue;
				}
				Error err = storage->add_region(brush_global_position);
				if (err) {
					continue;
				}
				new_region_index = storage->get_region_index(brush_global_position);
				_region_modified(brush_global_position);
			}

			if (new_region_index != region_index) {
				region_index = new_region_index;
				map = storage->get_map_region(map_type, region_index);
			}

			// Identify position on map image
			Vector2 uv_position = _get_uv_position(brush_global_position, region_size);
			Vector2i map_pixel_position = Vector2i(uv_position * region_size);

			if (_is_in_bounds(map_pixel_position, region_vsize)) {
				Vector2 brush_uv = Vector2(x, y) / real_t(brush_size);
				Vector2i brush_pixel_position = Vector2i(_rotate_uv(brush_uv, rot) * img_size);

				if (!_is_in_bounds(brush_pixel_position, Vector2i(img_size))) {
					continue;
				}

				Vector3 edited_position = brush_global_position;
				edited_position.y = storage->get_height(edited_position);
				edited_area = edited_area.expand(edited_position);

				// Start brushing on the map
				real_t brush_alpha = real_t(Math::pow(double(_brush.get_alpha(brush_pixel_position)), double(gamma)));
				Color src = map->get_pixelv(map_pixel_position);
				Color dest = src;

				if (map_type == Terrain3DStorage::TYPE_HEIGHT) {
					real_t srcf = src.r;
					real_t destf = dest.r;

					switch (_operation) {
						case ADD:
							destf = srcf + (brush_alpha * opacity * 10.f);
							break;
						case SUBTRACT:
							destf = srcf - (brush_alpha * opacity * 10.f);
							break;
						case MULTIPLY:
							destf = srcf * (brush_alpha * opacity * .01f + 1.0f);
							break;
						case DIVIDE:
							destf = srcf * (-brush_alpha * opacity * .01f + 1.0f);
							break;
						case REPLACE:
							destf = Math::lerp(srcf, height, brush_alpha * opacity);
							break;
						case AVERAGE: {
							Vector3 left_position = brush_global_position - Vector3(1, 0, 0);
							Vector3 right_position = brush_global_position + Vector3(1, 0, 0);
							Vector3 down_position = brush_global_position - Vector3(0, 0, 1);
							Vector3 up_position = brush_global_position + Vector3(0, 0, 1);

							real_t left = srcf, right = srcf, up = srcf, down = srcf;

							left = storage->get_pixel(map_type, left_position).r;
							right = storage->get_pixel(map_type, right_position).r;
							up = storage->get_pixel(map_type, up_position).r;
							down = storage->get_pixel(map_type, down_position).r;

							real_t avg = (srcf + left + right + up + down) * 0.2f;
							destf = Math::lerp(srcf, avg, brush_alpha * opacity);
							break;
						}
						case GRADIENT: {
							if (gradient_points.size() == 2) {
								Vector3 point_1 = gradient_points[0];
								Vector3 point_2 = gradient_points[1];

								Vector2 point_1_xz = Vector2(point_1.x, point_1.z);
								Vector2 point_2_xz = Vector2(point_2.x, point_2.z);
								Vector2 brush_xz = Vector2(brush_global_position.x, brush_global_position.z);

								if (_operation_movement.length_squared() > 0.f) {
									// Ramp up/down only in the direction of movement, to avoid giving winding
									// paths one edge higher than the other.
									Vector2 movement_xz = Vector2(_operation_movement.x, _operation_movement.z).normalized();
									Vector2 offset = movement_xz * Vector2(brush_offset).dot(movement_xz);
									brush_xz = Vector2(p_global_position.x + offset.x, p_global_position.z + offset.y);
								}

								Vector2 dir = point_2_xz - point_1_xz;
								real_t weight = dir.normalized().dot(brush_xz - point_1_xz) / dir.length();
								weight = CLAMP(weight, (real_t)0.0, (real_t)1.0);
								real_t height = Math::lerp(point_1.y, point_2.y, weight);

								destf = Math::lerp(srcf, height, brush_alpha * opacity);
							}
							break;
						}
						default:
							break;
					}
					dest = Color(destf, 0.f, 0.f, 1.f);
					storage->update_heights(destf);

					edited_position.y = destf;
					edited_area = edited_area.expand(edited_position);

				} else if (map_type == Terrain3DStorage::TYPE_CONTROL) {
					// Get bit field from pixel
					uint32_t base_id = Util::get_base(src.r);
					uint32_t overlay_id = Util::get_overlay(src.r);
					real_t blend = real_t(Util::get_blend(src.r)) / 255.f;
					bool hole = Util::is_hole(src.r);
					bool navigation = Util::is_nav(src.r);
					bool autoshader = Util::is_auto(src.r);

					real_t alpha_clip = (brush_alpha > 0.1f) ? 1.f : 0.f;
					uint32_t dest_id = uint32_t(Math::lerp(base_id, texture_id, alpha_clip));

					switch (_tool) {
						case TEXTURE:
							switch (_operation) {
								// Base Paint
								case REPLACE: {
									// Set base texture
									base_id = dest_id;
									// Erase blend value
									blend = Math::lerp(blend, real_t(0.f), alpha_clip);
									if (brush_alpha > 0.1f) {
										autoshader = false;
									}
								} break;

								// Overlay Spray
								case ADD: {
									real_t spray_opacity = CLAMP(opacity * 0.025f, 0.003f, 0.025f);
									real_t brush_value = CLAMP(brush_alpha * spray_opacity, 0.f, 1.f);
									// If overlay and base texture are the same, reduce blend value
									if (dest_id == base_id) {
										blend = CLAMP(blend - brush_value, 0.f, 1.f);
									} else {
										// Else overlay and base are separate, set overlay texture and increase blend value
										overlay_id = dest_id;
										blend = CLAMP(blend + brush_value, 0.f, 1.f);
									}
									if (brush_alpha * opacity * 11.f > 0.1f) {
										autoshader = false;
									}
								} break;

								default: {
								} break;
							}
							break;
						case AUTOSHADER:
							if (brush_alpha > 0.1f) {
								autoshader = enable;
							}
							break;
						case HOLES:
							if (brush_alpha > 0.1f) {
								hole = enable;
							}
							break;
						case NAVIGATION:
							if (brush_alpha > 0.1f) {
								navigation = enable;
							}
							break;
						default:
							break;
					}

					// Convert back to bitfield
					uint32_t blend_int = uint32_t(CLAMP(Math::round(blend * 255.f), 0.f, 255.f));
					uint32_t bits = Util::enc_base(base_id) | Util::enc_overlay(overlay_id) |
							Util::enc_blend(blend_int) | Util::enc_hole(hole) |
							Util::enc_nav(navigation) | Util::enc_auto(autoshader);

					// Write back to pixel in FORMAT_RF. Must be a 32-bit float
					dest = Color(*(float *)&bits, 0.f, 0.f, 1.f);

				} else if (map_type == Terrain3DStorage::TYPE_COLOR) {
					switch (_tool) {
						case COLOR:
							dest = src.lerp(color, brush_alpha * opacity);
							dest.a = src.a;
							break;
						case ROUGHNESS:
							/* Roughness received from UI is -100 to 100. Changed to 0,1 before storing.
							 * To convert 0,1 back to -100,100 use: 200 * (color.a - 0.5)
							 * However Godot stores values as 8-bit ints. Roundtrip is = int(a*255)/255.0
							 * Roughness 0 is saved as 0.5, but retreived is 0.498, or -0.4 roughness
							 * We round the final amount in tool_settings.gd:_on_picked().
							 */
							dest.a = Math::lerp(real_t(src.a), real_t(.5f) + real_t(.5f * .01f) * roughness, brush_alpha * opacity);
							break;
						default:
							break;
					}
				}

				map->set_pixelv(map_pixel_position, dest);
			}
		}
	}
	_modified = true;
	storage->force_update_maps(map_type);
	storage->add_edited_area(edited_area);
}

bool Terrain3DEditor::_is_in_bounds(Vector2i p_position, Vector2i p_max_position) {
	bool more_than_min = p_position.x >= 0 && p_position.y >= 0;
	bool less_than_max = p_position.x < p_max_position.x && p_position.y < p_max_position.y;
	return more_than_min && less_than_max;
}

Vector2 Terrain3DEditor::_get_uv_position(Vector3 p_global_position, int p_region_size) {
	Vector2 global_position_2d = Vector2(p_global_position.x, p_global_position.z);
	Vector2 region_position = global_position_2d / real_t(p_region_size);
	region_position = region_position.floor();
	Vector2 uv_position = (global_position_2d / real_t(p_region_size)) - region_position;

	return uv_position;
}

Vector2 Terrain3DEditor::_rotate_uv(Vector2 p_uv, real_t p_angle) {
	Vector2 rotation_offset = Vector2(0.5f, 0.5f);
	p_uv = (p_uv - rotation_offset).rotated(p_angle) + rotation_offset;
	return p_uv.clamp(Vector2(0.f, 0.f), Vector2(1.f, 1.f));
}

/* Stored in the _undo_set is:
 * 0-2: map 0,1,2
 * 3: Region offsets
 * 4: height range
 * 5: edited AABB
 */
void Terrain3DEditor::_setup_undo() {
	ERR_FAIL_COND_MSG(_terrain == nullptr, "terrain is null, returning");
	ERR_FAIL_COND_MSG(_terrain->get_plugin() == nullptr, "terrain->plugin is null, returning");
	if (_tool < 0 || _tool >= TOOL_MAX) {
		return;
	}
	LOG(INFO, "Setting up undo snapshot...");
	_undo_set.clear();
	_undo_set.resize(Terrain3DStorage::TYPE_MAX + 3);
	for (int i = 0; i < Terrain3DStorage::TYPE_MAX; i++) {
		_undo_set[i] = _terrain->get_storage()->get_maps_copy(static_cast<Terrain3DStorage::MapType>(i));
		LOG(DEBUG, "maps ", i, "(", static_cast<TypedArray<Image>>(_undo_set[i]).size(), "): ", _undo_set[i]);
	}
	_undo_set[Terrain3DStorage::TYPE_MAX] = _terrain->get_storage()->get_region_offsets().duplicate();
	LOG(DEBUG, "region_offsets(", static_cast<TypedArray<Vector2i>>(_undo_set[Terrain3DStorage::TYPE_MAX]).size(), "): ", _undo_set[Terrain3DStorage::TYPE_MAX]);
	_undo_set[Terrain3DStorage::TYPE_MAX + 1] = _terrain->get_storage()->get_height_range();

	_undo_set[Terrain3DStorage::TYPE_MAX + 2] = _terrain->get_storage()->get_edited_area();
}

void Terrain3DEditor::_store_undo() {
	ERR_FAIL_COND_MSG(_terrain == nullptr, "terrain is null, returning");
	ERR_FAIL_COND_MSG(_terrain->get_plugin() == nullptr, "terrain->plugin is null, returning");
	if (_tool < 0 || _tool >= TOOL_MAX) {
		return;
	}
	LOG(INFO, "Storing undo snapshot...");
	EditorUndoRedoManager *undo_redo = _terrain->get_plugin()->get_undo_redo();

	String action_name = String("Terrain3D ") + OPNAME[_operation] + String(" ") + TOOLNAME[_tool];
	LOG(DEBUG, "Creating undo action: '", action_name, "'");
	undo_redo->create_action(action_name);

	AABB edited_area = _terrain->get_storage()->get_edited_area();
	LOG(DEBUG, "Updating undo snapshot edited area: ", edited_area);
	_undo_set[Terrain3DStorage::TYPE_MAX + 2] = edited_area;

	LOG(DEBUG, "Storing undo snapshot: ", _undo_set);
	undo_redo->add_undo_method(this, "apply_undo", _undo_set.duplicate()); // Must be duplicated

	LOG(DEBUG, "Setting up redo snapshot...");
	Array redo_set;
	redo_set.resize(Terrain3DStorage::TYPE_MAX + 3);
	for (int i = 0; i < Terrain3DStorage::TYPE_MAX; i++) {
		redo_set[i] = _terrain->get_storage()->get_maps_copy(static_cast<Terrain3DStorage::MapType>(i));
		LOG(DEBUG, "maps ", i, "(", static_cast<TypedArray<Image>>(redo_set[i]).size(), "): ", redo_set[i]);
	}
	redo_set[Terrain3DStorage::TYPE_MAX] = _terrain->get_storage()->get_region_offsets().duplicate();
	LOG(DEBUG, "region_offsets(", static_cast<TypedArray<Vector2i>>(redo_set[Terrain3DStorage::TYPE_MAX]).size(), "): ", redo_set[Terrain3DStorage::TYPE_MAX]);
	redo_set[Terrain3DStorage::TYPE_MAX + 1] = _terrain->get_storage()->get_height_range();

	edited_area = _terrain->get_storage()->get_edited_area();
	LOG(DEBUG, "Storing edited area: ", edited_area);
	redo_set[Terrain3DStorage::TYPE_MAX + 2] = edited_area;

	LOG(DEBUG, "Storing redo snapshot: ", redo_set);
	undo_redo->add_do_method(this, "apply_undo", redo_set);

	LOG(DEBUG, "Committing undo action");
	undo_redo->commit_action(false);
}

void Terrain3DEditor::_apply_undo(const Array &p_set) {
	ERR_FAIL_COND_MSG(_terrain == nullptr, "terrain is null, returning");
	ERR_FAIL_COND_MSG(_terrain->get_plugin() == nullptr, "terrain->plugin is null, returning");
	LOG(INFO, "Applying Undo/Redo set. Array size: ", p_set.size());
	LOG(DEBUG, "Apply undo received: ", p_set);

	for (int i = 0; i < Terrain3DStorage::TYPE_MAX; i++) {
		Terrain3DStorage::MapType map_type = static_cast<Terrain3DStorage::MapType>(i);
		_terrain->get_storage()->set_maps(map_type, p_set[i]);
	}
	_terrain->get_storage()->set_region_offsets(p_set[Terrain3DStorage::TYPE_MAX]);
	_terrain->get_storage()->set_height_range(p_set[Terrain3DStorage::TYPE_MAX + 1]);

	if (_terrain->get_plugin()->has_method("update_grid")) {
		LOG(DEBUG, "Calling GDScript update_grid()");
		_terrain->get_plugin()->call("update_grid");
	}

	_pending_undo = false;
	_modified = false;

	AABB edited_area = p_set[Terrain3DStorage::TYPE_MAX + 2];
	_terrain->get_storage()->clear_edited_area();
	_terrain->get_storage()->add_edited_area(edited_area);
}

///////////////////////////
// Public Functions
///////////////////////////

Terrain3DEditor::Terrain3DEditor() {
}

Terrain3DEditor::~Terrain3DEditor() {
}

void Terrain3DEditor::set_brush_data(Dictionary p_data) {
	if (p_data.is_empty()) {
		return;
	}
	_brush.set_data(p_data);
}

void Terrain3DEditor::set_tool(Tool p_tool) {
	_tool = p_tool;
	_terrain->get_material()->set_show_navigation(_tool == NAVIGATION);
}

// Called on mouse click
void Terrain3DEditor::start_operation(Vector3 p_global_position) {
	_setup_undo();
	_pending_undo = true;
	_modified = false;
	_terrain->get_storage()->clear_edited_area();
	_operation_position = p_global_position;
	_operation_movement = Vector3();
	if (_tool == REGION) {
		_operate_region(p_global_position);
	}
}

// Called on mouse movement with left mouse button down
void Terrain3DEditor::operate(Vector3 p_global_position, real_t p_camera_direction) {
	if (!_pending_undo) {
		return;
	}

	_operation_movement = p_global_position - _operation_position;
	_operation_position = p_global_position;

	if (_tool == REGION) {
		_operate_region(p_global_position);
	} else if (_tool >= 0 && _tool < REGION) {
		_operate_map(p_global_position, p_camera_direction);
	}
}

// Called on left mouse button released
void Terrain3DEditor::stop_operation() {
	if (_pending_undo && _modified) {
		_store_undo();
		_pending_undo = false;
		_modified = false;
		_terrain->get_storage()->clear_edited_area();
	}
}

///////////////////////////
// Protected Functions
///////////////////////////

void Terrain3DEditor::_bind_methods() {
	BIND_ENUM_CONSTANT(ADD);
	BIND_ENUM_CONSTANT(SUBTRACT);
	BIND_ENUM_CONSTANT(MULTIPLY);
	BIND_ENUM_CONSTANT(DIVIDE);
	BIND_ENUM_CONSTANT(REPLACE);
	BIND_ENUM_CONSTANT(AVERAGE);
	BIND_ENUM_CONSTANT(GRADIENT);
	BIND_ENUM_CONSTANT(OP_MAX);

	BIND_ENUM_CONSTANT(HEIGHT);
	BIND_ENUM_CONSTANT(TEXTURE);
	BIND_ENUM_CONSTANT(COLOR);
	BIND_ENUM_CONSTANT(ROUGHNESS);
	BIND_ENUM_CONSTANT(AUTOSHADER);
	BIND_ENUM_CONSTANT(HOLES);
	BIND_ENUM_CONSTANT(NAVIGATION);
	BIND_ENUM_CONSTANT(REGION);
	BIND_ENUM_CONSTANT(TOOL_MAX);

	ClassDB::bind_method(D_METHOD("set_terrain", "terrain"), &Terrain3DEditor::set_terrain);
	ClassDB::bind_method(D_METHOD("get_terrain"), &Terrain3DEditor::get_terrain);

	ClassDB::bind_method(D_METHOD("set_brush_data", "data"), &Terrain3DEditor::set_brush_data);
	ClassDB::bind_method(D_METHOD("set_tool", "tool"), &Terrain3DEditor::set_tool);
	ClassDB::bind_method(D_METHOD("get_tool"), &Terrain3DEditor::get_tool);
	ClassDB::bind_method(D_METHOD("set_operation", "operation"), &Terrain3DEditor::set_operation);
	ClassDB::bind_method(D_METHOD("get_operation"), &Terrain3DEditor::get_operation);
	ClassDB::bind_method(D_METHOD("start_operation", "position"), &Terrain3DEditor::start_operation);
	ClassDB::bind_method(D_METHOD("operate", "position", "camera_direction"), &Terrain3DEditor::operate);
	ClassDB::bind_method(D_METHOD("stop_operation"), &Terrain3DEditor::stop_operation);
	ClassDB::bind_method(D_METHOD("is_operating"), &Terrain3DEditor::is_operating);

	ClassDB::bind_method(D_METHOD("apply_undo", "maps"), &Terrain3DEditor::_apply_undo);
}
