// Copyright © 2024 Cory Petkovsek, Roope Palmroos, and Contributors.

// #include <godot_cpp/classes/editor_undo_redo_manager.hpp>
// #include <godot_cpp/core/class_db.hpp>

#include "editor/editor_undo_redo_manager.h"

#include "logger.h"
#include "terrain_3d_editor.h"
#include "terrain_3d_util.h"

///////////////////////////
// Private Functions
///////////////////////////

void Terrain3DEditor::_region_modified(const Vector3 &p_global_position, const Vector2 &p_height_range) {
	Vector2i region_offset = _terrain->get_storage()->get_region_offset(p_global_position);
	Terrain3DStorage::RegionSize region_size = _terrain->get_storage()->get_region_size();

	AABB edited_area;
	edited_area.position = Vector3(region_offset.x * region_size, p_height_range.x, region_offset.y * region_size);
	edited_area.size = Vector3(region_size, p_height_range.y - p_height_range.x, region_size);
	edited_area.position *= _terrain->get_mesh_vertex_spacing();
	edited_area.size *= _terrain->get_mesh_vertex_spacing();

	_modified = true;
	_terrain->get_storage()->add_edited_area(edited_area);
}

void Terrain3DEditor::_operate_region(const Vector3 &p_global_position) {
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

void Terrain3DEditor::_operate_map(const Vector3 &p_global_position, const real_t p_camera_direction) {
	if (_brush_image.is_null()) {
		LOG(ERROR, "Invalid brush image. Returning");
	}
	LOG(DEBUG_CONT, "Operating at ", p_global_position, " tool type ", _tool, " op ", _operation);
	Ref<Terrain3DStorage> storage = _terrain->get_storage();
	int region_size = storage->get_region_size();
	Vector2i region_vsize = Vector2i(region_size, region_size);
	int region_index = storage->get_region_index(p_global_position);
	if (region_index == -1) {
		if (!_brush_data["auto_regions"] || _tool != HEIGHT) {
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
		case INSTANCER:
			map_type = Terrain3DStorage::TYPE_HEIGHT;
			break;
		case TEXTURE:
		case AUTOSHADER:
		case HOLES:
		case NAVIGATION:
		case ANGLE:
		case SCALE:
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
	real_t brush_size = _brush_data["size"];
	int asset_id = _brush_data["asset_id"];
	Vector2i img_size = _brush_data["brush_image_size"];
	real_t strength = _brush_data["strength"];
	real_t height = _brush_data["height"];
	Color color = _brush_data["color"];
	real_t roughness = _brush_data["roughness"];
	PackedVector3Array gradient_points = _brush_data["gradient_points"];
	bool enable = _brush_data["enable"];
	bool enable_texture = _brush_data["enable_texture"];
	bool enable_angle = _brush_data["enable_angle"];
	bool dynamic_angle = _brush_data["dynamic_angle"];
	real_t angle = _brush_data["angle"];
	bool enable_scale = _brush_data["enable_scale"];
	real_t scale = _brush_data["scale"];
	real_t gamma = _brush_data["gamma"];

	real_t randf = Math::randf();
	real_t rot = randf * Math_PI * real_t(_brush_data["jitter"]);
	if (_brush_data["align_to_view"]) {
		rot += p_camera_direction;
	}
	cast_to<Node>(_terrain->get_plugin()->get("ui"))->call("set_decal_rotation", rot);

	AABB edited_area;
	edited_area.position = p_global_position - Vector3(brush_size, 0.f, brush_size) * .5f;
	edited_area.size = Vector3(brush_size, 0.f, brush_size);

	if (_tool == INSTANCER) {
		if (enable) {
			_terrain->get_instancer()->add_instances(p_global_position, _brush_data);
		} else {
			_terrain->get_instancer()->remove_instances(p_global_position, _brush_data);
		}

		// TODO replace Enabled with Add/Subtract operation
		//switch (_operation) {
		//case ADD: {
		// Change to ADD/SUBTRACT
		//		break;
		//	}
		//	case SUBTRACT:
		//		break;
		//	default:
		//		return;
		//		break;
		//}

		_modified = true;
		storage->add_edited_area(edited_area);
		return;
	}

	// MAP Operations
	real_t vertex_spacing = _terrain->get_mesh_vertex_spacing();
	for (real_t x = 0.f; x < brush_size; x += vertex_spacing) {
		for (real_t y = 0.f; y < brush_size; y += vertex_spacing) {
			Vector2 brush_offset = Vector2(x, y) - (Vector2(brush_size, brush_size) / 2.f);
			Vector3 brush_global_position =
					Vector3(p_global_position.x + brush_offset.x + .5f, p_global_position.y,
							p_global_position.z + brush_offset.y + .5f);

			// If we're brushing across a region boundary, possibly add a region, and get the other map
			int new_region_index = storage->get_region_index(brush_global_position);
			if (new_region_index == -1) {
				if (!_brush_data["auto_regions"] || _tool != HEIGHT) {
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
				Vector2 brush_uv = Vector2(x, y) / brush_size;
				Vector2i brush_pixel_position = Vector2i(_get_rotated_uv(brush_uv, rot) * img_size);

				if (!_is_in_bounds(brush_pixel_position, img_size)) {
					continue;
				}

				Vector3 edited_position = brush_global_position;
				edited_position.y = storage->get_height(edited_position);
				edited_area = edited_area.expand(edited_position);

				// Start brushing on the map
				real_t brush_alpha = _get_brush_alpha(brush_pixel_position);
				if (std::isnan(brush_alpha)) {
					return;
				}
				brush_alpha = real_t(Math::pow(double(brush_alpha), double(gamma)));
				Color src = map->get_pixelv(map_pixel_position);
				Color dest = src;

				if (map_type == Terrain3DStorage::TYPE_HEIGHT) {
					real_t srcf = src.r;
					real_t destf = dest.r;

					switch (_operation) {
						case ADD:
							destf = srcf + (brush_alpha * strength * 10.f);
							break;
						case SUBTRACT:
							destf = srcf - (brush_alpha * strength * 10.f);
							break;
						case MULTIPLY:
							destf = srcf * (brush_alpha * strength * .01f + 1.0f);
							break;
						case DIVIDE:
							destf = srcf * (-brush_alpha * strength * .01f + 1.0f);
							break;
						case REPLACE:
							destf = Math::lerp(srcf, height, brush_alpha * strength);
							break;
						case AVERAGE: {
							Vector3 left_position = brush_global_position - Vector3(vertex_spacing, 0.f, 0.f);
							Vector3 right_position = brush_global_position + Vector3(vertex_spacing, 0.f, 0.f);
							Vector3 down_position = brush_global_position - Vector3(0.f, 0.f, vertex_spacing);
							Vector3 up_position = brush_global_position + Vector3(0.f, 0.f, vertex_spacing);

							real_t left = srcf, right = srcf, up = srcf, down = srcf;

							left = storage->get_pixel(map_type, left_position).r;
							if (std::isnan(left)) {
								left = 0.f;
							}
							right = storage->get_pixel(map_type, right_position).r;
							if (std::isnan(right)) {
								right = 0.f;
							}
							up = storage->get_pixel(map_type, up_position).r;
							if (std::isnan(up)) {
								up = 0.f;
							}
							down = storage->get_pixel(map_type, down_position).r;
							if (std::isnan(down)) {
								down = 0.f;
							}

							real_t avg = (srcf + left + right + up + down) * 0.2f;
							destf = Math::lerp(srcf, avg, brush_alpha * strength);
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
								weight = CLAMP(weight, (real_t)0.0f, (real_t)1.0f);
								real_t height = Math::lerp(point_1.y, point_2.y, weight);

								destf = Math::lerp(srcf, height, brush_alpha * strength);
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
					uint32_t base_id = get_base(src.r);
					uint32_t overlay_id = get_overlay(src.r);
					real_t blend = real_t(get_blend(src.r)) / 255.f;
					uint32_t uvrotation = get_uv_rotation(src.r);
					uint32_t uvscale = get_uv_scale(src.r);
					bool hole = is_hole(src.r);
					bool navigation = is_nav(src.r);
					bool autoshader = is_auto(src.r);

					real_t alpha_clip = (brush_alpha > 0.1f) ? 1.f : 0.f;
					uint32_t dest_id = uint32_t(Math::lerp(base_id, asset_id, alpha_clip));
					// Lookup to shift values saved to control map so that 0 (default) is the first entry
					// Shader scale array is aligned to match this.
					int scale_align[] = { 5, 6, 7, 0, 1, 2, 3, 4 };

					switch (_tool) {
						case TEXTURE:
							switch (_operation) {
								// Base Paint
								case REPLACE: {
									if (brush_alpha > 0.1f) {
										if (enable_texture) {
											// Set base texture
											base_id = dest_id;
											// Erase blend value
											blend = Math::lerp(blend, real_t(0.f), alpha_clip);
											autoshader = false;
										}
										// Set angle & scale
										if (enable_angle) {
											if (dynamic_angle) {
												// Angle from mouse movement.
												angle = Vector2(-_operation_movement.x, _operation_movement.z).angle();
												// Avoid negative, align texture "up" with mouse direction.
												angle = real_t(Math::fmod(Math::rad_to_deg(angle) + 450.f, 360.f));
											}
											// Convert from degrees to 0 - 15 value range
											uvrotation = uint32_t(CLAMP(Math::round(angle / 22.5f), 0.f, 15.f));
										}
										if (enable_scale) {
											// Offset negative and convert from percentage to 0 - 7 bit value range
											// Maintain 0 = 0, remap negatives to end.
											uvscale = scale_align[uint8_t(CLAMP(Math::round((scale + 60.f) / 20.f), 0.f, 7.f))];
										}
									}
								} break;

								// Overlay Spray
								case ADD: {
									real_t spray_strength = CLAMP(strength * 0.025f, 0.003f, 0.025f);
									real_t brush_value = CLAMP(brush_alpha * spray_strength, 0.f, 1.f);
									if (enable_texture) {
										// If overlay and base texture are the same, reduce blend value
										if (dest_id == base_id) {
											blend = CLAMP(blend - brush_value, 0.f, 1.f);
										} else {
											// Else overlay and base are separate, set overlay texture and increase blend value
											overlay_id = dest_id;
											blend = CLAMP(blend + brush_value, 0.f, 1.f);
										}
										autoshader = false;
									}
									if (brush_alpha * strength * 11.f > 0.1f) {
										// Set angle & scale
										if (enable_angle) {
											if (dynamic_angle) {
												// Angle from mouse movement.
												angle = Vector2(-_operation_movement.x, _operation_movement.z).angle();
												// Avoid negative, align texture "up" with mouse direction.
												angle = real_t(Math::fmod(Math::rad_to_deg(angle) + 450.f, 360.f));
											}
											// Convert from degrees to 0 - 15 value range
											uvrotation = uint32_t(CLAMP(Math::round(angle / 22.5f), 0.f, 15.f));
										}
										if (enable_scale) {
											// Offset negative and convert from percentage to 0 - 7 bit value range
											// Maintain 0 = 0, remap negatives to end.
											uvscale = scale_align[uint8_t(CLAMP(Math::round((scale + 60.f) / 20.f), 0.f, 7.f))];
										}
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
					uint32_t bits = enc_base(base_id) | enc_overlay(overlay_id) |
							enc_blend(blend_int) | enc_uv_rotation(uvrotation) |
							enc_uv_scale(uvscale) | enc_hole(hole) |
							enc_nav(navigation) | enc_auto(autoshader);

					// Write back to pixel in FORMAT_RF. Must be a 32-bit float
					dest = Color(as_float(bits), 0.f, 0.f, 1.f);

				} else if (map_type == Terrain3DStorage::TYPE_COLOR) {
					switch (_tool) {
						case COLOR:
							dest = src.lerp(color, brush_alpha * strength);
							dest.a = src.a;
							break;
						case ROUGHNESS:
							/* Roughness received from UI is -100 to 100. Changed to 0,1 before storing.
							 * To convert 0,1 back to -100,100 use: 200 * (color.a - 0.5)
							 * However Godot stores values as 8-bit ints. Roundtrip is = int(a*255)/255.0
							 * Roughness 0 is saved as 0.5, but retreived is 0.498, or -0.4 roughness
							 * We round the final amount in tool_settings.gd:_on_picked().
							 */
							dest.a = Math::lerp(real_t(src.a), real_t(.5f + .5f * roughness), brush_alpha * strength);
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

bool Terrain3DEditor::_is_in_bounds(const Vector2i &p_position, const Vector2i &p_max_position) const {
	bool more_than_min = p_position.x >= 0 && p_position.y >= 0;
	bool less_than_max = p_position.x < p_max_position.x && p_position.y < p_max_position.y;
	return more_than_min && less_than_max;
}

real_t Terrain3DEditor::_get_brush_alpha(const Vector2i &p_position) const {
	if (_brush_image.is_valid()) {
		return _brush_image->get_pixelv(p_position).r;
	}
	return NAN;
}

Vector2 Terrain3DEditor::_get_uv_position(const Vector3 &p_global_position, const int p_region_size) const {
	Vector2 descaled_position_2d = Vector2(p_global_position.x, p_global_position.z) / _terrain->get_mesh_vertex_spacing();
	Vector2 region_position = descaled_position_2d / real_t(p_region_size);
	region_position = region_position.floor();
	Vector2 uv_position = (descaled_position_2d / real_t(p_region_size)) - region_position;
	return uv_position;
}

Vector2 Terrain3DEditor::_get_rotated_uv(const Vector2 &p_uv, const real_t p_angle) const {
	Vector2 rotation_offset = Vector2(0.5f, 0.5f);
	Vector2 uv = (p_uv - rotation_offset).rotated(p_angle) + rotation_offset;
	return uv.clamp(Vector2(0.f, 0.f), Vector2(1.f, 1.f));
}

Dictionary Terrain3DEditor::_get_undo_data() const {
	Dictionary data;
	if (_tool < 0 || _tool >= TOOL_MAX) {
		return data;
	}
	switch (_tool) {
		case REGION:
			LOG(DEBUG, "Storing region offsets");
			data["region_offsets"] = _terrain->get_storage()->get_region_offsets().duplicate();
			if (_operation == SUBTRACT) {
				data["height_map"] = _terrain->get_storage()->get_maps_copy(Terrain3DStorage::TYPE_HEIGHT);
				data["control_map"] = _terrain->get_storage()->get_maps_copy(Terrain3DStorage::TYPE_CONTROL);
				data["color_map"] = _terrain->get_storage()->get_maps_copy(Terrain3DStorage::TYPE_COLOR);
				data["height_range"] = _terrain->get_storage()->get_height_range();
				data["edited_area"] = _terrain->get_storage()->get_edited_area();
			}
			break;

		case HEIGHT:
			LOG(DEBUG, "Storing height maps and range");
			data["region_offsets"] = _terrain->get_storage()->get_region_offsets().duplicate();
			data["height_map"] = _terrain->get_storage()->get_maps_copy(Terrain3DStorage::TYPE_HEIGHT);
			data["height_range"] = _terrain->get_storage()->get_height_range();
			data["edited_area"] = _terrain->get_storage()->get_edited_area();
			break;

		case TEXTURE:
		case AUTOSHADER:
		case HOLES:
		case NAVIGATION:
			LOG(DEBUG, "Storing control maps");
			data["control_map"] = _terrain->get_storage()->get_maps_copy(Terrain3DStorage::TYPE_CONTROL);
			break;

		case COLOR:
		case ROUGHNESS:
			LOG(DEBUG, "Storing color maps");
			data["color_map"] = _terrain->get_storage()->get_maps_copy(Terrain3DStorage::TYPE_COLOR);
			break;

		case INSTANCER:
			data["multimeshes"] = _terrain->get_storage()->get_multimeshes().duplicate(true);
			LOG(DEBUG, "Storing Multimesh: ", data["multimeshes"]);
			break;
	}
	return data;
}

void Terrain3DEditor::_store_undo() {
	IS_INIT_COND_MESG(_terrain->get_plugin() == nullptr, "_terrain isn't initialized, returning", VOID);
	if (_tool < 0 || _tool >= TOOL_MAX) {
		return;
	}
	LOG(INFO, "Storing undo snapshot...");
	EditorUndoRedoManager *undo_redo = _terrain->get_plugin()->get_undo_redo();

	String action_name = String("Terrain3D ") + OPNAME[_operation] + String(" ") + TOOLNAME[_tool];
	LOG(DEBUG, "Creating undo action: '", action_name, "'");
	undo_redo->create_action(action_name);

	if (_undo_data.has("edited_area")) {
		_undo_data["edited_area"] = _terrain->get_storage()->get_edited_area();
		LOG(DEBUG, "Updating undo snapshot edited area: ", _undo_data["edited_area"]);
	}

	LOG(DEBUG, "Storing undo snapshot: ", _undo_data);
	undo_redo->add_undo_method(this, "apply_undo", _undo_data.duplicate());

	LOG(DEBUG, "Setting up redo snapshot...");
	Dictionary redo_set = _get_undo_data();

	LOG(DEBUG, "Storing redo snapshot: ", redo_set);
	undo_redo->add_do_method(this, "apply_undo", redo_set);

	LOG(DEBUG, "Committing undo action");
	undo_redo->commit_action(false);
}

void Terrain3DEditor::_apply_undo(const Dictionary &p_set) {
	IS_INIT_COND_MESG(_terrain->get_plugin() == nullptr, "_terrain isn't initialized, returning", VOID);
	LOG(INFO, "Applying Undo/Redo set. Array size: ", p_set.size());
	LOG(DEBUG, "Apply undo received: ", p_set);

	Array keys = p_set.keys();
	for (int i = 0; i < keys.size(); i++) {
		String key = keys[i];
		if (key == "region_offsets") {
			_terrain->get_storage()->set_region_offsets(p_set[key]);
		} else if (key == "height_map") {
			_terrain->get_storage()->set_maps(Terrain3DStorage::TYPE_HEIGHT, p_set[key]);
		} else if (key == "control_map") {
			_terrain->get_storage()->set_maps(Terrain3DStorage::TYPE_CONTROL, p_set[key]);
		} else if (key == "color_map") {
			_terrain->get_storage()->set_maps(Terrain3DStorage::TYPE_COLOR, p_set[key]);
		} else if (key == "height_range") {
			_terrain->get_storage()->set_height_range(p_set[key]);
		} else if (key == "edited_area") {
			_terrain->get_storage()->clear_edited_area();
			_terrain->get_storage()->add_edited_area(p_set[key]);
		} else if (key == "multimeshes") {
			_terrain->get_storage()->set_multimeshes(p_set[key]);
		}
	}

	if (_terrain->get_plugin()->has_method("update_grid")) {
		LOG(DEBUG, "Calling GDScript update_grid()");
		_terrain->get_plugin()->call("update_grid");
	}

	_pending_undo = false;
	_modified = false;
}

///////////////////////////
// Public Functions
///////////////////////////

// Santize and set incoming brush data w/ defaults and clamps
// Only santizes data needed for the editor, other parameters (eg instancer) untouched here
void Terrain3DEditor::set_brush_data(const Dictionary &p_data) {
	_brush_data = p_data;

	// Setup image and textures
	_brush_image = Ref<Image>();
	_brush_data["brush_image_size"] = Vector2i(0, 0);
	_brush_data["brush_texture"] = Ref<ImageTexture>();
	Array brush = p_data["brush"];
	if (brush.size() == 2) {
		Ref<Image> img = brush[0];
		if (img.is_valid() && !img->is_empty()) {
			_brush_image = img;
			_brush_data["brush_image_size"] = img->get_size();
			_brush_data["brush_texture"] = brush[1];
		}
	}

	// Santize editor data
	_brush_data["size"] = CLAMP(real_t(p_data.get("size", 10.f)), 2.f, 4096); // Diameter in meters
	_brush_data["strength"] = CLAMP(real_t(p_data.get("strength", .1f)) * .01f, .01f, 100.f); // 1-10k%
	_brush_data["height"] = CLAMP(real_t(p_data.get("height", 0.f)), -65536.f, 65536.f); // Meters
	_brush_data["asset_id"] = CLAMP(int(p_data.get("asset_id", 0)), 0, Terrain3DAssets::MAX_MESHES);
	Color col = p_data.get("color", COLOR_ROUGHNESS);
	col.r = CLAMP(col.r, 0.f, 5.f);
	col.g = CLAMP(col.g, 0.f, 5.f);
	col.b = CLAMP(col.b, 0.f, 5.f);
	col.a = CLAMP(col.a, 0.f, 1.f);
	_brush_data["color"] = col;
	_brush_data["roughness"] = CLAMP(real_t(p_data.get("roughness", 0.f)), -100.f, 100.f) * .01f; // Percentage
	_brush_data["enable"] = bool(p_data.get("enable", true));
	_brush_data["auto_regions"] = bool(p_data.get("automatic_regions", true));
	_brush_data["align_to_view"] = bool(p_data.get("align_to_view", true));
	_brush_data["gamma"] = CLAMP(real_t(p_data.get("gamma", 1.f)), 0.1f, 2.f);
	_brush_data["jitter"] = CLAMP(real_t(p_data.get("jitter", 0.f)), 0.f, 1.f);
	_brush_data["gradient_points"] = p_data.get("gradient_points", PackedVector3Array());

	LOG(DEBUG_CONT, "Setting new, sanitized brush data: ");
	Array keys = _brush_data.keys();
	for (int i = 0; i < keys.size(); i++) {
		LOG(DEBUG_CONT, keys[i], ": ", _brush_data[keys[i]]);
	}
}

void Terrain3DEditor::set_tool(const Tool p_tool) {
	_tool = p_tool;
	if (_terrain) {
		_terrain->get_material()->set_show_navigation(_tool == NAVIGATION);
	}
}

// Called on mouse click
void Terrain3DEditor::start_operation(const Vector3 &p_global_position) {
	IS_STORAGE_INIT_MESG("Terrain isn't initialized", VOID);
	LOG(INFO, "Setting up undo snapshot...");
	_undo_data.clear();
	_undo_data = _get_undo_data();
	_pending_undo = true;
	_modified = false;
	// Reset counter at start to ensure first click places an instance
	_terrain->get_instancer()->reset_instance_counter();
	_terrain->get_storage()->clear_edited_area();
	_operation_position = p_global_position;
	_operation_movement = Vector3();
	if (_tool == REGION) {
		_operate_region(p_global_position);
	}
}

// Called on mouse movement with left mouse button down
void Terrain3DEditor::operate(const Vector3 &p_global_position, const real_t p_camera_direction) {
	IS_STORAGE_INIT_MESG("Terrain isn't initialized", VOID);
	if (!_pending_undo) {
		return;
	}
	_operation_movement = p_global_position - _operation_position;
	_operation_position = p_global_position;

	// Convolve the last 8 movement events, we dont clear on mouse release
	// so as to make repeated mouse strokes in the same direction consistent
	_operation_movement_history.append(_operation_movement);
	if (_operation_movement_history.size() > 8) {
		_operation_movement_history.pop_front();
	}
	// size -1, dont add the last appended entry
	for (int i = 0; i < _operation_movement_history.size() - 1; i++) {
		_operation_movement += _operation_movement_history[i];
	}
	_operation_movement *= 0.125; // 1/8th

	if (_tool == REGION) {
		_operate_region(p_global_position);
	} else if (_tool >= 0 && _tool < REGION) {
		_operate_map(p_global_position, p_camera_direction);
	}
}

// Called on left mouse button released
void Terrain3DEditor::stop_operation() {
	IS_STORAGE_INIT_MESG("Terrain isn't initialized", VOID);
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
	BIND_ENUM_CONSTANT(ANGLE);
	BIND_ENUM_CONSTANT(SCALE);
	BIND_ENUM_CONSTANT(AUTOSHADER);
	BIND_ENUM_CONSTANT(HOLES);
	BIND_ENUM_CONSTANT(NAVIGATION);
	BIND_ENUM_CONSTANT(INSTANCER);
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
