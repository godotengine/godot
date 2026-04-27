/**************************************************************************/
/*  visualizer_3d.cpp                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "visualizer_3d.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "editor/editor_node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/3d/world_3d.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/rendering_server_enums.h"

Visualizer3D *Visualizer3D::singleton = nullptr;

Visualizer3D *Visualizer3D::get_singleton() {
	return singleton;
}

void Visualizer3D::line(const Vector3 &from, const Vector3 &to, float duration, const Color &color, float width) {
	DebugLine line;
	line.from = from;
	line.to = to;
	line.color = color;
	line.width = MAX(0.0f, width);
	line.remaining_time = duration;

	lines.push_back(line);
}

void Visualizer3D::arrow(const Vector3 &from, const Vector3 &to, float duration, const Color &color, float width) {
	line(from, to, duration, color, width);

	const Vector3 dir = to - from;
	const float dir_len = dir.length();
	if (dir_len < 0.0001f) {
		return;
	}

	// Tip length is relative to width, but capped to a fraction of the line
	// length so very short lines still look like arrows.
	float tip_length = MAX(width, 1.0f) * 0.25f;
	tip_length = MIN(tip_length, dir_len * 0.4f);
	if (tip_length <= 0.0f) {
		return;
	}

	const Vector3 forward = dir / dir_len;

	// Pick a perpendicular axis. Use Y unless the line is nearly vertical.
	Vector3 up = Vector3(0, 1, 0);
	if (Math::abs(forward.dot(up)) > 0.95f) {
		up = Vector3(1, 0, 0);
	}
	const Vector3 side = forward.cross(up).normalized();
	const Vector3 vert = forward.cross(side).normalized();

	const Vector3 base = to - forward * tip_length;
	const float spread = tip_length * 0.5f;

	line(to, base + side * spread, duration, color, width);
	line(to, base - side * spread, duration, color, width);
	line(to, base + vert * spread, duration, color, width);
	line(to, base - vert * spread, duration, color, width);
}

void Visualizer3D::wire_box(const Vector3 &position, float size, const Vector3 &rotation, float duration, const Color &color, float width) {
	const float h = size * 0.5f;

	Vector3 corners[8] = {
		Vector3(-h, -h, -h), Vector3(h, -h, -h), Vector3(h, h, -h), Vector3(-h, h, -h),
		Vector3(-h, -h, h), Vector3(h, -h, h), Vector3(h, h, h), Vector3(-h, h, h)
	};

	Basis rot_basis;
	rot_basis.set_euler(rotation);

	for (int i = 0; i < 8; i++) {
		corners[i] = position + rot_basis.xform(corners[i]);
	}

	static const int edges[12][2] = {
		{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
		{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
		{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
	};

	for (int i = 0; i < 12; i++) {
		line(corners[edges[i][0]], corners[edges[i][1]], duration, color, width);
	}
}

void Visualizer3D::wire_sphere(const Vector3 &position, float radius, float duration, const Color &color, float width) {
	const int segments = 24;
	const float step = Math::TAU / segments;

	for (int axis = 0; axis < 3; axis++) {
		for (int i = 0; i < segments; i++) {
			float a0 = i * step;
			float a1 = (i + 1) * step;

			Vector3 p0;
			Vector3 p1;
			switch (axis) {
				case 0:
					p0 = Vector3(Math::cos(a0), Math::sin(a0), 0);
					p1 = Vector3(Math::cos(a1), Math::sin(a1), 0);
					break;
				case 1:
					p0 = Vector3(Math::cos(a0), 0, Math::sin(a0));
					p1 = Vector3(Math::cos(a1), 0, Math::sin(a1));
					break;
				case 2:
					p0 = Vector3(0, Math::cos(a0), Math::sin(a0));
					p1 = Vector3(0, Math::cos(a1), Math::sin(a1));
					break;
			}
			line(position + p0 * radius, position + p1 * radius, duration, color, width);
		}
	}
}


void Visualizer3D::clear() {
	lines.clear();
}

void Visualizer3D::process(double p_delta_time) {
	// Build the mesh from currently-alive lines.
	_rebuild_mesh();

	uint32_t i = 0;
	while (i < lines.size()) {
		DebugLine &l = lines[i];

		// 0 = infinite lifetime
		if (l.remaining_time == 0.0f) {
			i++;
			continue;
		}

		// -1 = one _process frame
		if (l.remaining_time == -1.0f) {
			lines.remove_at_unordered(i);
			continue;
		}

		// -2 = one _physics_process frame
		// if (is_physics && l.remaining_time == -2.0f) {
		// 	lines.remove_at_unordered(i);
		// 	continue;
		// }

		if (l.remaining_time > 0.0f) {
			l.remaining_time -= p_delta_time;
			if (l.remaining_time <= 0.0f) {
				lines.remove_at_unordered(i);
				continue;
			}
		}

		i++;
	}
}

RID Visualizer3D::_get_active_scenario() const {
	// Editor: use the edited scene root's world.
	if (Engine::get_singleton()->is_editor_hint()) {
		EditorNode *en = EditorNode::get_singleton();
		if (en) {
			Node *root = en->get_editor_data().get_edited_scene_root();
			if (root) {
				Viewport *vp = root->get_viewport();
				if (vp && vp->find_world_3d().is_valid()) {
					return vp->find_world_3d()->get_scenario();
				}
			}
		}
		return RID();
	}

	// Runtime: use the SceneTree root window's world.
	SceneTree *st = SceneTree::get_singleton();
	if (!st) {
		return RID();
	}

	Window *root = st->get_root();
	if (!root) {
		return RID();
	}

	Ref<World3D> world = root->find_world_3d();
	if (world.is_null()) {
		return RID();
	}

	return world->get_scenario();
}

void Visualizer3D::_ensure_rs_resources() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (!rs) {
		return;
	}

	if (!rs_initialized) {
		mesh_rid = rs->mesh_create();
		instance_rid = rs->instance_create();
		rs->instance_set_base(instance_rid, mesh_rid);
		rs->instance_geometry_set_cast_shadows_setting(instance_rid, RenderingServerEnums::SHADOW_CASTING_SETTING_OFF);
		rs->instance_set_ignore_culling(instance_rid, true);
		rs_initialized = true;
	}

	// Re-bind the scenario if the active scene has changed.
	RID active_scenario = _get_active_scenario();
	if (active_scenario.is_valid() && active_scenario != scenario_rid) {
		scenario_rid = active_scenario;
		rs->instance_set_scenario(instance_rid, scenario_rid);
	}
}

void Visualizer3D::_free_rs_resources() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (!rs) {
		return;
	}

	if (instance_rid.is_valid()) {
		rs->free_rid(instance_rid);
		instance_rid = RID();
	}

	if (mesh_rid.is_valid()) {
		rs->free_rid(mesh_rid);
		mesh_rid = RID();
	}

	scenario_rid = RID();
	rs_initialized = false;
}

void Visualizer3D::_rebuild_mesh() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (!rs) {
		return;
	}

	_ensure_rs_resources();
	if (!mesh_rid.is_valid()) {
		return;
	}

	rs->mesh_clear(mesh_rid);

	const uint32_t line_count = lines.size();
	if (line_count == 0) {
		return;
	}

	// Two triangles (6 vertices) per line, expanded into a screen-space quad
	// in the vertex shader.
	const uint32_t vert_count = line_count * 6;

	PackedVector3Array positions;
	PackedColorArray colors;
	PackedFloat32Array custom0; // (offset_x, offset_y, width, t)  - vec4
	PackedFloat32Array custom1; // (from.xyz, 0)                   - vec4
	PackedFloat32Array custom2; // (to.xyz,   0)                   - vec4

	positions.resize(vert_count);
	colors.resize(vert_count);
	custom0.resize(vert_count * 4);
	custom1.resize(vert_count * 4);
	custom2.resize(vert_count * 4);

	Vector3 *pos_w = positions.ptrw();
	Color *col_w = colors.ptrw();
	float *c0_w = custom0.ptrw();
	float *c1_w = custom1.ptrw();
	float *c2_w = custom2.ptrw();

	// Per-vertex (offset_x, offset_y, t) where t selects from(0)/to(1).
	// Triangles: (a-,a+,b+) and (a-,b+,b-) -- a proper quad.
	struct Corner {
		float ox;
		float oy;
		float t;
	};

	static const Corner corners[6] = {
		{ -1.0f, -1.0f, 0.0f }, // a-
		{ 1.0f, -1.0f, 1.0f }, // b- (a+ in screen space, but at t=1)
		{ 1.0f, 1.0f, 1.0f }, // b+
		{ -1.0f, -1.0f, 0.0f }, // a-
		{ 1.0f, 1.0f, 1.0f }, // b+
		{ -1.0f, 1.0f, 0.0f }, // a+
	};

	for (uint32_t i = 0; i < line_count; i++) {
		const DebugLine &line = lines[i];
		const uint32_t base_v = i * 6;

		for (uint32_t j = 0; j < 6; j++) {
			const uint32_t v = base_v + j;
			const Corner &c = corners[j];

			// The vertex shader only uses CUSTOM1/CUSTOM2 for the actual
			// world position, but we still need a non-degenerate vertex
			// position so the bounding sphere isn't zero-sized.
			pos_w[v] = (c.t < 0.5f) ? line.from : line.to;
			col_w[v] = line.color;

			c0_w[v * 4 + 0] = c.ox;
			c0_w[v * 4 + 1] = c.oy;
			c0_w[v * 4 + 2] = line.width;
			c0_w[v * 4 + 3] = c.t;

			c1_w[v * 4 + 0] = line.from.x;
			c1_w[v * 4 + 1] = line.from.y;
			c1_w[v * 4 + 2] = line.from.z;
			c1_w[v * 4 + 3] = 0.0f;

			c2_w[v * 4 + 0] = line.to.x;
			c2_w[v * 4 + 1] = line.to.y;
			c2_w[v * 4 + 2] = line.to.z;
			c2_w[v * 4 + 3] = 0.0f;
		}
	}

	Array arrays;
	arrays.resize(RenderingServerEnums::ARRAY_MAX);
	arrays[RenderingServerEnums::ARRAY_VERTEX] = positions;
	arrays[RenderingServerEnums::ARRAY_COLOR] = colors;
	arrays[RenderingServerEnums::ARRAY_CUSTOM0] = custom0;
	arrays[RenderingServerEnums::ARRAY_CUSTOM1] = custom1;
	arrays[RenderingServerEnums::ARRAY_CUSTOM2] = custom2;

	// Tell the RenderingServer how to interpret each custom array.
	// All three are 4 x float32 (RGBA_FLOAT).
	const uint64_t compress_flags =
			(uint64_t(RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT) << RenderingServerEnums::ARRAY_FORMAT_CUSTOM0_SHIFT) |
			(uint64_t(RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT) << RenderingServerEnums::ARRAY_FORMAT_CUSTOM1_SHIFT) |
			(uint64_t(RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT) << RenderingServerEnums::ARRAY_FORMAT_CUSTOM2_SHIFT);

	rs->mesh_add_surface_from_arrays(
			mesh_rid,
			RenderingServerEnums::PRIMITIVE_TRIANGLES,
			arrays,
			Array(),
			Dictionary(),
			compress_flags);

	if (debug_material.is_valid()) {
		rs->mesh_surface_set_material(mesh_rid, 0, debug_material->get_rid());
	}

	// Make sure the instance is never frustum-culled in weird ways: lines can
	// span very large distances, so we rely on a generous custom AABB.
	AABB aabb;
	if (line_count > 0) {
		aabb.position = lines[0].from;
		for (uint32_t i = 0; i < line_count; i++) {
			aabb.expand_to(lines[i].from);
			aabb.expand_to(lines[i].to);
		}
	}
	rs->instance_set_custom_aabb(instance_rid, aabb);
}

void Visualizer3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("line", "from", "to", "duration", "color", "width"), &Visualizer3D::line, DEFVAL(0.0f), DEFVAL(Color(1, 1, 1)), DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("arrow", "from", "to", "duration", "color", "width"), &Visualizer3D::arrow, DEFVAL(0.0f), DEFVAL(Color(1, 1, 1)), DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("wire_box", "position", "size", "rotation", "duration", "color", "width"), &Visualizer3D::wire_box, DEFVAL(Vector3()), DEFVAL(0.0f), DEFVAL(Color(1, 1, 1)), DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("wire_sphere", "position", "radius", "duration", "color", "width"), &Visualizer3D::wire_sphere, DEFVAL(0.0f), DEFVAL(Color(1, 1, 1)), DEFVAL(1.0f));

	ClassDB::bind_method(D_METHOD("clear"), &Visualizer3D::clear);
}

Visualizer3D::Visualizer3D() {
	singleton = this;

	// Screen-space line shader.
	// CUSTOM0.xy = quad corner offset (-1/+1)
	// CUSTOM0.z  = line width in pixels
	// CUSTOM0.w  = endpoint selector (0 = from, 1 = to)
	// CUSTOM1.xyz = world from
	// CUSTOM2.xyz = world to
	Ref<Shader> shader;
	shader.instantiate();
	shader->set_code(R"(
shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_opaque, blend_mix, fog_disabled, shadows_disabled;

void vertex() {
	vec3 world_a = CUSTOM1.xyz;
	vec3 world_b = CUSTOM2.xyz;

	vec4 clip_a = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(world_a, 1.0));
	vec4 clip_b = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(world_b, 1.0));

	// Convert to screen space (in pixels).
	vec2 screen_a = (clip_a.xy / clip_a.w) * 0.5 * VIEWPORT_SIZE;
	vec2 screen_b = (clip_b.xy / clip_b.w) * 0.5 * VIEWPORT_SIZE;

	// Build a 2D basis along the line in screen space.
	vec2 dir = screen_b - screen_a;
	float dir_len = length(dir);
	vec2 x_basis = dir_len > 0.0001 ? dir / dir_len : vec2(1.0, 0.0);
	vec2 y_basis = vec2(-x_basis.y, x_basis.x);

	float width_px = max(CUSTOM0.z, 1.0);
	float t = CUSTOM0.w;

	// Pick the correct endpoint, then push out perpendicular by half-width
	// and along the line by half-width (to cap the ends square).
	vec4 clip = mix(clip_a, clip_b, t);
	vec2 screen = mix(screen_a, screen_b, t);

	vec2 offset_px = (CUSTOM0.x * x_basis + CUSTOM0.y * y_basis) * (width_px * 0.5);
	vec2 screen_final = screen + offset_px;

	// Convert back to clip space, preserving original depth (clip.z/clip.w).
	vec2 ndc = (screen_final / (0.5 * VIEWPORT_SIZE));
	POSITION = vec4(ndc * clip.w, clip.z, clip.w);
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");

	Ref<ShaderMaterial> mat;
	mat.instantiate();
	mat->set_shader(shader);
	debug_material = mat;
}

Visualizer3D::~Visualizer3D() {
	_free_rs_resources();
	if (singleton == this) {
		singleton = nullptr;
	}
}
