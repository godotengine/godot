/*************************************************************************/
/*  spine.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifdef MODULE_SPINE_ENABLED

#include "core/io/resource_loader.h"
#include "scene/2d/collision_object_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "method_bind_ext.inc"

#include "spine.h"
#include <spine/spine.h>
#include <spine/extension.h>

Spine::SpineResource::SpineResource() {

	atlas = NULL;
	data = NULL;
}

Spine::SpineResource::~SpineResource() {

	if (atlas != NULL)
		spAtlas_dispose(atlas);

	if (data != NULL)
		spSkeletonData_dispose(data);
}


void Spine::spine_animation_callback(spAnimationState* p_state, int p_track, spEventType p_type, spEvent* p_event, int loop_count) {

	((Spine*)p_state->rendererObject)->_on_animation_state_event(p_track, p_type, p_event, loop_count);
}

void Spine::_on_animation_state_event(int p_track, spEventType p_type, spEvent *p_event, int p_loop_count) {

	switch (p_type) {
	case SP_ANIMATION_START:
		emit_signal("animation_start", p_track);
		break;
	case SP_ANIMATION_COMPLETE:
		emit_signal("animation_complete", p_track, p_loop_count);
		break;
	case SP_ANIMATION_EVENT: {
			Dictionary event;
			event["name"] = p_event->data->name;
			event["int"] = p_event->intValue;
			event["float"] = p_event->floatValue;
			event["string"] = p_event->stringValue ? p_event->stringValue : "";
			emit_signal("animation_event", p_track, event);
		}
		break;
	case SP_ANIMATION_END:
		emit_signal("animation_end", p_track);
		break;
	}
}

void Spine::_spine_dispose() {

	if (playing) {
		// stop first
		stop();
	}

	if (state) {

		spAnimationStateData_dispose(state->data);
		spAnimationState_dispose(state);
	}

	if (skeleton)
		spSkeleton_dispose(skeleton);

	state = NULL;
	skeleton = NULL;
	res = RES();

	for (AttachmentNodes::Element *E = attachment_nodes.front(); E; E = E->next()) {
		
		AttachmentNode& node = E->get();
		memdelete(node.ref);
	}
	attachment_nodes.clear();

	update();
}

static Ref<Texture> spine_get_texture(spRegionAttachment* attachment) {

	Ref<Texture> *ref = static_cast<Ref<Texture> *>(
		((spAtlasRegion*)attachment->rendererObject)->page->rendererObject
	);
	return *ref;
}

static Ref<Texture> spine_get_texture(spMeshAttachment* attachment) {

	Ref<Texture> *ref = static_cast<Ref<Texture> *>(
		((spAtlasRegion*)attachment->rendererObject)->page->rendererObject
		);
	return *ref;
}

static Ref<Texture> spine_get_texture(spSkinnedMeshAttachment* attachment) {

	Ref<Texture> *ref = static_cast<Ref<Texture> *>(
		((spAtlasRegion*)attachment->rendererObject)->page->rendererObject
		);
	return *ref;
}

void Spine::_on_fx_draw() {

	if (skeleton == NULL)
		return;
	fx_batcher.reset();
	RID eci = fx_node->get_canvas_item();
	VisualServer::get_singleton()->canvas_item_add_set_blend_mode(eci, VS::MaterialBlendMode(fx_node->get_blend_mode()));
	fx_batcher.flush();
}

void Spine::_animation_draw() {

	if (skeleton == NULL)
		return;

	skeleton->r = modulate.r;
	skeleton->g = modulate.g;
	skeleton->b = modulate.b;
	skeleton->a = modulate.a;

	int additive = 0;
	int fx_additive = 0;
	Color color;
	const float *uvs = NULL;
	int verties_count = 0;
	const int *triangles = NULL;
	int triangles_count = 0;
	float r = 0, g = 0, b = 0, a = 0;

	RID ci = this->get_canvas_item();
	batcher.reset();
	VisualServer::get_singleton()->canvas_item_add_set_blend_mode(ci, VS::MaterialBlendMode(get_blend_mode()));

	const char *fx_prefix = fx_slot_prefix.get_data();

	for (int i = 0, n = skeleton->slotsCount; i < n; i++) {

		spSlot* slot = skeleton->drawOrder[i];
		if (!slot->attachment) continue;
		bool is_fx = false;
		Ref<Texture> texture;
		switch (slot->attachment->type) {

			case SP_ATTACHMENT_REGION: {

				spRegionAttachment* attachment = (spRegionAttachment*)slot->attachment;
				is_fx = strstr(attachment->path, fx_prefix) != NULL;					
				spRegionAttachment_computeWorldVertices(attachment, slot->bone, world_verts.ptr());
				texture = spine_get_texture(attachment);
				uvs = attachment->uvs;
				verties_count = 8;
				static const int quadTriangles[6] = { 0, 1, 2, 2, 3, 0 };
				triangles = quadTriangles;
				triangles_count = 6;
				r = attachment->r;
				g = attachment->g;
				b = attachment->b;
				a = attachment->a;
				break;
			}
			case SP_ATTACHMENT_MESH: {

				spMeshAttachment* attachment = (spMeshAttachment*)slot->attachment;
				is_fx = strstr(attachment->path, fx_prefix) != NULL;					
				spMeshAttachment_computeWorldVertices(attachment, slot, world_verts.ptr());
				texture = spine_get_texture(attachment);
				uvs = attachment->uvs;
				verties_count = attachment->verticesCount;
				triangles = attachment->triangles;
				triangles_count = attachment->trianglesCount;
				r = attachment->r;
				g = attachment->g;
				b = attachment->b;
				a = attachment->a;
				break;
			}
			case SP_ATTACHMENT_SKINNED_MESH: {

				spSkinnedMeshAttachment* attachment = (spSkinnedMeshAttachment*)slot->attachment;
				is_fx = strstr(attachment->path, fx_prefix) != NULL;					
				spSkinnedMeshAttachment_computeWorldVertices(attachment, slot, world_verts.ptr());
				texture = spine_get_texture(attachment);
				uvs = attachment->uvs;
				verties_count = attachment->uvsCount;
				triangles = attachment->triangles;
				triangles_count = attachment->trianglesCount;
				r = attachment->r;
				g = attachment->g;
				b = attachment->b;
				a = attachment->a;
				break;
			}

			case SP_ATTACHMENT_BOUNDING_BOX: {

				continue;
			}
		}
		if (texture.is_null())
			continue;

		if (is_fx && slot->data->additiveBlending != fx_additive) {

			fx_batcher.add_set_blender_mode(slot->data->additiveBlending
				? VisualServer::MATERIAL_BLEND_MODE_ADD
				: get_blend_mode()
			);
			fx_additive = slot->data->additiveBlending;
		}
		else if (slot->data->additiveBlending != additive) {

			batcher.add_set_blender_mode(slot->data->additiveBlending
				? VisualServer::MATERIAL_BLEND_MODE_ADD
				: fx_node->get_blend_mode()
			);
			additive = slot->data->additiveBlending;
		}

		color.a = skeleton->a * slot->a * a * get_opacity();
		color.r = skeleton->r * slot->r * r;
		color.g = skeleton->g * slot->g * g;
		color.b = skeleton->b * slot->b * b;

		if (is_fx)
			fx_batcher.add(texture, world_verts.ptr(), uvs, verties_count, triangles, triangles_count, &color, flip_x, flip_y);
		else
			batcher.add(texture, world_verts.ptr(), uvs, verties_count, triangles, triangles_count, &color, flip_x, flip_y);
	}
	batcher.flush();
	fx_node->update();

	// Slots.
	if (debug_attachment_region || debug_attachment_mesh || debug_attachment_skinned_mesh || debug_attachment_bounding_box) {

		Color color(0, 0, 1, 1);
		for (int i = 0, n = skeleton->slotsCount; i < n; i++) {

			spSlot* slot = skeleton->drawOrder[i];
			if (!slot->attachment)
				continue;
			switch (slot->attachment->type) {

				case SP_ATTACHMENT_REGION:
				{
					if (!debug_attachment_region)
						continue;
					spRegionAttachment* attachment = (spRegionAttachment*)slot->attachment;
					verties_count = 8;
					spRegionAttachment_computeWorldVertices(attachment, slot->bone, world_verts.ptr());
					color = Color(0, 0, 1, 1);
					triangles = NULL;
					triangles_count = 0;
					break;
				}
				case SP_ATTACHMENT_MESH: {

					if (!debug_attachment_mesh)
						continue;
					spMeshAttachment* attachment = (spMeshAttachment*)slot->attachment;
					spMeshAttachment_computeWorldVertices(attachment, slot, world_verts.ptr());
					verties_count = attachment->verticesCount;
					color = Color(0, 1, 1, 1);
					triangles = attachment->triangles;
					triangles_count = attachment->trianglesCount;
					break;
				}
				case SP_ATTACHMENT_SKINNED_MESH: {

					if (!debug_attachment_skinned_mesh)
						continue;
					spSkinnedMeshAttachment* attachment = (spSkinnedMeshAttachment*)slot->attachment;
					spSkinnedMeshAttachment_computeWorldVertices(attachment, slot, world_verts.ptr());
					verties_count = attachment->uvsCount;
					color = Color(1, 0, 1, 1);
					triangles = attachment->triangles;
					triangles_count = attachment->trianglesCount;
					break;
				}
				case SP_ATTACHMENT_BOUNDING_BOX: {

					if (!debug_attachment_bounding_box)
						continue;
					spBoundingBoxAttachment* attachment = (spBoundingBoxAttachment*)slot->attachment;
					spBoundingBoxAttachment_computeWorldVertices(attachment, slot->bone, world_verts.ptr());
					verties_count = attachment->verticesCount;
					color = Color(0, 1, 0, 1);
					triangles = NULL;
					triangles_count = 0;
					break;
				}
			}

			Point2 *points = (Point2 *)world_verts.ptr();
			int points_size = verties_count / 2;

			for (int idx = 0; idx < points_size; idx++) {

				Point2& pt = points[idx];
				if (flip_x)
					pt.x = -pt.x;
				if (!flip_y)
					pt.y = -pt.y;
			}

			if (triangles == NULL || triangles_count == 0) {

				for (int idx = 0; idx < points_size; idx++) {

					if (idx == points_size - 1)
						draw_line(points[idx], points[0], color);
					else
						draw_line(points[idx], points[idx + 1], color);
				}
			} else {

				for (int idx = 0; idx < triangles_count - 2; idx += 3) {

					int a = triangles[idx];
					int b = triangles[idx + 1];
					int c = triangles[idx + 2];

					draw_line(points[a], points[b], color);
					draw_line(points[b], points[c], color);
					draw_line(points[c], points[a], color);
				}
			}
		}
	}

	if (debug_bones) {
		// Bone lengths.
		for (int i = 0; i < skeleton->bonesCount; i++) {
			spBone *bone = skeleton->bones[i];
			float x = bone->data->length * bone->m00 + bone->worldX;
			float y = bone->data->length * bone->m10 + bone->worldY;
			draw_line(Point2(flip_x ? -bone->worldX : bone->worldX,
				flip_y ? bone->worldY : -bone->worldY),
				Point2(flip_x ? -x : x, flip_y ? y : -y),
				Color(1, 0, 0, 1),
				2
			);
		}
		// Bone origins.
		for (int i = 0, n = skeleton->bonesCount; i < n; i++) {
			spBone *bone = skeleton->bones[i];
			Rect2 rt = Rect2(flip_x ? -bone->worldX - 1 : bone->worldX - 1,
				flip_y ? bone->worldY - 1 : -bone->worldY - 1,
				3,
				3
			);
			draw_rect(rt, (i == 0) ? Color(0, 1, 0, 1) : Color(0, 0, 1, 1));
		}
	}
}

void Spine::_animation_process(float p_delta) {

	if (speed_scale == 0)
		return;
	p_delta *= speed_scale;

	spAnimationState_update(state, forward ? p_delta : -p_delta);
	spAnimationState_apply(state, skeleton);
	spSkeleton_updateWorldTransform(skeleton);

	for (AttachmentNodes::Element *E = attachment_nodes.front(); E; E = E->next()) {

		AttachmentNode& info = E->get();
		WeakRef *ref = info.ref;
		Object *obj = ref->get_ref();
		Node2D *node = (obj != NULL) ? obj->cast_to<Node2D>() : NULL;
		if (obj == NULL || node == NULL) {

			AttachmentNodes::Element *NEXT = E->next();
			attachment_nodes.erase(E);
			E = NEXT;
			if (E == NULL)
				break;
			continue;
		}
		const spBone *bone = info.bone;
		node->call("set_pos", Vector2(bone->worldX + bone->skeleton->x, -bone->worldY + bone->skeleton->y) + info.ofs);
		node->call("set_scale", Vector2(bone->worldScaleX, bone->worldScaleY) * info.scale);
		node->call("set_rot", Math::atan2(bone->m10, bone->m11) + Math::deg2rad(info.rot));
	}
	update();
}

void Spine::_set_process(bool p_process, bool p_force) {

	if (processing == p_process && !p_force)
		return;

	switch (animation_process_mode) {

	case ANIMATION_PROCESS_FIXED: set_fixed_process(p_process && active); break;
	case ANIMATION_PROCESS_IDLE: set_process(p_process && active); break;
	}

	processing = p_process;
}

bool Spine::_set(const StringName& p_name, const Variant& p_value) {

	String name = p_name;

	if (name == "playback/play") {

		String which = p_value;
		if (skeleton != NULL) {

			if (which == "[stop]")
				stop();
			else if (has(which)) {
				reset();
				play(which, 1, loop);
			}
		} else
			current_animation = which;
	}
	else if (name == "playback/loop") {

		loop = p_value;
		if (skeleton != NULL && has(current_animation))
			play(current_animation, 1, loop);
	}
	else if (name == "playback/forward") {

		forward = p_value;
	}
	else if (name == "playback/skin") {

		skin = p_value;
		if (skeleton != NULL)
			set_skin(skin);
	}
	else if (name == "debug/region")
		set_debug_attachment(DEBUG_ATTACHMENT_REGION, p_value);
	else if (name == "debug/mesh")
		set_debug_attachment(DEBUG_ATTACHMENT_MESH, p_value);
	else if (name == "debug/skinned_mesh")
		set_debug_attachment(DEBUG_ATTACHMENT_SKINNED_MESH, p_value);
	else if (name == "debug/bounding_box")
		set_debug_attachment(DEBUG_ATTACHMENT_BOUNDING_BOX, p_value);

	return true;
}

bool Spine::_get(const StringName& p_name, Variant &r_ret) const {

	String name = p_name;

	if (name == "playback/play") {

		r_ret = current_animation;
	}
	else if (name == "playback/loop")
		r_ret = loop;
	else if (name == "playback/forward")
		r_ret = forward;
	else if (name == "playback/skin")
		r_ret = skin;
	else if (name == "debug/region")
		r_ret = is_debug_attachment(DEBUG_ATTACHMENT_REGION);
	else if (name == "debug/mesh")
		r_ret = is_debug_attachment(DEBUG_ATTACHMENT_MESH);
	else if (name == "debug/skinned_mesh")
		r_ret = is_debug_attachment(DEBUG_ATTACHMENT_SKINNED_MESH);
	else if (name == "debug/bounding_box")
		r_ret = is_debug_attachment(DEBUG_ATTACHMENT_BOUNDING_BOX);

	return true;
}

void Spine::_get_property_list(List<PropertyInfo> *p_list) const {

	List<String> names;

	if (state != NULL) {

		for (int i = 0; i < state->data->skeletonData->animationsCount; i++) {

			names.push_back(state->data->skeletonData->animations[i]->name);
		}
	}
	{
		names.sort();
		names.push_front("[stop]");
		String hint;
		for (List<String>::Element *E = names.front(); E; E = E->next()) {

			if (E != names.front())
				hint += ",";
			hint += E->get();
		}

		p_list->push_back(PropertyInfo(Variant::STRING, "playback/play", PROPERTY_HINT_ENUM, hint));
		p_list->push_back(PropertyInfo(Variant::BOOL, "playback/loop", PROPERTY_HINT_NONE));
		p_list->push_back(PropertyInfo(Variant::BOOL, "playback/forward", PROPERTY_HINT_NONE));
	}

	names.clear();
	{
		if (state != NULL) {

			for (int i = 0; i < state->data->skeletonData->skinsCount; i++) {

				names.push_back(state->data->skeletonData->skins[i]->name);
			}
		}

		String hint;
		for (List<String>::Element *E = names.front(); E; E = E->next()) {

			if (E != names.front())
				hint += ",";
			hint += E->get();
		}

		p_list->push_back(PropertyInfo(Variant::STRING, "playback/skin", PROPERTY_HINT_ENUM, hint));
	}
	p_list->push_back(PropertyInfo(Variant::BOOL, "debug/region", PROPERTY_HINT_NONE));
	p_list->push_back(PropertyInfo(Variant::BOOL, "debug/mesh", PROPERTY_HINT_NONE));
	p_list->push_back(PropertyInfo(Variant::BOOL, "debug/skinned_mesh", PROPERTY_HINT_NONE));
	p_list->push_back(PropertyInfo(Variant::BOOL, "debug/bounding_box", PROPERTY_HINT_NONE));
}

void Spine::_notification(int p_what) {

	switch (p_what) {

	case NOTIFICATION_ENTER_TREE: {

		if (!processing) {
			//make sure that a previous process state was not saved
			//only process if "processing" is set
			set_fixed_process(false);
			set_process(false);
		}
	} break;
	case NOTIFICATION_READY: {

		// add fx node as child
		fx_node->connect("draw", this, "_on_fx_draw");
		fx_node->set_z(1);
		fx_node->set_z_as_relative(false);
		add_child(fx_node);

		if (!get_tree()->is_editor_hint() && has(autoplay)) {
			play(autoplay);
		}
	} break;
	case NOTIFICATION_PROCESS: {
		if (animation_process_mode == ANIMATION_PROCESS_FIXED)
			break;

		if (processing)
			_animation_process(get_process_delta_time());
	} break;
	case NOTIFICATION_FIXED_PROCESS: {

		if (animation_process_mode == ANIMATION_PROCESS_IDLE)
			break;

		if (processing)
			_animation_process(get_fixed_process_delta_time());
	} break;

	case NOTIFICATION_DRAW: {

		_animation_draw();
	} break;

	case NOTIFICATION_EXIT_TREE: {

		stop_all();
	} break;
	}
}

void Spine::set_resource(Ref<Spine::SpineResource> p_data) {

	// cleanup
	_spine_dispose();

	res = p_data;
	if (res.is_null())
		return;

	skeleton = spSkeleton_create(res->data);
	root_bone = skeleton->bones[0];

	state = spAnimationState_create(spAnimationStateData_create(skeleton->data));
	state->rendererObject = this;
	state->listener = spine_animation_callback;

	if (skin != "")
		set_skin(skin);
	if (current_animation != "[stop]")
		play(current_animation, 1, loop);
	else
		reset();

	_change_notify();
}

Ref<Spine::SpineResource> Spine::get_resource() {

	return res;
}

bool Spine::has(const String& p_name) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	spAnimation* animation = spSkeletonData_findAnimation(skeleton->data, p_name.utf8().get_data());
	return animation != NULL;
}

void Spine::mix(const String& p_from, const String& p_to, real_t p_duration) {

	ERR_FAIL_COND(state == NULL);
	spAnimationStateData_setMixByName(state->data, p_from.utf8().get_data(), p_to.utf8().get_data(), p_duration);
}

bool Spine::play(const String& p_name, real_t p_cunstom_scale, bool p_loop, int p_track, int p_delay) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	spAnimation* animation = spSkeletonData_findAnimation(skeleton->data, p_name.utf8().get_data());
	ERR_FAIL_COND_V(animation == NULL, false);
	spTrackEntry *entry = spAnimationState_setAnimation(state, p_track, animation, p_loop);
	entry->delay = p_delay;
	current_animation = p_name;

	_set_process(true);
	playing = true;
	// update frame
	if (!is_active())
		_animation_process(0);

	return true;
}

bool Spine::add(const String& p_name, real_t p_cunstom_scale, bool p_loop, int p_track, int p_delay) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	spAnimation* animation = spSkeletonData_findAnimation(skeleton->data, p_name.utf8().get_data());
	ERR_FAIL_COND_V(animation == NULL, false);
	spTrackEntry *entry = spAnimationState_addAnimation(state, p_track, animation, p_loop, p_delay);

	_set_process(true);
	playing = true;

	return true;
}

void Spine::clear(int p_track) {

	ERR_FAIL_COND(state == NULL);
	if (p_track == -1)
		spAnimationState_clearTracks(state);
	else
		spAnimationState_clearTrack(state, p_track);
}

void Spine::stop() {

	_set_process(false);
	playing = false;
	current_animation = "[stop]";
	reset();
}

bool Spine::is_playing(int p_track) const {

	return playing && spAnimationState_getCurrent(state, p_track) != NULL;
}

void Spine::set_forward(bool p_forward) {

	forward = p_forward;
}

bool Spine::is_forward() const {

	return forward;
}

String Spine::get_current_animation(int p_track) const {

	ERR_FAIL_COND_V(state == NULL, "");
	spTrackEntry *entry = spAnimationState_getCurrent(state, p_track);
	if (entry == NULL || entry->animation == NULL)
		return "";
	return entry->animation->name;
}

void Spine::stop_all() {

	stop();

	_set_process(false); // always process when starting an animation
}

void Spine::reset() {

	ERR_FAIL_COND(skeleton == NULL);
	spSkeleton_setToSetupPose(skeleton);
	spAnimationState_update(state, 0);
	spAnimationState_apply(state, skeleton);
	spSkeleton_updateWorldTransform(skeleton);
}

void Spine::seek(float p_pos) {

	_animation_process(p_pos - current_pos);
}

float Spine::tell() const {

	return current_pos;
}

void Spine::set_active(bool p_active) {

	if (active == p_active)
		return;

	active = p_active;
	_set_process(processing, true);
}

bool Spine::is_active() const {

	return active;
}

void Spine::set_speed(float p_speed) {

	speed_scale = p_speed;
}

float Spine::get_speed() const {

	return speed_scale;
}

void Spine::set_autoplay(const String& p_name) {

	autoplay = p_name;
}

String Spine::get_autoplay() const {

	return autoplay;
}

void Spine::set_modulate(const Color& p_color) {

	modulate = p_color;
	update();
}

Color Spine::get_modulate() const{

	return modulate;
}

void Spine::set_flip_x(bool p_flip) {

	flip_x = p_flip;
	update();
}

void Spine::set_flip_y(bool p_flip) {

	flip_y = p_flip;
	update();
}

bool Spine::is_flip_x() const {

	return flip_x;
}

bool Spine::is_flip_y() const {

	return flip_y;
}

bool Spine::set_skin(const String& p_name) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	return spSkeleton_setSkinByName(skeleton, p_name.utf8().get_data()) ? true : false;
}

Dictionary Spine::get_skeleton() const {

	ERR_FAIL_COND_V(skeleton == NULL, Variant());
	Dictionary dict;

	dict["bonesCount"] = skeleton->bonesCount;
	dict["slotCount"] = skeleton->slotsCount;
	dict["ikConstraintsCount"] = skeleton->ikConstraintsCount;
	dict["time"] = skeleton->time;
	dict["flipX"] = skeleton->flipX;
	dict["flipY"] = skeleton->flipY;
	dict["x"] = skeleton->x;
	dict["y"] = skeleton->y;

	return dict;
}

Dictionary Spine::get_attachment(const String& p_slot_name, const String& p_attachment_name) const {

	ERR_FAIL_COND_V(skeleton == NULL, Variant());
	spAttachment *attachment = spSkeleton_getAttachmentForSlotName(skeleton, p_slot_name.utf8().get_data(), p_attachment_name.utf8().get_data());
	ERR_FAIL_COND_V(attachment == NULL, Variant());

	Dictionary dict;
	dict["name"] = attachment->name;

	switch (attachment->type) {
		case SP_ATTACHMENT_REGION: {

			spRegionAttachment *info = (spRegionAttachment *)attachment;
			dict["type"] = "region";
			dict["path"] = info->path;
			dict["x"] = info->x;
			dict["y"] = info->y;
			dict["scaleX"] = info->scaleX;
			dict["scaleY"] = info->scaleY;
			dict["rotation"] = info->rotation;
			dict["width"] = info->width;
			dict["height"] = info->height;
			dict["color"] = Color(info->r, info->g, info->b, info->a);
			dict["region"] = Rect2(info->regionOffsetX, info->regionOffsetY, info->regionWidth, info->regionHeight);
			dict["region_original_size"] = Size2(info->regionOriginalWidth, info->regionOriginalHeight);

			Vector2Array offset, uvs;
			for (int idx = 0; idx < 4; idx++) {
				offset.push_back(Vector2(info->offset[idx * 2], info->offset[idx * 2 + 1]));
				uvs.push_back(Vector2(info->uvs[idx * 2], info->uvs[idx * 2 + 1]));
			}
			dict["offset"] = offset;
			dict["uvs"] = uvs;

		} break;

		case SP_ATTACHMENT_BOUNDING_BOX: {

			spBoundingBoxAttachment *info = (spBoundingBoxAttachment *)attachment;
			dict["type"] = "bounding_box";

			Vector2Array vertices;
			for (int idx = 0; idx < info->verticesCount / 2; idx++)
				vertices.append(Vector2(info->vertices[idx * 2], -info->vertices[idx * 2 + 1]));
			dict["vertices"] = vertices;
		} break;

		case SP_ATTACHMENT_MESH:  {

			spMeshAttachment *info = (spMeshAttachment *)attachment;
			dict["type"] = "mesh";
			dict["path"] = info->path;
			dict["color"] = Color(info->r, info->g, info->b, info->a);
		} break;

		case SP_ATTACHMENT_SKINNED_MESH:  {

			spSkinnedMeshAttachment *info = (spSkinnedMeshAttachment *)attachment;
			dict["type"] = "skinned_mesh";
			dict["path"] = info->path;
			dict["color"] = Color(info->r, info->g, info->b, info->a);
		} break;
	}
	return dict;
}

Dictionary Spine::get_bone(const String& p_bone_name) const {

	ERR_FAIL_COND_V(skeleton == NULL, Variant());
	spBone *bone = spSkeleton_findBone(skeleton, p_bone_name.utf8().get_data());
	ERR_FAIL_COND_V(bone == NULL, Variant());
	Dictionary dict;
	dict["x"] = bone->x;
	dict["y"] = bone->y;
	dict["rotation"] = bone->rotation;
	dict["rotationIK"] = bone->rotationIK;
	dict["scaleX"] = bone->scaleX;
	dict["scaleY"] = bone->scaleY;
	dict["flipX"] = bone->flipX;
	dict["flipY"] = bone->flipY;
	dict["m00"] = bone->m00;
	dict["m01"] = bone->m01;
	dict["m10"] = bone->m10;
	dict["m11"] = bone->m11;
	dict["worldX"] = bone->worldX;
	dict["worldY"] = bone->worldY;
	dict["worldRotation"] = bone->worldRotation;
	dict["worldScaleX"] = bone->worldScaleX;
	dict["worldScaleY"] = bone->worldScaleY;
	dict["worldFlipX"] = bone->worldFlipX;
	dict["worldFlipY"] = bone->worldFlipY;

	return dict;
}

Dictionary Spine::get_slot(const String& p_slot_name) const {

	ERR_FAIL_COND_V(skeleton == NULL, Variant());
	spSlot *slot = spSkeleton_findSlot(skeleton, p_slot_name.utf8().get_data());
	ERR_FAIL_COND_V(slot == NULL, Variant());
	Dictionary dict;
	dict["color"] = Color(slot->r, slot->g, slot->b, slot->a);
	return dict;
}

bool Spine::set_attachment(const String& p_slot_name, const Variant& p_attachment) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	if (p_attachment.get_type() == Variant::STRING)
		return spSkeleton_setAttachment(skeleton, p_slot_name.utf8().get_data(), ((const String)p_attachment).utf8().get_data()) != 0;
	else
		return spSkeleton_setAttachment(skeleton, p_slot_name.utf8().get_data(), NULL) != 0;
}

bool Spine::has_attachment_node(const String& p_bone_name, const Variant& p_node) {

	return false;
}

bool Spine::add_attachment_node(const String& p_bone_name, const Variant& p_node, const Vector2& p_ofs, const Vector2& p_scale, const real_t p_rot) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	spBone *bone = spSkeleton_findBone(skeleton, p_bone_name.utf8().get_data());
	ERR_FAIL_COND_V(bone == NULL, false);
	Object *obj = p_node;
	ERR_FAIL_COND_V(obj == NULL, false);
	Node2D *node = obj->cast_to<Node2D>();
	ERR_FAIL_COND_V(node == NULL, false);

	if (obj->has_meta("spine_meta")) {

		AttachmentNode *info = (AttachmentNode *) ((size_t) obj->get_meta("spine_meta"));
		if (info->bone != bone) {
			// add to different bone, remove first
			remove_attachment_node(info->bone->data->name, p_node);
		} else {
			// add to same bone, update params
			info->ofs = p_ofs;
			info->scale = p_scale;
			info->rot = p_rot;
			return true;
		}
	}
	attachment_nodes.push_back(AttachmentNode());
	AttachmentNode& info = attachment_nodes.back()->get();
	info.E = attachment_nodes.back();
	info.bone = bone;
	info.ref = memnew(WeakRef);
	info.ref->set_obj(node);
	info.ofs = p_ofs;
	info.scale = p_scale;
	info.rot = p_rot;
	obj->set_meta("spine_meta", (size_t) &info);

	return true;
}

bool Spine::remove_attachment_node(const String& p_bone_name, const Variant& p_node) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	spBone *bone = spSkeleton_findBone(skeleton, p_bone_name.utf8().get_data());
	ERR_FAIL_COND_V(bone == NULL, false);
	Object *obj = p_node;
	ERR_FAIL_COND_V(obj == NULL, false);
	Node2D *node = obj->cast_to<Node2D>();
	ERR_FAIL_COND_V(node == NULL, false);

	if (!obj->has_meta("spine_meta"))
		return false;

	AttachmentNode *info = (AttachmentNode *)((size_t)obj->get_meta("spine_meta"));
	ERR_FAIL_COND_V(info->bone != bone, false);
	obj->set_meta("spine_meta", NULL);
	memdelete(info->ref);
	attachment_nodes.erase(info->E);

	return false;
}

Ref<Shape2D> Spine::get_bounding_box(const String& p_slot_name, const String& p_attachment_name) {

	ERR_FAIL_COND_V(skeleton == NULL, Ref<Shape2D>());
	spAttachment *attachment = spSkeleton_getAttachmentForSlotName(skeleton, p_slot_name.utf8().get_data(), p_attachment_name.utf8().get_data());
	ERR_FAIL_COND_V(attachment == NULL, Ref<Shape2D>());
	ERR_FAIL_COND_V(attachment->type != SP_ATTACHMENT_BOUNDING_BOX, Ref<Shape2D>());
	spBoundingBoxAttachment *info = (spBoundingBoxAttachment *)attachment;

	Vector<Vector2> points;
	points.resize(info->verticesCount / 2);
	for (int idx = 0; idx < info->verticesCount / 2; idx++)
		points[idx] = Vector2(info->vertices[idx * 2], -info->vertices[idx * 2 + 1]);

	ConvexPolygonShape2D *shape = memnew(ConvexPolygonShape2D);
	shape->set_points(points);

	return shape;
}

bool Spine::add_bounding_box(const String& p_bone_name, const String& p_slot_name, const String& p_attachment_name, const Variant& p_node, const Vector2& p_ofs, const Vector2& p_scale, const real_t p_rot) {

	ERR_FAIL_COND_V(skeleton == NULL, false);
	Object *obj = p_node;
	ERR_FAIL_COND_V(obj == NULL, false);
	CollisionObject2D *node = obj->cast_to<CollisionObject2D>();
	ERR_FAIL_COND_V(node == NULL, false);
	Ref<Shape2D> shape = get_bounding_box(p_slot_name, p_attachment_name);
	if (shape.is_null())
		return false;
	node->add_shape(shape);

	return add_attachment_node(p_bone_name, p_node);
}

bool Spine::remove_bounding_box(const String& p_bone_name, const Variant& p_node) {

	return remove_attachment_node(p_bone_name, p_node);
}

void Spine::set_animation_process_mode(Spine::AnimationProcessMode p_mode) {

	if (animation_process_mode == p_mode)
		return;

	bool pr = processing;
	if (pr)
		_set_process(false);
	animation_process_mode = p_mode;
	if (pr)
		_set_process(true);
}

Spine::AnimationProcessMode Spine::get_animation_process_mode() const {

	return animation_process_mode;
}

void Spine::set_fx_slot_prefix(const String& p_prefix) {

	fx_slot_prefix = p_prefix.utf8();
	update();
}

String Spine::get_fx_slot_prefix() const {

	String s;
	s.parse_utf8(fx_slot_prefix.get_data());
	return s;
}

void Spine::set_debug_bones(bool p_enable) {

	debug_bones = p_enable;
	update();
}

bool Spine::is_debug_bones() const {

	return debug_bones;
}

void Spine::set_debug_attachment(DebugAttachmentMode p_mode, bool p_enable) {

	switch (p_mode) {

	case DEBUG_ATTACHMENT_REGION:
		debug_attachment_region = p_enable;
		break;
	case DEBUG_ATTACHMENT_MESH:
		debug_attachment_mesh = p_enable;
		break;
	case DEBUG_ATTACHMENT_SKINNED_MESH:
		debug_attachment_skinned_mesh = p_enable;
		break;
	case DEBUG_ATTACHMENT_BOUNDING_BOX:
		debug_attachment_bounding_box = p_enable;
		break;
	};
	update();
}

bool Spine::is_debug_attachment(DebugAttachmentMode p_mode) const {

	switch (p_mode) {

		case DEBUG_ATTACHMENT_REGION:
			return debug_attachment_region;
		case DEBUG_ATTACHMENT_MESH:
			return debug_attachment_mesh;
		case DEBUG_ATTACHMENT_SKINNED_MESH:
			return debug_attachment_skinned_mesh;
		case DEBUG_ATTACHMENT_BOUNDING_BOX:
			return debug_attachment_bounding_box;
	};
	return false;
}

void Spine::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_resource", "spine"), &Spine::set_resource);
	ObjectTypeDB::bind_method(_MD("get_resource"), &Spine::get_resource);

	ObjectTypeDB::bind_method(_MD("has", "name"), &Spine::has);
	ObjectTypeDB::bind_method(_MD("mix", "from", "to", "duration"), &Spine::mix, 0);
	ObjectTypeDB::bind_method(_MD("play", "name", "cunstom_scale", "loop", "track", "delay"), &Spine::play, 1.0f, false, 0, 0);
	ObjectTypeDB::bind_method(_MD("add", "name", "cunstom_scale", "loop", "track", "delay"), &Spine::add, 1.0f, false, 0, 0);
	ObjectTypeDB::bind_method(_MD("clear", "track"), &Spine::clear);
	ObjectTypeDB::bind_method(_MD("stop"), &Spine::stop);
	ObjectTypeDB::bind_method(_MD("is_playing", "track"), &Spine::is_playing);
	ObjectTypeDB::bind_method(_MD("get_current_animation"), &Spine::get_current_animation);
	ObjectTypeDB::bind_method(_MD("stop_all"), &Spine::stop_all);
	ObjectTypeDB::bind_method(_MD("reset"), &Spine::reset);
	ObjectTypeDB::bind_method(_MD("seek", "pos"), &Spine::seek);
	ObjectTypeDB::bind_method(_MD("tell"), &Spine::tell);
	ObjectTypeDB::bind_method(_MD("set_active", "active"), &Spine::set_active);
	ObjectTypeDB::bind_method(_MD("is_active"), &Spine::is_active);
	ObjectTypeDB::bind_method(_MD("set_speed", "speed"), &Spine::set_speed);
	ObjectTypeDB::bind_method(_MD("get_speed"), &Spine::get_speed);
	ObjectTypeDB::bind_method(_MD("set_modulate", "modulate"), &Spine::set_modulate);
	ObjectTypeDB::bind_method(_MD("get_modulate"), &Spine::get_modulate);
	ObjectTypeDB::bind_method(_MD("set_flip_x", "modulate"), &Spine::set_flip_x);
	ObjectTypeDB::bind_method(_MD("is_flip_x"), &Spine::is_flip_x);
	ObjectTypeDB::bind_method(_MD("set_flip_y", "modulate"), &Spine::set_flip_y);
	ObjectTypeDB::bind_method(_MD("is_flip_y"), &Spine::is_flip_y);
	ObjectTypeDB::bind_method(_MD("set_skin", "skin"), &Spine::set_skin);
	ObjectTypeDB::bind_method(_MD("set_animation_process_mode","mode"),&Spine::set_animation_process_mode);
	ObjectTypeDB::bind_method(_MD("get_animation_process_mode"),&Spine::get_animation_process_mode);
	ObjectTypeDB::bind_method(_MD("get_skeleton"), &Spine::get_skeleton);
	ObjectTypeDB::bind_method(_MD("get_attachment", "slot_name", "attachment_name"), &Spine::get_attachment);
	ObjectTypeDB::bind_method(_MD("get_bone", "bone_name"), &Spine::get_bone);
	ObjectTypeDB::bind_method(_MD("get_slot", "slot_name"), &Spine::get_slot);
	ObjectTypeDB::bind_method(_MD("set_attachment", "slot_name", "attachment"), &Spine::set_attachment);
	ObjectTypeDB::bind_method(_MD("has_attachment_node", "bone_name", "node"), &Spine::has_attachment_node);
	ObjectTypeDB::bind_method(_MD("add_attachment_node", "bone_name", "node", "ofs", "scale", "rot"), &Spine::add_attachment_node, Vector2(0, 0), Vector2(1, 1), 0);
	ObjectTypeDB::bind_method(_MD("remove_attachment_node", "p_bone_name", "node"), &Spine::remove_attachment_node);
	ObjectTypeDB::bind_method(_MD("get_bounding_box", "slot_name", "attachment_name"), &Spine::get_bounding_box);
	ObjectTypeDB::bind_method(_MD("add_bounding_box", "bone_name", "slot_name", "attachment_name", "collision_object_2d", "ofs", "scale", "rot"), &Spine::add_bounding_box, Vector2(0, 0), Vector2(1, 1), 0);
	ObjectTypeDB::bind_method(_MD("remove_bounding_box", "bone_name", "collision_object_2d"), &Spine::remove_bounding_box);

	ObjectTypeDB::bind_method(_MD("set_fx_slot_prefix", "prefix"), &Spine::set_fx_slot_prefix);
	ObjectTypeDB::bind_method(_MD("get_fx_slot_prefix"), &Spine::get_fx_slot_prefix);
	
	ObjectTypeDB::bind_method(_MD("set_debug_bones", "enable"), &Spine::set_debug_bones);
	ObjectTypeDB::bind_method(_MD("is_debug_bones"), &Spine::is_debug_bones);
	ObjectTypeDB::bind_method(_MD("set_debug_attachment", "mode", "enable"), &Spine::set_debug_attachment);
	ObjectTypeDB::bind_method(_MD("is_debug_attachment", "mode"), &Spine::is_debug_attachment);

	ObjectTypeDB::bind_method(_MD("_on_fx_draw"), &Spine::_on_fx_draw);

	ADD_PROPERTY( PropertyInfo( Variant::INT, "playback/process_mode", PROPERTY_HINT_ENUM, "Fixed,Idle"), _SCS("set_animation_process_mode"), _SCS("get_animation_process_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "playback/speed", PROPERTY_HINT_RANGE, "-64,64,0.01"), _SCS("set_speed"), _SCS("get_speed"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playback/active"), _SCS("set_active"), _SCS("is_active"));
	//ADD_PROPERTY(PropertyInfo(Variant::REAL, "playback/pos", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), _SCS("seek"), _SCS("tell"));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/bones"), _SCS("set_debug_bones"), _SCS("is_debug_bones"));

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "modulate"), _SCS("set_modulate"), _SCS("get_modulate"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_x"), _SCS("set_flip_x"), _SCS("is_flip_x"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_y"), _SCS("set_flip_y"), _SCS("is_flip_y"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "fx_prefix"), _SCS("set_fx_slot_prefix"), _SCS("get_fx_slot_prefix"));
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "SpineResource"), _SCS("set_resource"), _SCS("get_resource")); //, PROPERTY_USAGE_NOEDITOR));

	ADD_SIGNAL(MethodInfo("animation_start", PropertyInfo(Variant::INT, "track")));
	ADD_SIGNAL(MethodInfo("animation_complete", PropertyInfo(Variant::INT, "track"), PropertyInfo(Variant::INT, "loop_count")));
	ADD_SIGNAL(MethodInfo("animation_event", PropertyInfo(Variant::INT, "track"), PropertyInfo(Variant::DICTIONARY, "event")));
	ADD_SIGNAL(MethodInfo("animation_end", PropertyInfo(Variant::INT, "track")));

	BIND_CONSTANT(ANIMATION_PROCESS_FIXED);
	BIND_CONSTANT(ANIMATION_PROCESS_IDLE);

	BIND_CONSTANT(DEBUG_ATTACHMENT_REGION);
	BIND_CONSTANT(DEBUG_ATTACHMENT_MESH);
	BIND_CONSTANT(DEBUG_ATTACHMENT_SKINNED_MESH);
	BIND_CONSTANT(DEBUG_ATTACHMENT_BOUNDING_BOX);
}

Rect2 Spine::get_item_rect() const {

	if (skeleton == NULL)
		return Node2D::get_item_rect();

	float minX = 65535, minY = 65535, maxX = -65535, maxY = -65535;
	bool attached = false;
	for (int i = 0; i < skeleton->slotsCount; ++i) {

		spSlot* slot = skeleton->slots[i];
		if (!slot->attachment) continue;
		int verticesCount;
		if (slot->attachment->type == SP_ATTACHMENT_REGION) {
			spRegionAttachment* attachment = (spRegionAttachment*)slot->attachment;
			spRegionAttachment_computeWorldVertices(attachment, slot->bone, world_verts.ptr());
			verticesCount = 8;
		}
		else if (slot->attachment->type == SP_ATTACHMENT_MESH) {
			spMeshAttachment* mesh = (spMeshAttachment*)slot->attachment;
			spMeshAttachment_computeWorldVertices(mesh, slot, world_verts.ptr());
			verticesCount = mesh->verticesCount;
		}
		else if (slot->attachment->type == SP_ATTACHMENT_SKINNED_MESH) {
			spSkinnedMeshAttachment* mesh = (spSkinnedMeshAttachment*)slot->attachment;
			spSkinnedMeshAttachment_computeWorldVertices(mesh, slot, world_verts.ptr());
			verticesCount = mesh->uvsCount;
		}
		else
			continue;

		attached = true;

		for (int ii = 0; ii < verticesCount; ii += 2) {
			float x = world_verts[ii] * 1, y = world_verts[ii + 1] * 1;
			minX = MIN(minX, x);
			minY = MIN(minY, y);
			maxX = MAX(maxX, x);
			maxY = MAX(maxY, y);
		}
	}

	int h = maxY - minY;
	return attached ? Rect2(minX, -minY - h, maxX - minX, h) : Node2D::get_item_rect();
}

Spine::Spine()
	: batcher(this)
	, fx_node(memnew(Node2D))
	, fx_batcher(fx_node)
{

	skeleton = NULL;
	root_bone = NULL;
	state = NULL;
	res = RES();
	world_verts.resize(1000); // Max number of vertices per mesh.

	speed_scale = 1;
	autoplay = "";
	animation_process_mode = ANIMATION_PROCESS_IDLE;
	processing = false;
	active = false;
	playing = false;
	forward = true;

	debug_bones = false;
	debug_attachment_region = false;
	debug_attachment_mesh = false;
	debug_attachment_skinned_mesh = false;
	debug_attachment_bounding_box = false;

	skin = "";
	current_animation = "[stop]";
	loop = true;
	fx_slot_prefix = String("fx/").utf8();

	modulate = Color(1, 1, 1, 1);
	flip_x = false;
	flip_y = false;
}

Spine::~Spine() {

	// cleanup
	_spine_dispose();
}

#endif // MODULE_SPINE_ENABLED
