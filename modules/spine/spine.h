/*************************************************************************/
/*  spine.h                                                              */
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
#ifndef SPINE_H
#define SPINE_H

#include "scene/2d/node_2d.h"
#include "scene/resources/shape_2d.h"
#include <spine/spine.h>
#include "spine_batcher.h"

class CollisionObject2D;

class Spine : public Node2D {

	OBJ_TYPE(Spine, Node2D);
public:
	enum AnimationProcessMode {
		ANIMATION_PROCESS_FIXED,
		ANIMATION_PROCESS_IDLE,
	};
	enum DebugAttachmentMode {
		DEBUG_ATTACHMENT_REGION,
		DEBUG_ATTACHMENT_MESH,
		DEBUG_ATTACHMENT_SKINNED_MESH,
		DEBUG_ATTACHMENT_BOUNDING_BOX,
	};

	class SpineResource : public Resource {

		OBJ_TYPE(SpineResource, Resource);

	public:
		SpineResource();
		~SpineResource();

		spAtlas *atlas;
		spSkeletonData *data;
	};

private:


	Ref<SpineResource> res;

	spSkeleton* skeleton;
	spBone* root_bone;
	spAnimationState* state;
	mutable Vector<float> world_verts;

	float speed_scale;
	String autoplay;
	AnimationProcessMode animation_process_mode;
	bool processing;
	bool active;
	bool playing;
	bool forward;
	bool debug_bones;
	bool debug_attachment_region;
	bool debug_attachment_mesh;
	bool debug_attachment_skinned_mesh;
	bool debug_attachment_bounding_box;
	String current_animation;
	bool loop;
	String skin;

	Color modulate;
	bool flip_x, flip_y;
	SpineBatcher batcher;

	// fx slots (always show on top)
	Node2D *fx_node;
	SpineBatcher fx_batcher;
	CharString fx_slot_prefix;

	float current_pos;

	typedef struct AttachmentNode {
		List<AttachmentNode>::Element *E;
		spBone *bone;
		WeakRef *ref;
		Vector2 ofs;
		Vector2 scale;
		real_t rot;
	} AttachmentNode;
	typedef List<AttachmentNode> AttachmentNodes;
	AttachmentNodes attachment_nodes;

	static void spine_animation_callback(spAnimationState* p_state, int p_track, spEventType p_type, spEvent* p_event, int loop_count);
	void _on_animation_state_event(int p_track, spEventType p_type, spEvent *p_event, int p_loop_count);

	void _spine_dispose();
	void _animation_process(float p_delta);
	void _animation_draw();
	void _set_process(bool p_process, bool p_force = false);
	void _on_fx_draw();

protected:
	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

public:

	// set/get spine resource
	void set_resource(Ref<SpineResource> p_data);
	Ref<SpineResource> get_resource();

	bool has(const String& p_name);
	void mix(const String& p_from, const String& p_to, real_t p_duration);

	bool play(const String& p_name, real_t p_cunstom_scale = 1.0f, bool p_loop = false, int p_track = 0, int p_delay = 0);
	bool add(const String& p_name, real_t p_cunstom_scale = 1.0f, bool p_loop = false, int p_track = 0, int p_delay = 0);
	void clear(int p_track = -1);
	void stop();
	bool is_playing(int p_track = 0) const;
	void set_forward(bool p_forward = true);
	bool is_forward() const;
	String get_current_animation(int p_track = 0) const;
	void stop_all();
	void reset();
	void seek(float p_pos);
	float tell() const;

	void set_active(bool p_active);
	bool is_active() const;

	void set_speed(float p_speed);
	float get_speed() const;

	void set_autoplay(const String& p_name);
	String get_autoplay() const;

	void set_modulate(const Color& p_color);
	Color get_modulate() const;

	void set_flip_x(bool p_flip);
	void set_flip_y(bool p_flip);
	bool is_flip_x() const;
	bool is_flip_y() const;

	void set_animation_process_mode(AnimationProcessMode p_mode);
	AnimationProcessMode get_animation_process_mode() const;

	/* Sets the skin used to look up attachments not found in the SkeletonData defaultSkin. Attachments from the new skin are
	* attached if the corresponding attachment from the old skin was attached. If there was no old skin, each slot's setup mode
	* attachment is attached from the new skin. Returns false if the skin was not found.
	* @param skin May be 0.*/
	bool set_skin(const String& p_name);

	//spAttachment* get_attachment(const char* slotName, const char* attachmentName) const;
	Dictionary get_skeleton() const;
	/* Returns null if the slot or attachment was not found. */
	Dictionary get_attachment(const String& p_slot_name, const String& p_attachment_name) const;
	/* Returns null if the bone was not found. */
	Dictionary get_bone(const String& p_bone_name) const;
	/* Returns null if the slot was not found. */
	Dictionary get_slot(const String& p_slot_name) const;
	/* Returns false if the slot or attachment was not found. */
	bool set_attachment(const String& p_slot_name, const Variant& p_attachment);
	// bind node to bone, auto update pos/rotate/scale
	bool has_attachment_node(const String& p_bone_name, const Variant& p_node);
	bool add_attachment_node(const String& p_bone_name, const Variant& p_node, const Vector2& p_ofs = Vector2(0, 0), const Vector2& p_scale = Vector2(1, 1), const real_t p_rot = 0);
	bool remove_attachment_node(const String& p_bone_name, const Variant& p_node);
	// get spine bounding box
	Ref<Shape2D> get_bounding_box(const String& p_slot_name, const String& p_attachment_name);
	// bind collision object 2d to spine bounding box
	bool add_bounding_box(const String& p_bone_name, const String& p_slot_name, const String& p_attachment_name, const Variant& p_node, const Vector2& p_ofs = Vector2(0, 0), const Vector2& p_scale = Vector2(1, 1), const real_t p_rot = 0);
	bool remove_bounding_box(const String& p_bone_name, const Variant& p_node);

	void set_fx_slot_prefix(const String& p_prefix);
	String get_fx_slot_prefix() const;

	void set_debug_bones(bool p_enable);
	bool is_debug_bones() const;
	void set_debug_attachment(DebugAttachmentMode p_mode, bool p_enable);
	bool is_debug_attachment(DebugAttachmentMode p_mode) const;

	//void seek(float p_time, bool p_update = false);
	//void seek_delta(float p_time, float p_delta);
	//float get_current_animation_pos() const;
	//float get_current_animation_length() const;

	//void advance(float p_time);

	virtual Rect2 get_item_rect() const;
	
	Spine();
	virtual ~Spine();
};

VARIANT_ENUM_CAST(Spine::AnimationProcessMode);
VARIANT_ENUM_CAST(Spine::DebugAttachmentMode);

#endif // SPINE_H
#endif // MODULE_SPINE_ENABLED

