#include "animation_blend_space1d.h"

void AnimationNodeBlendSpace1D::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("blend_point_")) {
		String left = property.name.get_slicec('/', 0);
		int idx = left.get_slicec('_', 1).to_int();
		if (idx >= blend_points_used) {
			property.usage = 0;
		}
	}
	AnimationRootNode::_validate_property(property);
}

void AnimationNodeBlendSpace1D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_blend_point", "node", "pos", "at_index"), &AnimationNodeBlendSpace1D::add_blend_point, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("set_blend_point_position", "point", "pos"), &AnimationNodeBlendSpace1D::set_blend_point_position);
	ClassDB::bind_method(D_METHOD("get_blend_point_position", "point"), &AnimationNodeBlendSpace1D::get_blend_point_position);
	ClassDB::bind_method(D_METHOD("set_blend_point_node", "point", "node"), &AnimationNodeBlendSpace1D::set_blend_point_node);
	ClassDB::bind_method(D_METHOD("get_blend_point_node", "point"), &AnimationNodeBlendSpace1D::get_blend_point_node);
	ClassDB::bind_method(D_METHOD("remove_blend_point", "point"), &AnimationNodeBlendSpace1D::remove_blend_point);
	ClassDB::bind_method(D_METHOD("get_blend_point_count"), &AnimationNodeBlendSpace1D::get_blend_point_count);

	ClassDB::bind_method(D_METHOD("set_min_space", "min_space"), &AnimationNodeBlendSpace1D::set_min_space);
	ClassDB::bind_method(D_METHOD("get_min_space"), &AnimationNodeBlendSpace1D::get_min_space);

	ClassDB::bind_method(D_METHOD("set_max_space", "max_space"), &AnimationNodeBlendSpace1D::set_max_space);
	ClassDB::bind_method(D_METHOD("get_max_space"), &AnimationNodeBlendSpace1D::get_max_space);

	ClassDB::bind_method(D_METHOD("set_snap", "snap"), &AnimationNodeBlendSpace1D::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &AnimationNodeBlendSpace1D::get_snap);

	ClassDB::bind_method(D_METHOD("set_blend_pos", "pos"), &AnimationNodeBlendSpace1D::set_blend_pos);
	ClassDB::bind_method(D_METHOD("get_blend_pos"), &AnimationNodeBlendSpace1D::get_blend_pos);

	ClassDB::bind_method(D_METHOD("set_value_label", "text"), &AnimationNodeBlendSpace1D::set_value_label);
	ClassDB::bind_method(D_METHOD("get_value_label"), &AnimationNodeBlendSpace1D::get_value_label);

	ClassDB::bind_method(D_METHOD("_add_blend_point", "index", "node"), &AnimationNodeBlendSpace1D::_add_blend_point);

	for (int i = 0; i < MAX_BLEND_POINTS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "blend_point_" + itos(i) + "/node", PROPERTY_HINT_RESOURCE_TYPE, "AnimationRootNode", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE), "_add_blend_point", "get_blend_point_node", i);
		ADD_PROPERTYI(PropertyInfo(Variant::REAL, "blend_point_" + itos(i) + "/pos", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_blend_point_position", "get_blend_point_position", i);
	}

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "min_space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_min_space", "get_min_space");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_max_space", "get_max_space");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "snap", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_snap", "get_snap");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "blend_pos", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_blend_pos", "get_blend_pos");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "value_label", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_value_label", "get_value_label");
}

void AnimationNodeBlendSpace1D::add_blend_point(const Ref<AnimationRootNode> &p_node, float p_position, int p_at_index) {
	ERR_FAIL_COND(blend_points_used >= MAX_BLEND_POINTS);
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_node->get_parent().is_valid());
	ERR_FAIL_COND(p_at_index < -1 || p_at_index > blend_points_used);

	if (p_at_index == -1 || p_at_index == blend_points_used) {
		p_at_index = blend_points_used;
	} else {
		for (int i = blend_points_used; i > p_at_index; i++) {
			blend_points[i] = blend_points[i - 1];
		}
	}

	blend_points[p_at_index].node = p_node;
	blend_points[p_at_index].position = p_position;

	blend_points[p_at_index].node->set_parent(this);
	blend_points[p_at_index].node->set_graph_player(get_graph_player());

	blend_points_used++;
}

void AnimationNodeBlendSpace1D::set_blend_point_position(int p_point, float p_position) {
	ERR_FAIL_INDEX(p_point, blend_points_used);

	blend_points[p_point].position = p_position;
}

void AnimationNodeBlendSpace1D::set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node) {
	ERR_FAIL_INDEX(p_point, blend_points_used);
	ERR_FAIL_COND(p_node.is_null());

	if (blend_points[p_point].node.is_valid()) {
		blend_points[p_point].node->set_parent(NULL);
		blend_points[p_point].node->set_graph_player(NULL);
	}

	blend_points[p_point].node = p_node;
	blend_points[p_point].node->set_parent(this);
	blend_points[p_point].node->set_graph_player(get_graph_player());
}

float AnimationNodeBlendSpace1D::get_blend_point_position(int p_point) const {
	ERR_FAIL_INDEX_V(p_point, blend_points_used, 0);
	return blend_points[p_point].position;
}

Ref<AnimationRootNode> AnimationNodeBlendSpace1D::get_blend_point_node(int p_point) const {
	ERR_FAIL_INDEX_V(p_point, blend_points_used, Ref<AnimationRootNode>());
	return blend_points[p_point].node;
}

void AnimationNodeBlendSpace1D::remove_blend_point(int p_point) {
	ERR_FAIL_INDEX(p_point, blend_points_used);

	blend_points[p_point].node->set_parent(NULL);
	blend_points[p_point].node->set_graph_player(NULL);

	for (int i = p_point; i < blend_points_used - 1; i++) {
		blend_points[i] = blend_points[i + 1];
	}

	blend_points_used--;
}

int AnimationNodeBlendSpace1D::get_blend_point_count() const {

	return blend_points_used;
}

void AnimationNodeBlendSpace1D::set_min_space(float p_min) {
	min_space = p_min;

	if (min_space >= max_space) {
		min_space = max_space - 1;
	}
}

float AnimationNodeBlendSpace1D::get_min_space() const {
	return min_space;
}

void AnimationNodeBlendSpace1D::set_max_space(float p_max) {
	max_space = p_max;

	if (max_space <= min_space) {
		max_space = min_space + 1;
	}
}

float AnimationNodeBlendSpace1D::get_max_space() const {
	return max_space;
}

void AnimationNodeBlendSpace1D::set_snap(float p_snap) {
	snap = p_snap;
}

float AnimationNodeBlendSpace1D::get_snap() const {
	return snap;
}

void AnimationNodeBlendSpace1D::set_blend_pos(float p_pos) {
	blend_pos = p_pos;
}

float AnimationNodeBlendSpace1D::get_blend_pos() const {
	return blend_pos;
}

void AnimationNodeBlendSpace1D::set_value_label(const String &p_label) {
	value_label = p_label;
}

String AnimationNodeBlendSpace1D::get_value_label() const {
	return value_label;
}

void AnimationNodeBlendSpace1D::_add_blend_point(int p_index, const Ref<AnimationRootNode> &p_node) {
	if (p_index == blend_points_used) {
		add_blend_point(p_node, 0);
	} else {
		set_blend_point_node(p_index, p_node);
	}
}

float AnimationNodeBlendSpace1D::process(float p_time, bool p_seek) {

	if (blend_points_used == 0) {
		return 0.0;
	}

	if (blend_points_used == 1) {
		// only one point available, just play that animation
		return blend_node(blend_points[0].node, p_time, p_seek, 1.0, FILTER_IGNORE, false);
	}

	int point_lower = -1;
	float pos_lower = 0.0;
	int point_higher = -1;
	float pos_higher = 0.0;

	// find the closest two points to blend between
	for (int i = 0; i < blend_points_used; i++) {

		float pos = blend_points[i].position;

		if (pos <= blend_pos) {
			if (point_lower == -1) {
				point_lower = i;
				pos_lower = pos;
			} else if ((blend_pos - pos) < (blend_pos - pos_lower)) {
				point_lower = i;
				pos_lower = pos;
			}
		} else {
			if (point_higher == -1) {
				point_higher = i;
				pos_higher = pos;
			} else if ((pos - blend_pos) < (pos_higher - blend_pos)) {
				point_higher = i;
				pos_higher = pos;
			}
		}
	}

	if (point_lower == -1) {
		// we are on the left side, no other point to the left
		// we just play the next point.

		return blend_node(blend_points[point_higher].node, p_time, p_seek, 1.0, FILTER_IGNORE, false);
	} else if (point_higher == -1) {
		// we are on the right side, no other point to the right
		// we just play the previous point
		return blend_node(blend_points[point_lower].node, p_time, p_seek, 1.0, FILTER_IGNORE, false);
	} else {

		//w we are between two points.
		// figure out weights, then blend the animations

		float distance_between_points = pos_higher - pos_lower;

		float current_pos_inbetween = blend_pos - pos_lower;

		float blend_percentage = current_pos_inbetween / distance_between_points;

		float blend_lower = 1.0 - blend_percentage;
		float blend_higher = blend_percentage;

		float time_remaining_lower = 0.0;
		float time_remaining_higher = 0.0;

		time_remaining_lower = blend_node(blend_points[point_lower].node, p_time, p_seek, blend_lower, FILTER_IGNORE, false);
		time_remaining_higher = blend_node(blend_points[point_higher].node, p_time, p_seek, blend_higher, FILTER_IGNORE, false);

		return MAX(time_remaining_lower, time_remaining_higher);
	}
}

String AnimationNodeBlendSpace1D::get_caption() const {
	return "BlendSpace1D";
}

AnimationNodeBlendSpace1D::AnimationNodeBlendSpace1D() {

	blend_points_used = 0;
	max_space = 1;
	min_space = -1;

	snap = 0.1;
	value_label = "value";
}

AnimationNodeBlendSpace1D::~AnimationNodeBlendSpace1D() {

	for (int i = 0; i < blend_points_used; i++) {
		blend_points[i].node->set_parent(this);
		blend_points[i].node->set_graph_player(get_graph_player());
	}
}
