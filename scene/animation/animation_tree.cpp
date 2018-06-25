#include "animation_tree.h"
#include "animation_blend_tree.h"
#include "core/method_bind_ext.gen.inc"
#include "engine.h"
#include "scene/scene_string_names.h"
#include "servers/audio/audio_stream.h"

void AnimationNode::blend_animation(const StringName &p_animation, float p_time, float p_delta, bool p_seeked, float p_blend) {

	ERR_FAIL_COND(!state);
	ERR_FAIL_COND(!state->player->has_animation(p_animation));

	Ref<Animation> animation = state->player->get_animation(p_animation);

	if (animation.is_null()) {

		Ref<AnimationNodeBlendTree> btree = get_parent();
		if (btree.is_valid()) {
			String name = btree->get_node_name(Ref<AnimationNodeAnimation>(this));
			make_invalid(vformat(RTR("In node '%s', invalid animation: '%s'."), name, p_animation));
		} else {
			make_invalid(vformat(RTR("Invalid animation: '%s'."), p_animation));
		}
		return;
	}

	ERR_FAIL_COND(!animation.is_valid());

	AnimationState anim_state;
	anim_state.blend = p_blend;
	anim_state.track_blends = &blends;
	anim_state.delta = p_delta;
	anim_state.time = p_time;
	anim_state.animation = animation;
	anim_state.seeked = p_seeked;

	state->animation_states.push_back(anim_state);
}

float AnimationNode::_pre_process(State *p_state, float p_time, bool p_seek) {
	state = p_state;
	float t = process(p_time, p_seek);
	state = NULL;
	return t;
}

void AnimationNode::make_invalid(const String &p_reason) {
	ERR_FAIL_COND(!state);
	state->valid = false;
	if (state->invalid_reasons != String()) {
		state->invalid_reasons += "\n";
	}
	state->invalid_reasons += "- " + p_reason;
}

float AnimationNode::blend_input(int p_input, float p_time, bool p_seek, float p_blend, FilterAction p_filter, bool p_optimize) {
	ERR_FAIL_INDEX_V(p_input, inputs.size(), 0);
	ERR_FAIL_COND_V(!state, 0);
	ERR_FAIL_COND_V(!get_tree(), 0); //should not happen, but used to catch bugs

	Ref<AnimationNodeBlendTree> tree = get_parent();

	if (!tree.is_valid() && get_tree()->get_graph_root().ptr() != this) {
		make_invalid(RTR("Can't blend input because node is not in a tree"));
		return 0;
	}

	ERR_FAIL_COND_V(!tree.is_valid(), 0); //should not happen

	StringName anim_name = inputs[p_input].connected_to;

	Ref<AnimationNode> node = tree->get_node(anim_name);

	if (node.is_null()) {

		String name = tree->get_node_name(Ref<AnimationNodeAnimation>(this));
		make_invalid(vformat(RTR("Nothing connected to input '%s' of node '%s'."), get_input_name(p_input), name));
		return 0;
	}

	inputs[p_input].last_pass = state->last_pass;

	return _blend_node(node, p_time, p_seek, p_blend, p_filter, p_optimize, &inputs[p_input].activity);
}

float AnimationNode::blend_node(Ref<AnimationNode> p_node, float p_time, bool p_seek, float p_blend, FilterAction p_filter, bool p_optimize) {

	return _blend_node(p_node, p_time, p_seek, p_blend, p_filter, p_optimize);
}

float AnimationNode::_blend_node(Ref<AnimationNode> p_node, float p_time, bool p_seek, float p_blend, FilterAction p_filter, bool p_optimize, float *r_max) {

	ERR_FAIL_COND_V(!p_node.is_valid(), 0);
	ERR_FAIL_COND_V(!state, 0);

	int blend_count = blends.size();

	if (p_node->blends.size() != blend_count) {
		p_node->blends.resize(blend_count);
	}

	float *blendw = p_node->blends.ptrw();
	const float *blendr = blends.ptr();

	bool any_valid = false;

	if (has_filter() && is_filter_enabled() && p_filter != FILTER_IGNORE) {

		for (int i = 0; i < blend_count; i++) {
			blendw[i] = 0.0; //all to zero by default
		}

		const NodePath *K = NULL;
		while ((K = filter.next(K))) {
			if (!state->track_map.has(*K)) {
				continue;
			}
			int idx = state->track_map[*K];
			blendw[idx] = 1.0; //filtered goes to one
		}

		switch (p_filter) {
			case FILTER_IGNORE:
				break; //will not happen anyway
			case FILTER_PASS: {
				//values filtered pass, the rest dont
				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] == 0) //not filtered, does not pass
						continue;

					blendw[i] = blendr[i] * p_blend;
					if (blendw[i] > CMP_EPSILON) {
						any_valid = true;
					}
				}

			} break;
			case FILTER_STOP: {

				//values filtered dont pass, the rest are blended

				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] > 0) //filtered, does not pass
						continue;

					blendw[i] = blendr[i] * p_blend;
					if (blendw[i] > CMP_EPSILON) {
						any_valid = true;
					}
				}

			} break;
			case FILTER_BLEND: {

				//filtered values are blended, the rest are passed without blending

				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] == 1.0) {
						blendw[i] = blendr[i] * p_blend; //filtered, blend
					} else {
						blendw[i] = blendr[i]; //not filtered, do not blend
					}

					if (blendw[i] > CMP_EPSILON) {
						any_valid = true;
					}
				}

			} break;
		}
	} else {
		for (int i = 0; i < blend_count; i++) {

			//regular blend
			blendw[i] = blendr[i] * p_blend;
			if (blendw[i] > CMP_EPSILON) {
				any_valid = true;
			}
		}
	}

	if (r_max) {
		*r_max = 0;
		for (int i = 0; i < blend_count; i++) {
			*r_max = MAX(*r_max, blendw[i]);
		}
	}

	if (!p_seek && p_optimize && !any_valid) //pointless to go on, all are zero
		return 0;

	return p_node->_pre_process(state, p_time, p_seek);
}

int AnimationNode::get_input_count() const {

	return inputs.size();
}
String AnimationNode::get_input_name(int p_input) {
	ERR_FAIL_INDEX_V(p_input, inputs.size(), String());
	return inputs[p_input].name;
}

float AnimationNode::get_input_activity(int p_input) const {

	ERR_FAIL_INDEX_V(p_input, inputs.size(), 0);
	if (!get_tree())
		return 0;

	if (get_tree()->get_last_process_pass() != inputs[p_input].last_pass) {
		return 0;
	}
	return inputs[p_input].activity;
}
StringName AnimationNode::get_input_connection(int p_input) {

	ERR_FAIL_INDEX_V(p_input, inputs.size(), StringName());
	return inputs[p_input].connected_to;
}

void AnimationNode::set_input_connection(int p_input, const StringName &p_connection) {

	ERR_FAIL_INDEX(p_input, inputs.size());
	inputs[p_input].connected_to = p_connection;
}

String AnimationNode::get_caption() const {
	return "Node";
}

void AnimationNode::add_input(const String &p_name) {
	//root nodes cant add inputs
	ERR_FAIL_COND(Object::cast_to<AnimationRootNode>(this) != NULL)
	Input input;
	ERR_FAIL_COND(p_name.find(".") != -1 || p_name.find("/") != -1);
	input.name = p_name;
	input.activity = 0;
	input.last_pass = 0;
	inputs.push_back(input);
	emit_changed();
}

void AnimationNode::set_input_name(int p_input, const String &p_name) {
	ERR_FAIL_INDEX(p_input, inputs.size());
	ERR_FAIL_COND(p_name.find(".") != -1 || p_name.find("/") != -1);
	inputs[p_input].name = p_name;
	emit_changed();
}

void AnimationNode::remove_input(int p_index) {
	ERR_FAIL_INDEX(p_index, inputs.size());
	inputs.remove(p_index);
	emit_changed();
}

void AnimationNode::set_parent(AnimationNode *p_parent) {
	parent = p_parent; //do not use ref because parent contains children
}

Ref<AnimationNode> AnimationNode::get_parent() const {
	if (parent) {
		return Ref<AnimationNode>(parent);
	}

	return Ref<AnimationNode>();
}

AnimationTree *AnimationNode::get_tree() const {

	return player;
}

AnimationPlayer *AnimationNode::get_player() const {
	ERR_FAIL_COND_V(!state, NULL);
	return state->player;
}

float AnimationNode::process(float p_time, bool p_seek) {

	if (get_script_instance()) {
		return get_script_instance()->call("process", p_time, p_seek);
	}

	return 0;
}

void AnimationNode::set_filter_path(const NodePath &p_path, bool p_enable) {
	if (p_enable) {
		filter[p_path] = true;
	} else {
		filter.erase(p_path);
	}
}

void AnimationNode::set_filter_enabled(bool p_enable) {
	filter_enabled = p_enable;
}

bool AnimationNode::is_filter_enabled() const {
	return filter_enabled;
}

bool AnimationNode::is_path_filtered(const NodePath &p_path) const {
	return filter.has(p_path);
}

bool AnimationNode::has_filter() const {
	return false;
}

void AnimationNode::set_position(const Vector2 &p_position) {
	position = p_position;
}

Vector2 AnimationNode::get_position() const {
	return position;
}

void AnimationNode::set_tree(AnimationTree *p_player) {

	if (player != NULL && p_player == NULL) {
		emit_signal("removed_from_graph");
	}
	player = p_player;
}

Array AnimationNode::_get_filters() const {

	Array paths;

	const NodePath *K = NULL;
	while ((K = filter.next(K))) {
		paths.push_back(String(*K)); //use strings, so sorting is possible
	}
	paths.sort(); //done so every time the scene is saved, it does not change

	return paths;
}
void AnimationNode::_set_filters(const Array &p_filters) {
	filter.clear();
	for (int i = 0; i < p_filters.size(); i++) {
		set_filter_path(p_filters[i], true);
	}
}

void AnimationNode::_validate_property(PropertyInfo &property) const {
	if (!has_filter() && (property.name == "filter_enabled" || property.name == "filters")) {
		property.usage = 0;
	}
}

void AnimationNode::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_input_count"), &AnimationNode::get_input_count);
	ClassDB::bind_method(D_METHOD("get_input_name", "input"), &AnimationNode::get_input_name);
	ClassDB::bind_method(D_METHOD("get_input_connection", "input"), &AnimationNode::get_input_connection);
	ClassDB::bind_method(D_METHOD("get_input_activity", "input"), &AnimationNode::get_input_activity);

	ClassDB::bind_method(D_METHOD("add_input", "name"), &AnimationNode::add_input);
	ClassDB::bind_method(D_METHOD("remove_input", "index"), &AnimationNode::remove_input);

	ClassDB::bind_method(D_METHOD("set_filter_path", "path", "enable"), &AnimationNode::set_filter_path);
	ClassDB::bind_method(D_METHOD("is_path_filtered", "path"), &AnimationNode::is_path_filtered);

	ClassDB::bind_method(D_METHOD("set_filter_enabled", "enable"), &AnimationNode::set_filter_enabled);
	ClassDB::bind_method(D_METHOD("is_filter_enabled"), &AnimationNode::is_filter_enabled);

	ClassDB::bind_method(D_METHOD("set_position", "position"), &AnimationNode::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &AnimationNode::get_position);

	ClassDB::bind_method(D_METHOD("_set_filters", "filters"), &AnimationNode::_set_filters);
	ClassDB::bind_method(D_METHOD("_get_filters"), &AnimationNode::_get_filters);

	ClassDB::bind_method(D_METHOD("blend_animation", "animation", "time", "delta", "seeked", "blend"), &AnimationNode::blend_animation);
	ClassDB::bind_method(D_METHOD("blend_node", "node", "time", "seek", "blend", "filter", "optimize"), &AnimationNode::blend_node, DEFVAL(FILTER_IGNORE), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("blend_input", "input_index", "time", "seek", "blend", "filter", "optimize"), &AnimationNode::blend_input, DEFVAL(FILTER_IGNORE), DEFVAL(true));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_filter_enabled", "is_filter_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "filters", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_filters", "_get_filters");

	BIND_VMETHOD(MethodInfo("process", PropertyInfo(Variant::REAL, "time"), PropertyInfo(Variant::BOOL, "seek")));

	ADD_SIGNAL(MethodInfo("removed_from_graph"));
	BIND_ENUM_CONSTANT(FILTER_IGNORE);
	BIND_ENUM_CONSTANT(FILTER_PASS);
	BIND_ENUM_CONSTANT(FILTER_STOP);
	BIND_ENUM_CONSTANT(FILTER_BLEND);
}

AnimationNode::AnimationNode() {

	state = NULL;
	parent = NULL;
	player = NULL;
	set_local_to_scene(true);
	filter_enabled = false;
}

////////////////////

void AnimationTree::set_graph_root(const Ref<AnimationNode> &p_root) {

	if (root.is_valid()) {
		root->set_tree(NULL);
	}
	if (p_root.is_valid()) {
		ERR_EXPLAIN("root node already set to another player");
		ERR_FAIL_COND(p_root->player);
	}
	root = p_root;

	if (root.is_valid()) {
		root->set_tree(this);
	}

	update_configuration_warning();
}

Ref<AnimationNode> AnimationTree::get_graph_root() const {
	return root;
}

void AnimationTree::set_active(bool p_active) {

	if (active == p_active)
		return;

	active = p_active;
	started = active;

	if (process_mode == ANIMATION_PROCESS_IDLE) {
		set_process_internal(active);
	} else {

		set_physics_process_internal(active);
	}

	if (!active && is_inside_tree()) {
		for (Set<TrackCache *>::Element *E = playing_caches.front(); E; E = E->next()) {

			if (ObjectDB::get_instance(E->get()->object_id)) {
				E->get()->object->call("stop");
			}
		}

		playing_caches.clear();
	}
}

bool AnimationTree::is_active() const {

	return active;
}

void AnimationTree::set_process_mode(AnimationProcessMode p_mode) {

	if (process_mode == p_mode)
		return;

	bool was_active = is_active();
	if (was_active) {
		set_active(false);
	}

	process_mode = p_mode;

	if (was_active) {
		set_active(true);
	}
}

AnimationTree::AnimationProcessMode AnimationTree::get_process_mode() const {
	return process_mode;
}

void AnimationTree::_node_removed(Node *p_node) {
	cache_valid = false;
}

bool AnimationTree::_update_caches(AnimationPlayer *player) {

	setup_pass++;

	if (!player->has_node(player->get_root())) {
		ERR_PRINT("AnimationTree: AnimationPlayer root is invalid.");
		set_active(false);
		return false;
	}
	Node *parent = player->get_node(player->get_root());

	List<StringName> sname;
	player->get_animation_list(&sname);

	for (List<StringName>::Element *E = sname.front(); E; E = E->next()) {
		Ref<Animation> anim = player->get_animation(E->get());
		for (int i = 0; i < anim->get_track_count(); i++) {
			NodePath path = anim->track_get_path(i);
			Animation::TrackType track_type = anim->track_get_type(i);

			TrackCache *track = NULL;
			if (track_cache.has(path)) {
				track = track_cache.get(path);
			}

			//if not valid, delete track
			if (track && (track->type != track_type || ObjectDB::get_instance(track->object_id) == NULL)) {
				playing_caches.erase(track);
				memdelete(track);
				track_cache.erase(path);
				track = NULL;
			}

			if (!track) {

				RES resource;
				Vector<StringName> leftover_path;
				Node *child = parent->get_node_and_resource(path, resource, leftover_path);

				if (!child) {
					ERR_PRINTS("AnimationTree: '" + String(E->get()) + "', couldn't resolve track:  '" + String(path) + "'");
					continue;
				}

				if (!child->is_connected("tree_exited", this, "_node_removed")) {
					child->connect("tree_exited", this, "_node_removed", varray(child));
				}

				switch (track_type) {
					case Animation::TYPE_VALUE: {

						TrackCacheValue *track_value = memnew(TrackCacheValue);

						if (resource.is_valid()) {
							track_value->object = resource.ptr();
						} else {
							track_value->object = child;
						}

						track_value->subpath = leftover_path;
						track_value->object_id = track_value->object->get_instance_id();

						track = track_value;

					} break;
					case Animation::TYPE_TRANSFORM: {

						Spatial *spatial = Object::cast_to<Spatial>(child);

						if (!spatial) {
							ERR_PRINTS("AnimationTree: '" + String(E->get()) + "', transform track does not point to spatial:  '" + String(path) + "'");
							continue;
						}

						TrackCacheTransform *track_xform = memnew(TrackCacheTransform);

						track_xform->spatial = spatial;
						track_xform->skeleton = NULL;
						track_xform->bone_idx = -1;

						if (path.get_subname_count() == 1 && Object::cast_to<Skeleton>(spatial)) {

							Skeleton *sk = Object::cast_to<Skeleton>(spatial);
							int bone_idx = sk->find_bone(path.get_subname(0));
							if (bone_idx != -1 && !sk->is_bone_ignore_animation(bone_idx)) {

								track_xform->skeleton = sk;
								track_xform->bone_idx = bone_idx;
							}
						}

						track_xform->object = spatial;
						track_xform->object_id = track_xform->object->get_instance_id();

						track = track_xform;

					} break;
					case Animation::TYPE_METHOD: {

						TrackCacheMethod *track_method = memnew(TrackCacheMethod);

						if (resource.is_valid()) {
							track_method->object = resource.ptr();
						} else {
							track_method->object = child;
						}

						track_method->object_id = track_method->object->get_instance_id();

						track = track_method;

					} break;
					case Animation::TYPE_BEZIER: {

						TrackCacheBezier *track_bezier = memnew(TrackCacheBezier);

						if (resource.is_valid()) {
							track_bezier->object = resource.ptr();
						} else {
							track_bezier->object = child;
						}

						track_bezier->subpath = leftover_path;
						track_bezier->object_id = track_bezier->object->get_instance_id();

						track = track_bezier;
					} break;
					case Animation::TYPE_AUDIO: {

						TrackCacheAudio *track_audio = memnew(TrackCacheAudio);

						track_audio->object = child;
						track_audio->object_id = track_audio->object->get_instance_id();

						track = track_audio;

					} break;
					case Animation::TYPE_ANIMATION: {

						TrackCacheAnimation *track_animation = memnew(TrackCacheAnimation);

						track_animation->object = child;
						track_animation->object_id = track_animation->object->get_instance_id();

						track = track_animation;

					} break;
				}

				track_cache[path] = track;
			}

			track->setup_pass = setup_pass;
		}
	}

	List<NodePath> to_delete;

	const NodePath *K = NULL;
	while ((K = track_cache.next(K))) {
		TrackCache *tc = track_cache[*K];
		if (tc->setup_pass != setup_pass) {
			to_delete.push_back(*K);
		}
	}

	while (to_delete.front()) {
		NodePath np = to_delete.front()->get();
		memdelete(track_cache[np]);
		track_cache.erase(np);
		to_delete.pop_front();
	}

	state.track_map.clear();

	K = NULL;
	int idx = 0;
	while ((K = track_cache.next(K))) {
		state.track_map[*K] = idx;
		idx++;
	}

	state.track_count = idx;

	cache_valid = true;

	return true;
}

void AnimationTree::_clear_caches() {

	const NodePath *K = NULL;
	while ((K = track_cache.next(K))) {
		memdelete(track_cache[*K]);
	}
	playing_caches.clear();

	track_cache.clear();
	cache_valid = false;
}

void AnimationTree::_process_graph(float p_delta) {

	//check all tracks, see if they need modification

	if (!root.is_valid()) {
		ERR_PRINT("AnimationTree: root AnimationNode is not set, disabling playback.");
		set_active(false);
		cache_valid = false;
		return;
	}

	if (!has_node(animation_player)) {
		ERR_PRINT("AnimationTree: no valid AnimationPlayer path set, disabling playback");
		set_active(false);
		cache_valid = false;
		return;
	}

	AnimationPlayer *player = Object::cast_to<AnimationPlayer>(get_node(animation_player));

	if (!player) {
		ERR_PRINT("AnimationTree: path points to a node not an AnimationPlayer, disabling playback");
		set_active(false);
		cache_valid = false;
		return;
	}

	if (!cache_valid) {
		if (!_update_caches(player)) {
			return;
		}
	}

	{ //setup

		process_pass++;

		state.valid = true;
		state.invalid_reasons = "";
		state.animation_states.clear(); //will need to be re-created
		state.valid = true;
		state.player = player;
		state.last_pass = process_pass;

		// root source blends

		root->blends.resize(state.track_count);
		float *src_blendsw = root->blends.ptrw();
		for (int i = 0; i < state.track_count; i++) {
			src_blendsw[i] = 1.0; //by default all go to 1 for the root input
		}
	}

	//process

	{

		if (started) {
			//if started, seek
			root->_pre_process(&state, 0, true);
			started = false;
		}

		root->_pre_process(&state, p_delta, false);
	}

	if (!state.valid) {
		return; //state is not valid. do nothing.
	}
	//apply value/transform/bezier blends to track caches and execute method/audio/animation tracks

	{

		bool can_call = is_inside_tree() && !Engine::get_singleton()->is_editor_hint();

		for (List<AnimationNode::AnimationState>::Element *E = state.animation_states.front(); E; E = E->next()) {

			const AnimationNode::AnimationState &as = E->get();

			Ref<Animation> a = as.animation;
			float time = as.time;
			float delta = as.delta;
			bool seeked = as.seeked;

			for (int i = 0; i < a->get_track_count(); i++) {

				NodePath path = a->track_get_path(i);
				TrackCache *track = track_cache[path];
				if (track->type != a->track_get_type(i)) {
					continue; //may happen should not
				}

				ERR_CONTINUE(!state.track_map.has(path));
				int blend_idx = state.track_map[path];

				ERR_CONTINUE(blend_idx < 0 || blend_idx >= state.track_count);

				float blend = (*as.track_blends)[blend_idx];

				if (blend < CMP_EPSILON)
					continue; //nothing to blend

				switch (track->type) {

					case Animation::TYPE_TRANSFORM: {

						TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

						Vector3 loc;
						Quat rot;
						Vector3 scale;

						Error err = a->transform_track_interpolate(i, time, &loc, &rot, &scale);
						//ERR_CONTINUE(err!=OK); //used for testing, should be removed

						scale -= Vector3(1.0, 1.0, 1.0); //helps make it work properly with Add nodes

						if (err != OK)
							continue;

						if (t->process_pass != process_pass) {

							t->process_pass = process_pass;
							t->loc = Vector3();
							t->rot = Quat();
							t->scale = Vector3();
						}

						t->loc = t->loc.linear_interpolate(loc, blend);
						t->rot = t->rot.slerp(rot, blend);
						t->scale = t->scale.linear_interpolate(scale, blend);

					} break;
					case Animation::TYPE_VALUE: {

						TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

						Animation::UpdateMode update_mode = a->value_track_get_update_mode(i);

						if (update_mode == Animation::UPDATE_CONTINUOUS || update_mode == Animation::UPDATE_CAPTURE) { //delta == 0 means seek

							Variant value = a->value_track_interpolate(i, time);

							if (value == Variant())
								continue;

							if (t->process_pass != process_pass) {
								Variant::CallError ce;
								t->value = Variant::construct(value.get_type(), NULL, 0, ce); //reset
								t->process_pass = process_pass;
							}

							Variant::interpolate(t->value, value, blend, t->value);

						} else if (delta != 0) {

							List<int> indices;
							a->value_track_get_key_indices(i, time, delta, &indices);

							for (List<int>::Element *F = indices.front(); F; F = F->next()) {

								Variant value = a->track_get_key_value(i, F->get());
								t->object->set_indexed(t->subpath, value);
							}
						}

					} break;
					case Animation::TYPE_METHOD: {

						if (delta == 0) {
							continue;
						}
						TrackCacheMethod *t = static_cast<TrackCacheMethod *>(track);

						List<int> indices;

						a->method_track_get_key_indices(i, time, delta, &indices);

						for (List<int>::Element *E = indices.front(); E; E = E->next()) {

							StringName method = a->method_track_get_name(i, E->get());
							Vector<Variant> params = a->method_track_get_params(i, E->get());

							int s = params.size();

							ERR_CONTINUE(s > VARIANT_ARG_MAX);
							if (can_call) {
								t->object->call_deferred(
										method,
										s >= 1 ? params[0] : Variant(),
										s >= 2 ? params[1] : Variant(),
										s >= 3 ? params[2] : Variant(),
										s >= 4 ? params[3] : Variant(),
										s >= 5 ? params[4] : Variant());
							}
						}

					} break;
					case Animation::TYPE_BEZIER: {

						TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);

						float bezier = a->bezier_track_interpolate(i, time);

						if (t->process_pass != process_pass) {
							t->value = 0;
							t->process_pass = process_pass;
						}

						t->value = Math::lerp(t->value, bezier, blend);

					} break;
					case Animation::TYPE_AUDIO: {

						TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);

						if (seeked) {
							//find whathever should be playing
							int idx = a->track_find_key(i, time);
							if (idx < 0)
								continue;

							Ref<AudioStream> stream = a->audio_track_get_key_stream(i, idx);
							if (!stream.is_valid()) {
								t->object->call("stop");
								t->playing = false;
								playing_caches.erase(t);
							} else {
								float start_ofs = a->audio_track_get_key_start_offset(i, idx);
								start_ofs += time - a->track_get_key_time(i, idx);
								float end_ofs = a->audio_track_get_key_end_offset(i, idx);
								float len = stream->get_length();

								if (start_ofs > len - end_ofs) {
									t->object->call("stop");
									t->playing = false;
									playing_caches.erase(t);
									continue;
								}

								t->object->call("set_stream", stream);
								t->object->call("play", start_ofs);

								t->playing = true;
								playing_caches.insert(t);
								if (len && end_ofs > 0) { //force a end at a time
									t->len = len - start_ofs - end_ofs;
								} else {
									t->len = 0;
								}

								t->start = time;
							}

						} else {
							//find stuff to play
							List<int> to_play;
							a->track_get_key_indices_in_range(i, time, delta, &to_play);
							if (to_play.size()) {
								int idx = to_play.back()->get();

								Ref<AudioStream> stream = a->audio_track_get_key_stream(i, idx);
								if (!stream.is_valid()) {
									t->object->call("stop");
									t->playing = false;
									playing_caches.erase(t);
								} else {
									float start_ofs = a->audio_track_get_key_start_offset(i, idx);
									float end_ofs = a->audio_track_get_key_end_offset(i, idx);
									float len = stream->get_length();

									t->object->call("set_stream", stream);
									t->object->call("play", start_ofs);

									t->playing = true;
									playing_caches.insert(t);
									if (len && end_ofs > 0) { //force a end at a time
										t->len = len - start_ofs - end_ofs;
									} else {
										t->len = 0;
									}

									t->start = time;
								}
							} else if (t->playing) {
								if (t->start > time || (t->len > 0 && time - t->start < t->len)) {
									//time to stop
									t->object->call("stop");
									t->playing = false;
									playing_caches.erase(t);
								}
							}
						}

					} break;
					case Animation::TYPE_ANIMATION: {

						TrackCacheAnimation *t = static_cast<TrackCacheAnimation *>(track);

						AnimationPlayer *player = Object::cast_to<AnimationPlayer>(t->object);

						if (!player)
							continue;

						if (delta == 0 || seeked) {
							//seek
							int idx = a->track_find_key(i, time);
							if (idx < 0)
								continue;

							float pos = a->track_get_key_time(i, idx);

							StringName anim_name = a->animation_track_get_key_animation(i, idx);
							if (String(anim_name) == "[stop]" || !player->has_animation(anim_name))
								continue;

							Ref<Animation> anim = player->get_animation(anim_name);

							float at_anim_pos;

							if (anim->has_loop()) {
								at_anim_pos = Math::fposmod(time - pos, anim->get_length()); //seek to loop
							} else {
								at_anim_pos = MAX(anim->get_length(), time - pos); //seek to end
							}

							if (player->is_playing() || seeked) {
								player->play(anim_name);
								player->seek(at_anim_pos);
								t->playing = true;
								playing_caches.insert(t);
							} else {
								player->set_assigned_animation(anim_name);
								player->seek(at_anim_pos, true);
							}
						} else {
							//find stuff to play
							List<int> to_play;
							a->track_get_key_indices_in_range(i, time, delta, &to_play);
							if (to_play.size()) {
								int idx = to_play.back()->get();

								StringName anim_name = a->animation_track_get_key_animation(i, idx);
								if (String(anim_name) == "[stop]" || !player->has_animation(anim_name)) {

									if (playing_caches.has(t)) {
										playing_caches.erase(t);
										player->stop();
										t->playing = false;
									}
								} else {
									player->play(anim_name);
									t->playing = true;
									playing_caches.insert(t);
								}
							}
						}

					} break;
				}
			}
		}
	}

	{
		// finally, set the tracks
		const NodePath *K = NULL;
		while ((K = track_cache.next(K))) {
			TrackCache *track = track_cache[*K];
			if (track->process_pass != process_pass)
				continue; //not processed, ignore

			switch (track->type) {

				case Animation::TYPE_TRANSFORM: {

					TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

					Transform xform;
					xform.origin = t->loc;

					t->scale += Vector3(1.0, 1.0, 1.0); //helps make it work properly with Add nodes

					xform.basis.set_quat_scale(t->rot, t->scale);

					if (t->skeleton && t->bone_idx >= 0) {

						t->skeleton->set_bone_pose(t->bone_idx, xform);

					} else {

						t->spatial->set_transform(xform);
					}

				} break;
				case Animation::TYPE_VALUE: {

					TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

					t->object->set_indexed(t->subpath, t->value);

				} break;
				case Animation::TYPE_BEZIER: {

					TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);

					t->object->set_indexed(t->subpath, t->value);

				} break;
				default: {} //the rest dont matter
			}
		}
	}
}

void AnimationTree::_notification(int p_what) {

	if (active && p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS && process_mode == ANIMATION_PROCESS_PHYSICS) {
		_process_graph(get_physics_process_delta_time());
	}

	if (active && p_what == NOTIFICATION_INTERNAL_PROCESS && process_mode == ANIMATION_PROCESS_IDLE) {
		_process_graph(get_process_delta_time());
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		_clear_caches();
	}
}

void AnimationTree::set_animation_player(const NodePath &p_player) {
	animation_player = p_player;
	update_configuration_warning();
}

NodePath AnimationTree::get_animation_player() const {
	return animation_player;
}

bool AnimationTree::is_state_invalid() const {

	return !state.valid;
}
String AnimationTree::get_invalid_state_reason() const {

	return state.invalid_reasons;
}

uint64_t AnimationTree::get_last_process_pass() const {
	return process_pass;
}

String AnimationTree::get_configuration_warning() const {

	String warning = Node::get_configuration_warning();

	if (!root.is_valid()) {
		if (warning != String()) {
			warning += "\n";
		}
		warning += TTR("A root AnimationNode for the graph is not set.");
	}

	if (!has_node(animation_player)) {

		if (warning != String()) {
			warning += "\n";
		}

		warning += TTR("Path to an AnimationPlayer node containing animations is not set.");
		return warning;
	}

	AnimationPlayer *player = Object::cast_to<AnimationPlayer>(get_node(animation_player));

	if (!player) {
		if (warning != String()) {
			warning += "\n";
		}

		warning += TTR("Path set for AnimationPlayer does not lead to an AnimationPlayer node.");
		return warning;
	}

	if (!player->has_node(player->get_root())) {
		if (warning != String()) {
			warning += "\n";
		}

		warning += TTR("AnimationPlayer root is not a valid node.");
		return warning;
	}

	return warning;
}

void AnimationTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_active", "active"), &AnimationTree::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &AnimationTree::is_active);

	ClassDB::bind_method(D_METHOD("set_graph_root", "root"), &AnimationTree::set_graph_root);
	ClassDB::bind_method(D_METHOD("get_graph_root"), &AnimationTree::get_graph_root);

	ClassDB::bind_method(D_METHOD("set_process_mode", "mode"), &AnimationTree::set_process_mode);
	ClassDB::bind_method(D_METHOD("get_process_mode"), &AnimationTree::get_process_mode);

	ClassDB::bind_method(D_METHOD("set_animation_player", "root"), &AnimationTree::set_animation_player);
	ClassDB::bind_method(D_METHOD("get_animation_player"), &AnimationTree::get_animation_player);

	ClassDB::bind_method(D_METHOD("_node_removed"), &AnimationTree::_node_removed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "graph_root", PROPERTY_HINT_RESOURCE_TYPE, "AnimationRootNode", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE), "set_graph_root", "get_graph_root");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "anim_player"), "set_animation_player", "get_animation_player");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_process_mode", "get_process_mode");
}

AnimationTree::AnimationTree() {

	process_mode = ANIMATION_PROCESS_IDLE;
	active = false;
	cache_valid = false;
	setup_pass = 1;
	started = true;
}

AnimationTree::~AnimationTree() {
	if (root.is_valid()) {
		root->player = NULL;
	}
}
