/**************************************************************************/
/*  animation_tree_player.cpp                                             */
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

#include "animation_tree_player.h"
#include "animation_player.h"

#include "core/os/os.h"
#include "scene/scene_string_names.h"

void AnimationTreePlayer::set_animation_process_mode(AnimationProcessMode p_mode) {
	if (animation_process_mode == p_mode) {
		return;
	}

	bool pr = processing;
	if (pr) {
		_set_process(false);
	}
	animation_process_mode = p_mode;
	if (pr) {
		_set_process(true);
	}
}

AnimationTreePlayer::AnimationProcessMode AnimationTreePlayer::get_animation_process_mode() const {
	return animation_process_mode;
}

void AnimationTreePlayer::_set_process(bool p_process, bool p_force) {
	if (processing == p_process && !p_force) {
		return;
	}

	switch (animation_process_mode) {
		case ANIMATION_PROCESS_PHYSICS:
			set_physics_process_internal(p_process && active);
			break;
		case ANIMATION_PROCESS_IDLE:
			set_process_internal(p_process && active);
			break;
	}

	processing = p_process;
}

bool AnimationTreePlayer::_set(const StringName &p_name, const Variant &p_value) {
	if (String(p_name) == "base_path") {
		set_base_path(p_value);
		return true;
	}

	if (String(p_name) == "master_player") {
		set_master_player(p_value);
		return true;
	}

	if (String(p_name) == SceneStringNames::get_singleton()->playback_active) {
		set_active(p_value);
		return true;
	}

	if (String(p_name) != "data") {
		return false;
	}

	Dictionary data = p_value;

	Array nodes = data.get_valid("nodes");

	for (int i = 0; i < nodes.size(); i++) {
		Dictionary node = nodes[i];

		StringName id = node.get_valid("id");
		Point2 pos = node.get_valid("position");

		NodeType nt = NODE_MAX;
		String type = node.get_valid("type");

		if (type == "output") {
			nt = NODE_OUTPUT;
		} else if (type == "animation") {
			nt = NODE_ANIMATION;
		} else if (type == "oneshot") {
			nt = NODE_ONESHOT;
		} else if (type == "mix") {
			nt = NODE_MIX;
		} else if (type == "blend2") {
			nt = NODE_BLEND2;
		} else if (type == "blend3") {
			nt = NODE_BLEND3;
		} else if (type == "blend4") {
			nt = NODE_BLEND4;
		} else if (type == "timescale") {
			nt = NODE_TIMESCALE;
		} else if (type == "timeseek") {
			nt = NODE_TIMESEEK;
		} else if (type == "transition") {
			nt = NODE_TRANSITION;
		}

		ERR_FAIL_COND_V(nt == NODE_MAX, false);

		if (nt != NODE_OUTPUT) {
			add_node(nt, id);
		}
		node_set_position(id, pos);

		switch (nt) {
			case NODE_OUTPUT: {
			} break;
			case NODE_ANIMATION: {
				if (node.has("from")) {
					animation_node_set_master_animation(id, node.get_valid("from"));
				} else {
					animation_node_set_animation(id, node.get_valid("animation"));
				}
				Array filters = node.get_valid("filter");
				for (int j = 0; j < filters.size(); j++) {
					animation_node_set_filter_path(id, filters[j], true);
				}
			} break;
			case NODE_ONESHOT: {
				oneshot_node_set_fadein_time(id, node.get_valid("fade_in"));
				oneshot_node_set_fadeout_time(id, node.get_valid("fade_out"));
				oneshot_node_set_mix_mode(id, node.get_valid("mix"));
				oneshot_node_set_autorestart(id, node.get_valid("autorestart"));
				oneshot_node_set_autorestart_delay(id, node.get_valid("autorestart_delay"));
				oneshot_node_set_autorestart_random_delay(id, node.get_valid("autorestart_random_delay"));
				Array filters = node.get_valid("filter");
				for (int j = 0; j < filters.size(); j++) {
					oneshot_node_set_filter_path(id, filters[j], true);
				}

			} break;
			case NODE_MIX: {
				mix_node_set_amount(id, node.get_valid("mix"));
			} break;
			case NODE_BLEND2: {
				blend2_node_set_amount(id, node.get_valid("blend"));
				Array filters = node.get_valid("filter");
				for (int j = 0; j < filters.size(); j++) {
					blend2_node_set_filter_path(id, filters[j], true);
				}
			} break;
			case NODE_BLEND3: {
				blend3_node_set_amount(id, node.get_valid("blend"));
			} break;
			case NODE_BLEND4: {
				blend4_node_set_amount(id, node.get_valid("blend"));
			} break;
			case NODE_TIMESCALE: {
				timescale_node_set_scale(id, node.get_valid("scale"));
			} break;
			case NODE_TIMESEEK: {
			} break;
			case NODE_TRANSITION: {
				transition_node_set_xfade_time(id, node.get_valid("xfade"));

				Array transitions = node.get_valid("transitions");
				transition_node_set_input_count(id, transitions.size());

				for (int x = 0; x < transitions.size(); x++) {
					Dictionary d = transitions[x];
					bool aa = d.get_valid("auto_advance");
					transition_node_set_input_auto_advance(id, x, aa);
				}

			} break;
			default: {
			};
		}
	}

	Array connections = data.get_valid("connections");
	ERR_FAIL_COND_V(connections.size() % 3, false);

	int cc = connections.size() / 3;

	for (int i = 0; i < cc; i++) {
		StringName src = connections[i * 3 + 0];
		StringName dst = connections[i * 3 + 1];
		int dst_in = connections[i * 3 + 2];
		connect_nodes(src, dst, dst_in);
	}

	set_active(data.get_valid("active"));
	set_master_player(data.get_valid("master"));

	return true;
}

bool AnimationTreePlayer::_get(const StringName &p_name, Variant &r_ret) const {
	if (String(p_name) == "base_path") {
		r_ret = base_path;
		return true;
	}

	if (String(p_name) == "master_player") {
		r_ret = master;
		return true;
	}

	if (String(p_name) == "playback/active") {
		r_ret = is_active();
		return true;
	}

	if (String(p_name) != "data") {
		return false;
	}

	Dictionary data;

	Array nodes;

	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		NodeBase *n = node_map[E->key()];

		Dictionary node;
		node["id"] = E->key();
		node["position"] = n->pos;

		switch (n->type) {
			case NODE_OUTPUT:
				node["type"] = "output";
				break;
			case NODE_ANIMATION:
				node["type"] = "animation";
				break;
			case NODE_ONESHOT:
				node["type"] = "oneshot";
				break;
			case NODE_MIX:
				node["type"] = "mix";
				break;
			case NODE_BLEND2:
				node["type"] = "blend2";
				break;
			case NODE_BLEND3:
				node["type"] = "blend3";
				break;
			case NODE_BLEND4:
				node["type"] = "blend4";
				break;
			case NODE_TIMESCALE:
				node["type"] = "timescale";
				break;
			case NODE_TIMESEEK:
				node["type"] = "timeseek";
				break;
			case NODE_TRANSITION:
				node["type"] = "transition";
				break;
			default:
				node["type"] = "";
				break;
		}

		switch (n->type) {
			case NODE_OUTPUT: {
			} break;
			case NODE_ANIMATION: {
				AnimationNode *an = static_cast<AnimationNode *>(n);
				if (master != NodePath() && an->from != "") {
					node["from"] = an->from;
				} else {
					node["animation"] = an->animation;
				}
				Array k;
				List<NodePath> keys;
				an->filter.get_key_list(&keys);
				k.resize(keys.size());
				int i = 0;
				for (List<NodePath>::Element *F = keys.front(); F; F = F->next()) {
					k[i++] = F->get();
				}
				node["filter"] = k;
			} break;
			case NODE_ONESHOT: {
				OneShotNode *osn = static_cast<OneShotNode *>(n);
				node["fade_in"] = osn->fade_in;
				node["fade_out"] = osn->fade_out;
				node["mix"] = osn->mix;
				node["autorestart"] = osn->autorestart;
				node["autorestart_delay"] = osn->autorestart_delay;
				node["autorestart_random_delay"] = osn->autorestart_random_delay;

				Array k;
				List<NodePath> keys;
				osn->filter.get_key_list(&keys);
				k.resize(keys.size());
				int i = 0;
				for (List<NodePath>::Element *F = keys.front(); F; F = F->next()) {
					k[i++] = F->get();
				}
				node["filter"] = k;

			} break;
			case NODE_MIX: {
				MixNode *mn = static_cast<MixNode *>(n);
				node["mix"] = mn->amount;
			} break;
			case NODE_BLEND2: {
				Blend2Node *bn = static_cast<Blend2Node *>(n);
				node["blend"] = bn->value;
				Array k;
				List<NodePath> keys;
				bn->filter.get_key_list(&keys);
				k.resize(keys.size());
				int i = 0;
				for (List<NodePath>::Element *F = keys.front(); F; F = F->next()) {
					k[i++] = F->get();
				}
				node["filter"] = k;

			} break;
			case NODE_BLEND3: {
				Blend3Node *bn = static_cast<Blend3Node *>(n);
				node["blend"] = bn->value;
			} break;
			case NODE_BLEND4: {
				Blend4Node *bn = static_cast<Blend4Node *>(n);
				node["blend"] = bn->value;

			} break;
			case NODE_TIMESCALE: {
				TimeScaleNode *tsn = static_cast<TimeScaleNode *>(n);
				node["scale"] = tsn->scale;
			} break;
			case NODE_TIMESEEK: {
			} break;
			case NODE_TRANSITION: {
				TransitionNode *tn = static_cast<TransitionNode *>(n);
				node["xfade"] = tn->xfade;
				Array transitions;

				for (int i = 0; i < tn->input_data.size(); i++) {
					Dictionary d;
					d["auto_advance"] = tn->input_data[i].auto_advance;
					transitions.push_back(d);
				}

				node["transitions"] = transitions;

			} break;
			default: {
			};
		}

		nodes.push_back(node);
	}

	data["nodes"] = nodes;
	//connectiosn

	List<Connection> connections;
	get_connection_list(&connections);
	Array connections_arr;
	connections_arr.resize(connections.size() * 3);

	int idx = 0;
	for (List<Connection>::Element *E = connections.front(); E; E = E->next()) {
		connections_arr.set(idx + 0, E->get().src_node);
		connections_arr.set(idx + 1, E->get().dst_node);
		connections_arr.set(idx + 2, E->get().dst_input);

		idx += 3;
	}

	data["connections"] = connections_arr;
	data["active"] = active;
	data["master"] = master;

	r_ret = data;

	return true;
}

void AnimationTreePlayer::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_NETWORK));
}

void AnimationTreePlayer::advance(float p_time) {
	_process_animation(p_time);
}

void AnimationTreePlayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			WARN_DEPRECATED_MSG("AnimationTreePlayer has been deprecated. Use AnimationTree instead.");

			if (!processing) {
				//make sure that a previous process state was not saved
				//only process if "processing" is set
				set_physics_process_internal(false);
				set_process_internal(false);
			}
		} break;
		case NOTIFICATION_READY: {
			dirty_caches = true;
			if (master != NodePath()) {
				_update_sources();
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (animation_process_mode == ANIMATION_PROCESS_PHYSICS) {
				break;
			}

			if (processing && OS::get_singleton()->is_update_pending()) {
				_process_animation(get_process_delta_time());
			}
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (animation_process_mode == ANIMATION_PROCESS_IDLE) {
				break;
			}

			if (processing && OS::get_singleton()->is_update_pending()) {
				_process_animation(get_physics_process_delta_time());
			}
		} break;
	}
}

void AnimationTreePlayer::_compute_weights(float *p_fallback_weight, HashMap<NodePath, float> *p_weights, float p_coeff, const HashMap<NodePath, bool> *p_filter, float p_filtered_coeff) {
	if (p_filter != nullptr) {
		List<NodePath> key_list;
		p_filter->get_key_list(&key_list);

		for (List<NodePath>::Element *E = key_list.front(); E; E = E->next()) {
			if ((*p_filter)[E->get()]) {
				if (p_weights->has(E->get())) {
					(*p_weights)[E->get()] *= p_filtered_coeff;
				} else {
					p_weights->set(E->get(), *p_fallback_weight * p_filtered_coeff);
				}

			} else if (p_weights->has(E->get())) {
				(*p_weights)[E->get()] *= p_coeff;
			}
		}
	}

	List<NodePath> key_list;
	p_weights->get_key_list(&key_list);

	for (List<NodePath>::Element *E = key_list.front(); E; E = E->next()) {
		if (p_filter == nullptr || !p_filter->has(E->get())) {
			(*p_weights)[E->get()] *= p_coeff;
		}
	}

	*p_fallback_weight *= p_coeff;
}

float AnimationTreePlayer::_process_node(const StringName &p_node, AnimationNode **r_prev_anim, float p_time, bool p_seek, float p_fallback_weight, HashMap<NodePath, float> *p_weights) {
	ERR_FAIL_COND_V(!node_map.has(p_node), 0);
	NodeBase *nb = node_map[p_node];

	//transform to seconds...

	switch (nb->type) {
		case NODE_OUTPUT: {
			NodeOut *on = static_cast<NodeOut *>(nb);
			HashMap<NodePath, float> weights;

			return _process_node(on->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, &weights);

		} break;
		case NODE_ANIMATION: {
			AnimationNode *an = static_cast<AnimationNode *>(nb);

			float rem = 0;
			if (!an->animation.is_null()) {
				//float pos = an->time;
				//float delta = p_time;

				//const Animation *a = an->animation.operator->();

				if (p_seek) {
					an->time = p_time;
					an->step = 0;
				} else {
					an->time = MAX(0, an->time + p_time);
					an->step = p_time;
				}

				float anim_size = an->animation->get_length();

				if (an->animation->has_loop()) {
					if (anim_size) {
						an->time = Math::fposmod(an->time, anim_size);
					}

				} else if (an->time > anim_size) {
					an->time = anim_size;
				}

				an->skip = true;

				for (List<AnimationNode::TrackRef>::Element *E = an->tref.front(); E; E = E->next()) {
					NodePath track_path = an->animation->track_get_path(E->get().local_track);
					if (an->filter.has(track_path) && an->filter[track_path]) {
						E->get().weight = 0;
					} else {
						if (p_weights->has(track_path)) {
							float weight = (*p_weights)[track_path];
							E->get().weight = weight;
						} else {
							E->get().weight = p_fallback_weight;
						}
					}
					if (E->get().weight > CMP_EPSILON) {
						an->skip = false;
					}
				}

				rem = anim_size - an->time;
			}

			if (!(*r_prev_anim)) {
				active_list = an;
			} else {
				(*r_prev_anim)->next = an;
			}

			an->next = nullptr;
			*r_prev_anim = an;

			return rem;

		} break;
		case NODE_ONESHOT: {
			OneShotNode *osn = static_cast<OneShotNode *>(nb);

			if (!osn->active) {
				//make it as if this node doesn't exist, pass input 0 by.
				return _process_node(osn->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
			}

			bool os_seek = p_seek;

			if (p_seek) {
				osn->time = p_time;
			}
			if (osn->start) {
				osn->time = 0;
				os_seek = true;
			}

			float blend;

			if (osn->time < osn->fade_in) {
				if (osn->fade_in > 0) {
					blend = osn->time / osn->fade_in;
				} else {
					blend = 0; //wtf
				}

			} else if (!osn->start && osn->remaining < osn->fade_out) {
				if (osn->fade_out) {
					blend = (osn->remaining / osn->fade_out);
				} else {
					blend = 1.0;
				}
			} else {
				blend = 1.0;
			}

			float main_rem;
			float os_rem;

			HashMap<NodePath, float> os_weights(*p_weights);
			float os_fallback_weight = p_fallback_weight;
			_compute_weights(&p_fallback_weight, p_weights, osn->mix ? 1.0 : 1.0 - blend, &osn->filter, 1.0);
			_compute_weights(&os_fallback_weight, &os_weights, blend, &osn->filter, 0.0);

			main_rem = _process_node(osn->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
			os_rem = _process_node(osn->inputs[1].node, r_prev_anim, p_time, os_seek, os_fallback_weight, &os_weights);

			if (osn->start) {
				osn->remaining = os_rem;
				osn->start = false;
			}

			if (!p_seek) {
				osn->time += p_time;
				osn->remaining = os_rem;
				if (osn->remaining <= 0) {
					osn->active = false;
				}
			}

			return MAX(main_rem, osn->remaining);
		} break;
		case NODE_MIX: {
			MixNode *mn = static_cast<MixNode *>(nb);

			HashMap<NodePath, float> mn_weights(*p_weights);
			float mn_fallback_weight = p_fallback_weight;
			_compute_weights(&mn_fallback_weight, &mn_weights, mn->amount);
			float rem = _process_node(mn->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
			_process_node(mn->inputs[1].node, r_prev_anim, p_time, p_seek, mn_fallback_weight, &mn_weights);
			return rem;

		} break;
		case NODE_BLEND2: {
			Blend2Node *bn = static_cast<Blend2Node *>(nb);

			HashMap<NodePath, float> bn_weights(*p_weights);
			float bn_fallback_weight = p_fallback_weight;
			_compute_weights(&p_fallback_weight, p_weights, 1.0 - bn->value, &bn->filter, 1.0);
			_compute_weights(&bn_fallback_weight, &bn_weights, bn->value, &bn->filter, 0.0);
			float rem = _process_node(bn->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
			_process_node(bn->inputs[1].node, r_prev_anim, p_time, p_seek, bn_fallback_weight, &bn_weights);

			return rem;
		} break;
		case NODE_BLEND3: {
			Blend3Node *bn = static_cast<Blend3Node *>(nb);

			float rem;
			float blend, lower_blend, upper_blend;
			if (bn->value < 0) {
				lower_blend = -bn->value;
				blend = 1.0 - lower_blend;
				upper_blend = 0;
			} else {
				lower_blend = 0;
				blend = 1.0 - bn->value;
				upper_blend = bn->value;
			}

			HashMap<NodePath, float> upper_weights(*p_weights);
			float upper_fallback_weight = p_fallback_weight;
			HashMap<NodePath, float> lower_weights(*p_weights);
			float lower_fallback_weight = p_fallback_weight;
			_compute_weights(&upper_fallback_weight, &upper_weights, upper_blend);
			_compute_weights(&p_fallback_weight, p_weights, blend);
			_compute_weights(&lower_fallback_weight, &lower_weights, lower_blend);

			rem = _process_node(bn->inputs[1].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
			_process_node(bn->inputs[0].node, r_prev_anim, p_time, p_seek, lower_fallback_weight, &lower_weights);
			_process_node(bn->inputs[2].node, r_prev_anim, p_time, p_seek, upper_fallback_weight, &upper_weights);

			return rem;
		} break;
		case NODE_BLEND4: {
			Blend4Node *bn = static_cast<Blend4Node *>(nb);

			HashMap<NodePath, float> weights1(*p_weights);
			float fallback_weight1 = p_fallback_weight;
			HashMap<NodePath, float> weights2(*p_weights);
			float fallback_weight2 = p_fallback_weight;
			HashMap<NodePath, float> weights3(*p_weights);
			float fallback_weight3 = p_fallback_weight;

			_compute_weights(&p_fallback_weight, p_weights, 1.0 - bn->value.x);
			_compute_weights(&fallback_weight1, &weights1, bn->value.x);
			_compute_weights(&fallback_weight2, &weights2, 1.0 - bn->value.y);
			_compute_weights(&fallback_weight3, &weights3, bn->value.y);

			float rem = _process_node(bn->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
			_process_node(bn->inputs[1].node, r_prev_anim, p_time, p_seek, fallback_weight1, &weights1);
			float rem2 = _process_node(bn->inputs[2].node, r_prev_anim, p_time, p_seek, fallback_weight2, &weights2);
			_process_node(bn->inputs[3].node, r_prev_anim, p_time, p_seek, fallback_weight3, &weights3);

			return MAX(rem, rem2);

		} break;
		case NODE_TIMESCALE: {
			TimeScaleNode *tsn = static_cast<TimeScaleNode *>(nb);
			float rem;
			if (p_seek) {
				rem = _process_node(tsn->inputs[0].node, r_prev_anim, p_time, true, p_fallback_weight, p_weights);
			} else {
				rem = _process_node(tsn->inputs[0].node, r_prev_anim, p_time * tsn->scale, false, p_fallback_weight, p_weights);
			}
			if (tsn->scale == 0) {
				return Math_INF;
			} else {
				return rem / tsn->scale;
			}

		} break;
		case NODE_TIMESEEK: {
			TimeSeekNode *tsn = static_cast<TimeSeekNode *>(nb);
			if (tsn->seek_pos >= 0 && !p_seek) {
				p_time = tsn->seek_pos;
				p_seek = true;
			}
			tsn->seek_pos = -1;

			return _process_node(tsn->inputs[0].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);

		} break;
		case NODE_TRANSITION: {
			TransitionNode *tn = static_cast<TransitionNode *>(nb);
			HashMap<NodePath, float> prev_weights(*p_weights);
			float prev_fallback_weight = p_fallback_weight;

			if (tn->prev < 0) { // process current animation, check for transition

				float rem = _process_node(tn->inputs[tn->current].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
				if (p_seek) {
					tn->time = p_time;
				} else {
					tn->time += p_time;
				}

				if (tn->input_data[tn->current].auto_advance && rem <= tn->xfade) {
					tn->set_current((tn->current + 1) % tn->inputs.size());
				}

				return rem;
			} else { // cross-fading from tn->prev to tn->current

				float blend = tn->xfade ? (tn->prev_xfading / tn->xfade) : 1;

				float rem;

				_compute_weights(&p_fallback_weight, p_weights, 1.0 - blend);
				_compute_weights(&prev_fallback_weight, &prev_weights, blend);

				if (!p_seek && tn->switched) { //just switched, seek to start of current

					rem = _process_node(tn->inputs[tn->current].node, r_prev_anim, 0, true, p_fallback_weight, p_weights);
				} else {
					rem = _process_node(tn->inputs[tn->current].node, r_prev_anim, p_time, p_seek, p_fallback_weight, p_weights);
				}

				tn->switched = false;

				if (p_seek) { // don't seek prev animation
					_process_node(tn->inputs[tn->prev].node, r_prev_anim, 0, false, prev_fallback_weight, &prev_weights);
					tn->time = p_time;
				} else {
					_process_node(tn->inputs[tn->prev].node, r_prev_anim, p_time, false, prev_fallback_weight, &prev_weights);
					tn->time += p_time;
					tn->prev_xfading -= p_time;
					if (tn->prev_xfading < 0) {
						tn->prev = -1;
					}
				}

				return rem;
			}

		} break;
		default: {
		}
	}

	return 0;
}

void AnimationTreePlayer::_process_animation(float p_delta) {
	if (last_error != CONNECT_OK) {
		return;
	}

	if (dirty_caches) {
		_recompute_caches();
	}

	active_list = nullptr;
	AnimationNode *prev = nullptr;

	if (reset_request) {
		_process_node(out_name, &prev, 0, true);
		reset_request = false;
	} else {
		_process_node(out_name, &prev, p_delta);
	}

	if (dirty_caches) {
		//some animation changed.. ignore this pass
		return;
	}

	//update the tracks..

	/* STEP 1 CLEAR TRACKS */

	for (TrackMap::Element *E = track_map.front(); E; E = E->next()) {
		Track &t = E->get();

		t.loc.zero();
		t.rot = Quat();
		t.scale.x = 0;
		t.scale.y = 0;
		t.scale.z = 0;

		t.value = t.object->get_indexed(t.subpath);
		t.value.zero();

		t.skip = false;
	}

	/* STEP 2 PROCESS ANIMATIONS */

	AnimationNode *anim_list = active_list;
	Quat empty_rot;

	while (anim_list) {
		if (!anim_list->animation.is_null() && !anim_list->skip) {
			//check if animation is meaningful
			Animation *a = anim_list->animation.operator->();

			for (List<AnimationNode::TrackRef>::Element *E = anim_list->tref.front(); E; E = E->next()) {
				AnimationNode::TrackRef &tr = E->get();
				if (tr.track == nullptr || tr.local_track < 0 || tr.weight < CMP_EPSILON || !a->track_is_enabled(tr.local_track)) {
					continue;
				}

				switch (a->track_get_type(tr.local_track)) {
					case Animation::TYPE_TRANSFORM: { ///< Transform a node or a bone.

						Vector3 loc;
						Quat rot;
						Vector3 scale;
						a->transform_track_interpolate(tr.local_track, anim_list->time, &loc, &rot, &scale);

						tr.track->loc += loc * tr.weight;

						scale.x -= 1.0;
						scale.y -= 1.0;
						scale.z -= 1.0;
						tr.track->scale += scale * tr.weight;

						tr.track->rot = tr.track->rot * empty_rot.slerp(rot, tr.weight);

					} break;
					case Animation::TYPE_VALUE: { ///< Set a value in a property, can be interpolated.

						if (a->value_track_get_update_mode(tr.local_track) == Animation::UPDATE_CONTINUOUS) {
							Variant value = a->value_track_interpolate(tr.local_track, anim_list->time);
							Variant::blend(tr.track->value, value, tr.weight, tr.track->value);
						} else {
							int index = a->track_find_key(tr.local_track, anim_list->time);
							tr.track->value = a->track_get_key_value(tr.local_track, index);
						}
					} break;
					case Animation::TYPE_METHOD: { ///< Call any method on a specific node.

						List<int> indices;
						a->method_track_get_key_indices(tr.local_track, anim_list->time, anim_list->step, &indices);
						for (List<int>::Element *F = indices.front(); F; F = F->next()) {
							StringName method = a->method_track_get_name(tr.local_track, F->get());
							Vector<Variant> args = a->method_track_get_params(tr.local_track, F->get());
							args.resize(VARIANT_ARG_MAX);
							tr.track->object->call(method, args[0], args[1], args[2], args[3], args[4]);
						}
					} break;
					default: {
					}
				}
			}
		}

		anim_list = anim_list->next;
	}

	/* STEP 3 APPLY TRACKS */

	for (TrackMap::Element *E = track_map.front(); E; E = E->next()) {
		Track &t = E->get();

		if (t.skip || !t.object) {
			continue;
		}

		if (t.subpath.size()) { // value track
			t.object->set_indexed(t.subpath, t.value);
			continue;
		}

		Transform xform;
		xform.origin = t.loc;

		t.scale.x += 1.0;
		t.scale.y += 1.0;
		t.scale.z += 1.0;
		xform.basis.set_quat_scale(t.rot, t.scale);

		if (t.bone_idx >= 0) {
			if (t.skeleton) {
				t.skeleton->set_bone_pose(t.bone_idx, xform);
			}

		} else if (t.spatial) {
			t.spatial->set_transform(xform);
		}
	}
}

void AnimationTreePlayer::add_node(NodeType p_type, const StringName &p_node) {
	ERR_FAIL_COND(p_type == NODE_OUTPUT);
	ERR_FAIL_COND(node_map.has(p_node));
	ERR_FAIL_INDEX(p_type, NODE_MAX);
	NodeBase *n = nullptr;

	switch (p_type) {
		case NODE_ANIMATION: {
			n = memnew(AnimationNode);
		} break;
		case NODE_ONESHOT: {
			n = memnew(OneShotNode);

		} break;
		case NODE_MIX: {
			n = memnew(MixNode);

		} break;
		case NODE_BLEND2: {
			n = memnew(Blend2Node);

		} break;
		case NODE_BLEND3: {
			n = memnew(Blend3Node);

		} break;
		case NODE_BLEND4: {
			n = memnew(Blend4Node);

		} break;
		case NODE_TIMESCALE: {
			n = memnew(TimeScaleNode);

		} break;
		case NODE_TIMESEEK: {
			n = memnew(TimeSeekNode);

		} break;
		case NODE_TRANSITION: {
			n = memnew(TransitionNode);

		} break;
		default: {
		}
	}

	//n->name+=" "+itos(p_node);
	node_map[p_node] = n;
}

StringName AnimationTreePlayer::node_get_input_source(const StringName &p_node, int p_input) const {
	ERR_FAIL_COND_V(!node_map.has(p_node), StringName());
	ERR_FAIL_INDEX_V(p_input, node_map[p_node]->inputs.size(), StringName());
	return node_map[p_node]->inputs[p_input].node;
}

int AnimationTreePlayer::node_get_input_count(const StringName &p_node) const {
	ERR_FAIL_COND_V(!node_map.has(p_node), -1);
	return node_map[p_node]->inputs.size();
}
#define GET_NODE(m_type, m_cast)                                                             \
	ERR_FAIL_COND(!node_map.has(p_node));                                                    \
	ERR_FAIL_COND_MSG(node_map[p_node]->type != m_type, "Invalid parameter for node type."); \
	m_cast *n = static_cast<m_cast *>(node_map[p_node]);

void AnimationTreePlayer::animation_node_set_animation(const StringName &p_node, const Ref<Animation> &p_animation) {
	GET_NODE(NODE_ANIMATION, AnimationNode);
	n->animation = p_animation;
	dirty_caches = true;
}

void AnimationTreePlayer::animation_node_set_master_animation(const StringName &p_node, const String &p_master_animation) {
	GET_NODE(NODE_ANIMATION, AnimationNode);
	n->from = p_master_animation;
	dirty_caches = true;
	if (master != NodePath()) {
		_update_sources();
	}
}

void AnimationTreePlayer::animation_node_set_filter_path(const StringName &p_node, const NodePath &p_track_path, bool p_filter) {
	GET_NODE(NODE_ANIMATION, AnimationNode);

	if (p_filter) {
		n->filter[p_track_path] = true;
	} else {
		n->filter.erase(p_track_path);
	}
}

void AnimationTreePlayer::animation_node_set_get_filtered_paths(const StringName &p_node, List<NodePath> *r_paths) const {
	GET_NODE(NODE_ANIMATION, AnimationNode);

	n->filter.get_key_list(r_paths);
}

void AnimationTreePlayer::oneshot_node_set_fadein_time(const StringName &p_node, float p_time) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->fade_in = p_time;
}

void AnimationTreePlayer::oneshot_node_set_fadeout_time(const StringName &p_node, float p_time) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->fade_out = p_time;
}

void AnimationTreePlayer::oneshot_node_set_mix_mode(const StringName &p_node, bool p_mix) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->mix = p_mix;
}

void AnimationTreePlayer::oneshot_node_set_autorestart(const StringName &p_node, bool p_active) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->autorestart = p_active;
}

void AnimationTreePlayer::oneshot_node_set_autorestart_delay(const StringName &p_node, float p_time) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->autorestart_delay = p_time;
}
void AnimationTreePlayer::oneshot_node_set_autorestart_random_delay(const StringName &p_node, float p_time) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->autorestart_random_delay = p_time;
}

void AnimationTreePlayer::oneshot_node_start(const StringName &p_node) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->active = true;
	n->start = true;
}

void AnimationTreePlayer::oneshot_node_stop(const StringName &p_node) {
	GET_NODE(NODE_ONESHOT, OneShotNode);
	n->active = false;
}

void AnimationTreePlayer::oneshot_node_set_filter_path(const StringName &p_node, const NodePath &p_filter, bool p_enable) {
	GET_NODE(NODE_ONESHOT, OneShotNode);

	if (p_enable) {
		n->filter[p_filter] = true;
	} else {
		n->filter.erase(p_filter);
	}
}

void AnimationTreePlayer::oneshot_node_set_get_filtered_paths(const StringName &p_node, List<NodePath> *r_paths) const {
	GET_NODE(NODE_ONESHOT, OneShotNode);

	n->filter.get_key_list(r_paths);
}

void AnimationTreePlayer::mix_node_set_amount(const StringName &p_node, float p_amount) {
	GET_NODE(NODE_MIX, MixNode);
	n->amount = p_amount;
}

void AnimationTreePlayer::blend2_node_set_amount(const StringName &p_node, float p_amount) {
	GET_NODE(NODE_BLEND2, Blend2Node);
	n->value = p_amount;
}

void AnimationTreePlayer::blend2_node_set_filter_path(const StringName &p_node, const NodePath &p_filter, bool p_enable) {
	GET_NODE(NODE_BLEND2, Blend2Node);

	if (p_enable) {
		n->filter[p_filter] = true;
	} else {
		n->filter.erase(p_filter);
	}
}

void AnimationTreePlayer::blend2_node_set_get_filtered_paths(const StringName &p_node, List<NodePath> *r_paths) const {
	GET_NODE(NODE_BLEND2, Blend2Node);

	n->filter.get_key_list(r_paths);
}

void AnimationTreePlayer::blend3_node_set_amount(const StringName &p_node, float p_amount) {
	GET_NODE(NODE_BLEND3, Blend3Node);
	n->value = p_amount;
}
void AnimationTreePlayer::blend4_node_set_amount(const StringName &p_node, const Vector2 &p_amount) {
	GET_NODE(NODE_BLEND4, Blend4Node);
	n->value = p_amount;
}
void AnimationTreePlayer::timescale_node_set_scale(const StringName &p_node, float p_scale) {
	GET_NODE(NODE_TIMESCALE, TimeScaleNode);
	n->scale = p_scale;
}
void AnimationTreePlayer::timeseek_node_seek(const StringName &p_node, float p_pos) {
	GET_NODE(NODE_TIMESEEK, TimeSeekNode);
	n->seek_pos = p_pos;
}
void AnimationTreePlayer::transition_node_set_input_count(const StringName &p_node, int p_inputs) {
	GET_NODE(NODE_TRANSITION, TransitionNode);
	ERR_FAIL_COND(p_inputs < 1);

	n->inputs.resize(p_inputs);
	n->input_data.resize(p_inputs);

	_clear_cycle_test();

	last_error = _cycle_test(out_name);
}
void AnimationTreePlayer::transition_node_set_input_auto_advance(const StringName &p_node, int p_input, bool p_auto_advance) {
	GET_NODE(NODE_TRANSITION, TransitionNode);
	ERR_FAIL_INDEX(p_input, n->input_data.size());

	n->input_data.write[p_input].auto_advance = p_auto_advance;
}
void AnimationTreePlayer::transition_node_set_xfade_time(const StringName &p_node, float p_time) {
	GET_NODE(NODE_TRANSITION, TransitionNode);
	n->xfade = p_time;
}

void AnimationTreePlayer::TransitionNode::set_current(int p_current) {
	ERR_FAIL_INDEX(p_current, inputs.size());

	if (current == p_current) {
		return;
	}

	prev = current;
	prev_xfading = xfade;
	prev_time = time;
	time = 0;
	current = p_current;
	switched = true;
}

void AnimationTreePlayer::transition_node_set_current(const StringName &p_node, int p_current) {
	GET_NODE(NODE_TRANSITION, TransitionNode);
	n->set_current(p_current);
}

void AnimationTreePlayer::node_set_position(const StringName &p_node, const Vector2 &p_pos) {
	ERR_FAIL_COND(!node_map.has(p_node));
	node_map[p_node]->pos = p_pos;
}

AnimationTreePlayer::NodeType AnimationTreePlayer::node_get_type(const StringName &p_node) const {
	ERR_FAIL_COND_V(!node_map.has(p_node), NODE_OUTPUT);
	return node_map[p_node]->type;
}
Point2 AnimationTreePlayer::node_get_position(const StringName &p_node) const {
	ERR_FAIL_COND_V(!node_map.has(p_node), Point2());
	return node_map[p_node]->pos;
}

#define GET_NODE_V(m_type, m_cast, m_ret)                                                             \
	ERR_FAIL_COND_V(!node_map.has(p_node), m_ret);                                                    \
	ERR_FAIL_COND_V_MSG(node_map[p_node]->type != m_type, m_ret, "Invalid parameter for node type."); \
	m_cast *n = static_cast<m_cast *>(node_map[p_node]);

Ref<Animation> AnimationTreePlayer::animation_node_get_animation(const StringName &p_node) const {
	GET_NODE_V(NODE_ANIMATION, AnimationNode, Ref<Animation>());
	return n->animation;
}

String AnimationTreePlayer::animation_node_get_master_animation(const StringName &p_node) const {
	GET_NODE_V(NODE_ANIMATION, AnimationNode, String());
	return n->from;
}

float AnimationTreePlayer::animation_node_get_position(const StringName &p_node) const {
	GET_NODE_V(NODE_ANIMATION, AnimationNode, 0);
	return n->time;
}

bool AnimationTreePlayer::animation_node_is_path_filtered(const StringName &p_node, const NodePath &p_path) const {
	GET_NODE_V(NODE_ANIMATION, AnimationNode, 0);
	return n->filter.has(p_path);
}

float AnimationTreePlayer::oneshot_node_get_fadein_time(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->fade_in;
}

float AnimationTreePlayer::oneshot_node_get_fadeout_time(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->fade_out;
}

bool AnimationTreePlayer::oneshot_node_get_mix_mode(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->mix;
}
bool AnimationTreePlayer::oneshot_node_has_autorestart(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->autorestart;
}
float AnimationTreePlayer::oneshot_node_get_autorestart_delay(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->autorestart_delay;
}
float AnimationTreePlayer::oneshot_node_get_autorestart_random_delay(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->autorestart_random_delay;
}

bool AnimationTreePlayer::oneshot_node_is_active(const StringName &p_node) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->active;
}

bool AnimationTreePlayer::oneshot_node_is_path_filtered(const StringName &p_node, const NodePath &p_path) const {
	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0);
	return n->filter.has(p_path);
}

float AnimationTreePlayer::mix_node_get_amount(const StringName &p_node) const {
	GET_NODE_V(NODE_MIX, MixNode, 0);
	return n->amount;
}
float AnimationTreePlayer::blend2_node_get_amount(const StringName &p_node) const {
	GET_NODE_V(NODE_BLEND2, Blend2Node, 0);
	return n->value;
}

bool AnimationTreePlayer::blend2_node_is_path_filtered(const StringName &p_node, const NodePath &p_path) const {
	GET_NODE_V(NODE_BLEND2, Blend2Node, 0);
	return n->filter.has(p_path);
}

float AnimationTreePlayer::blend3_node_get_amount(const StringName &p_node) const {
	GET_NODE_V(NODE_BLEND3, Blend3Node, 0);
	return n->value;
}
Vector2 AnimationTreePlayer::blend4_node_get_amount(const StringName &p_node) const {
	GET_NODE_V(NODE_BLEND4, Blend4Node, Vector2());
	return n->value;
}

float AnimationTreePlayer::timescale_node_get_scale(const StringName &p_node) const {
	GET_NODE_V(NODE_TIMESCALE, TimeScaleNode, 0);
	return n->scale;
}

void AnimationTreePlayer::transition_node_delete_input(const StringName &p_node, int p_input) {
	GET_NODE(NODE_TRANSITION, TransitionNode);
	ERR_FAIL_INDEX(p_input, n->inputs.size());

	if (n->inputs.size() <= 1) {
		return;
	}

	n->inputs.remove(p_input);
	n->input_data.remove(p_input);
	last_error = _cycle_test(out_name);
}

int AnimationTreePlayer::transition_node_get_input_count(const StringName &p_node) const {
	GET_NODE_V(NODE_TRANSITION, TransitionNode, 0);
	return n->inputs.size();
}

bool AnimationTreePlayer::transition_node_has_input_auto_advance(const StringName &p_node, int p_input) const {
	GET_NODE_V(NODE_TRANSITION, TransitionNode, false);
	ERR_FAIL_INDEX_V(p_input, n->inputs.size(), false);
	return n->input_data[p_input].auto_advance;
}
float AnimationTreePlayer::transition_node_get_xfade_time(const StringName &p_node) const {
	GET_NODE_V(NODE_TRANSITION, TransitionNode, 0);
	return n->xfade;
}

int AnimationTreePlayer::transition_node_get_current(const StringName &p_node) const {
	GET_NODE_V(NODE_TRANSITION, TransitionNode, -1);
	return n->current;
}

/*misc  */
void AnimationTreePlayer::get_node_list(List<StringName> *p_node_list) const {
	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		p_node_list->push_back(E->key());
	}
}

void AnimationTreePlayer::remove_node(const StringName &p_node) {
	ERR_FAIL_COND(!node_map.has(p_node));
	ERR_FAIL_COND_MSG(p_node == out_name, "Node 0 (output) can't be removed.");

	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		NodeBase *nb = E->get();
		for (int i = 0; i < nb->inputs.size(); i++) {
			if (nb->inputs[i].node == p_node) {
				nb->inputs.write[i].node = StringName();
			}
		}
	}

	memdelete(node_map[p_node]);
	node_map.erase(p_node);

	_clear_cycle_test();

	// compute last error again, just in case
	last_error = _cycle_test(out_name);
	dirty_caches = true;
}

AnimationTreePlayer::ConnectError AnimationTreePlayer::_cycle_test(const StringName &p_at_node) {
	ERR_FAIL_COND_V(!node_map.has(p_at_node), CONNECT_INCOMPLETE);

	NodeBase *nb = node_map[p_at_node];
	if (nb->cycletest) {
		return CONNECT_CYCLE;
	}

	nb->cycletest = true;

	for (int i = 0; i < nb->inputs.size(); i++) {
		if (nb->inputs[i].node == StringName()) {
			return CONNECT_INCOMPLETE;
		}

		ConnectError _err = _cycle_test(nb->inputs[i].node);
		if (_err) {
			return _err;
		}
	}

	return CONNECT_OK;
}

// Use this function to not alter next complete _cycle_test().
void AnimationTreePlayer::_clear_cycle_test() {
	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		NodeBase *nb = E->get();
		nb->cycletest = false;
	}
}

Error AnimationTreePlayer::connect_nodes(const StringName &p_src_node, const StringName &p_dst_node, int p_dst_input) {
	ERR_FAIL_COND_V(!node_map.has(p_src_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!node_map.has(p_dst_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_src_node == p_dst_node, ERR_INVALID_PARAMETER);

	//NodeBase *src = node_map[p_src_node];
	NodeBase *dst = node_map[p_dst_node];
	ERR_FAIL_INDEX_V(p_dst_input, dst->inputs.size(), ERR_INVALID_PARAMETER);

	//int oldval = dst->inputs[p_dst_input].node;

	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		NodeBase *nb = E->get();
		for (int i = 0; i < nb->inputs.size(); i++) {
			if (nb->inputs[i].node == p_src_node) {
				nb->inputs.write[i].node = StringName();
			}
		}
	}

	dst->inputs.write[p_dst_input].node = p_src_node;

	_clear_cycle_test();

	last_error = _cycle_test(out_name);
	if (last_error) {
		if (last_error == CONNECT_INCOMPLETE) {
			return ERR_UNCONFIGURED;
		} else if (last_error == CONNECT_CYCLE) {
			return ERR_CYCLIC_LINK;
		}
	}
	dirty_caches = true;
	return OK;
}

bool AnimationTreePlayer::are_nodes_connected(const StringName &p_src_node, const StringName &p_dst_node, int p_dst_input) const {
	ERR_FAIL_COND_V(!node_map.has(p_src_node), false);
	ERR_FAIL_COND_V(!node_map.has(p_dst_node), false);
	ERR_FAIL_COND_V(p_src_node == p_dst_node, false);

	NodeBase *dst = node_map[p_dst_node];
	ERR_FAIL_INDEX_V(p_dst_input, dst->inputs.size(), false);

	return dst->inputs[p_dst_input].node == p_src_node;
}

void AnimationTreePlayer::disconnect_nodes(const StringName &p_node, int p_input) {
	ERR_FAIL_COND(!node_map.has(p_node));

	NodeBase *dst = node_map[p_node];
	ERR_FAIL_INDEX(p_input, dst->inputs.size());
	dst->inputs.write[p_input].node = StringName();
	last_error = CONNECT_INCOMPLETE;
	dirty_caches = true;
}

void AnimationTreePlayer::get_connection_list(List<Connection> *p_connections) const {
	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		NodeBase *nb = E->get();
		for (int i = 0; i < nb->inputs.size(); i++) {
			if (nb->inputs[i].node != StringName()) {
				Connection c;
				c.src_node = nb->inputs[i].node;
				c.dst_node = E->key();
				c.dst_input = i;
				p_connections->push_back(c);
			}
		}
	}
}

AnimationTreePlayer::Track *AnimationTreePlayer::_find_track(const NodePath &p_path) {
	Node *parent = get_node(base_path);
	ERR_FAIL_COND_V(!parent, nullptr);

	RES resource;
	Vector<StringName> leftover_path;
	Node *child = parent->get_node_and_resource(p_path, resource, leftover_path);
	if (!child) {
		String err = "Animation track references unknown Node: '" + String(p_path) + "'.";
		WARN_PRINT(err.ascii().get_data());
		return nullptr;
	}

	ObjectID id = child->get_instance_id();
	int bone_idx = -1;

	if (p_path.get_subname_count()) {
		if (Object::cast_to<Skeleton>(child)) {
			bone_idx = Object::cast_to<Skeleton>(child)->find_bone(p_path.get_subname(0));
		}
	}

	TrackKey key;
	key.id = id;
	key.bone_idx = bone_idx;
	key.subpath_concatenated = p_path.get_concatenated_subnames();

	if (!track_map.has(key)) {
		Track tr;
		tr.id = id;
		tr.object = resource.is_valid() ? (Object *)resource.ptr() : (Object *)child;
		tr.skeleton = Object::cast_to<Skeleton>(child);
		tr.spatial = Object::cast_to<Spatial>(child);
		tr.bone_idx = bone_idx;
		if (bone_idx == -1) {
			tr.subpath = leftover_path;
		}

		track_map[key] = tr;
	}

	return &track_map[key];
}

void AnimationTreePlayer::_recompute_caches() {
	track_map.clear();
	_recompute_caches(out_name);
	dirty_caches = false;
}

void AnimationTreePlayer::_recompute_caches(const StringName &p_node) {
	ERR_FAIL_COND(!node_map.has(p_node));

	NodeBase *nb = node_map[p_node];

	if (nb->type == NODE_ANIMATION) {
		AnimationNode *an = static_cast<AnimationNode *>(nb);
		an->tref.clear();

		if (!an->animation.is_null()) {
			Ref<Animation> a = an->animation;

			for (int i = 0; i < an->animation->get_track_count(); i++) {
				Track *tr = _find_track(a->track_get_path(i));
				if (!tr) {
					continue;
				}

				AnimationNode::TrackRef tref;
				tref.local_track = i;
				tref.track = tr;
				tref.weight = 0;

				an->tref.push_back(tref);
			}
		}
	}

	for (int i = 0; i < nb->inputs.size(); i++) {
		_recompute_caches(nb->inputs[i].node);
	}
}

void AnimationTreePlayer::recompute_caches() {
	dirty_caches = true;
}

/* playback */

void AnimationTreePlayer::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}

	active = p_active;
	processing = active;
	reset_request = p_active;
	_set_process(processing, true);
}

bool AnimationTreePlayer::is_active() const {
	return active;
}

AnimationTreePlayer::ConnectError AnimationTreePlayer::get_last_error() const {
	return last_error;
}

void AnimationTreePlayer::reset() {
	reset_request = true;
}

void AnimationTreePlayer::set_base_path(const NodePath &p_path) {
	base_path = p_path;
	recompute_caches();
}

NodePath AnimationTreePlayer::get_base_path() const {
	return base_path;
}

void AnimationTreePlayer::set_master_player(const NodePath &p_path) {
	if (p_path == master) {
		return;
	}

	master = p_path;
	_update_sources();
	recompute_caches();
}

NodePath AnimationTreePlayer::get_master_player() const {
	return master;
}

PoolVector<String> AnimationTreePlayer::_get_node_list() {
	List<StringName> nl;
	get_node_list(&nl);
	PoolVector<String> ret;
	ret.resize(nl.size());
	int idx = 0;
	for (List<StringName>::Element *E = nl.front(); E; E = E->next()) {
		ret.set(idx++, E->get());
	}

	return ret;
}

void AnimationTreePlayer::_update_sources() {
	if (master == NodePath()) {
		return;
	}
	if (!is_inside_tree()) {
		return;
	}

	Node *m = get_node(master);
	if (!m) {
		master = NodePath();
		ERR_FAIL_COND(!m);
	}

	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(m);

	if (!ap) {
		master = NodePath();
		ERR_FAIL_COND(!ap);
	}

	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		if (E->get()->type == NODE_ANIMATION) {
			AnimationNode *an = static_cast<AnimationNode *>(E->get());

			if (an->from != "") {
				an->animation = ap->get_animation(an->from);
			}
		}
	}
}

bool AnimationTreePlayer::node_exists(const StringName &p_name) const {
	return (node_map.has(p_name));
}

Error AnimationTreePlayer::node_rename(const StringName &p_node, const StringName &p_new_name) {
	if (p_new_name == p_node) {
		return OK;
	}
	ERR_FAIL_COND_V(!node_map.has(p_node), ERR_ALREADY_EXISTS);
	ERR_FAIL_COND_V(node_map.has(p_new_name), ERR_ALREADY_EXISTS);
	ERR_FAIL_COND_V(p_new_name == StringName(), ERR_INVALID_DATA);
	ERR_FAIL_COND_V(p_node == out_name, ERR_INVALID_DATA);
	ERR_FAIL_COND_V(p_new_name == out_name, ERR_INVALID_DATA);

	for (Map<StringName, NodeBase *>::Element *E = node_map.front(); E; E = E->next()) {
		NodeBase *nb = E->get();
		for (int i = 0; i < nb->inputs.size(); i++) {
			if (nb->inputs[i].node == p_node) {
				nb->inputs.write[i].node = p_new_name;
			}
		}
	}

	node_map[p_new_name] = node_map[p_node];
	node_map.erase(p_node);

	return OK;
}

String AnimationTreePlayer::get_configuration_warning() const {
	return TTR("This node has been deprecated. Use AnimationTree instead.");
}

void AnimationTreePlayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_node", "type", "id"), &AnimationTreePlayer::add_node);

	ClassDB::bind_method(D_METHOD("node_exists", "node"), &AnimationTreePlayer::node_exists);
	ClassDB::bind_method(D_METHOD("node_rename", "node", "new_name"), &AnimationTreePlayer::node_rename);

	ClassDB::bind_method(D_METHOD("node_get_type", "id"), &AnimationTreePlayer::node_get_type);
	ClassDB::bind_method(D_METHOD("node_get_input_count", "id"), &AnimationTreePlayer::node_get_input_count);
	ClassDB::bind_method(D_METHOD("node_get_input_source", "id", "idx"), &AnimationTreePlayer::node_get_input_source);

	ClassDB::bind_method(D_METHOD("animation_node_set_animation", "id", "animation"), &AnimationTreePlayer::animation_node_set_animation);
	ClassDB::bind_method(D_METHOD("animation_node_get_animation", "id"), &AnimationTreePlayer::animation_node_get_animation);

	ClassDB::bind_method(D_METHOD("animation_node_set_master_animation", "id", "source"), &AnimationTreePlayer::animation_node_set_master_animation);
	ClassDB::bind_method(D_METHOD("animation_node_get_master_animation", "id"), &AnimationTreePlayer::animation_node_get_master_animation);
	ClassDB::bind_method(D_METHOD("animation_node_get_position", "id"), &AnimationTreePlayer::animation_node_get_position);
	ClassDB::bind_method(D_METHOD("animation_node_set_filter_path", "id", "path", "enable"), &AnimationTreePlayer::animation_node_set_filter_path);

	ClassDB::bind_method(D_METHOD("oneshot_node_set_fadein_time", "id", "time_sec"), &AnimationTreePlayer::oneshot_node_set_fadein_time);
	ClassDB::bind_method(D_METHOD("oneshot_node_get_fadein_time", "id"), &AnimationTreePlayer::oneshot_node_get_fadein_time);

	ClassDB::bind_method(D_METHOD("oneshot_node_set_fadeout_time", "id", "time_sec"), &AnimationTreePlayer::oneshot_node_set_fadeout_time);
	ClassDB::bind_method(D_METHOD("oneshot_node_get_fadeout_time", "id"), &AnimationTreePlayer::oneshot_node_get_fadeout_time);

	ClassDB::bind_method(D_METHOD("oneshot_node_set_autorestart", "id", "enable"), &AnimationTreePlayer::oneshot_node_set_autorestart);
	ClassDB::bind_method(D_METHOD("oneshot_node_set_autorestart_delay", "id", "delay_sec"), &AnimationTreePlayer::oneshot_node_set_autorestart_delay);
	ClassDB::bind_method(D_METHOD("oneshot_node_set_autorestart_random_delay", "id", "rand_sec"), &AnimationTreePlayer::oneshot_node_set_autorestart_random_delay);

	ClassDB::bind_method(D_METHOD("oneshot_node_has_autorestart", "id"), &AnimationTreePlayer::oneshot_node_has_autorestart);
	ClassDB::bind_method(D_METHOD("oneshot_node_get_autorestart_delay", "id"), &AnimationTreePlayer::oneshot_node_get_autorestart_delay);
	ClassDB::bind_method(D_METHOD("oneshot_node_get_autorestart_random_delay", "id"), &AnimationTreePlayer::oneshot_node_get_autorestart_random_delay);

	ClassDB::bind_method(D_METHOD("oneshot_node_start", "id"), &AnimationTreePlayer::oneshot_node_start);
	ClassDB::bind_method(D_METHOD("oneshot_node_stop", "id"), &AnimationTreePlayer::oneshot_node_stop);
	ClassDB::bind_method(D_METHOD("oneshot_node_is_active", "id"), &AnimationTreePlayer::oneshot_node_is_active);
	ClassDB::bind_method(D_METHOD("oneshot_node_set_filter_path", "id", "path", "enable"), &AnimationTreePlayer::oneshot_node_set_filter_path);

	ClassDB::bind_method(D_METHOD("mix_node_set_amount", "id", "ratio"), &AnimationTreePlayer::mix_node_set_amount);
	ClassDB::bind_method(D_METHOD("mix_node_get_amount", "id"), &AnimationTreePlayer::mix_node_get_amount);

	ClassDB::bind_method(D_METHOD("blend2_node_set_amount", "id", "blend"), &AnimationTreePlayer::blend2_node_set_amount);
	ClassDB::bind_method(D_METHOD("blend2_node_get_amount", "id"), &AnimationTreePlayer::blend2_node_get_amount);
	ClassDB::bind_method(D_METHOD("blend2_node_set_filter_path", "id", "path", "enable"), &AnimationTreePlayer::blend2_node_set_filter_path);

	ClassDB::bind_method(D_METHOD("blend3_node_set_amount", "id", "blend"), &AnimationTreePlayer::blend3_node_set_amount);
	ClassDB::bind_method(D_METHOD("blend3_node_get_amount", "id"), &AnimationTreePlayer::blend3_node_get_amount);

	ClassDB::bind_method(D_METHOD("blend4_node_set_amount", "id", "blend"), &AnimationTreePlayer::blend4_node_set_amount);
	ClassDB::bind_method(D_METHOD("blend4_node_get_amount", "id"), &AnimationTreePlayer::blend4_node_get_amount);

	ClassDB::bind_method(D_METHOD("timescale_node_set_scale", "id", "scale"), &AnimationTreePlayer::timescale_node_set_scale);
	ClassDB::bind_method(D_METHOD("timescale_node_get_scale", "id"), &AnimationTreePlayer::timescale_node_get_scale);

	ClassDB::bind_method(D_METHOD("timeseek_node_seek", "id", "seconds"), &AnimationTreePlayer::timeseek_node_seek);

	ClassDB::bind_method(D_METHOD("transition_node_set_input_count", "id", "count"), &AnimationTreePlayer::transition_node_set_input_count);
	ClassDB::bind_method(D_METHOD("transition_node_get_input_count", "id"), &AnimationTreePlayer::transition_node_get_input_count);
	ClassDB::bind_method(D_METHOD("transition_node_delete_input", "id", "input_idx"), &AnimationTreePlayer::transition_node_delete_input);

	ClassDB::bind_method(D_METHOD("transition_node_set_input_auto_advance", "id", "input_idx", "enable"), &AnimationTreePlayer::transition_node_set_input_auto_advance);
	ClassDB::bind_method(D_METHOD("transition_node_has_input_auto_advance", "id", "input_idx"), &AnimationTreePlayer::transition_node_has_input_auto_advance);

	ClassDB::bind_method(D_METHOD("transition_node_set_xfade_time", "id", "time_sec"), &AnimationTreePlayer::transition_node_set_xfade_time);
	ClassDB::bind_method(D_METHOD("transition_node_get_xfade_time", "id"), &AnimationTreePlayer::transition_node_get_xfade_time);

	ClassDB::bind_method(D_METHOD("transition_node_set_current", "id", "input_idx"), &AnimationTreePlayer::transition_node_set_current);
	ClassDB::bind_method(D_METHOD("transition_node_get_current", "id"), &AnimationTreePlayer::transition_node_get_current);

	ClassDB::bind_method(D_METHOD("node_set_position", "id", "screen_position"), &AnimationTreePlayer::node_set_position);
	ClassDB::bind_method(D_METHOD("node_get_position", "id"), &AnimationTreePlayer::node_get_position);

	ClassDB::bind_method(D_METHOD("remove_node", "id"), &AnimationTreePlayer::remove_node);
	ClassDB::bind_method(D_METHOD("connect_nodes", "id", "dst_id", "dst_input_idx"), &AnimationTreePlayer::connect_nodes);
	ClassDB::bind_method(D_METHOD("are_nodes_connected", "id", "dst_id", "dst_input_idx"), &AnimationTreePlayer::are_nodes_connected);
	ClassDB::bind_method(D_METHOD("disconnect_nodes", "id", "dst_input_idx"), &AnimationTreePlayer::disconnect_nodes);

	ClassDB::bind_method(D_METHOD("set_active", "enabled"), &AnimationTreePlayer::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &AnimationTreePlayer::is_active);

	ClassDB::bind_method(D_METHOD("set_base_path", "path"), &AnimationTreePlayer::set_base_path);
	ClassDB::bind_method(D_METHOD("get_base_path"), &AnimationTreePlayer::get_base_path);

	ClassDB::bind_method(D_METHOD("set_master_player", "nodepath"), &AnimationTreePlayer::set_master_player);
	ClassDB::bind_method(D_METHOD("get_master_player"), &AnimationTreePlayer::get_master_player);

	ClassDB::bind_method(D_METHOD("get_node_list"), &AnimationTreePlayer::_get_node_list);

	ClassDB::bind_method(D_METHOD("set_animation_process_mode", "mode"), &AnimationTreePlayer::set_animation_process_mode);
	ClassDB::bind_method(D_METHOD("get_animation_process_mode"), &AnimationTreePlayer::get_animation_process_mode);

	ClassDB::bind_method(D_METHOD("advance", "delta"), &AnimationTreePlayer::advance);

	ClassDB::bind_method(D_METHOD("reset"), &AnimationTreePlayer::reset);

	ClassDB::bind_method(D_METHOD("recompute_caches"), &AnimationTreePlayer::recompute_caches);

	ADD_GROUP("Playback", "playback_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "playback_process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_animation_process_mode", "get_animation_process_mode");

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "master_player", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "AnimationPlayer"), "set_master_player", "get_master_player");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "base_path"), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");

	BIND_ENUM_CONSTANT(NODE_OUTPUT);
	BIND_ENUM_CONSTANT(NODE_ANIMATION);
	BIND_ENUM_CONSTANT(NODE_ONESHOT);
	BIND_ENUM_CONSTANT(NODE_MIX);
	BIND_ENUM_CONSTANT(NODE_BLEND2);
	BIND_ENUM_CONSTANT(NODE_BLEND3);
	BIND_ENUM_CONSTANT(NODE_BLEND4);
	BIND_ENUM_CONSTANT(NODE_TIMESCALE);
	BIND_ENUM_CONSTANT(NODE_TIMESEEK);
	BIND_ENUM_CONSTANT(NODE_TRANSITION);

	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_IDLE);
}

AnimationTreePlayer::AnimationTreePlayer() {
	active_list = nullptr;
	out = memnew(NodeOut);
	out_name = "out";
	out->pos = Point2(40, 40);
	node_map.insert(out_name, out);
	animation_process_mode = ANIMATION_PROCESS_IDLE;
	processing = false;
	active = false;
	dirty_caches = true;
	reset_request = true;
	last_error = CONNECT_INCOMPLETE;
	base_path = String("..");
}

AnimationTreePlayer::~AnimationTreePlayer() {
	while (node_map.size()) {
		memdelete(node_map.front()->get());
		node_map.erase(node_map.front());
	}
}
