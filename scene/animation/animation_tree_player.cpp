/*************************************************************************/
/*  animation_tree_player.cpp                                            */
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
#include "animation_tree_player.h"
#include "animation_player.h"


bool AnimationTreePlayer::_set(const StringName& p_name, const Variant& p_value) {

	if (String(p_name)=="base_path") {
		set_base_path(p_value);
		return true;
	}

	if (String(p_name)=="master_player") {
		set_master_player(p_value);
		return true;
	}

	if (String(p_name)!="data")
		return false;


	Dictionary data=p_value;

	Array nodes=data.get_valid("nodes");

	for(int i=0;i<nodes.size();i++) {

		Dictionary node = nodes[i];

		StringName id = node.get_valid("id");
		Point2 pos = node.get_valid("pos");

		NodeType nt=NODE_MAX;
		String type = node.get_valid("type");

		if (type=="output")
			nt=NODE_OUTPUT;
		else if (type=="animation")
			nt=NODE_ANIMATION;
		else if (type=="oneshot")
			nt=NODE_ONESHOT;
		else if (type=="mix")
			nt=NODE_MIX;
		else if (type=="blend2")
			nt=NODE_BLEND2;
		else if (type=="blend3")
			nt=NODE_BLEND3;
		else if (type=="blend4")
			nt=NODE_BLEND4;
		else if (type=="timescale")
			nt=NODE_TIMESCALE;
		else if (type=="timeseek")
			nt=NODE_TIMESEEK;
		else if (type=="transition")
			nt=NODE_TRANSITION;

		ERR_FAIL_COND_V(nt==NODE_MAX,false);

		if (nt!=NODE_OUTPUT)
			add_node(nt,id);
		node_set_pos(id,pos);


		switch(nt) {
			case NODE_OUTPUT: {

			} break;
			case NODE_ANIMATION: {

				if (node.has("from"))
					animation_node_set_master_animation(id,node.get_valid("from"));
				else
					animation_node_set_animation(id,node.get_valid("animation"));
			   } break;
			case NODE_ONESHOT: {

				oneshot_node_set_fadein_time(id,node.get_valid("fade_in"));
				oneshot_node_set_fadeout_time(id,node.get_valid("fade_out"));
				oneshot_node_set_mix_mode(id,node.get_valid("mix"));
				oneshot_node_set_autorestart(id,node.get_valid("autorestart"));
				oneshot_node_set_autorestart_delay(id,node.get_valid("autorestart_delay"));
				oneshot_node_set_autorestart_random_delay(id,node.get_valid("autorestart_random_delay"));
				Array filters= node.get_valid("filter");
				for(int i=0;i<filters.size();i++) {

					oneshot_node_set_filter_path(id,filters[i],true);
				}

			} break;
			case NODE_MIX: {
				mix_node_set_amount(id,node.get_valid("mix"));
			} break;
			case NODE_BLEND2: {
				blend2_node_set_amount(id,node.get_valid("blend"));
				Array filters= node.get_valid("filter");
				for(int i=0;i<filters.size();i++) {

					blend2_node_set_filter_path(id,filters[i],true);
				}
			} break;
			case NODE_BLEND3: {
				blend3_node_set_amount(id,node.get_valid("blend"));
			} break;
			case NODE_BLEND4: {
				blend4_node_set_amount(id,node.get_valid("blend"));
			} break;
			case NODE_TIMESCALE: {
				timescale_node_set_scale(id,node.get_valid("scale"));
			} break;
			case NODE_TIMESEEK: {
			} break;
			case NODE_TRANSITION: {

				transition_node_set_xfade_time(id,node.get_valid("xfade"));

				Array transitions = node.get_valid("transitions");
				transition_node_set_input_count(id,transitions.size());

				for(int x=0;x<transitions.size();x++) {

					Dictionary d =transitions[x];
					bool aa = d.get_valid("auto_advance");
					transition_node_set_input_auto_advance(id,x,aa);

				}

			} break;
			default: {};
		}

	}


	Array connections = data.get_valid("connections");
	ERR_FAIL_COND_V(connections.size()%3,false);

	int cc=connections.size()/3;

	for(int i=0;i<cc;i++) {

		StringName src = connections[i*3+0];
		StringName dst = connections[i*3+1];
		int dst_in = connections[i*3+2];
		connect(src,dst,dst_in);
	}

	set_active(data.get_valid("active"));
	set_master_player(data.get_valid("master"));

	return true;

}

bool AnimationTreePlayer::_get(const StringName& p_name,Variant &r_ret) const {

	if (String(p_name)=="base_path") {
		r_ret=base_path;
		return true;
	}

	if (String(p_name)=="master_player") {
		r_ret=master;
		return true;
	}

	if (String(p_name)!="data")
		return false;

	Dictionary data;

	Array nodes;

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		NodeBase *n = node_map[E->key()];

		Dictionary node;
		node["id"]=E->key();
		node["pos"]=n->pos;

		switch(n->type) {
			case NODE_OUTPUT: node["type"]= "output"; break;
			case NODE_ANIMATION: node["type"]= "animation"; break;
			case NODE_ONESHOT: node["type"]= "oneshot"; break;
			case NODE_MIX: node["type"]= "mix"; break;
			case NODE_BLEND2: node["type"]= "blend2"; break;
			case NODE_BLEND3: node["type"]= "blend3"; break;
			case NODE_BLEND4: node["type"]= "blend4"; break;
			case NODE_TIMESCALE: node["type"]= "timescale"; break;
			case NODE_TIMESEEK: node["type"]= "timeseek"; break;
			case NODE_TRANSITION: node["type"]= "transition"; break;
			default: node["type"]= ""; break;
		}

		switch(n->type) {
			case NODE_OUTPUT: {

			} break;
			case NODE_ANIMATION: {
				AnimationNode *an = static_cast<AnimationNode*>(n);
				if (master!=NodePath() && an->from!="") {
					node["from"]=an->from;
				} else {
					node["animation"]=an->animation;
				}
			   } break;
			case NODE_ONESHOT: {
				OneShotNode *osn = static_cast<OneShotNode*>(n);
				node["fade_in"]=osn->fade_in;
				node["fade_out"]=osn->fade_out;
				node["mix"]=osn->mix;
				node["autorestart"]=osn->autorestart;
				node["autorestart_delay"]=osn->autorestart_delay;
				node["autorestart_random_delay"]=osn->autorestart_random_delay;

				Array k;
				List<NodePath> keys;
				osn->filter.get_key_list(&keys);
				k.resize(keys.size());
				int i=0;
				for(List<NodePath>::Element *E=keys.front();E;E=E->next()) {
					k[i++]=E->get();
				}
				node["filter"]=k;

			} break;
			case NODE_MIX: {
				MixNode *mn = static_cast<MixNode*>(n);
				node["mix"]=mn->amount;
			} break;
			case NODE_BLEND2: {
				Blend2Node *bn = static_cast<Blend2Node*>(n);
				node["blend"]=bn->value;
				Array k;
				List<NodePath> keys;
				bn->filter.get_key_list(&keys);
				k.resize(keys.size());
				int i=0;
				for(List<NodePath>::Element *E=keys.front();E;E=E->next()) {
					k[i++]=E->get();
				}
				node["filter"]=k;

			} break;
			case NODE_BLEND3: {
				Blend3Node *bn = static_cast<Blend3Node*>(n);
				node["blend"]=bn->value;
			} break;
			case NODE_BLEND4: {
				Blend4Node *bn = static_cast<Blend4Node*>(n);
				node["blend"]=bn->value;

			} break;
			case NODE_TIMESCALE: {
				TimeScaleNode *tsn = static_cast<TimeScaleNode*>(n);
				node["scale"]=tsn->scale;
			} break;
			case NODE_TIMESEEK: {
			} break;
			case NODE_TRANSITION: {

				TransitionNode *tn = static_cast<TransitionNode*>(n);
				node["xfade"]=tn->xfade;
				Array transitions;

				for(int i=0;i<tn->input_data.size();i++) {

					Dictionary d;
					d["auto_advance"]=tn->input_data[i].auto_advance;
					transitions.push_back(d);

				}

				node["transitions"]=transitions;

			} break;
			default: {};
		}

		nodes.push_back(node);
	}

	data["nodes"]=nodes;
	//connectiosn

	List<Connection> connections;
	get_connection_list(&connections);
	Array connections_arr;
	connections_arr.resize(connections.size()*3);

	int idx=0;
	for (List<Connection>::Element *E=connections.front();E;E=E->next()) {

		connections_arr.set(idx+0,E->get().src_node);
		connections_arr.set(idx+1,E->get().dst_node);
		connections_arr.set(idx+2,E->get().dst_input);

		idx+=3;
	}

	data["connections"]=connections_arr;
	data["active"]=active;
	data["master"]=master;

	r_ret=data;

	return true;

}

void AnimationTreePlayer::_get_property_list( List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo(Variant::NODE_PATH,"base_path" ) );
	p_list->push_back( PropertyInfo(Variant::NODE_PATH,"master_player" ) );
	p_list->push_back( PropertyInfo(Variant::DICTIONARY,"data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_NETWORK) );
}


void AnimationTreePlayer::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_READY: {
			dirty_caches=true;
			if (master!=NodePath()) {
				_update_sources();
			}
		} break;
		case NOTIFICATION_PROCESS: {
			_process_animation();
		} break;
	}

}


float AnimationTreePlayer::_process_node(const StringName& p_node,AnimationNode **r_prev_anim,float p_weight, float p_time, bool p_seek,const HashMap<NodePath,bool> *p_filter, float p_reverse_weight) {

	ERR_FAIL_COND_V(!node_map.has(p_node), 0);
	NodeBase *nb=node_map[p_node];

	//transform to seconds...


	switch(nb->type) {

		case NODE_OUTPUT: {

			NodeOut *on = static_cast<NodeOut*>(nb);
			return _process_node(on->inputs[0].node,r_prev_anim,p_weight,p_time,p_seek);

		} break;
		case NODE_ANIMATION: {

			AnimationNode *an = static_cast<AnimationNode*>(nb);

			float rem = 0;
			if (!an->animation.is_null()) {

		//		float pos = an->time;
//				float delta = p_time;

	//			const Animation *a = an->animation.operator->();

				if (p_seek) {
					an->time=p_time;
					an->step=0;
				} else {
					an->time+=p_time;
					an->step=p_time;
				}

				float anim_size = an->animation->get_length();

				if (an->animation->has_loop()) {

					if (anim_size)
						an->time=Math::fposmod(an->time,anim_size);

				} else if (an->time > anim_size) {

					an->time=anim_size;
				}

				an->skip=true;
				for (List<AnimationNode::TrackRef>::Element *E=an->tref.front();E;E=E->next()) {

					if (p_filter && p_filter->has(an->animation->track_get_path(E->get().local_track))) {

						if (p_reverse_weight<0)
							E->get().weight=0;
						else
							E->get().weight=p_reverse_weight;

					} else {
						E->get().weight=p_weight;
					}
					if (E->get().weight>CMP_EPSILON)
						an->skip=false;
				}

				rem = anim_size - an->time;

			}


			if (!(*r_prev_anim))
				active_list=an;
			else 
				(*r_prev_anim)->next=an;

			an->next=NULL;
			*r_prev_anim=an;

			return rem;


		} break;
		case NODE_ONESHOT: {

			OneShotNode *osn = static_cast<OneShotNode*>(nb);

			if (!osn->active) {
				//make it as if this node doesn't exist, pass input 0 by.
				return _process_node(osn->inputs[0].node,r_prev_anim,p_weight,p_time,p_seek,p_filter,p_reverse_weight);
			}

			if (p_seek)
				osn->time=p_time;
			if (osn->start)
				osn->time=0;

			float blend;

			if (osn->time<osn->fade_in) {

				if (osn->fade_in>0)
					blend = osn->time/osn->fade_in;
				else
					blend=0; //wtf

			} else if (!osn->start && osn->remaining<osn->fade_out) {

				if (osn->fade_out)
					blend=(osn->remaining/osn->fade_out);
				else
					blend=1.0;
			} else
				blend=1.0;

			float main_rem;
			float os_rem;

			if (!osn->filter.empty()) {

				main_rem = _process_node(osn->inputs[0].node,r_prev_anim,(osn->mix?p_weight:p_weight*(1.0-blend)),p_time,p_seek,&osn->filter,p_weight);
				os_rem = _process_node(osn->inputs[1].node,r_prev_anim,p_weight*blend,p_time,p_seek,&osn->filter,-1);

			} else {

				main_rem = _process_node(osn->inputs[0].node,r_prev_anim,(osn->mix?p_weight:p_weight*(1.0-blend)),p_time,p_seek);
				os_rem = _process_node(osn->inputs[1].node,r_prev_anim,p_weight*blend,p_time,p_seek);
			}

			if (osn->start) {
				osn->remaining=os_rem;
				osn->start=false;
			}

			if (!p_seek) {
				osn->time+=p_time;
				osn->remaining-=p_time;
				if (osn->remaining<0)
					osn->active=false;
			}

			return MAX(main_rem,osn->remaining);
		} break;
		case NODE_MIX: {
			MixNode *mn = static_cast<MixNode*>(nb);


			float rem = _process_node(mn->inputs[0].node,r_prev_anim,p_weight,p_time,p_seek,p_filter,p_reverse_weight);
			_process_node(mn->inputs[1].node,r_prev_anim,p_weight*mn->amount,p_time,p_seek,p_filter,p_reverse_weight);
			return rem;

		} break;
		case NODE_BLEND2: {

			Blend2Node *bn = static_cast<Blend2Node*>(nb);

			float rem;
			if (!bn->filter.empty()) {

				rem = _process_node(bn->inputs[0].node,r_prev_anim,p_weight*(1.0-bn->value),p_time,p_seek,&bn->filter,p_weight);
				_process_node(bn->inputs[1].node,r_prev_anim,p_weight*bn->value,p_time,p_seek,&bn->filter,-1);

			} else {
				rem = _process_node(bn->inputs[0].node,r_prev_anim,p_weight*(1.0-bn->value),p_time,p_seek,p_filter,p_reverse_weight*(1.0-bn->value));
				_process_node(bn->inputs[1].node,r_prev_anim,p_weight*bn->value,p_time,p_seek,p_filter,p_reverse_weight*bn->value);
			}

			return rem;
		} break;
		case NODE_BLEND3: {
			Blend3Node *bn = static_cast<Blend3Node*>(nb);

			float rem;

			if (bn->value==0) {
				rem = _process_node(bn->inputs[1].node,r_prev_anim,p_weight,p_time,p_seek,p_filter,p_reverse_weight);
			} else if (bn->value>0) {

				rem = _process_node(bn->inputs[1].node,r_prev_anim,p_weight*(1.0-bn->value),p_time,p_seek,p_filter,p_reverse_weight*(1.0-bn->value));
				_process_node(bn->inputs[2].node,r_prev_anim,p_weight*bn->value,p_time,p_seek,p_filter,p_reverse_weight*bn->value);

			} else {

				rem = _process_node(bn->inputs[1].node,r_prev_anim,p_weight*(1.0+bn->value),p_time,p_seek,p_filter,p_reverse_weight*(1.0+bn->value));
				_process_node(bn->inputs[0].node,r_prev_anim,p_weight*-bn->value,p_time,p_seek,p_filter,p_reverse_weight*-bn->value);
			}

			return rem;
		} break;
		case NODE_BLEND4: {
			Blend4Node *bn = static_cast<Blend4Node*>(nb);

			float rem = _process_node(bn->inputs[0].node,r_prev_anim,p_weight*(1.0-bn->value.x),p_time,p_seek,p_filter,p_reverse_weight*(1.0-bn->value.x));
			_process_node(bn->inputs[1].node,r_prev_anim,p_weight*bn->value.x,p_time,p_seek,p_filter,p_reverse_weight*bn->value.x);
			float rem2 = _process_node(bn->inputs[2].node,r_prev_anim,p_weight*(1.0-bn->value.y),p_time,p_seek,p_filter,p_reverse_weight*(1.0-bn->value.y));
			_process_node(bn->inputs[3].node,r_prev_anim,p_weight*bn->value.y,p_time,p_seek,p_filter,p_reverse_weight*bn->value.y);

			return MAX(rem,rem2);

		} break;
		case NODE_TIMESCALE: {
			TimeScaleNode *tsn = static_cast<TimeScaleNode*>(nb);
			if (p_seek)
				return _process_node(tsn->inputs[0].node,r_prev_anim,p_weight,p_time,true,p_filter,p_reverse_weight);
			else
				return _process_node(tsn->inputs[0].node,r_prev_anim,p_weight,p_time*tsn->scale,false,p_filter,p_reverse_weight);

		} break;
		case NODE_TIMESEEK: {

			TimeSeekNode *tsn = static_cast<TimeSeekNode*>(nb);
			if (tsn->seek_pos>=0) {

				float res = _process_node(tsn->inputs[0].node,r_prev_anim,p_weight,tsn->seek_pos,true,p_filter,p_reverse_weight);
				tsn->seek_pos=-1;
				return res;

			} else
				return _process_node(tsn->inputs[0].node,r_prev_anim,p_weight,p_time,p_seek);

		}   break;
		case NODE_TRANSITION: {

			TransitionNode *tn = static_cast<TransitionNode*>(nb);

			if (tn->prev<0) {

				float rem = _process_node(tn->inputs[tn->current].node,r_prev_anim,p_weight,p_time,p_seek,p_filter,p_reverse_weight);
				if (p_seek)
					tn->time=p_time;
				else
					tn->time+=p_time;

				if (tn->input_data[tn->current].auto_advance && rem < tn->xfade) {

					tn->prev=tn->current;
					tn->current++;
					if (tn->current>=tn->inputs.size())
						tn->current=0;
					tn->prev_xfading=tn->xfade;
					tn->prev_time=tn->time;
					tn->time=0;
					tn->switched=true;
				}


				return rem;
			} else {


				float blend = tn->xfade? (tn->prev_xfading/tn->xfade) : 1;

				float rem;

				if (!p_seek && tn->switched) { //just switched

					rem = _process_node(tn->inputs[tn->current].node,r_prev_anim,p_weight*(1.0-blend),0,true,p_filter,p_reverse_weight*(1.0-blend));
				} else {

					rem = _process_node(tn->inputs[tn->current].node,r_prev_anim,p_weight*(1.0-blend),p_time,p_seek,p_filter,p_reverse_weight*(1.0-blend));

				}

				tn->switched=false;

				//if (!p_seek)


				if (p_seek) {
					_process_node(tn->inputs[tn->prev].node,r_prev_anim,p_weight*blend,0,false,p_filter,p_reverse_weight*blend);
					tn->time=p_time;
				} else {
					_process_node(tn->inputs[tn->prev].node,r_prev_anim,p_weight*blend,p_time,false,p_filter,p_reverse_weight*blend);
					tn->time+=p_time;
					tn->prev_xfading-=p_time;
					if (tn->prev_xfading<0) {

						tn->prev=-1;
					}

				}

				return rem;
			}


		} break;
		default: {}
	}



	return 0;
}


void AnimationTreePlayer::_process_animation() {

	if (!active)
		return;

	if (last_error!=CONNECT_OK)
		return;

	if (dirty_caches)
		_recompute_caches();


	active_list=NULL;
	AnimationNode *prev=NULL;

	if (reset_request) {
		_process_node(out_name,&prev, 1.0, 0, true );
		reset_request=false;
	} else
		_process_node(out_name,&prev, 1.0, get_process_delta_time(), false );

	if (dirty_caches) {
		//some animation changed.. ignore this pass
		return;
	}

	//update the tracks..



	/* STEP 1 CLEAR TRACKS */

	for(TrackMap::Element *E=track_map.front();E;E=E->next()) {

		Track &t = E->get();

		t.loc.zero();
		t.rot=Quat();
		t.scale.x=0;
		t.scale.y=0;
		t.scale.z=0;
	}


	/* STEP 2 PROCESS ANIMATIONS */

	AnimationNode *anim_list=active_list;
	Quat empty_rot;


	int total = 0;
	while(anim_list) {

		if (!anim_list->animation.is_null() && !anim_list->skip) {
			++total;
			//check if animation is meaningful
			Animation *a = anim_list->animation.operator->();

			for(List<AnimationNode::TrackRef>::Element *E=anim_list->tref.front();E;E=E->next()) {


				AnimationNode::TrackRef &tr = E->get();
				if (tr.track==NULL || tr.local_track<0 || tr.weight < CMP_EPSILON)
					continue;

				float blend=tr.weight;

				switch(a->track_get_type(tr.local_track)) {
					case Animation::TYPE_TRANSFORM: { ///< Transform a node or a bone.

						Vector3 loc;
						Quat rot;
						Vector3 scale;
						a->transform_track_interpolate(tr.local_track,anim_list->time,&loc,&rot,&scale);

						tr.track->loc+=loc*blend;

						scale.x-=1.0;
						scale.y-=1.0;
						scale.z-=1.0;
						tr.track->scale+=scale*blend;

						tr.track->rot = tr.track->rot * empty_rot.slerp(rot,blend);


					} break;
					case Animation::TYPE_VALUE: { ///< Set a value in a property, can be interpolated.

						if (a->value_track_is_continuous(tr.local_track)) {
							Variant value = a->value_track_interpolate(tr.local_track,anim_list->time);
							tr.track->node->set(tr.track->property,value);
						} else {

							List<int> indices;
							a->value_track_get_key_indices(tr.local_track,anim_list->time,anim_list->step,&indices);
							for(List<int>::Element *E=indices.front();E;E=E->next()) {

								Variant value = a->track_get_key_value(tr.local_track,E->get());
								tr.track->node->set(tr.track->property,value);
							}
						}
					} break;
					case Animation::TYPE_METHOD: { ///< Call any method on a specific node.

						List<int> indices;
						a->method_track_get_key_indices(tr.local_track,anim_list->time,anim_list->step,&indices);
						for(List<int>::Element *E=indices.front();E;E=E->next()) {

							StringName method = a->method_track_get_name(tr.local_track,E->get());
							Vector<Variant> args=a->method_track_get_params(tr.local_track,E->get());
							ERR_CONTINUE(args.size()!=VARIANT_ARG_MAX);
							tr.track->node->call(method,args[0],args[1],args[2],args[3],args[4]);
						}
					} break;
				}

			}
		}

		anim_list=anim_list->next;
	}

	/* STEP 3 APPLY TRACKS */

	for(TrackMap::Element *E=track_map.front();E;E=E->next()) {

		Track &t = E->get();

		if (!t.node)
			continue;
		//if (E->get()->t.type!=Animation::TYPE_TRANSFORM)
		//	continue;

		Transform xform;
		xform.basis=t.rot;
		xform.origin=t.loc;

		t.scale.x+=1.0;
		t.scale.y+=1.0;
		t.scale.z+=1.0;
		xform.basis.scale(t.scale);

		if (t.bone_idx>=0) {
			if (t.skeleton)
				t.skeleton->set_bone_pose(t.bone_idx,xform);

		} else if (t.spatial) {

			t.spatial->set_transform(xform);
		}
	}



}


void AnimationTreePlayer::add_node(NodeType p_type, const StringName& p_node) {

	ERR_FAIL_COND( p_type == NODE_OUTPUT );
	ERR_FAIL_COND( node_map.has(p_node));

	NodeBase *n=NULL;

	switch(p_type) {

		case NODE_ANIMATION: {

			n = memnew( AnimationNode );
		} break;
		case NODE_ONESHOT: {

			n = memnew( OneShotNode );

		} break;
		case NODE_MIX: {
			n = memnew( MixNode );

		} break;
		case NODE_BLEND2: {
			n = memnew( Blend2Node );

		} break;
		case NODE_BLEND3: {
			n = memnew( Blend3Node );

		} break;
		case NODE_BLEND4: {
			n = memnew( Blend4Node );

		} break;
		case NODE_TIMESCALE: {
			n = memnew( TimeScaleNode );


		} break;
		case NODE_TIMESEEK: {
			n = memnew( TimeSeekNode );

		} break;
		case NODE_TRANSITION: {
			n = memnew( TransitionNode );


		} break;
		default: {}
	}

	//n->name+=" "+itos(p_node);
	node_map[p_node]=n;
}


StringName AnimationTreePlayer::node_get_input_source(const StringName& p_node,int p_input) const {

	ERR_FAIL_COND_V(!node_map.has(p_node),StringName());
	ERR_FAIL_INDEX_V( p_input,node_map[p_node]->inputs.size(),StringName() );
	return node_map[p_node]->inputs[p_input].node;

}


int AnimationTreePlayer::node_get_input_count(const StringName& p_node) const {

	ERR_FAIL_COND_V(!node_map.has(p_node),-1);
	return node_map[p_node]->inputs.size();

}
#define GET_NODE( m_type, m_cast )\
	ERR_FAIL_COND(!node_map.has(p_node));\
	ERR_EXPLAIN("Invalid parameter for node type.");\
	ERR_FAIL_COND(node_map[p_node]->type!=m_type);\
	m_cast *n = static_cast<m_cast*>( node_map[p_node] );\



void AnimationTreePlayer::animation_node_set_animation(const StringName& p_node,const Ref<Animation>& p_animation) {

	GET_NODE( NODE_ANIMATION, AnimationNode );
	n->animation=p_animation;
	dirty_caches=true;


}

void AnimationTreePlayer::animation_node_set_master_animation(const StringName& p_node,const String& p_master_animation) {

	GET_NODE( NODE_ANIMATION, AnimationNode );
	n->from=p_master_animation;
	dirty_caches=true;
	if (master!=NodePath())
		_update_sources();


}

void AnimationTreePlayer::oneshot_node_set_fadein_time(const StringName& p_node,float p_time) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->fade_in=p_time;

}


void AnimationTreePlayer::oneshot_node_set_fadeout_time(const StringName& p_node,float p_time) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->fade_out=p_time;


}

void AnimationTreePlayer::oneshot_node_set_mix_mode(const StringName& p_node,bool p_mix) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->mix=p_mix;
}


void AnimationTreePlayer::oneshot_node_set_autorestart(const StringName& p_node,bool p_active) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->autorestart=p_active;

}

void AnimationTreePlayer::oneshot_node_set_autorestart_delay(const StringName& p_node,float p_time) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->autorestart_delay=p_time;

}
void AnimationTreePlayer::oneshot_node_set_autorestart_random_delay(const StringName& p_node,float p_time) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->autorestart_random_delay=p_time;

}

void AnimationTreePlayer::oneshot_node_start(const StringName& p_node) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->active=true;
	n->start=true;

}


void AnimationTreePlayer::oneshot_node_stop(const StringName& p_node) {

	GET_NODE( NODE_ONESHOT, OneShotNode );
	n->active=false;

}


void AnimationTreePlayer::oneshot_node_set_filter_path(const StringName& p_node,const NodePath& p_filter,bool p_enable) {

	GET_NODE( NODE_ONESHOT, OneShotNode );

	if (p_enable)
		n->filter[p_filter]=true;
	else
		n->filter.erase(p_filter);

}

void AnimationTreePlayer::oneshot_node_set_get_filtered_paths(const StringName& p_node,List<NodePath> *r_paths) const{

	GET_NODE( NODE_ONESHOT, OneShotNode );

	n->filter.get_key_list(r_paths);
}


void AnimationTreePlayer::mix_node_set_amount(const StringName& p_node,float p_amount) {

	GET_NODE( NODE_MIX, MixNode );
	n->amount=p_amount;

}


void AnimationTreePlayer::blend2_node_set_amount(const StringName& p_node,float p_amount) {

	GET_NODE( NODE_BLEND2, Blend2Node );
	n->value=p_amount;

}

void AnimationTreePlayer::blend2_node_set_filter_path(const StringName& p_node,const NodePath& p_filter,bool p_enable) {

	GET_NODE( NODE_BLEND2, Blend2Node );

	if (p_enable)
		n->filter[p_filter]=true;
	else
		n->filter.erase(p_filter);

}

void AnimationTreePlayer::blend2_node_set_get_filtered_paths(const StringName& p_node,List<NodePath> *r_paths) const{

	GET_NODE( NODE_BLEND2, Blend2Node );

	n->filter.get_key_list(r_paths);
}


void AnimationTreePlayer::blend3_node_set_amount(const StringName& p_node,float p_amount) {

	GET_NODE( NODE_BLEND3, Blend3Node );
	n->value=p_amount;

}
void AnimationTreePlayer::blend4_node_set_amount(const StringName& p_node,const Vector2& p_amount) {

	GET_NODE( NODE_BLEND4, Blend4Node );
	n->value=p_amount;

}
void AnimationTreePlayer::timescale_node_set_scale(const StringName& p_node,float p_scale) {


	GET_NODE( NODE_TIMESCALE, TimeScaleNode );
	n->scale=p_scale;

}
void AnimationTreePlayer::timeseek_node_seek(const StringName& p_node,float p_pos) {


	GET_NODE( NODE_TIMESEEK, TimeSeekNode );
	n->seek_pos=p_pos;

}
void AnimationTreePlayer::transition_node_set_input_count(const StringName& p_node, int p_inputs) {


	GET_NODE( NODE_TRANSITION, TransitionNode );
	ERR_FAIL_COND(p_inputs<1);

	n->inputs.resize(p_inputs);
	n->input_data.resize(p_inputs);
	last_error=_cycle_test(out_name);

}
void AnimationTreePlayer::transition_node_set_input_auto_advance(const StringName& p_node, int p_input,bool p_auto_advance) {

	GET_NODE( NODE_TRANSITION, TransitionNode );
	ERR_FAIL_INDEX(p_input,n->input_data.size());

	n->input_data[p_input].auto_advance=p_auto_advance;

}
void AnimationTreePlayer::transition_node_set_xfade_time(const StringName& p_node, float p_time) {


	GET_NODE( NODE_TRANSITION, TransitionNode );
	n->xfade=p_time;
}


void AnimationTreePlayer::transition_node_set_current(const StringName& p_node, int p_current) {

	GET_NODE( NODE_TRANSITION, TransitionNode );
	ERR_FAIL_INDEX(p_current,n->inputs.size());

	if (n->current==p_current)
		return;

	n->prev=n->current;
	n->prev_xfading=n->xfade;
	n->prev_time=n->time;
	n->time=0;
	n->current=p_current;

}


void AnimationTreePlayer::node_set_pos(const StringName& p_node, const Vector2& p_pos) {

	ERR_FAIL_COND(!node_map.has(p_node));
	node_map[p_node]->pos=p_pos;

}

AnimationTreePlayer::NodeType AnimationTreePlayer::node_get_type(const StringName& p_node) const {

	ERR_FAIL_COND_V(!node_map.has(p_node),NODE_OUTPUT);
	return node_map[p_node]->type;

}
Point2 AnimationTreePlayer::node_get_pos(const StringName& p_node) const {

	ERR_FAIL_COND_V(!node_map.has(p_node),Point2());
	return node_map[p_node]->pos;


}

#define GET_NODE_V( m_type, m_cast, m_ret )\
	ERR_FAIL_COND_V(!node_map.has(p_node),m_ret);\
	ERR_EXPLAIN("Invalid parameter for node type.");\
	ERR_FAIL_COND_V(node_map[p_node]->type!=m_type,m_ret);\
	m_cast *n = static_cast<m_cast*>( node_map[p_node] );\

Ref<Animation> AnimationTreePlayer::animation_node_get_animation(const StringName& p_node) const {

	GET_NODE_V(NODE_ANIMATION, AnimationNode, Ref<Animation>());
	return n->animation;

}

String AnimationTreePlayer::animation_node_get_master_animation(const StringName& p_node) const {

	GET_NODE_V(NODE_ANIMATION, AnimationNode, String());
	return n->from;

}

float AnimationTreePlayer::oneshot_node_get_fadein_time(const StringName& p_node) const {


	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->fade_in;

}

float AnimationTreePlayer::oneshot_node_get_fadeout_time(const StringName& p_node) const {

	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->fade_out;

}

bool AnimationTreePlayer::oneshot_node_get_mix_mode(const StringName& p_node) const {

	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->mix;

}
bool AnimationTreePlayer::oneshot_node_has_autorestart(const StringName& p_node) const {

	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->autorestart;

}
float AnimationTreePlayer::oneshot_node_get_autorestart_delay(const StringName& p_node) const {

	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->autorestart_delay;

}
float AnimationTreePlayer::oneshot_node_get_autorestart_random_delay(const StringName& p_node) const {

	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->autorestart_random_delay;

}

bool AnimationTreePlayer::oneshot_node_is_active(const StringName& p_node) const {


	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->active;

}

bool AnimationTreePlayer::oneshot_node_is_path_filtered(const StringName& p_node,const NodePath& p_path) const {

	GET_NODE_V(NODE_ONESHOT, OneShotNode, 0 );
	return n->filter.has(p_path);
}


float AnimationTreePlayer::mix_node_get_amount(const StringName& p_node) const {

	GET_NODE_V(NODE_MIX, MixNode, 0 );
	return n->amount;

}
float AnimationTreePlayer::blend2_node_get_amount(const StringName& p_node) const {

	GET_NODE_V(NODE_BLEND2, Blend2Node, 0 );
	return n->value;

}

bool AnimationTreePlayer::blend2_node_is_path_filtered(const StringName& p_node,const NodePath& p_path) const {

	GET_NODE_V(NODE_BLEND2, Blend2Node, 0 );
	return n->filter.has(p_path);
}

float AnimationTreePlayer::blend3_node_get_amount(const StringName& p_node) const  {


	GET_NODE_V(NODE_BLEND3, Blend3Node, 0 );
	return n->value;

}
Vector2 AnimationTreePlayer::blend4_node_get_amount(const StringName& p_node) const {

	GET_NODE_V(NODE_BLEND4, Blend4Node, Vector2() );
	return n->value;

}

float AnimationTreePlayer::timescale_node_get_scale(const StringName& p_node) const {

	GET_NODE_V(NODE_TIMESCALE, TimeScaleNode, 0 );
	return n->scale;

}

void AnimationTreePlayer::transition_node_delete_input(const StringName& p_node, int p_input) {

	GET_NODE(NODE_TRANSITION, TransitionNode);
	ERR_FAIL_INDEX(p_input,n->inputs.size());

	if (n->inputs.size()<=1)
		return;


	n->inputs.remove(p_input);
	n->input_data.remove(p_input);
	last_error=_cycle_test(out_name);
}


int AnimationTreePlayer::transition_node_get_input_count(const StringName& p_node) const {

	GET_NODE_V(NODE_TRANSITION, TransitionNode, 0 );
	return n->inputs.size();
}

bool AnimationTreePlayer::transition_node_has_input_auto_advance(const StringName& p_node, int p_input) const {

	GET_NODE_V(NODE_TRANSITION, TransitionNode, false );
	ERR_FAIL_INDEX_V(p_input,n->inputs.size(),false);
	return n->input_data[p_input].auto_advance;

}
float AnimationTreePlayer::transition_node_get_xfade_time(const StringName& p_node) const {

	GET_NODE_V(NODE_TRANSITION, TransitionNode, 0 );
	return n->xfade;

}

int AnimationTreePlayer::transition_node_get_current(const StringName& p_node) const {

	GET_NODE_V(NODE_TRANSITION, TransitionNode, -1 );
	return n->current;

}

	/*misc  */
void AnimationTreePlayer::get_node_list(List<StringName> *p_node_list) const {

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		p_node_list->push_back( E->key() );
	}
}

void AnimationTreePlayer::remove_node(const StringName& p_node) {

	ERR_FAIL_COND( !node_map.has(p_node) );
	ERR_EXPLAIN("Node 0 (output) can't be removed.");
	ERR_FAIL_COND( p_node == out_name );

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		NodeBase *nb = E->get();
		for(int i=0;i<nb->inputs.size();i++) {

			if (nb->inputs[i].node==p_node)
				nb->inputs[i].node=StringName();
		}
	}

	node_map.erase(p_node);

	// compute last error again, just in case
	last_error=_cycle_test(out_name);
	dirty_caches=true;
}


AnimationTreePlayer::ConnectError AnimationTreePlayer::_cycle_test(const StringName& p_at_node)  {

	ERR_FAIL_COND_V(!node_map.has(p_at_node), CONNECT_INCOMPLETE);

	NodeBase *nb = node_map[p_at_node];
	if (nb->cycletest)
		return CONNECT_CYCLE;


	nb->cycletest=true;

	for(int i=0;i<nb->inputs.size();i++) {
		if (nb->inputs[i].node==StringName())
			return CONNECT_INCOMPLETE;

		ConnectError _err = _cycle_test(nb->inputs[i].node);
		if (_err)
			return _err;
	}

	return CONNECT_OK;
}


Error AnimationTreePlayer::connect(const StringName& p_src_node,const StringName& p_dst_node, int p_dst_input) {

	ERR_FAIL_COND_V( !node_map.has(p_src_node) , ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V( !node_map.has(p_dst_node) , ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V( p_src_node==p_dst_node , ERR_INVALID_PARAMETER);

//	NodeBase *src = node_map[p_src_node];
	NodeBase *dst = node_map[p_dst_node];
	ERR_FAIL_INDEX_V( p_dst_input, dst->inputs.size(), ERR_INVALID_PARAMETER);

//	int oldval = dst->inputs[p_dst_input].node;

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		NodeBase *nb = E->get();
		for(int i=0;i<nb->inputs.size();i++) {

			if (nb->inputs[i].node==p_src_node)
				nb->inputs[i].node=StringName();
		}
	}

	dst->inputs[p_dst_input].node=p_src_node;

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		NodeBase *nb = E->get();
		nb->cycletest=false;
	}

	last_error=_cycle_test(out_name);
	if (last_error) {

		if (last_error==CONNECT_INCOMPLETE)
			return ERR_UNCONFIGURED;
		else if (last_error==CONNECT_CYCLE)
			return ERR_CYCLIC_LINK;
	}
	dirty_caches=true;
	return OK;
}

bool AnimationTreePlayer::is_connected(const StringName& p_src_node,const StringName& p_dst_node, int p_dst_input) const {

	ERR_FAIL_COND_V( !node_map.has(p_src_node) , false);
	ERR_FAIL_COND_V( !node_map.has(p_dst_node) , false);
	ERR_FAIL_COND_V( p_src_node==p_dst_node , false);

	NodeBase *dst = node_map[p_dst_node];

	return dst->inputs[p_dst_input].node==p_src_node;

}

void AnimationTreePlayer::disconnect(const StringName& p_node, int p_input) {

	ERR_FAIL_COND( !node_map.has(p_node));

	NodeBase *dst = node_map[p_node];
	ERR_FAIL_INDEX(p_input,dst->inputs.size());
	dst->inputs[p_input].node=StringName();
	last_error=CONNECT_INCOMPLETE;
	dirty_caches=true;
}


void AnimationTreePlayer::get_connection_list( List<Connection> *p_connections) const {

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		NodeBase *nb = E->get();
		for(int i=0;i<nb->inputs.size();i++) {

			if (nb->inputs[i].node!=StringName()) {
				Connection c;
				c.src_node=nb->inputs[i].node;
				c.dst_node=E->key();
				c.dst_input=i;
				p_connections->push_back(c);
			}
		}
	}
}

AnimationTreePlayer::Track* AnimationTreePlayer::_find_track(const NodePath& p_path) {

	Node *parent=get_node(base_path);
	ERR_FAIL_COND_V(!parent,NULL);

	Node *child=parent->get_node(p_path);
	if (!child) {
		String err = "Animation track references unknown Node: '"+String(p_path)+"'.";
		WARN_PRINT(err.ascii().get_data());
		return NULL;
	}

	ObjectID id=child->get_instance_ID();
	StringName property;
	int bone_idx=-1;

	if (p_path.get_property()) {

		if (child->cast_to<Skeleton>())
			bone_idx = child->cast_to<Skeleton>()->find_bone( p_path.get_property() );
		if (bone_idx==-1)
			property=p_path.get_property();
	}

	TrackKey key;
	key.id=id;
	key.bone_idx=bone_idx;
	key.property=property;

	if (!track_map.has(key)) {

		Track tr;
		tr.id=id;
		tr.node=child;
		tr.skeleton=child->cast_to<Skeleton>();
		tr.spatial=child->cast_to<Spatial>();
		tr.bone_idx=bone_idx;
		tr.property=property;

		track_map[key]=tr;
	}

	return &track_map[key];

}

void AnimationTreePlayer::_recompute_caches() {

	track_map.clear();
	_recompute_caches(out_name);
	dirty_caches=false;
}

void AnimationTreePlayer::_recompute_caches(const StringName& p_node) {

	ERR_FAIL_COND( !node_map.has(p_node) );

	NodeBase *nb = node_map[p_node];

	if (nb->type==NODE_ANIMATION) {

		AnimationNode *an = static_cast<AnimationNode*>(nb);
		an->tref.clear();;

		if (!an->animation.is_null()) {


			Ref<Animation> a = an->animation;

			for(int i=0;i<an->animation->get_track_count();i++) {


				Track *tr = _find_track(a->track_get_path(i));
				if (!tr)
					continue;

				AnimationNode::TrackRef tref;
				tref.local_track=i;
				tref.track=tr;
				tref.weight=0;

				an->tref.push_back(tref);

			}
		}
	}

	for(int i=0;i<nb->inputs.size();i++) {

		_recompute_caches(nb->inputs[i].node);
	}

}

void AnimationTreePlayer::recompute_caches() {

	dirty_caches=true;

}



	/* playback */

void AnimationTreePlayer::set_active(bool p_active) {

	active=p_active;
	set_process(active);
}

bool AnimationTreePlayer::is_active() const {

	return active;

}

AnimationTreePlayer::ConnectError AnimationTreePlayer::get_last_error() const {

	return last_error;
}

void AnimationTreePlayer::reset() {


	reset_request=false;
}


void AnimationTreePlayer::set_base_path(const NodePath& p_path) {

	base_path=p_path;
	recompute_caches();
}

NodePath AnimationTreePlayer::get_base_path() const{

	return base_path;
}

void AnimationTreePlayer::set_master_player(const NodePath& p_path) {

	if (p_path==master)
		return;

	master=p_path;
	_update_sources();
	recompute_caches();
}

NodePath AnimationTreePlayer::get_master_player() const{

	return master;
}

DVector<String> AnimationTreePlayer::_get_node_list() {

	List<StringName> nl;
	get_node_list(&nl);
	DVector<String> ret;
	ret.resize(nl.size());
	int idx=0;
	for(List<StringName>::Element *E=nl.front();E;E=E->next()) {
		ret.set(idx++,E->get());
	}

	return ret;
}


void AnimationTreePlayer::_update_sources() {

	if (master==NodePath())
		return;
	if (!is_inside_tree())
		return;

	Node *m = get_node(master);
	if (!m) {
		master=NodePath();
		ERR_FAIL_COND(!m);
	}

	AnimationPlayer *ap = m->cast_to<AnimationPlayer>();

	if (!ap) {

		master=NodePath();
		ERR_FAIL_COND(!ap);
	}

	for (Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		if (E->get()->type==NODE_ANIMATION) {

			AnimationNode *an = static_cast<AnimationNode*>(E->get());

			if (an->from!="") {

				an->animation = ap->get_animation(an->from);
			}
		}
	}


}

bool AnimationTreePlayer::node_exists(const StringName& p_name) const {

	return (node_map.has(p_name));
}

Error AnimationTreePlayer::node_rename(const StringName& p_node,const StringName& p_new_name) {

	if (p_new_name==p_node)
		return OK;
	ERR_FAIL_COND_V(!node_map.has(p_node),ERR_ALREADY_EXISTS);
	ERR_FAIL_COND_V(node_map.has(p_new_name),ERR_ALREADY_EXISTS);
	ERR_FAIL_COND_V(p_new_name==StringName(),ERR_INVALID_DATA);
	ERR_FAIL_COND_V(p_node==out_name,ERR_INVALID_DATA);
	ERR_FAIL_COND_V(p_new_name==out_name,ERR_INVALID_DATA);

	for(Map<StringName,NodeBase*>::Element *E=node_map.front();E;E=E->next()) {

		NodeBase *nb = E->get();
		for(int i=0;i<nb->inputs.size();i++) {

			if (nb->inputs[i].node==p_node) {
				nb->inputs[i].node=p_new_name;
			}
		}
	}

	node_map[p_new_name]=node_map[p_node];
	node_map.erase(p_node);

	return OK;

}


void AnimationTreePlayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("add_node","type","id"),&AnimationTreePlayer::add_node);

	ObjectTypeDB::bind_method(_MD("node_exists","node"),&AnimationTreePlayer::node_exists);
	ObjectTypeDB::bind_method(_MD("node_rename","node","new_name"),&AnimationTreePlayer::node_rename);

	ObjectTypeDB::bind_method(_MD("node_get_type","id"),&AnimationTreePlayer::node_get_type);
	ObjectTypeDB::bind_method(_MD("node_get_input_count","id"),&AnimationTreePlayer::node_get_input_count);
	ObjectTypeDB::bind_method(_MD("node_get_input_source","id","idx"),&AnimationTreePlayer::node_get_input_source);

	ObjectTypeDB::bind_method(_MD("animation_node_set_animation","id","animation:Animation"),&AnimationTreePlayer::animation_node_set_animation);
	ObjectTypeDB::bind_method(_MD("animation_node_get_animation:Animation","id"),&AnimationTreePlayer::animation_node_get_animation);

	ObjectTypeDB::bind_method(_MD("animation_node_set_master_animation","id","source"),&AnimationTreePlayer::animation_node_set_master_animation);
	ObjectTypeDB::bind_method(_MD("animation_node_get_master_animation","id"),&AnimationTreePlayer::animation_node_get_master_animation);

	ObjectTypeDB::bind_method(_MD("oneshot_node_set_fadein_time","id","time_sec"),&AnimationTreePlayer::oneshot_node_set_fadein_time);
	ObjectTypeDB::bind_method(_MD("oneshot_node_get_fadein_time","id"),&AnimationTreePlayer::oneshot_node_get_fadein_time);

	ObjectTypeDB::bind_method(_MD("oneshot_node_set_fadeout_time","id","time_sec"),&AnimationTreePlayer::oneshot_node_set_fadeout_time);
	ObjectTypeDB::bind_method(_MD("oneshot_node_get_fadeout_time","id"),&AnimationTreePlayer::oneshot_node_get_fadeout_time);


	ObjectTypeDB::bind_method(_MD("oneshot_node_set_autorestart","id","enable"),&AnimationTreePlayer::oneshot_node_set_autorestart);
	ObjectTypeDB::bind_method(_MD("oneshot_node_set_autorestart_delay","id","delay_sec"),&AnimationTreePlayer::oneshot_node_set_autorestart_delay);
	ObjectTypeDB::bind_method(_MD("oneshot_node_set_autorestart_random_delay","id","rand_sec"),&AnimationTreePlayer::oneshot_node_set_autorestart_random_delay);


	ObjectTypeDB::bind_method(_MD("oneshot_node_has_autorestart","id"),&AnimationTreePlayer::oneshot_node_has_autorestart);
	ObjectTypeDB::bind_method(_MD("oneshot_node_get_autorestart_delay","id"),&AnimationTreePlayer::oneshot_node_get_autorestart_delay);
	ObjectTypeDB::bind_method(_MD("oneshot_node_get_autorestart_random_delay","id"),&AnimationTreePlayer::oneshot_node_get_autorestart_random_delay);

	ObjectTypeDB::bind_method(_MD("oneshot_node_start","id"),&AnimationTreePlayer::oneshot_node_start);
	ObjectTypeDB::bind_method(_MD("oneshot_node_stop","id"),&AnimationTreePlayer::oneshot_node_stop);
	ObjectTypeDB::bind_method(_MD("oneshot_node_is_active","id"),&AnimationTreePlayer::oneshot_node_is_active);
	ObjectTypeDB::bind_method(_MD("oneshot_node_set_filter_path","id","path","enable"),&AnimationTreePlayer::oneshot_node_set_filter_path);

	ObjectTypeDB::bind_method(_MD("mix_node_set_amount","id","ratio"),&AnimationTreePlayer::mix_node_set_amount);
	ObjectTypeDB::bind_method(_MD("mix_node_get_amount","id"),&AnimationTreePlayer::mix_node_get_amount);

	ObjectTypeDB::bind_method(_MD("blend2_node_set_amount","id","blend"),&AnimationTreePlayer::blend2_node_set_amount);
	ObjectTypeDB::bind_method(_MD("blend2_node_get_amount","id"),&AnimationTreePlayer::blend2_node_get_amount);
	ObjectTypeDB::bind_method(_MD("blend2_node_set_filter_path","id","path","enable"),&AnimationTreePlayer::blend2_node_set_filter_path);

	ObjectTypeDB::bind_method(_MD("blend3_node_set_amount","id","blend"),&AnimationTreePlayer::blend3_node_set_amount);
	ObjectTypeDB::bind_method(_MD("blend3_node_get_amount","id"),&AnimationTreePlayer::blend3_node_get_amount);

	ObjectTypeDB::bind_method(_MD("blend4_node_set_amount","id","blend"),&AnimationTreePlayer::blend4_node_set_amount);
	ObjectTypeDB::bind_method(_MD("blend4_node_get_amount","id"),&AnimationTreePlayer::blend4_node_get_amount);

	ObjectTypeDB::bind_method(_MD("timescale_node_set_scale","id","scale"),&AnimationTreePlayer::timescale_node_set_scale);
	ObjectTypeDB::bind_method(_MD("timescale_node_get_scale","id"),&AnimationTreePlayer::timescale_node_get_scale);

	ObjectTypeDB::bind_method(_MD("timeseek_node_seek","id","pos_sec"),&AnimationTreePlayer::timeseek_node_seek);

	ObjectTypeDB::bind_method(_MD("transition_node_set_input_count","id","count"),&AnimationTreePlayer::transition_node_set_input_count);
	ObjectTypeDB::bind_method(_MD("transition_node_get_input_count","id"),&AnimationTreePlayer::transition_node_get_input_count);
	ObjectTypeDB::bind_method(_MD("transition_node_delete_input","id","input_idx"),&AnimationTreePlayer::transition_node_delete_input);

	ObjectTypeDB::bind_method(_MD("transition_node_set_input_auto_advance","id","input_idx","enable"),&AnimationTreePlayer::transition_node_set_input_auto_advance);
	ObjectTypeDB::bind_method(_MD("transition_node_has_input_auto_advance","id","input_idx"),&AnimationTreePlayer::transition_node_has_input_auto_advance);

	ObjectTypeDB::bind_method(_MD("transition_node_set_xfade_time","id","time_sec"),&AnimationTreePlayer::transition_node_set_xfade_time);
	ObjectTypeDB::bind_method(_MD("transition_node_get_xfade_time","id"),&AnimationTreePlayer::transition_node_get_xfade_time);

	ObjectTypeDB::bind_method(_MD("transition_node_set_current","id","input_idx"),&AnimationTreePlayer::transition_node_set_current);
	ObjectTypeDB::bind_method(_MD("transition_node_get_current","id"),&AnimationTreePlayer::transition_node_get_current);


	ObjectTypeDB::bind_method(_MD("node_set_pos","id","screen_pos"),&AnimationTreePlayer::node_set_pos);
	ObjectTypeDB::bind_method(_MD("node_get_pos","id"),&AnimationTreePlayer::node_get_pos);

	ObjectTypeDB::bind_method(_MD("remove_node","id"),&AnimationTreePlayer::remove_node);
	ObjectTypeDB::bind_method(_MD("connect","id","dst_id","dst_input_idx"),&AnimationTreePlayer::connect);
	ObjectTypeDB::bind_method(_MD("is_connected","id","dst_id","dst_input_idx"),&AnimationTreePlayer::is_connected);
	ObjectTypeDB::bind_method(_MD("disconnect","id","dst_input_idx"),&AnimationTreePlayer::disconnect);

	ObjectTypeDB::bind_method(_MD("set_active","enabled"),&AnimationTreePlayer::set_active);
	ObjectTypeDB::bind_method(_MD("is_active"),&AnimationTreePlayer::is_active);

	ObjectTypeDB::bind_method(_MD("set_base_path","path"),&AnimationTreePlayer::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&AnimationTreePlayer::get_base_path);

	ObjectTypeDB::bind_method(_MD("get_node_list"),&AnimationTreePlayer::_get_node_list);

	ObjectTypeDB::bind_method(_MD("reset"),&AnimationTreePlayer::reset);

	ObjectTypeDB::bind_method(_MD("recompute_caches"),&AnimationTreePlayer::recompute_caches);	

	BIND_CONSTANT( NODE_OUTPUT );
	BIND_CONSTANT( NODE_ANIMATION );
	BIND_CONSTANT( NODE_ONESHOT );
	BIND_CONSTANT( NODE_MIX );
	BIND_CONSTANT( NODE_BLEND2 );
	BIND_CONSTANT( NODE_BLEND3 );
	BIND_CONSTANT( NODE_BLEND4 );
	BIND_CONSTANT( NODE_TIMESCALE );
	BIND_CONSTANT( NODE_TIMESEEK );
	BIND_CONSTANT( NODE_TRANSITION );
}


AnimationTreePlayer::AnimationTreePlayer() {

	active_list=NULL;
	out = memnew( NodeOut ) ;
	out_name="out";
	out->pos=Point2(40,40);
	node_map.insert( out_name , out);
	active=false;
	dirty_caches=true;
	reset_request=false;
	last_error=CONNECT_INCOMPLETE;
	base_path=String("..");
}


AnimationTreePlayer::~AnimationTreePlayer() {

	while(node_map.size()) {
		memdelete( node_map.front()->get() );
		node_map.erase( node_map.front() );
	}
}



