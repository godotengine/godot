#include "animation_blend_tree.h"
#include "scene/scene_string_names.h"

void AnimationNodeAnimation::set_animation(const StringName &p_name) {
	animation = p_name;
}

StringName AnimationNodeAnimation::get_animation() const {
	return animation;
}

float AnimationNodeAnimation::get_playback_time() const {
	return time;
}

void AnimationNodeAnimation::_validate_property(PropertyInfo &property) const {

	if (property.name == "animation") {
		AnimationGraphPlayer *gp = get_graph_player();
		if (gp && gp->has_node(gp->get_animation_player())) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(gp->get_node(gp->get_animation_player()));
			if (ap) {
				List<StringName> names;
				ap->get_animation_list(&names);
				String anims;
				for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
					if (E != names.front()) {
						anims += ",";
					}
					anims += String(E->get());
				}
				if (anims != String()) {
					property.hint = PROPERTY_HINT_ENUM;
					property.hint_string = anims;
				}
			}
		}
	}

	AnimationRootNode::_validate_property(property);
}

float AnimationNodeAnimation::process(float p_time, bool p_seek) {

	AnimationPlayer *ap = get_player();
	ERR_FAIL_COND_V(!ap, 0);

	Ref<Animation> anim = ap->get_animation(animation);
	if (!anim.is_valid()) {

		Ref<AnimationNodeBlendTree> tree = get_parent();
		if (tree.is_valid()) {
			String name = tree->get_node_name(Ref<AnimationNodeAnimation>(this));
			make_invalid(vformat(RTR("On BlendTree node '%s', animation not found: '%s'"), name, animation));

		} else {
			make_invalid(vformat(RTR("Animation not found: '%s'"), animation));
		}

		return 0;
	}

	if (p_seek) {
		time = p_time;
		step = 0;
	} else {
		time = MAX(0, time + p_time);
		step = p_time;
	}

	float anim_size = anim->get_length();

	if (anim->has_loop()) {

		if (anim_size) {
			time = Math::fposmod(time, anim_size);
		}

	} else if (time > anim_size) {

		time = anim_size;
	}

	blend_animation(animation, time, step, p_seek, 1.0);

	return anim_size - time;
}

String AnimationNodeAnimation::get_caption() const {
	return "Animation";
}

void AnimationNodeAnimation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_animation", "name"), &AnimationNodeAnimation::set_animation);
	ClassDB::bind_method(D_METHOD("get_animation"), &AnimationNodeAnimation::get_animation);

	ClassDB::bind_method(D_METHOD("get_playback_time"), &AnimationNodeAnimation::get_playback_time);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "animation"), "set_animation", "get_animation");
}

AnimationNodeAnimation::AnimationNodeAnimation() {
	last_version = 0;
	skip = false;
	time = 0;
	step = 0;
}

////////////////////////////////////////////////////////

void AnimationNodeOneShot::set_fadein_time(float p_time) {

	fade_in = p_time;
}

void AnimationNodeOneShot::set_fadeout_time(float p_time) {

	fade_out = p_time;
}

float AnimationNodeOneShot::get_fadein_time() const {

	return fade_in;
}
float AnimationNodeOneShot::get_fadeout_time() const {

	return fade_out;
}

void AnimationNodeOneShot::set_autorestart(bool p_active) {

	autorestart = p_active;
}
void AnimationNodeOneShot::set_autorestart_delay(float p_time) {

	autorestart_delay = p_time;
}
void AnimationNodeOneShot::set_autorestart_random_delay(float p_time) {

	autorestart_random_delay = p_time;
}

bool AnimationNodeOneShot::has_autorestart() const {

	return autorestart;
}
float AnimationNodeOneShot::get_autorestart_delay() const {

	return autorestart_delay;
}
float AnimationNodeOneShot::get_autorestart_random_delay() const {

	return autorestart_random_delay;
}

void AnimationNodeOneShot::set_mix_mode(MixMode p_mix) {

	mix = p_mix;
}
AnimationNodeOneShot::MixMode AnimationNodeOneShot::get_mix_mode() const {

	return mix;
}

void AnimationNodeOneShot::start() {
	active = true;
	do_start = true;
}
void AnimationNodeOneShot::stop() {
	active = false;
}
bool AnimationNodeOneShot::is_active() const {

	return active;
}

String AnimationNodeOneShot::get_caption() const {
	return "OneShot";
}

bool AnimationNodeOneShot::has_filter() const {
	return true;
}

float AnimationNodeOneShot::process(float p_time, bool p_seek) {

	if (!active) {
		//make it as if this node doesn't exist, pass input 0 by.
		return blend_input(0, p_time, p_seek, 1.0, FILTER_IGNORE, !sync);
	}

	bool os_seek = p_seek;

	if (p_seek)
		time = p_time;
	if (do_start) {
		time = 0;
		os_seek = true;
	}

	float blend;

	if (time < fade_in) {

		if (fade_in > 0)
			blend = time / fade_in;
		else
			blend = 0; //wtf

	} else if (!do_start && remaining < fade_out) {

		if (fade_out)
			blend = (remaining / fade_out);
		else
			blend = 1.0;
	} else
		blend = 1.0;

	float main_rem;
	if (mix == MIX_MODE_ADD) {
		main_rem = blend_input(0, p_time, p_seek, 1.0, FILTER_IGNORE, !sync);
	} else {
		main_rem = blend_input(0, p_time, p_seek, 1.0 - blend, FILTER_BLEND, !sync);
	}

	float os_rem = blend_input(1, os_seek ? time : p_time, os_seek, blend, FILTER_PASS, false);

	if (do_start) {
		remaining = os_rem;
		do_start = false;
	}

	if (!p_seek) {
		time += p_time;
		remaining = os_rem;
		if (remaining <= 0)
			active = false;
	}

	return MAX(main_rem, remaining);
}
void AnimationNodeOneShot::set_use_sync(bool p_sync) {

	sync = p_sync;
}

bool AnimationNodeOneShot::is_using_sync() const {

	return sync;
}

void AnimationNodeOneShot::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_fadein_time", "time"), &AnimationNodeOneShot::set_fadein_time);
	ClassDB::bind_method(D_METHOD("get_fadein_time"), &AnimationNodeOneShot::get_fadein_time);

	ClassDB::bind_method(D_METHOD("set_fadeout_time", "time"), &AnimationNodeOneShot::set_fadeout_time);
	ClassDB::bind_method(D_METHOD("get_fadeout_time"), &AnimationNodeOneShot::get_fadeout_time);

	ClassDB::bind_method(D_METHOD("set_autorestart", "enable"), &AnimationNodeOneShot::set_autorestart);
	ClassDB::bind_method(D_METHOD("has_autorestart"), &AnimationNodeOneShot::has_autorestart);

	ClassDB::bind_method(D_METHOD("set_autorestart_delay", "enable"), &AnimationNodeOneShot::set_autorestart_delay);
	ClassDB::bind_method(D_METHOD("get_autorestart_delay"), &AnimationNodeOneShot::get_autorestart_delay);

	ClassDB::bind_method(D_METHOD("set_autorestart_random_delay", "enable"), &AnimationNodeOneShot::set_autorestart_random_delay);
	ClassDB::bind_method(D_METHOD("get_autorestart_random_delay"), &AnimationNodeOneShot::get_autorestart_random_delay);

	ClassDB::bind_method(D_METHOD("set_mix_mode", "mode"), &AnimationNodeOneShot::set_mix_mode);
	ClassDB::bind_method(D_METHOD("get_mix_mode"), &AnimationNodeOneShot::get_mix_mode);

	ClassDB::bind_method(D_METHOD("start"), &AnimationNodeOneShot::start);
	ClassDB::bind_method(D_METHOD("stop"), &AnimationNodeOneShot::stop);
	ClassDB::bind_method(D_METHOD("is_active"), &AnimationNodeOneShot::is_active);

	ClassDB::bind_method(D_METHOD("set_use_sync", "enable"), &AnimationNodeOneShot::set_use_sync);
	ClassDB::bind_method(D_METHOD("is_using_sync"), &AnimationNodeOneShot::is_using_sync);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "fadein_time", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater"), "set_fadein_time", "get_fadein_time");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "fadeout_time", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater"), "set_fadeout_time", "get_fadeout_time");

	ADD_GROUP("autorestart_", "Auto Restart");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autorestart"), "set_autorestart", "has_autorestart");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "autorestart_delay", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater"), "set_autorestart_delay", "get_autorestart_delay");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "autorestart_random_delay", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater"), "set_autorestart_random_delay", "get_autorestart_random_delay");

	ADD_GROUP("", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync"), "set_use_sync", "is_using_sync");

	BIND_CONSTANT(MIX_MODE_BLEND)
	BIND_CONSTANT(MIX_MODE_ADD)
}

AnimationNodeOneShot::AnimationNodeOneShot() {

	add_input("in");
	add_input("shot");

	time = 0;
	fade_in = 0.1;
	fade_out = 0.1;
	autorestart = false;
	autorestart_delay = 1;
	autorestart_remaining = 0;
	mix = MIX_MODE_BLEND;
	active = false;
	do_start = false;
	sync = false;
}

////////////////////////////////////////////////

void AnimationNodeAdd::set_amount(float p_amount) {
	amount = p_amount;
}

float AnimationNodeAdd::get_amount() const {
	return amount;
}

String AnimationNodeAdd::get_caption() const {
	return "Add";
}
void AnimationNodeAdd::set_use_sync(bool p_sync) {

	sync = p_sync;
}

bool AnimationNodeAdd::is_using_sync() const {

	return sync;
}

bool AnimationNodeAdd::has_filter() const {

	return true;
}

float AnimationNodeAdd::process(float p_time, bool p_seek) {

	float rem0 = blend_input(0, p_time, p_seek, 1.0, FILTER_IGNORE, !sync);
	blend_input(1, p_time, p_seek, amount, FILTER_PASS, !sync);

	return rem0;
}

void AnimationNodeAdd::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_amount", "amount"), &AnimationNodeAdd::set_amount);
	ClassDB::bind_method(D_METHOD("get_amount"), &AnimationNodeAdd::get_amount);

	ClassDB::bind_method(D_METHOD("set_use_sync", "enable"), &AnimationNodeAdd::set_use_sync);
	ClassDB::bind_method(D_METHOD("is_using_sync"), &AnimationNodeAdd::is_using_sync);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_amount", "get_amount");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync"), "set_use_sync", "is_using_sync");
}

AnimationNodeAdd::AnimationNodeAdd() {

	add_input("in");
	add_input("add");
	amount = 0;
	sync = false;
}

/////////////////////////////////////////////

void AnimationNodeBlend2::set_amount(float p_amount) {
	amount = p_amount;
}

float AnimationNodeBlend2::get_amount() const {
	return amount;
}
String AnimationNodeBlend2::get_caption() const {
	return "Blend2";
}

float AnimationNodeBlend2::process(float p_time, bool p_seek) {

	float rem0 = blend_input(0, p_time, p_seek, 1.0 - amount, FILTER_BLEND, !sync);
	float rem1 = blend_input(1, p_time, p_seek, amount, FILTER_PASS, !sync);

	return amount > 0.5 ? rem1 : rem0; //hacky but good enough
}

void AnimationNodeBlend2::set_use_sync(bool p_sync) {

	sync = p_sync;
}

bool AnimationNodeBlend2::is_using_sync() const {

	return sync;
}

bool AnimationNodeBlend2::has_filter() const {

	return true;
}
void AnimationNodeBlend2::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_amount", "amount"), &AnimationNodeBlend2::set_amount);
	ClassDB::bind_method(D_METHOD("get_amount"), &AnimationNodeBlend2::get_amount);

	ClassDB::bind_method(D_METHOD("set_use_sync", "enable"), &AnimationNodeBlend2::set_use_sync);
	ClassDB::bind_method(D_METHOD("is_using_sync"), &AnimationNodeBlend2::is_using_sync);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_amount", "get_amount");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync"), "set_use_sync", "is_using_sync");
}
AnimationNodeBlend2::AnimationNodeBlend2() {
	add_input("in");
	add_input("blend");
	sync = false;

	amount = 0;
}

//////////////////////////////////////

void AnimationNodeBlend3::set_amount(float p_amount) {
	amount = p_amount;
}

float AnimationNodeBlend3::get_amount() const {
	return amount;
}

String AnimationNodeBlend3::get_caption() const {
	return "Blend3";
}

void AnimationNodeBlend3::set_use_sync(bool p_sync) {

	sync = p_sync;
}

bool AnimationNodeBlend3::is_using_sync() const {

	return sync;
}

float AnimationNodeBlend3::process(float p_time, bool p_seek) {

	float rem0 = blend_input(0, p_time, p_seek, MAX(0, -amount), FILTER_IGNORE, !sync);
	float rem1 = blend_input(1, p_time, p_seek, 1.0 - ABS(amount), FILTER_IGNORE, !sync);
	float rem2 = blend_input(2, p_time, p_seek, MAX(0, amount), FILTER_IGNORE, !sync);

	return amount > 0.5 ? rem2 : (amount < -0.5 ? rem0 : rem1); //hacky but good enough
}

void AnimationNodeBlend3::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_amount", "amount"), &AnimationNodeBlend3::set_amount);
	ClassDB::bind_method(D_METHOD("get_amount"), &AnimationNodeBlend3::get_amount);

	ClassDB::bind_method(D_METHOD("set_use_sync", "enable"), &AnimationNodeBlend3::set_use_sync);
	ClassDB::bind_method(D_METHOD("is_using_sync"), &AnimationNodeBlend3::is_using_sync);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "amount", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_amount", "get_amount");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync"), "set_use_sync", "is_using_sync");
}
AnimationNodeBlend3::AnimationNodeBlend3() {
	add_input("-blend");
	add_input("in");
	add_input("+blend");
	sync = false;
	amount = 0;
}

/////////////////////////////////

void AnimationNodeTimeScale::set_scale(float p_scale) {
	scale = p_scale;
}

float AnimationNodeTimeScale::get_scale() const {
	return scale;
}

String AnimationNodeTimeScale::get_caption() const {
	return "TimeScale";
}

float AnimationNodeTimeScale::process(float p_time, bool p_seek) {

	if (p_seek) {
		return blend_input(0, p_time, true, 1.0, FILTER_IGNORE, false);
	} else {
		return blend_input(0, p_time * scale, false, 1.0, FILTER_IGNORE, false);
	}
}

void AnimationNodeTimeScale::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &AnimationNodeTimeScale::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &AnimationNodeTimeScale::get_scale);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "scale", PROPERTY_HINT_RANGE, "0,32,0.01,or_greater"), "set_scale", "get_scale");
}
AnimationNodeTimeScale::AnimationNodeTimeScale() {
	add_input("in");
	scale = 1.0;
}

////////////////////////////////////

void AnimationNodeTimeSeek::set_seek_pos(float p_seek_pos) {
	seek_pos = p_seek_pos;
}

float AnimationNodeTimeSeek::get_seek_pos() const {
	return seek_pos;
}

String AnimationNodeTimeSeek::get_caption() const {
	return "Seek";
}

float AnimationNodeTimeSeek::process(float p_time, bool p_seek) {

	if (p_seek) {
		return blend_input(0, p_time, true, 1.0, FILTER_IGNORE, false);
	} else if (seek_pos >= 0) {
		float ret = blend_input(0, seek_pos, true, 1.0, FILTER_IGNORE, false);
		seek_pos = -1;
		_change_notify("seek_pos");
		return ret;
	} else {
		return blend_input(0, p_time, false, 1.0, FILTER_IGNORE, false);
	}
}

void AnimationNodeTimeSeek::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_seek_pos", "seek_pos"), &AnimationNodeTimeSeek::set_seek_pos);
	ClassDB::bind_method(D_METHOD("get_seek_pos"), &AnimationNodeTimeSeek::get_seek_pos);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "seek_pos", PROPERTY_HINT_RANGE, "-1,3600,0.01,or_greater"), "set_seek_pos", "get_seek_pos");
}
AnimationNodeTimeSeek::AnimationNodeTimeSeek() {
	add_input("in");
	seek_pos = -1;
}

/////////////////////////////////////////////////

String AnimationNodeTransition::get_caption() const {
	return "Transition";
}

void AnimationNodeTransition::_update_inputs() {
	while (get_input_count() < enabled_inputs) {
		add_input(inputs[get_input_count()].name);
	}

	while (get_input_count() > enabled_inputs) {
		remove_input(get_input_count() - 1);
	}
}

void AnimationNodeTransition::set_enabled_inputs(int p_inputs) {
	ERR_FAIL_INDEX(p_inputs, MAX_INPUTS);
	enabled_inputs = p_inputs;
	_update_inputs();
}

int AnimationNodeTransition::get_enabled_inputs() {
	return enabled_inputs;
}

void AnimationNodeTransition::set_input_as_auto_advance(int p_input, bool p_enable) {
	ERR_FAIL_INDEX(p_input, MAX_INPUTS);
	inputs[p_input].auto_advance = p_enable;
}

bool AnimationNodeTransition::is_input_set_as_auto_advance(int p_input) const {
	ERR_FAIL_INDEX_V(p_input, MAX_INPUTS, false);
	return inputs[p_input].auto_advance;
}

void AnimationNodeTransition::set_input_caption(int p_input, const String &p_name) {
	ERR_FAIL_INDEX(p_input, MAX_INPUTS);
	inputs[p_input].name = p_name;
	set_input_name(p_input, p_name);
}

String AnimationNodeTransition::get_input_caption(int p_input) const {
	ERR_FAIL_INDEX_V(p_input, MAX_INPUTS, String());
	return inputs[p_input].name;
}

void AnimationNodeTransition::set_current(int p_current) {

	if (current == p_current)
		return;
	ERR_FAIL_INDEX(p_current, enabled_inputs);

	Ref<AnimationNodeBlendTree> tree = get_parent();

	if (tree.is_valid() && current >= 0) {
		prev = current;
		prev_xfading = xfade;
		prev_time = time;
		time = 0;
		current = p_current;
		switched = true;
		_change_notify("current");
	} else {
		current = p_current;
	}
}

int AnimationNodeTransition::get_current() const {
	return current;
}
void AnimationNodeTransition::set_cross_fade_time(float p_fade) {
	xfade = p_fade;
}

float AnimationNodeTransition::get_cross_fade_time() const {
	return xfade;
}

float AnimationNodeTransition::process(float p_time, bool p_seek) {

	if (prev < 0) { // process current animation, check for transition

		float rem = blend_input(current, p_time, p_seek, 1.0, FILTER_IGNORE, false);

		if (p_seek)
			time = p_time;
		else
			time += p_time;

		if (inputs[current].auto_advance && rem <= xfade) {

			set_current((current + 1) % enabled_inputs);
		}

		return rem;
	} else { // cross-fading from prev to current

		float blend = xfade ? (prev_xfading / xfade) : 1;

		float rem;

		if (!p_seek && switched) { //just switched, seek to start of current

			rem = blend_input(current, 0, true, 1.0 - blend, FILTER_IGNORE, false);
		} else {

			rem = blend_input(current, p_time, p_seek, 1.0 - blend, FILTER_IGNORE, false);
		}

		switched = false;

		if (p_seek) { // don't seek prev animation
			blend_input(prev, 0, false, blend, FILTER_IGNORE, false);
			time = p_time;
		} else {
			blend_input(prev, p_time, false, blend, FILTER_IGNORE, false);
			time += p_time;
			prev_xfading -= p_time;
			if (prev_xfading < 0) {
				prev = -1;
			}
		}

		return rem;
	}
}

void AnimationNodeTransition::_validate_property(PropertyInfo &property) const {

	if (property.name == "current" && enabled_inputs > 0) {
		property.hint = PROPERTY_HINT_ENUM;
		String anims;
		for (int i = 0; i < enabled_inputs; i++) {
			if (i > 0) {
				anims += ",";
			}
			anims += inputs[i].name;
		}
		property.hint_string = anims;
	}

	if (property.name.begins_with("input_")) {
		String n = property.name.get_slicec('/', 0).get_slicec('_', 1);
		if (n != "count") {
			int idx = n.to_int();
			if (idx >= enabled_inputs) {
				property.usage = 0;
			}
		}
	}

	AnimationNode::_validate_property(property);
}

void AnimationNodeTransition::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_enabled_inputs", "amount"), &AnimationNodeTransition::set_enabled_inputs);
	ClassDB::bind_method(D_METHOD("get_enabled_inputs"), &AnimationNodeTransition::get_enabled_inputs);

	ClassDB::bind_method(D_METHOD("set_input_as_auto_advance", "input", "enable"), &AnimationNodeTransition::set_input_as_auto_advance);
	ClassDB::bind_method(D_METHOD("is_input_set_as_auto_advance", "input"), &AnimationNodeTransition::is_input_set_as_auto_advance);

	ClassDB::bind_method(D_METHOD("set_input_caption", "input", "caption"), &AnimationNodeTransition::set_input_caption);
	ClassDB::bind_method(D_METHOD("get_input_caption", "input"), &AnimationNodeTransition::get_input_caption);

	ClassDB::bind_method(D_METHOD("set_current", "index"), &AnimationNodeTransition::set_current);
	ClassDB::bind_method(D_METHOD("get_current"), &AnimationNodeTransition::get_current);

	ClassDB::bind_method(D_METHOD("set_cross_fade_time", "time"), &AnimationNodeTransition::set_cross_fade_time);
	ClassDB::bind_method(D_METHOD("get_cross_fade_time"), &AnimationNodeTransition::get_cross_fade_time);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "input_count", PROPERTY_HINT_RANGE, "0,64,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_enabled_inputs", "get_enabled_inputs");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current", PROPERTY_HINT_RANGE, "0,64,1"), "set_current", "get_current");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "xfade_time", PROPERTY_HINT_RANGE, "0,120,0.01"), "set_cross_fade_time", "get_cross_fade_time");

	for (int i = 0; i < MAX_INPUTS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "input_" + itos(i) + "/name"), "set_input_caption", "get_input_caption", i);
		ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "input_" + itos(i) + "/auto_advance"), "set_input_as_auto_advance", "is_input_set_as_auto_advance", i);
	}
}

AnimationNodeTransition::AnimationNodeTransition() {
	enabled_inputs = 0;
	xfade = 0;
	current = -1;
	prev = -1;
	prev_time = 0;
	prev_xfading = 0;
	switched = false;
	for (int i = 0; i < MAX_INPUTS; i++) {
		inputs[i].auto_advance = false;
		inputs[i].name = itos(i + 1);
	}
}

/////////////////////

String AnimationNodeOutput::get_caption() const {
	return "Output";
}

float AnimationNodeOutput::process(float p_time, bool p_seek) {
	return blend_input(0, p_time, p_seek, 1.0);
}

AnimationNodeOutput::AnimationNodeOutput() {
	add_input("output");
}

///////////////////////////////////////////////////////
void AnimationNodeBlendTree::add_node(const StringName &p_name, Ref<AnimationNode> p_node) {

	ERR_FAIL_COND(nodes.has(p_name));
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_node->get_parent().is_valid());
	ERR_FAIL_COND(p_node->get_graph_player() != NULL);
	ERR_FAIL_COND(p_name == SceneStringNames::get_singleton()->output);
	ERR_FAIL_COND(String(p_name).find("/") != -1);
	nodes[p_name] = p_node;

	p_node->set_parent(this);
	p_node->set_graph_player(get_graph_player());

	emit_changed();
}

Ref<AnimationNode> AnimationNodeBlendTree::get_node(const StringName &p_name) const {

	ERR_FAIL_COND_V(!nodes.has(p_name), Ref<AnimationNode>());

	return nodes[p_name];
}

StringName AnimationNodeBlendTree::get_node_name(const Ref<AnimationNode> &p_node) const {
	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		if (E->get() == p_node) {
			return E->key();
		}
	}

	ERR_FAIL_V(StringName());
}
bool AnimationNodeBlendTree::has_node(const StringName &p_name) const {
	return nodes.has(p_name);
}
void AnimationNodeBlendTree::remove_node(const StringName &p_name) {

	ERR_FAIL_COND(!nodes.has(p_name));
	ERR_FAIL_COND(p_name == SceneStringNames::get_singleton()->output); //can't delete output

	{
		//erase node connections
		Ref<AnimationNode> node = nodes[p_name];
		for (int i = 0; i < node->get_input_count(); i++) {
			node->set_input_connection(i, StringName());
		}
		node->set_parent(NULL);
		node->set_graph_player(NULL);
	}

	nodes.erase(p_name);

	//erase connections to name
	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> node = E->get();
		for (int i = 0; i < node->get_input_count(); i++) {
			if (node->get_input_connection(i) == p_name) {
				node->set_input_connection(i, StringName());
			}
		}
	}

	emit_changed();
}

void AnimationNodeBlendTree::rename_node(const StringName &p_name, const StringName &p_new_name) {

	ERR_FAIL_COND(!nodes.has(p_name));
	ERR_FAIL_COND(nodes.has(p_new_name));
	ERR_FAIL_COND(p_name == SceneStringNames::get_singleton()->output);
	ERR_FAIL_COND(p_new_name == SceneStringNames::get_singleton()->output);

	nodes[p_new_name] = nodes[p_name];
	nodes.erase(p_name);

	//rename connections
	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> node = E->get();
		for (int i = 0; i < node->get_input_count(); i++) {
			if (node->get_input_connection(i) == p_name) {
				node->set_input_connection(i, p_new_name);
			}
		}
	}
}

void AnimationNodeBlendTree::connect_node(const StringName &p_input_node, int p_input_index, const StringName &p_output_node) {

	ERR_FAIL_COND(!nodes.has(p_output_node));
	ERR_FAIL_COND(!nodes.has(p_input_node));
	ERR_FAIL_COND(p_output_node == SceneStringNames::get_singleton()->output);
	ERR_FAIL_COND(p_input_node == p_output_node);

	Ref<AnimationNode> input = nodes[p_input_node];
	ERR_FAIL_INDEX(p_input_index, input->get_input_count());

	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> node = E->get();
		for (int i = 0; i < node->get_input_count(); i++) {
			StringName output = node->get_input_connection(i);
			ERR_FAIL_COND(output == p_output_node);
		}
	}

	input->set_input_connection(p_input_index, p_output_node);
	emit_changed();
}

void AnimationNodeBlendTree::disconnect_node(const StringName &p_node, int p_input_index) {

	ERR_FAIL_COND(!nodes.has(p_node));

	Ref<AnimationNode> input = nodes[p_node];
	ERR_FAIL_INDEX(p_input_index, input->get_input_count());

	input->set_input_connection(p_input_index, StringName());
}

float AnimationNodeBlendTree::get_connection_activity(const StringName &p_input_node, int p_input_index) const {

	ERR_FAIL_COND_V(!nodes.has(p_input_node), 0);

	Ref<AnimationNode> input = nodes[p_input_node];
	ERR_FAIL_INDEX_V(p_input_index, input->get_input_count(), 0);

	return input->get_input_activity(p_input_index);
}

AnimationNodeBlendTree::ConnectionError AnimationNodeBlendTree::can_connect_node(const StringName &p_input_node, int p_input_index, const StringName &p_output_node) const {

	if (!nodes.has(p_output_node) || p_output_node == SceneStringNames::get_singleton()->output) {
		return CONNECTION_ERROR_NO_OUTPUT;
	}

	if (!nodes.has(p_input_node)) {
		return CONNECTION_ERROR_NO_INPUT;
	}

	if (!nodes.has(p_input_node)) {
		return CONNECTION_ERROR_SAME_NODE;
	}

	Ref<AnimationNode> input = nodes[p_input_node];

	if (p_input_index < 0 || p_input_index >= input->get_input_count()) {
		return CONNECTION_ERROR_NO_INPUT_INDEX;
	}

	if (input->get_input_connection(p_input_index) != StringName()) {
		return CONNECTION_ERROR_CONNECTION_EXISTS;
	}

	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> node = E->get();
		for (int i = 0; i < node->get_input_count(); i++) {
			StringName output = node->get_input_connection(i);
			if (output == p_output_node) {
				return CONNECTION_ERROR_CONNECTION_EXISTS;
			}
		}
	}
	return CONNECTION_OK;
}

void AnimationNodeBlendTree::get_node_connections(List<NodeConnection> *r_connections) const {

	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> node = E->get();
		for (int i = 0; i < node->get_input_count(); i++) {
			StringName output = node->get_input_connection(i);
			if (output != StringName()) {
				NodeConnection nc;
				nc.input_node = E->key();
				nc.input_index = i;
				nc.output_node = output;
				r_connections->push_back(nc);
			}
		}
	}
}

String AnimationNodeBlendTree::get_caption() const {
	return "BlendTree";
}

float AnimationNodeBlendTree::process(float p_time, bool p_seek) {

	Ref<AnimationNodeOutput> output = nodes[SceneStringNames::get_singleton()->output];
	return blend_node(output, p_time, p_seek, 1.0);
}

void AnimationNodeBlendTree::get_node_list(List<StringName> *r_list) {

	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		r_list->push_back(E->key());
	}
}

void AnimationNodeBlendTree::set_graph_offset(const Vector2 &p_graph_offset) {

	graph_offset = p_graph_offset;
}

Vector2 AnimationNodeBlendTree::get_graph_offset() const {

	return graph_offset;
}

void AnimationNodeBlendTree::set_graph_player(AnimationGraphPlayer *p_player) {

	AnimationNode::set_graph_player(p_player);

	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		Ref<AnimationNode> node = E->get();
		node->set_graph_player(p_player);
	}
}

bool AnimationNodeBlendTree::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name;
	if (name.begins_with("nodes/")) {
		String node_name = name.get_slicec('/', 1);
		String what = name.get_slicec('/', 2);

		if (what == "node") {
			Ref<AnimationNode> anode = p_value;
			if (anode.is_valid()) {
				add_node(node_name, p_value);
			}
			return true;
		}

		if (what == "position") {

			if (nodes.has(node_name)) {
				nodes[node_name]->set_position(p_value);
			}
			return true;
		}
	} else if (name == "node_connections") {

		Array conns = p_value;
		ERR_FAIL_COND_V(conns.size() % 3 != 0, false);

		for (int i = 0; i < conns.size(); i += 3) {
			connect_node(conns[i], conns[i + 1], conns[i + 2]);
		}
		return true;
	}

	return false;
}

bool AnimationNodeBlendTree::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;
	if (name.begins_with("nodes/")) {
		String node_name = name.get_slicec('/', 1);
		String what = name.get_slicec('/', 2);

		if (what == "node") {
			if (nodes.has(node_name)) {
				r_ret = nodes[node_name];
				return true;
			}
		}

		if (what == "position") {

			if (nodes.has(node_name)) {
				r_ret = nodes[node_name]->get_position();
				return true;
			}
		}
	} else if (name == "node_connections") {
		List<NodeConnection> nc;
		get_node_connections(&nc);
		Array conns;
		conns.resize(nc.size() * 3);

		int idx = 0;
		for (List<NodeConnection>::Element *E = nc.front(); E; E = E->next()) {
			conns[idx * 3 + 0] = E->get().input_node;
			conns[idx * 3 + 1] = E->get().input_index;
			conns[idx * 3 + 2] = E->get().output_node;
			idx++;
		}

		r_ret = conns;
		return true;
	}

	return false;
}
void AnimationNodeBlendTree::_get_property_list(List<PropertyInfo> *p_list) const {

	List<StringName> names;
	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		names.push_back(E->key());
	}
	names.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		String name = E->get();
		if (name != "output") {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "nodes/" + name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "AnimationNode", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
		}
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "nodes/" + name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "node_connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
}

void AnimationNodeBlendTree::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_node", "name", "node"), &AnimationNodeBlendTree::add_node);
	ClassDB::bind_method(D_METHOD("get_node", "name"), &AnimationNodeBlendTree::get_node);
	ClassDB::bind_method(D_METHOD("remove_node", "name"), &AnimationNodeBlendTree::remove_node);
	ClassDB::bind_method(D_METHOD("rename_node", "name", "new_name"), &AnimationNodeBlendTree::rename_node);
	ClassDB::bind_method(D_METHOD("has_node", "name"), &AnimationNodeBlendTree::has_node);
	ClassDB::bind_method(D_METHOD("connect_node", "input_node", "input_index", "output_node"), &AnimationNodeBlendTree::connect_node);
	ClassDB::bind_method(D_METHOD("disconnect_node", "input_node", "input_index"), &AnimationNodeBlendTree::disconnect_node);

	ClassDB::bind_method(D_METHOD("set_graph_offset", "offset"), &AnimationNodeBlendTree::set_graph_offset);
	ClassDB::bind_method(D_METHOD("get_graph_offset"), &AnimationNodeBlendTree::get_graph_offset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "graph_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_graph_offset", "get_graph_offset");

	BIND_CONSTANT(CONNECTION_OK);
	BIND_CONSTANT(CONNECTION_ERROR_NO_INPUT);
	BIND_CONSTANT(CONNECTION_ERROR_NO_INPUT_INDEX);
	BIND_CONSTANT(CONNECTION_ERROR_NO_OUTPUT);
	BIND_CONSTANT(CONNECTION_ERROR_SAME_NODE);
	BIND_CONSTANT(CONNECTION_ERROR_CONNECTION_EXISTS);
}

AnimationNodeBlendTree::AnimationNodeBlendTree() {

	Ref<AnimationNodeOutput> output;
	output.instance();
	output->set_position(Vector2(300, 150));
	output->set_parent(this);
	nodes["output"] = output;
}

AnimationNodeBlendTree::~AnimationNodeBlendTree() {

	for (Map<StringName, Ref<AnimationNode> >::Element *E = nodes.front(); E; E = E->next()) {
		E->get()->set_parent(NULL);
		E->get()->set_graph_player(NULL);
	}
}
