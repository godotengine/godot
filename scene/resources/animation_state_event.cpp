#include "animation_state_event.h"

#include "core/object/class_db.h"
#include "scene/animation/animation_mixer.h"
#include "scene/resources/animation.h"

// ----------------------------------------------------
// AnimationStateContext
// ----------------------------------------------------

void AnimationStateContext::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_target"), &AnimationStateContext::get_target);
	ClassDB::bind_method(D_METHOD("get_mixer"), &AnimationStateContext::get_mixer);
	ClassDB::bind_method(D_METHOD("get_animation"), &AnimationStateContext::get_animation);
	ClassDB::bind_method(D_METHOD("get_delta"), &AnimationStateContext::get_delta);
	ClassDB::bind_method(D_METHOD("get_elapsed"), &AnimationStateContext::get_elapsed);
	ClassDB::bind_method(D_METHOD("get_duration"), &AnimationStateContext::get_duration);
	ClassDB::bind_method(D_METHOD("get_weight"), &AnimationStateContext::get_weight);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "target"), "", "get_target");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mixer"), "", "get_mixer");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation"), "", "get_animation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "delta"), "", "get_delta");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "elapsed"), "", "get_elapsed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "duration"), "", "get_duration");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "weight"), "", "get_weight");
}

void AnimationStateContext::set_target(Object *p_target) {
	target_id = p_target ? p_target->get_instance_id() : ObjectID();
}

Object *AnimationStateContext::get_target() const {
	return ObjectDB::get_instance(target_id);
}

void AnimationStateContext::set_mixer(AnimationMixer *p_mixer) {
	mixer_id = p_mixer ? p_mixer->get_instance_id() : ObjectID();
}

AnimationMixer *AnimationStateContext::get_mixer() const {
	return ObjectDB::get_instance<AnimationMixer>(mixer_id);
}

void AnimationStateContext::set_animation(const Ref<Animation> &p_animation) {
	animation = p_animation;
}

Ref<Animation> AnimationStateContext::get_animation() const {
	return animation;
}

void AnimationStateContext::set_delta(double p_delta) {
	delta = p_delta;
}

double AnimationStateContext::get_delta() const {
	return delta;
}

void AnimationStateContext::set_elapsed(double p_elapsed) {
	elapsed = p_elapsed;
}

double AnimationStateContext::get_elapsed() const {
	return elapsed;
}

void AnimationStateContext::set_duration(double p_duration) {
	duration = p_duration;
}

double AnimationStateContext::get_duration() const {
	return duration;
}

void AnimationStateContext::set_weight(real_t p_weight) {
	weight = p_weight;
}

real_t AnimationStateContext::get_weight() const {
	return weight;
}

// ----------------------------------------------------
// AnimationStateEvent
// ----------------------------------------------------

void AnimationStateEvent::_bind_methods() {
	GDVIRTUAL_BIND(_start, "context");
	GDVIRTUAL_BIND(_update, "context", "delta");
	GDVIRTUAL_BIND(_end, "context");
	GDVIRTUAL_BIND(_cancel, "context");

	ClassDB::bind_method(D_METHOD("set_event_name", "name"), &AnimationStateEvent::set_event_name);
	ClassDB::bind_method(D_METHOD("get_event_name"), &AnimationStateEvent::get_event_name);

	ClassDB::bind_method(D_METHOD("set_tag_color", "color"), &AnimationStateEvent::set_tag_color);
	ClassDB::bind_method(D_METHOD("get_tag_color"), &AnimationStateEvent::get_tag_color);

	ClassDB::bind_method(D_METHOD("set_trigger_weight_threshold", "threshold"), &AnimationStateEvent::set_trigger_weight_threshold);
	ClassDB::bind_method(D_METHOD("get_trigger_weight_threshold"), &AnimationStateEvent::get_trigger_weight_threshold);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "event_name"), "set_event_name", "get_event_name");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "tag_color"), "set_tag_color", "get_tag_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "trigger_weight_threshold", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_trigger_weight_threshold", "get_trigger_weight_threshold");
}

void AnimationStateEvent::set_event_name(const StringName &p_name) {
	event_name = p_name;
}

StringName AnimationStateEvent::get_event_name() const {
	return event_name;
}

void AnimationStateEvent::set_tag_color(const Color &p_color) {
	tag_color = p_color;
}

Color AnimationStateEvent::get_tag_color() const {
	return tag_color;
}

void AnimationStateEvent::set_trigger_weight_threshold(double p_threshold) {
	trigger_weight_threshold = CLAMP(p_threshold, 0.0, 1.0);
}

double AnimationStateEvent::get_trigger_weight_threshold() const {
	return trigger_weight_threshold;
}

void AnimationStateEvent::start(const Ref<AnimationStateContext> &p_context) {
	GDVIRTUAL_CALL(_start, p_context);
}

void AnimationStateEvent::update(const Ref<AnimationStateContext> &p_context, double p_delta) {
	GDVIRTUAL_CALL(_update, p_context, p_delta);
}

void AnimationStateEvent::end(const Ref<AnimationStateContext> &p_context) {
	GDVIRTUAL_CALL(_end, p_context);
}

void AnimationStateEvent::cancel(const Ref<AnimationStateContext> &p_context) {
	GDVIRTUAL_CALL(_cancel, p_context);
}