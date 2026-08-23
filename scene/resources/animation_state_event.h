#pragma once

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.h"

class AnimationMixer;
class Animation;

class AnimationStateContext : public RefCounted {
	GDCLASS(AnimationStateContext, RefCounted);

private:
	ObjectID target_id;
	ObjectID mixer_id;
	Ref<Animation> animation;
	double delta = 0.0;
	double elapsed = 0.0;
	double duration = 0.0;
	real_t weight = 1.0;

protected:
	static void _bind_methods();

public:
	void set_target(Object *p_target);
	Object *get_target() const;

	void set_mixer(AnimationMixer *p_mixer);
	AnimationMixer *get_mixer() const;

	void set_animation(const Ref<Animation> &p_animation);
	Ref<Animation> get_animation() const;

	void set_delta(double p_delta);
	double get_delta() const;

	void set_elapsed(double p_elapsed);
	double get_elapsed() const;

	void set_duration(double p_duration);
	double get_duration() const;

	void set_weight(real_t p_weight);
	real_t get_weight() const;
};

class AnimationStateEvent : public Resource {
	GDCLASS(AnimationStateEvent, Resource);

protected:
	static void _bind_methods();

	GDVIRTUAL1(_start, Ref<AnimationStateContext>)
	GDVIRTUAL2(_update, Ref<AnimationStateContext>, double)
	GDVIRTUAL1(_end, Ref<AnimationStateContext>)
	GDVIRTUAL1(_cancel, Ref<AnimationStateContext>)

public:
	StringName event_name;
	Color tag_color = Color(0.3, 0.6, 0.9, 0.8);

	void set_event_name(const StringName &p_name);
	StringName get_event_name() const;

	void set_tag_color(const Color &p_color);
	Color get_tag_color() const;

	virtual void start(const Ref<AnimationStateContext> &p_context);
	virtual void update(const Ref<AnimationStateContext> &p_context, double p_delta);
	virtual void end(const Ref<AnimationStateContext> &p_context);
	virtual void cancel(const Ref<AnimationStateContext> &p_context);
};