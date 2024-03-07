/**
 * bt_stop_animation.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_STOP_ANIMATION_H
#define BT_STOP_ANIMATION_H

#include "../bt_action.h"

#include "../../../blackboard/bb_param/bb_node.h"

#ifdef LIMBOAI_MODULE
#include "scene/animation/animation_player.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/animation_player.hpp>
#endif

class BTStopAnimation : public BTAction {
	GDCLASS(BTStopAnimation, BTAction);
	TASK_CATEGORY(Scene);

private:
	Ref<BBNode> animation_player_param;
	StringName animation_name;
	bool keep_state = false;

	AnimationPlayer *animation_player = nullptr;
	bool setup_failed = false;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _setup() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_animation_player(Ref<BBNode> p_animation_player);
	Ref<BBNode> get_animation_player() const { return animation_player_param; }

	void set_animation_name(StringName p_animation_name);
	StringName get_animation_name() const { return animation_name; }

	void set_keep_state(bool p_keep_state);
	bool get_keep_state() const { return keep_state; }

	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_STOP_ANIMATION_H
