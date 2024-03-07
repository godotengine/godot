/**
 * bt_cooldown.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_COOLDOWN_H
#define BT_COOLDOWN_H

#include "../bt_decorator.h"

#ifdef LIMBOAI_MODULE
#include "scene/main/scene_tree.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/scene_tree_timer.hpp>
#endif

class BTCooldown : public BTDecorator {
	GDCLASS(BTCooldown, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	double duration = 10.0;
	bool process_pause = false;
	bool start_cooled = false;
	bool trigger_on_failure = false;
	StringName cooldown_state_var = "";

	Ref<SceneTreeTimer> timer = nullptr;

	void _chill();
	void _on_timeout();

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _setup() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_duration(double p_value);
	double get_duration() const { return duration; }

	void set_process_pause(bool p_value);
	bool get_process_pause() const { return process_pause; }

	void set_start_cooled(bool p_value);
	bool get_start_cooled() const { return start_cooled; }

	void set_trigger_on_failure(bool p_value);
	bool get_trigger_on_failure() const { return trigger_on_failure; }

	void set_cooldown_state_var(const StringName &p_value);
	StringName get_cooldown_state_var() const { return cooldown_state_var; }
};

#endif // BT_COOLDOWN_H
