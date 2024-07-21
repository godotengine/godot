/**
 * register_types.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "register_types.h"

#include "blackboard/bb_param/bb_aabb.h"
#include "blackboard/bb_param/bb_array.h"
#include "blackboard/bb_param/bb_basis.h"
#include "blackboard/bb_param/bb_bool.h"
#include "blackboard/bb_param/bb_byte_array.h"
#include "blackboard/bb_param/bb_color.h"
#include "blackboard/bb_param/bb_color_array.h"
#include "blackboard/bb_param/bb_dictionary.h"
#include "blackboard/bb_param/bb_float.h"
#include "blackboard/bb_param/bb_float32_array.h"
#include "blackboard/bb_param/bb_float64_array.h"
#include "blackboard/bb_param/bb_int.h"
#include "blackboard/bb_param/bb_int32_array.h"
#include "blackboard/bb_param/bb_int64_array.h"
#include "blackboard/bb_param/bb_node.h"
#include "blackboard/bb_param/bb_param.h"
#include "blackboard/bb_param/bb_plane.h"
#include "blackboard/bb_param/bb_projection.h"
#include "blackboard/bb_param/bb_quaternion.h"
#include "blackboard/bb_param/bb_rect2.h"
#include "blackboard/bb_param/bb_rect2i.h"
#include "blackboard/bb_param/bb_string.h"
#include "blackboard/bb_param/bb_string_array.h"
#include "blackboard/bb_param/bb_string_name.h"
#include "blackboard/bb_param/bb_transform2d.h"
#include "blackboard/bb_param/bb_transform3d.h"
#include "blackboard/bb_param/bb_variant.h"
#include "blackboard/bb_param/bb_vector2.h"
#include "blackboard/bb_param/bb_vector2_array.h"
#include "blackboard/bb_param/bb_vector2i.h"
#include "blackboard/bb_param/bb_vector3.h"
#include "blackboard/bb_param/bb_vector3_array.h"
#include "blackboard/bb_param/bb_vector3i.h"
#include "blackboard/bb_param/bb_vector4.h"
#include "blackboard/bb_param/bb_vector4i.h"
#include "blackboard/blackboard.h"
#include "blackboard/blackboard_plan.h"
#include "bt/behavior_tree.h"
#include "bt/bt_player.h"
#include "bt/bt_state.h"
#include "bt/tasks/blackboard/bt_check_trigger.h"
#include "bt/tasks/blackboard/bt_check_var.h"
#include "bt/tasks/blackboard/bt_set_var.h"
#include "bt/tasks/bt_action.h"
#include "bt/tasks/bt_comment.h"
#include "bt/tasks/bt_composite.h"
#include "bt/tasks/bt_condition.h"
#include "bt/tasks/bt_decorator.h"
#include "bt/tasks/bt_task.h"
#include "bt/tasks/composites/bt_dynamic_selector.h"
#include "bt/tasks/composites/bt_dynamic_sequence.h"
#include "bt/tasks/composites/bt_parallel.h"
#include "bt/tasks/composites/bt_probability_selector.h"
#include "bt/tasks/composites/bt_random_selector.h"
#include "bt/tasks/composites/bt_random_sequence.h"
#include "bt/tasks/composites/bt_selector.h"
#include "bt/tasks/composites/bt_sequence.h"
#include "bt/tasks/decorators/bt_always_fail.h"
#include "bt/tasks/decorators/bt_always_succeed.h"
#include "bt/tasks/decorators/bt_cooldown.h"
#include "bt/tasks/decorators/bt_delay.h"
#include "bt/tasks/decorators/bt_for_each.h"
#include "bt/tasks/decorators/bt_invert.h"
#include "bt/tasks/decorators/bt_new_scope.h"
#include "bt/tasks/decorators/bt_probability.h"
#include "bt/tasks/decorators/bt_repeat.h"
#include "bt/tasks/decorators/bt_repeat_until_failure.h"
#include "bt/tasks/decorators/bt_repeat_until_success.h"
#include "bt/tasks/decorators/bt_run_limit.h"
#include "bt/tasks/decorators/bt_subtree.h"
#include "bt/tasks/decorators/bt_time_limit.h"
#include "bt/tasks/scene/bt_await_animation.h"
#include "bt/tasks/scene/bt_check_agent_property.h"
#include "bt/tasks/scene/bt_pause_animation.h"
#include "bt/tasks/scene/bt_play_animation.h"
#include "bt/tasks/scene/bt_set_agent_property.h"
#include "bt/tasks/scene/bt_stop_animation.h"
#include "bt/tasks/utility/bt_call_method.h"
#include "bt/tasks/utility/bt_console_print.h"
#include "bt/tasks/utility/bt_evaluate_expression.h"
#include "bt/tasks/utility/bt_fail.h"
#include "bt/tasks/utility/bt_random_wait.h"
#include "bt/tasks/utility/bt_wait.h"
#include "bt/tasks/utility/bt_wait_ticks.h"
#include "editor/action_banner.h"
#include "editor/blackboard_plan_editor.h"
#include "editor/debugger/behavior_tree_data.h"
#include "editor/debugger/limbo_debugger.h"
#include "editor/debugger/limbo_debugger_plugin.h"
#include "editor/mode_switch_button.h"
#include "hsm/limbo_hsm.h"
#include "hsm/limbo_state.h"
#include "util/limbo_string_names.h"
#include "util/limbo_task_db.h"
#include "util/limbo_utility.h"

#ifdef TOOLS_ENABLED
#include "editor/debugger/behavior_tree_view.h"
#include "editor/limbo_ai_editor_plugin.h"
#endif // TOOLS_ENABLED

#ifdef LIMBOAI_MODULE
#include "core/object/class_db.h"
#include "core/os/memory.h"
#include "core/string/print_string.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/memory.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

static LimboUtility *_limbo_utility = nullptr;

void initialize_limboai_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
#ifdef TOOLS_ENABLED
		GDREGISTER_CLASS(BehaviorTreeView);
#endif // TOOLS_ENABLED
		GDREGISTER_CLASS(BehaviorTreeData);
#ifdef LIMBOAI_GDEXTENSION
		GDREGISTER_CLASS(LimboDebugger);
#endif
		LimboDebugger::initialize();

		GDREGISTER_CLASS(LimboUtility);
		GDREGISTER_CLASS(Blackboard);
		GDREGISTER_CLASS(BlackboardPlan);

		GDREGISTER_CLASS(LimboState);
		GDREGISTER_CLASS(LimboHSM);

		GDREGISTER_ABSTRACT_CLASS(BT);
		GDREGISTER_ABSTRACT_CLASS(BTTask);
		GDREGISTER_CLASS(BehaviorTree);
		GDREGISTER_CLASS(BTPlayer);
		GDREGISTER_CLASS(BTState);

		LIMBO_REGISTER_TASK(BTComment);

		GDREGISTER_CLASS(BTComposite);
		LIMBO_REGISTER_TASK(BTSequence);
		LIMBO_REGISTER_TASK(BTSelector);
		LIMBO_REGISTER_TASK(BTParallel);
		LIMBO_REGISTER_TASK(BTDynamicSequence);
		LIMBO_REGISTER_TASK(BTDynamicSelector);
		LIMBO_REGISTER_TASK(BTProbabilitySelector);
		LIMBO_REGISTER_TASK(BTRandomSequence);
		LIMBO_REGISTER_TASK(BTRandomSelector);

		GDREGISTER_CLASS(BTDecorator);
		LIMBO_REGISTER_TASK(BTInvert);
		LIMBO_REGISTER_TASK(BTAlwaysFail);
		LIMBO_REGISTER_TASK(BTAlwaysSucceed);
		LIMBO_REGISTER_TASK(BTDelay);
		LIMBO_REGISTER_TASK(BTRepeat);
		LIMBO_REGISTER_TASK(BTRepeatUntilFailure);
		LIMBO_REGISTER_TASK(BTRepeatUntilSuccess);
		LIMBO_REGISTER_TASK(BTRunLimit);
		LIMBO_REGISTER_TASK(BTTimeLimit);
		LIMBO_REGISTER_TASK(BTCooldown);
		LIMBO_REGISTER_TASK(BTProbability);
		LIMBO_REGISTER_TASK(BTForEach);
		LIMBO_REGISTER_TASK(BTNewScope);
		LIMBO_REGISTER_TASK(BTSubtree);

		GDREGISTER_CLASS(BTAction);
		GDREGISTER_CLASS(BTCondition);
		LIMBO_REGISTER_TASK(BTAwaitAnimation);
		LIMBO_REGISTER_TASK(BTCallMethod);
		LIMBO_REGISTER_TASK(BTEvaluateExpression);
		LIMBO_REGISTER_TASK(BTConsolePrint);
		LIMBO_REGISTER_TASK(BTFail);
		LIMBO_REGISTER_TASK(BTPauseAnimation);
		LIMBO_REGISTER_TASK(BTPlayAnimation);
		LIMBO_REGISTER_TASK(BTRandomWait);
		LIMBO_REGISTER_TASK(BTSetAgentProperty);
		LIMBO_REGISTER_TASK(BTSetVar);
		LIMBO_REGISTER_TASK(BTStopAnimation);
		LIMBO_REGISTER_TASK(BTWait);
		LIMBO_REGISTER_TASK(BTWaitTicks);
		LIMBO_REGISTER_TASK(BTCheckAgentProperty);
		LIMBO_REGISTER_TASK(BTCheckTrigger);
		LIMBO_REGISTER_TASK(BTCheckVar);

		GDREGISTER_ABSTRACT_CLASS(BBParam);
		GDREGISTER_CLASS(BBAabb);
		GDREGISTER_CLASS(BBArray);
		GDREGISTER_CLASS(BBBasis);
		GDREGISTER_CLASS(BBBool);
		GDREGISTER_CLASS(BBByteArray);
		GDREGISTER_CLASS(BBColor);
		GDREGISTER_CLASS(BBColorArray);
		GDREGISTER_CLASS(BBDictionary);
		GDREGISTER_CLASS(BBFloat);
		GDREGISTER_CLASS(BBFloat32Array);
		GDREGISTER_CLASS(BBFloat64Array);
		GDREGISTER_CLASS(BBInt);
		GDREGISTER_CLASS(BBInt32Array);
		GDREGISTER_CLASS(BBInt64Array);
		GDREGISTER_CLASS(BBNode);
		GDREGISTER_CLASS(BBPlane);
		GDREGISTER_CLASS(BBProjection);
		GDREGISTER_CLASS(BBQuaternion);
		GDREGISTER_CLASS(BBRect2);
		GDREGISTER_CLASS(BBRect2i);
		GDREGISTER_CLASS(BBString);
		GDREGISTER_CLASS(BBStringArray);
		GDREGISTER_CLASS(BBStringName);
		GDREGISTER_CLASS(BBTransform2D);
		GDREGISTER_CLASS(BBTransform3D);
		GDREGISTER_CLASS(BBVariant);
		GDREGISTER_CLASS(BBVector2);
		GDREGISTER_CLASS(BBVector2Array);
		GDREGISTER_CLASS(BBVector2i);
		GDREGISTER_CLASS(BBVector3);
		GDREGISTER_CLASS(BBVector3Array);
		GDREGISTER_CLASS(BBVector3i);
		GDREGISTER_CLASS(BBVector4);
		GDREGISTER_CLASS(BBVector4i);

		_limbo_utility = memnew(LimboUtility);

#ifdef LIMBOAI_MODULE
		Engine::get_singleton()->add_singleton(Engine::Singleton("LimboUtility", LimboUtility::get_singleton()));
#elif LIMBOAI_GDEXTENSION
		Engine::get_singleton()->register_singleton("LimboUtility", LimboUtility::get_singleton());
#endif

		LimboStringNames::create();
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
#ifdef LIMBOAI_GDEXTENSION
		GDREGISTER_CLASS(TaskTree);
		GDREGISTER_CLASS(TaskButton);
		GDREGISTER_CLASS(TaskPaletteSection);
		GDREGISTER_CLASS(TaskPalette);
		GDREGISTER_CLASS(ActionBanner);
		GDREGISTER_CLASS(ModeSwitchButton);
		GDREGISTER_CLASS(CompatShortcutBin);
		GDREGISTER_CLASS(CompatScreenSelect);
		GDREGISTER_CLASS(CompatWindowWrapper);
		GDREGISTER_CLASS(LimboDebuggerTab);
		GDREGISTER_CLASS(LimboDebuggerPlugin);
		GDREGISTER_CLASS(BlackboardPlanEditor);
		GDREGISTER_CLASS(EditorInspectorPluginBBPlan);
		GDREGISTER_CLASS(EditorPropertyVariableName);
		GDREGISTER_CLASS(EditorInspectorPluginVariableName);
		GDREGISTER_CLASS(OwnerPicker);
		GDREGISTER_CLASS(LimboAIEditor);
		GDREGISTER_CLASS(LimboAIEditorPlugin);
#endif // LIMBOAI_GDEXTENSION

		EditorPlugins::add_by_type<LimboAIEditorPlugin>();
	}

#endif // ! TOOLS_ENABLED
}

void uninitialize_limboai_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		LimboDebugger::deinitialize();
		LimboStringNames::free();
		memdelete(_limbo_utility);
	}
}

#ifdef LIMBOAI_GDEXTENSION
extern "C" {
// Initialization.
GDExtensionBool GDE_EXPORT limboai_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_limboai_module);
	init_obj.register_terminator(uninitialize_limboai_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}
}
#endif // LIMBOAI_GDEXTENSION
