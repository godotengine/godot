/**************************************************************************/
/*  register_types.cpp                                                    */
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

#ifdef TOOLS_ENABLED
#include "editor/plugins/editor_plugin.h"
#include "editor/unity_link_server_editor_plugin.h"
#endif
#include "register_types.h"


#include "core/object/class_db.h"
#include "modules/game_help/Terrain3D/src/register_types.h"
#include "modules/game_help/foliage_manager/register_types.h"
#include "modules/game_help/game_gui/game_gui_compoent.h"
#include "modules/game_help/logic/character_ai/character_ai.h"
#include "modules/game_help/logic/character_ai/animator_condition.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_bool.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_float.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_int.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_string.h"
#include "modules/game_help/logic/beehave/beehave_node.h"
#include "modules/game_help/logic/beehave/beehave_tree.h"
#include "modules/game_help/logic/beehave/decorators/cooldown.h"
#include "modules/game_help/logic/beehave/decorators/delayer.h"
#include "modules/game_help/logic/beehave/decorators/limeter_timer.h"
#include "modules/game_help/logic/beehave/decorators/limiter_count.h"
#include "modules/game_help/logic/beehave/decorators/repeater.h"

#include "modules/game_help/logic/beehave/composites/parallel.h"
#include "modules/game_help/logic/beehave/composites/selector.h"
#include "modules/game_help/logic/beehave/composites/sequence_restart.h"
#include "modules/game_help/logic/beehave/composites/sequence.h"

#include "modules/game_help/logic/beehave/leaves/blaclboard_set.h"
#include "modules/game_help/logic/beehave/leaves/blackboard_condition.h"
#include "modules/game_help/logic/beehave/leaves/move_position.h"



#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_bool.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_float.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_int.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_string.h"


#include "modules/game_help/logic/animator/animation_help.h"
#include "modules/game_help/logic/body_main.h"
#include "modules/game_help/logic/character_shape/character_body_part.h"
#include "modules/game_help/logic/data_table_manager.h"
#include "modules/game_help/logic/path_manager.h"
#include "modules/game_help/logic/character_manager.h"
#include "modules/game_help/csv/CSV_EditorImportPlugin.h"

#include "modules/game_help/unity/unity_link_server.h"




#include "modules/game_help/MTerrain/gdextension/src/register_types.h"

#include "logic/blackboard/blackboard.h"
#include "logic/blackboard/blackboard_plan.h"
#include "logic/blackboard/bb_param/bb_aabb.h"
#include "logic/blackboard/bb_param/bb_array.h"
#include "logic/blackboard/bb_param/bb_basis.h"
#include "logic/blackboard/bb_param/bb_bool.h"
#include "logic/blackboard/bb_param/bb_byte_array.h"
#include "logic/blackboard/bb_param/bb_color.h"
#include "logic/blackboard/bb_param/bb_color_array.h"
#include "logic/blackboard/bb_param/bb_dictionary.h"
#include "logic/blackboard/bb_param/bb_float.h"
#include "logic/blackboard/bb_param/bb_float32_array.h"
#include "logic/blackboard/bb_param/bb_float64_array.h"
#include "logic/blackboard/bb_param/bb_int.h"
#include "logic/blackboard/bb_param/bb_int32_array.h"
#include "logic/blackboard/bb_param/bb_int64_array.h"
#include "logic/blackboard/bb_param/bb_node.h"
#include "logic/blackboard/bb_param/bb_param.h"
#include "logic/blackboard/bb_param/bb_plane.h"
#include "logic/blackboard/bb_param/bb_projection.h"
#include "logic/blackboard/bb_param/bb_quaternion.h"
#include "logic/blackboard/bb_param/bb_rect2.h"
#include "logic/blackboard/bb_param/bb_rect2i.h"
#include "logic/blackboard/bb_param/bb_string.h"
#include "logic/blackboard/bb_param/bb_string_array.h"
#include "logic/blackboard/bb_param/bb_string_name.h"
#include "logic/blackboard/bb_param/bb_transform2d.h"
#include "logic/blackboard/bb_param/bb_transform3d.h"
#include "logic/blackboard/bb_param/bb_variant.h"
#include "logic/blackboard/bb_param/bb_vector2.h"
#include "logic/blackboard/bb_param/bb_vector2_array.h"
#include "logic/blackboard/bb_param/bb_vector2i.h"
#include "logic/blackboard/bb_param/bb_vector3.h"
#include "logic/blackboard/bb_param/bb_vector3_array.h"
#include "logic/blackboard/bb_param/bb_vector3i.h"
#include "logic/blackboard/bb_param/bb_vector4.h"
#include "logic/blackboard/bb_param/bb_vector4i.h"






static AnimationManager* animation_help = nullptr;
static CSV_EditorImportPlugin * csv_editor_import = nullptr;
static DataTableManager * data_table_manager = nullptr;
static PathManager* path_manager = nullptr;
static CharacterManager* character_manager = nullptr;

void initialize_game_help_module(ModuleInitializationLevel p_level) {
#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR)
	{
		Ref<CSV_EditorImportPlugin> mp3_import;
		mp3_import.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(mp3_import);
		
		UnityLinkServerEditorPluginRegister::initialize();
	}
#endif
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		data_table_manager = memnew(DataTableManager);
		path_manager = memnew(PathManager);

		ClassDB::register_class<CSVData>();
		initialize_mterrain_module(p_level);
		//initialize_terrain_3d(p_level);
		initialize_filiage_manager(p_level);
		ClassDB::register_class<AnimationManager>();

		ClassDB::register_class<DataTableManager>();
		ClassDB::register_class<PathManager>();

		ClassDB::register_class<CharacterManager>();
		
		ClassDB::register_class<CharacterAnimationLibraryItem>();
		ClassDB::register_class<CharacterAnimationLibrary>();
		ClassDB::register_class<CharacterBodyMain>();
		ClassDB::register_class<CharacterBodyPart>();
		ClassDB::register_class<CharacterBodyPartInstane>();
		ClassDB::register_class<CharacterBodyPrefab>();
		ClassDB::register_class<CharacterNavigationAgent3D>();
		//ClassDB::register_class<BTPlaySkill>();


		ClassDB::register_abstract_class<CharacterMovement>();

		
		ClassDB::register_abstract_class<AnimatorAIStateConditionBase>();
		ClassDB::register_class<AnimatorAIStateFloatCondition>();
		ClassDB::register_class<AnimatorAIStateIntCondition>();
		ClassDB::register_class<AnimatorAIStateStringNameCondition>();
		ClassDB::register_class<AnimatorAIStateBoolCondition>();

		
		ClassDB::register_class<CharacterAnimatorCondition>();
		ClassDB::register_class<CharacterAnimationLogicNode>();

		
		ClassDB::register_abstract_class<AnimatorBlackboardSetItemBase>();
		ClassDB::register_class<AnimatorBlackboardSetItemBool>();
		ClassDB::register_class<AnimatorBlackboardSetItemFloat>();
		ClassDB::register_class<AnimatorBlackboardSetItemInt>();
		ClassDB::register_class<AnimatorBlackboardSetItemString>();

		ClassDB::register_class<AnimatorBlackboardSet>();
		
		ClassDB::register_class<CharacterAnimatorNodeBase>();
		ClassDB::register_class<CharacterAnimatorMask>();
		ClassDB::register_class<CharacterBoneMap>();
		ClassDB::register_class<CharacterAnimationItem>();
		ClassDB::register_class<CharacterAnimatorNode1D>();
		ClassDB::register_class<CharacterAnimatorNode2D>();
		ClassDB::register_class<CharacterAnimatorLayerConfig>();
		ClassDB::register_class<CharacterAnimatorLayer>();
		ClassDB::register_class<CharacterAnimatorLayerConfigInstance>();
		ClassDB::register_class<CharacterAnimator>();
		ClassDB::register_class<CharacterCheckArea3DResult>();
		ClassDB::register_class<CharacterCheckArea3D>();


		ClassDB::register_class<CharacterAI_CheckBase>();
		ClassDB::register_class<CharacterAI_CheckGround>();
		ClassDB::register_class<CharacterAI_CheckEnemy>();
		ClassDB::register_class<CharacterAI_CheckJump>();
		ClassDB::register_class<CharacterAI_CheckJump2>();
		ClassDB::register_class<CharacterAI_CheckPatrol>();

		ClassDB::register_class<CharacterAILogicNode>();


		ClassDB::register_class<BeehaveNode>(true);
		ClassDB::register_class<BeehaveComposite>(true);
		ClassDB::register_class<BeehaveDecorator>(true);
		ClassDB::register_class<BeehaveLeaf>(true);
		ClassDB::register_class<BeehaveAction>(true);

		
		ClassDB::register_class<BeehaveListener>();
		ClassDB::register_class<BeehaveTree>();

		ClassDB::register_class<BeehaveCompositeParallel>();
		ClassDB::register_class<BeehaveCompositeSelector>();
		ClassDB::register_class<BeehaveCompositeSequenceRestart>();
		ClassDB::register_class<BeehaveCompositeSequence>();

		
		ClassDB::register_class<BeehaveDecoratorCooldown>();
		ClassDB::register_class<BeehaveDecoratorDelayer>();
		ClassDB::register_class<BeehaveDecoratorLimiterTimer>();
		ClassDB::register_class<BeehaveDecoratorLimiterCount>();
		ClassDB::register_class<BeehaveDecoratorRepeater>();


		ClassDB::register_class<BeehaveLeafBlackboardCondition>();
		ClassDB::register_class<BeehaveLeafBlackboardSet>();
		ClassDB::register_class<BeehaveLeafMovePosition>();

		ClassDB::register_class<CharacterAI_Inductor>();
		ClassDB::register_class<CharacterAI_Brain>();
		ClassDB::register_class<CharacterAI>();


		ClassDB::register_class<Blackboard>();
		ClassDB::register_class<BlackboardPlan>();
		
		ClassDB::register_abstract_class<BBParam>();
		ClassDB::register_class<BBAabb>();
		ClassDB::register_class<BBArray>();
		ClassDB::register_class<BBBasis>();
		ClassDB::register_class<BBBool>();
		ClassDB::register_class<BBByteArray>();
		ClassDB::register_class<BBColor>();
		ClassDB::register_class<BBColorArray>();
		ClassDB::register_class<BBDictionary>();
		ClassDB::register_class<BBFloat>();
		ClassDB::register_class<BBFloat32Array>();
		ClassDB::register_class<BBFloat64Array>();
		ClassDB::register_class<BBInt>();
		ClassDB::register_class<BBInt32Array>();
		ClassDB::register_class<BBInt64Array>();
		ClassDB::register_class<BBNode>();
		ClassDB::register_class<BBPlane>();
		ClassDB::register_class<BBProjection>();
		ClassDB::register_class<BBQuaternion>();
		ClassDB::register_class<BBRect2>();
		ClassDB::register_class<BBRect2i>();
		ClassDB::register_class<BBString>();
		ClassDB::register_class<BBStringArray>();
		ClassDB::register_class<BBStringName>();
		ClassDB::register_class<BBTransform2D>();
		ClassDB::register_class<BBTransform3D>();
		ClassDB::register_class<BBVariant>();
		ClassDB::register_class<BBVector2>();
		ClassDB::register_class<BBVector2Array>();
		ClassDB::register_class<BBVector2i>();
		ClassDB::register_class<BBVector3>();
		ClassDB::register_class<BBVector3Array>();
		ClassDB::register_class<BBVector3i>();
		ClassDB::register_class<BBVector4>();
		ClassDB::register_class<BBVector4i>();

		animation_help = memnew(AnimationManager);

		character_manager = memnew(CharacterManager);
		CharacterManager::singleton = character_manager;

		Engine::get_singleton()->add_singleton(Engine::Singleton("AnimationManager", animation_help));
		Engine::get_singleton()->add_singleton(Engine::Singleton("DataTableManager", data_table_manager));
		Engine::get_singleton()->add_singleton(Engine::Singleton("PathManager", path_manager));
		Engine::get_singleton()->add_singleton(Engine::Singleton("CharacterManager", character_manager));



		if (Engine::get_singleton())
		{
			Engine::get_singleton()->add_globale_ticker(character_manager);
		}
	}



	// 技能释放
	//LimboTaskDB::register_task<BTPlaySkill>();

}

void uninitialize_game_help_module(ModuleInitializationLevel p_level) {
	
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Engine::get_singleton()->remove_singleton("DataTableManager");
	Engine::get_singleton()->remove_singleton("PathManager");
	CharacterManager::singleton = nullptr;
	if(Engine::get_singleton() != nullptr)
	{
		Engine::get_singleton()->remove_globale_ticker(character_manager);
	}


	memdelete(animation_help);
	animation_help = nullptr;

	memdelete(data_table_manager);
	data_table_manager = nullptr;
	memdelete(path_manager);
	path_manager = nullptr;

	memdelete(character_manager);
	character_manager = nullptr;

}
