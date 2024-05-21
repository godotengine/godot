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


#include "modules/game_help/logic/animation_help.h"
#include "modules/game_help/logic/body_main.h"
#include "modules/game_help/logic/body_part.h"
#include "modules/game_help/logic/data_table_manager.h"
#include "modules/game_help/logic/path_manager.h"
#include "modules/game_help/csv/CSV_EditorImportPlugin.h"

#include "modules/game_help/unity/unity_link_server.h"




#include "modules/game_help/MTerrain/gdextension/src/mterrain.h"
#include "modules/game_help/MTerrain/gdextension/src/mgrid.h"
#include "modules/game_help/MTerrain/gdextension/src/mresource.h"
#include "modules/game_help/MTerrain/gdextension/src/mchunk_generator.h"
#include "modules/game_help/MTerrain/gdextension/src/mchunks.h"
#include "modules/game_help/MTerrain/gdextension/src/mtool.h"
#include "modules/game_help/MTerrain/gdextension/src/mregion.h"
#include "modules/game_help/MTerrain/gdextension/src/mbrush_manager.h"
#include "modules/game_help/MTerrain/gdextension/src/mcollision.h"

#include "modules/game_help/MTerrain/gdextension/src/grass/mgrass.h"
#include "modules/game_help/MTerrain/gdextension/src/grass/mgrass_data.h"
#include "modules/game_help/MTerrain/gdextension/src/grass/mgrass_lod_setting.h"
#include "modules/game_help/MTerrain/gdextension/src/navmesh/mnavigation_region_3d.h"
#include "modules/game_help/MTerrain/gdextension/src/navmesh/mnavigation_mesh_data.h"
#include "modules/game_help/MTerrain/gdextension/src/navmesh/mobstacle.h"
#include "modules/game_help/MTerrain/gdextension/src/mbrush_layers.h"
#include "modules/game_help/MTerrain/gdextension/src/mterrain_material.h"






static AnimationManager* animation_help = nullptr;
static CSV_EditorImportPlugin * csv_editor_import = nullptr;
static DataTableManager * data_table_manager = nullptr;
static PathManager* path_manager = nullptr;

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

	initialize_terrain_3d(p_level);
	initialize_filiage_manager(p_level);
	ClassDB::register_class<AnimationManager>();

	ClassDB::register_class<DataTableManager>();
	ClassDB::register_class<PathManager>();
	
	ClassDB::register_class<CharacterBodyMain>();
	ClassDB::register_class<CharacterBodyPart>();
	ClassDB::register_class<CharacterBodyPartInstane>();
	ClassDB::register_class<CharacterController>();
	//ClassDB::register_class<BTPlaySkill>();

	
	ClassDB::register_abstract_class<AnimatorAIStateConditionBase>();
	ClassDB::register_class<AnimatorAIStateFloatCondition>();
	ClassDB::register_class<AnimatorAIStateIntCondition>();
	ClassDB::register_class<AnimatorAIStateStringNameCondition>();
	ClassDB::register_class<AnimatorAIStateBoolCondition>();

	
	ClassDB::register_class<CharacterAnimatorConditionList>();
	ClassDB::register_class<CharacterAnimatorCondition>();
	ClassDB::register_class<CharacterAnimationLogicNode>();

	
	ClassDB::register_class<CharacterAnimatorNodeBase>();
	ClassDB::register_class<CharacterAnimatorMask>();
	ClassDB::register_class<CharacterBoneMap>();
	ClassDB::register_class<CharacterAnimationItem>();
	ClassDB::register_class<CharacterAnimatorNode1D>();
	ClassDB::register_class<CharacterAnimatorNode2D>();
	ClassDB::register_class<CharacterAnimatorLayerConfig>();
	ClassDB::register_class<CharacterAnimatorLayer>();
	ClassDB::register_class<CharacterAnimator>();

	animation_help = memnew(AnimationManager);

	Engine::get_singleton()->add_singleton(Engine::Singleton("AnimationManager", animation_help));
	Engine::get_singleton()->add_singleton(Engine::Singleton("DataTableManager", data_table_manager));
	Engine::get_singleton()->add_singleton(Engine::Singleton("PathManager", path_manager));



	ClassDB::register_class<MTerrain>();
	ClassDB::register_class<MGrid>();
	ClassDB::register_class<MResource>();
	ClassDB::register_class<MChunkGenerator>();
	ClassDB::register_class<MChunks>();
	ClassDB::register_class<MRegion>();
	ClassDB::register_class<MTool>();
	ClassDB::register_class<MBrushManager>();
	ClassDB::register_class<MCollision>();
	ClassDB::register_class<MGrass>();
	ClassDB::register_class<MGrassData>();
	ClassDB::register_class<MGrassLodSetting>();
	ClassDB::register_class<MNavigationRegion3D>();
	ClassDB::register_class<MNavigationMeshData>();
	ClassDB::register_class<MObstacle>();
	ClassDB::register_class<MBrushLayers>();
	ClassDB::register_class<MTerrainMaterial>();
	}



	// 技能释放
	//LimboTaskDB::register_task<BTPlaySkill>();

}

void uninitialize_game_help_module(ModuleInitializationLevel p_level) {
	
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Engine::get_singleton()->remove_singleton("AnimationHelp");
	Engine::get_singleton()->remove_singleton("DataTableManager");
	Engine::get_singleton()->remove_singleton("PathManager");


	memdelete(animation_help);
	animation_help = nullptr;

	memdelete(data_table_manager);
	data_table_manager = nullptr;
	memdelete(path_manager);
	path_manager = nullptr;

}
