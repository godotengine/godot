/* godot-cpp integration testing project.
 *
 * This is free and unencumbered software released into the public domain.
 */

#include "register_types.h"



#include "mterrain.h"
#include "mresource.h"
#include "mchunk_generator.h"
#include "mchunks.h"
#include "mtool.h"
#include "mbrush_manager.h"
#include "mcollision.h"

#include "grass/mgrass.h"
#include "grass/mgrass_data.h"
#include "grass/mgrass_lod_setting.h"
#include "navmesh/mnavigation_region_3d.h"
#include "navmesh/mnavigation_mesh_data.h"
#include "navmesh/mobstacle.h"
#include "mbrush_layers.h"
#include "mterrain_material.h"

#include "moctree.h"
#include "octmesh/mmesh_lod.h"
#include "octmesh/moctmesh.h"

#include "path/mpath.h"
#include "path/mcurve.h"
#include "path/mintersection.h"
#include "path/mcurve_mesh.h"
#include "path/mcurve_mesh_override.h"
#include "path/mcurve_terrain.h"


using namespace godot;

void initialize_mterrain_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	
	ClassDB::register_class<MTerrain>();
	ClassDB::register_class<MResource>();
	ClassDB::register_class<MChunkGenerator>();
	ClassDB::register_class<MChunks>();
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

	ClassDB::register_class<MOctree>();
	ClassDB::register_class<MMeshLod>();
	ClassDB::register_class<MOctMesh>();

	ClassDB::register_class<MPath>();
	ClassDB::register_class<MCurve>();
	ClassDB::register_class<MIntersection>();
	ClassDB::register_class<MCurveMesh>();
	ClassDB::register_class<MCurveMeshOverride>();
	ClassDB::register_class<MCurveTerrain>();
}

void uninitialize_mterrain_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {}

}

// extern "C" {
// // Initialization.
// GDExtensionBool GDE_EXPORT test_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
// 	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

// 	init_obj.register_initializer(initialize_test_module);
// 	init_obj.register_terminator(uninitialize_test_module);
// 	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

// 	return init_obj.init();
// }
// }