
#include "register_types.h"
#include "foliage_manager.h"
using namespace Foliage;

void initialize_filiage_manager(ModuleInitializationLevel p_level)
{
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
    
	ClassDB::register_class<FoliageMapChunkConfig>();
	ClassDB::register_class<FoliageManager>();

}
void uninitialize_filiage_manager(ModuleInitializationLevel p_level)
{

}
