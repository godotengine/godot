#include "register_types.h" 
#include "tree_functions/BranchFunction.hpp"
#include "tree_functions/GrowthFunction.hpp"
#include "tree_functions/TrunkFunction.hpp"
#include "tree_functions/PipeRadiusFunction.hpp"

void initialize_mtree_module(ModuleInitializationLevel p_level)
{
    
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE)
    {
	    ClassDB::register_abstract_class<Mtree::TreeFunction>();
	    ClassDB::register_class<Mtree::BranchFunction>();
	    ClassDB::register_class<Mtree::GrowthFunction>();
	    ClassDB::register_class<Mtree::TrunkFunction>();
	    ClassDB::register_class<Mtree::PipeRadiusFunction>();
    }
}
void uninitialize_mtree_module(ModuleInitializationLevel p_level)
{
    
}