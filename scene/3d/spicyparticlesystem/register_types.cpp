// #include "register_types.h"

// #include "SpicyParticleSystem.hpp"
// #include "SpicyParticleSystemNode.h"
// #include "SpicyParticleSystemPlugin.hpp"
// #include "SpicyParticleGenerator.hpp"
// #include "SpicyParticleRenderer.hpp"

// #include <gdextension_interface.h>
// #include <godot_cpp/core/defs.hpp>
// #include <godot_cpp/core/class_db.hpp>
// #include <godot_cpp/godot.hpp>

// using namespace godot;

// void initialize_spicyparticlesystem_module(ModuleInitializationLevel p_level) {
// 	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
// 		ClassDB::register_class<SpicyParticleSystemNode>();
// 		ClassDB::register_class<SpicyParticleEmitter>();
// 		ClassDB::register_class<SpicyParticleSystem>();
// 		ClassDB::register_class<ParticleData>();
// 		ClassDB::register_class<SpicyParticleBurst>();
// 		ClassDB::register_class<SpicyParticleGenerator>(true);
// 		ClassDB::register_class<LifetimeGenerator>();
// 		ClassDB::register_class<PositionGenerator>();
// 		ClassDB::register_class<ColorGenerator>();
// 		ClassDB::register_class<SizeGenerator>();
// 		ClassDB::register_class<VelocityGenerator>();
// 		ClassDB::register_class<RotationGenerator>();
// 		ClassDB::register_class<SpicyParticleUpdater>(true);
// 		ClassDB::register_class<LifetimeUpdater>(true);
// 		ClassDB::register_class<PositionUpdater>(true);
// 		ClassDB::register_class<ColorUpdater>();
// 		ClassDB::register_class<VelocityUpdater>();
// 		ClassDB::register_class<AccelerationUpdater>();
// 		ClassDB::register_class<RotationUpdater>();
// 		ClassDB::register_class<SizeUpdater>();
// 		ClassDB::register_class<CustomDataUpdater>();

// 		ClassDB::register_class<SpicyFloatProperty>();
// 		ClassDB::register_class<SpicyVector3Property>();
// 		ClassDB::register_class<SpicyColorProperty>();

// 		ClassDB::register_class<SpicyProperty>(true);
// 		ClassDB::register_class<EmissionShape>(true);
// 		ClassDB::register_class<PointEmissionShape>();
// 		ClassDB::register_class<BoxEmissionShape>();
// 		ClassDB::register_class<SphereEmissionShape>();

// 		ClassDB::register_class<MultiMeshParticleRenderer>();
// 	}

// 	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
// 		ClassDB::register_class<SpicyParticleSystemModuleInspectorPlugin>();
// 		ClassDB::register_class<SpicyParticleSystemInspectorPlugin>();
// 		ClassDB::register_class<EditorPropertyRandomInteger>();
// 		ClassDB::register_class<SpicyParticleSystemPlugin>();
// 		EditorPlugins::add_by_type<SpicyParticleSystemPlugin>();
// 	}


// }

// void uninitialize_spicyparticlesystem_module(ModuleInitializationLevel p_level) {

// 	//if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
// 	//}

// 	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {

// 		EditorPlugins::remove_by_type<SpicyParticleSystemPlugin>();
// 		return;
// 	}
// }

// extern "C" {
// 	// Initialization.
// 	GDExtensionBool GDE_EXPORT spicyparticlesystem_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization* r_initialization) {
// 		godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

// 		init_obj.register_initializer(initialize_spicyparticlesystem_module);
// 		init_obj.register_terminator(uninitialize_spicyparticlesystem_module);
// 		init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

// 		return init_obj.init();
// 	}
// }