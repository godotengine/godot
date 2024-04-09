#include <version_generated.gen.h>

#include "core/config/engine.h"
#include "register_types.h"

#include "src/godotShareData.h"

void initialize_ios_module(ModuleInitializationLevel p_level) {
    Engine::get_singleton()->add_singleton(Engine::Singleton("GodotShareData", memnew(GodotShareData)));
}

void uninitialize_ios_module(ModuleInitializationLevel p_level) {
}
