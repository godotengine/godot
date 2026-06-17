#include "register_types.h"
#include "core/object/class_db.h"
#include "Signals/EventSignal.h"
#include "FrameCallback/DotnetBridge.h"

void initialize_cross_runtime_module(ModuleInitializationLevel p_level) {
#ifdef WEB_ENABLED
    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        memnew(CrossRuntimeEventSignal); // Emits signals
        memnew(DotnetBridge); // FrameCallback
    }
#endif
}

void uninitialize_cross_runtime_module(ModuleInitializationLevel p_level) {
    // …
}