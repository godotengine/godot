#include "register_types.h"

#include "ai_chat_interface.h"
#include "core/object/class_db.h"
#include "editor/editor_node.h"


static void _editor_init_callback() {
    EditorNode::add_editor_plugin(memnew(AIChatInterfacePlugin));
}

void initialize_ai_chat_interface_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_EDITOR) {
        return;
    }
    ClassDB::register_class<AIChatInterfacePlugin>();
    EditorNode::add_init_callback(_editor_init_callback);
}

void uninitialize_ai_chat_interface_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_EDITOR) {
        return;
    }
}
