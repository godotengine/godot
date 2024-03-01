#include "register_types.h"
#include "core/object/ref_counted.h"

#ifdef TOOLS_ENABLED

#include "editor/editor_node.h"
#include "editor/post_import_plugin_animation_mirror.h"

#endif

#ifdef TOOLS_ENABLED

static void _editor_init() {
	const Ref<PostImportPluginAnimationMirror> post_importer_animation_mirror = memnew(PostImportPluginAnimationMirror);
	ResourceImporterScene::add_post_importer_plugin(post_importer_animation_mirror);
}

#endif

void initialize_mirror_animations_module(ModuleInitializationLevel p_level) {
#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		// Editor-specific API.
		ClassDB::APIType prev_api = ClassDB::get_current_api();
		ClassDB::set_current_api(ClassDB::API_EDITOR);

		GDREGISTER_CLASS(PostImportPluginAnimationMirror)

		ClassDB::set_current_api(prev_api);
		EditorNode::add_init_callback(_editor_init);
	}
#endif
}

void uninitialize_mirror_animations_module(ModuleInitializationLevel p_level) {

}
