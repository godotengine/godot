#include "register_types.h"

#include "core/gaussian_data.h"
#include "core/gaussian_splat_asset.h"
#include "core/gaussian_splat_world.h"
#include "core/gaussian_splat_scene_director.h"
#include "core/gaussian_splat_config_registry.h"
#include "core/gaussian_streaming.h"
#include "renderer/gaussian_splat_renderer.h"
#include "renderer/gpu_buffer_manager.h"
#include "core/gaussian_splat_manager.h"
#include "core/config/engine.h"
#include "io/ply_loader.h"
#include "io/spz_loader.h"
#include "io/i_gaussian_loader.h"
#include "io/gaussian_splat_world_io.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "resources/color_grading_resource.h"
#include "renderer/gpu_sorter.h"
#include "renderer/gpu_memory_stream.h"
#include "renderer/rendering_diagnostics.h"
#include "painterly/painterly_material.h"
#include "nodes/gaussian_splat_node_3d.h"
#include "nodes/gaussian_splat_debug_hud.h"
#include "nodes/gaussian_splat_container.h"
#include "nodes/gaussian_splat_dynamic_instance_3d.h"
#include "nodes/gaussian_splat_world_3d.h"
#include "interfaces/cluster_culler.h"

// Animation and Persistence (v0.6.0)
#include "animation/animation_state_machine.h"
#include "animation/keyframe_interpolator.h"
#include "persistence/gaussian_scene_serializer.h"
#include "persistence/incremental_saver.h"

// Asset Management System (v0.7.0)
#include "asset_management/asset_dependency_manager.h"
#include "core/performance_monitors.h"

#ifdef TOOLS_ENABLED
#include "editor/gaussian_editor_plugin.h"
#include "io/resource_importer_ply.h"
#include "io/resource_importer_spz.h"
#include "io/resource_importer_gsplatworld.h"
#endif


// Global resource loader instance
static Ref<ResourceFormatLoaderGaussianSplat> gaussian_format_loader;
static Ref<ResourceFormatLoaderGaussianSplatWorld> gaussian_world_format_loader;
static Ref<ResourceFormatSaverGaussianSplatWorld> gaussian_world_format_saver;

static GaussianSplatManager *gaussian_splat_manager_singleton = nullptr;
static GaussianSplatSceneDirector *gaussian_splat_scene_director_singleton = nullptr;

#ifdef TOOLS_ENABLED
// PLY and SPZ importer instances
static Ref<ResourceImporterPLY> ply_importer;
static Ref<ResourceImporterSPZ> spz_importer;
static Ref<ResourceImporterGSplatWorld> gsplatworld_importer;
#endif

void initialize_gaussian_splatting_module(ModuleInitializationLevel p_level) {
    switch (p_level) {
        case MODULE_INITIALIZATION_LEVEL_SCENE: {
            // Initialize configuration systems ahead of renderer setup.
            GaussianSplatConfigRegistry::initialize_all();
            // Core data structures
            GDREGISTER_CLASS(GaussianData);
            GDREGISTER_CLASS(GaussianSplatAsset);
            GDREGISTER_CLASS(GaussianSplatWorld);
            GDREGISTER_CLASS(GaussianStreamingSystem);
            GDREGISTER_CLASS(VRAMBudgetRegulator);

            // Node classes
            GDREGISTER_CLASS(GaussianSplatNode3D);
            GDREGISTER_CLASS(GaussianSplatDebugHUD);
            GDREGISTER_CLASS(GaussianSplatContainer);
            GDREGISTER_CLASS(GaussianSplatDynamicInstance3D);
            GDREGISTER_CLASS(GaussianSplatWorld3D);

            // Rendering components
            GDREGISTER_CLASS(GaussianSplatRenderer);
            GDREGISTER_CLASS(GaussianMemoryStream);
            GDREGISTER_CLASS(PainterlyMaterial);
            GDREGISTER_CLASS(GPUBufferManager);
            GDREGISTER_CLASS(GaussianSplatManager);
            GDREGISTER_CLASS(GaussianSplatSceneDirector);
            GDREGISTER_CLASS(ColorGradingResource);

            if (!gaussian_splat_manager_singleton) {
                gaussian_splat_manager_singleton = memnew(GaussianSplatManager);
                gaussian_splat_manager_singleton->initialize_module();

                Engine::Singleton singleton_info;
                singleton_info.name = "GaussianSplatManager";
                singleton_info.ptr = gaussian_splat_manager_singleton;
                Engine::get_singleton()->add_singleton(singleton_info);
            }

            if (!gaussian_splat_scene_director_singleton) {
                gaussian_splat_scene_director_singleton = memnew(GaussianSplatSceneDirector);

                Engine::Singleton director_info;
                director_info.name = "GaussianSplatSceneDirector";
                director_info.ptr = gaussian_splat_scene_director_singleton;
                Engine::get_singleton()->add_singleton(director_info);
            }

            GaussianRenderingDiagnostics::ensure_singleton();
            if (GaussianRenderingDiagnostics::get_singleton()) {
                GaussianRenderingDiagnostics::get_singleton()->process_command_line_requests();
            }

            // Initialize Custom Performance Monitors for editor debugger
            GaussianSplattingPerformanceMonitors::create_singleton();

            // IO components
            GDREGISTER_CLASS(PLYLoader);
            GDREGISTER_CLASS(SPZLoader);
            GDREGISTER_ABSTRACT_CLASS(IGaussianLoader);

            // Modular GPU sorting implementation
            GDREGISTER_ABSTRACT_CLASS(IGPUSorter);
            GDREGISTER_CLASS(BitonicSort);
            GDREGISTER_CLASS(RadixSort);
            GDREGISTER_CLASS(OneSweepSort);

            // Cluster-level coarse culling (LiteGS-style)
            GDREGISTER_CLASS(ClusterCuller);

            // Animation and Persistence (v0.6.0)
            GDREGISTER_CLASS(GaussianSplatting::GaussianAnimationStateMachine);
            GDREGISTER_CLASS(GaussianSplatting::GaussianSceneSerializer);
            GDREGISTER_CLASS(GaussianSplatting::GaussianIncrementalSaver);

            // Asset Management System (v0.7.0)
            GDREGISTER_CLASS(AssetDependencyManager);
            // AssetDependencyManager is currently the only compiled asset management
            // type. Keep this list aligned with asset_management/*.cpp sources to
            // prevent unresolved symbol errors when new implementations land.

            if (!gaussian_format_loader.is_valid()) {
                gaussian_format_loader.instantiate();
                ResourceLoader::add_resource_format_loader(gaussian_format_loader);
            }
            if (!gaussian_world_format_loader.is_valid()) {
                gaussian_world_format_loader.instantiate();
                ResourceLoader::add_resource_format_loader(gaussian_world_format_loader, true);
            }
            if (!gaussian_world_format_saver.is_valid()) {
                gaussian_world_format_saver.instantiate();
                ResourceSaver::add_resource_format_saver(gaussian_world_format_saver, true);
            }
        } break;

#ifdef TOOLS_ENABLED
        case MODULE_INITIALIZATION_LEVEL_EDITOR: {
            EditorPlugins::add_by_type<GaussianEditorPlugin>();

            // Register PLY importer
            ply_importer.instantiate();
            ResourceFormatImporter::get_singleton()->add_importer(ply_importer);

            // Register SPZ importer
            spz_importer.instantiate();
            ResourceFormatImporter::get_singleton()->add_importer(spz_importer);

            // Register gsplatworld importer
            gsplatworld_importer.instantiate();
            ResourceFormatImporter::get_singleton()->add_importer(gsplatworld_importer);
        } break;
#endif

        default:
            break;
    }
}

void uninitialize_gaussian_splatting_module(ModuleInitializationLevel p_level) {
    // Cleanup when module is unloaded
    switch (p_level) {
        case MODULE_INITIALIZATION_LEVEL_SCENE:
            if (gaussian_format_loader.is_valid()) {
                ResourceLoader::remove_resource_format_loader(gaussian_format_loader);
                gaussian_format_loader.unref();
            }
            if (gaussian_world_format_loader.is_valid()) {
                ResourceLoader::remove_resource_format_loader(gaussian_world_format_loader);
                gaussian_world_format_loader.unref();
            }
            if (gaussian_world_format_saver.is_valid()) {
                ResourceSaver::remove_resource_format_saver(gaussian_world_format_saver);
                gaussian_world_format_saver.unref();
            }
            if (gaussian_splat_manager_singleton) {
                if (Engine::get_singleton()->has_singleton("GaussianSplatManager")) {
                    Engine::get_singleton()->remove_singleton("GaussianSplatManager");
                }
                gaussian_splat_manager_singleton->finalize_module();
                memdelete(gaussian_splat_manager_singleton);
                gaussian_splat_manager_singleton = nullptr;
            }
            if (gaussian_splat_scene_director_singleton) {
                if (Engine::get_singleton()->has_singleton("GaussianSplatSceneDirector")) {
                    Engine::get_singleton()->remove_singleton("GaussianSplatSceneDirector");
                }
                memdelete(gaussian_splat_scene_director_singleton);
                gaussian_splat_scene_director_singleton = nullptr;
            }
            // Cleanup Custom Performance Monitors
            GaussianSplattingPerformanceMonitors::destroy_singleton();
            break;

#ifdef TOOLS_ENABLED
        case MODULE_INITIALIZATION_LEVEL_EDITOR:
            // Cleanup PLY importer
            if (ply_importer.is_valid()) {
                ResourceFormatImporter::get_singleton()->remove_importer(ply_importer);
                ply_importer.unref();
            }
            // Cleanup SPZ importer
            if (spz_importer.is_valid()) {
                ResourceFormatImporter::get_singleton()->remove_importer(spz_importer);
                spz_importer.unref();
            }
            if (gsplatworld_importer.is_valid()) {
                ResourceFormatImporter::get_singleton()->remove_importer(gsplatworld_importer);
                gsplatworld_importer.unref();
            }
            break;
#endif

        default:
            break;
    }
}
