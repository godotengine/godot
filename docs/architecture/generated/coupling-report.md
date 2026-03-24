# Generated Coupling Report

This report is generated from static source heuristics. It is useful for architecture reasoning, not as a perfect semantic model.

## Scope

- Source files scanned: `272`
- Subsystems scanned: `16`
- Local include edges: `836`
- Symbol-reference edges: `1685`

## Top Outgoing Include Dependencies

| File | Local include edges |
| --- | ---: |
| `renderer/gaussian_splat_renderer.cpp` | 38 |
| `register_types.cpp` | 35 |
| `renderer/gaussian_splat_renderer.h` | 32 |
| `renderer/render_pipeline_stages.cpp` | 19 |
| `interfaces/gpu_sorting_pipeline.cpp` | 15 |
| `renderer/render_sorting_orchestrator.cpp` | 15 |
| `core/gaussian_streaming.cpp` | 14 |
| `renderer/tile_render_binning.cpp` | 14 |
| `renderer/tile_render_debug_stats.cpp` | 14 |
| `renderer/tile_render_prefix_scan.cpp` | 14 |
| `renderer/tile_render_rasterizer_stage.cpp` | 14 |
| `renderer/tile_render_resolve.cpp` | 14 |
| `renderer/tile_renderer.cpp` | 14 |
| `nodes/gaussian_splat_node_3d.cpp` | 13 |
| `renderer/render_streaming_orchestrator.cpp` | 13 |

## Top Incoming Include Dependencies

| File | Incoming local includes |
| --- | ---: |
| `logger/gs_logger.h` | 76 |
| `core/gaussian_data.h` | 47 |
| `renderer/gaussian_splat_renderer.h` | 35 |
| `renderer/gaussian_gpu_layout.h` | 25 |
| `core/gaussian_splat_asset.h` | 23 |
| `core/gs_project_settings.h` | 22 |
| `core/gaussian_splat_manager.h` | 20 |
| `interfaces/sync_policy.h` | 19 |
| `renderer/gpu_sorting_config.h` | 18 |
| `renderer/gpu_debug_utils.h` | 17 |
| `renderer/tile_renderer.h` | 15 |
| `renderer/gpu_sorter.h` | 14 |
| `core/gaussian_streaming.h` | 13 |
| `logger/gs_debug_trace.h` | 13 |
| `renderer/pipeline_io_contracts.h` | 13 |

## GaussianSplatRenderer Direct-Access Hotspots

Files with explicit `GaussianSplatRenderer *`, renderer header includes, or `get_*_state()` calls.

| File | Renderer* mentions | `get_*_state()` calls | Includes renderer header |
| --- | ---: | ---: | :---: |
| `core/gaussian_splat_merge_utils.h` | 0 | 0 | yes |
| `core/gaussian_splat_scene_director.cpp` | 6 | 1 |  |
| `core/gaussian_splat_scene_director.h` | 6 | 0 | yes |
| `core/gaussian_splat_world.h` | 0 | 0 | yes |
| `core/gaussian_streaming.cpp` | 0 | 2 |  |
| `core/gaussian_streaming.h` | 0 | 7 |  |
| `core/performance_monitors.cpp` | 77 | 3 | yes |
| `core/performance_monitors.h` | 5 | 0 |  |
| `editor/gaussian_editor_plugin.cpp` | 1 | 0 |  |
| `editor/gaussian_editor_plugin.h` | 0 | 0 | yes |
| `editor/gaussian_editor_services.cpp` | 0 | 0 | yes |
| `editor/gaussian_inspector_plugins.cpp` | 1 | 0 | yes |
| `interfaces/debug_overlay_macros.h` | 2 | 2 |  |
| `interfaces/debug_overlay_system.cpp` | 5 | 9 | yes |
| `interfaces/debug_overlay_system.h` | 11 | 0 |  |
| `interfaces/gpu_sorting_pipeline.cpp` | 3 | 9 | yes |
| `interfaces/gpu_sorting_pipeline.h` | 3 | 0 |  |
| `interfaces/interactive_state_manager.cpp` | 11 | 1 | yes |
| `interfaces/interactive_state_manager.h` | 11 | 0 |  |
| `interfaces/output_compositor.cpp` | 1 | 1 | yes |
| `interfaces/output_compositor.h` | 1 | 0 |  |
| `interfaces/painterly_renderer.cpp` | 14 | 20 | yes |
| `interfaces/painterly_renderer.h` | 14 | 0 |  |
| `nodes/gaussian_splat_container.h` | 0 | 0 | yes |
| `nodes/gaussian_splat_debug_hud.cpp` | 0 | 0 | yes |
| `nodes/gaussian_splat_node_3d.cpp` | 0 | 0 | yes |
| `nodes/gaussian_splat_node_helpers.cpp` | 0 | 2 | yes |
| `nodes/gaussian_splat_world_3d.cpp` | 0 | 2 |  |
| `nodes/gaussian_splat_world_3d.h` | 0 | 0 | yes |
| `register_types.cpp` | 0 | 0 | yes |
| `renderer/debug_overlay_methods.cpp` | 0 | 0 | yes |
| `renderer/gaussian_splat_renderer.cpp` | 1 | 27 | yes |
| `renderer/gaussian_splat_renderer.h` | 3 | 4 |  |
| `renderer/gaussian_splat_renderer_bindings.cpp` | 0 | 0 | yes |
| `renderer/instance_pipeline_contract.h` | 0 | 0 | yes |
| `renderer/render_config_orchestrator.cpp` | 1 | 0 |  |
| `renderer/render_config_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_data_orchestrator.cpp` | 2 | 28 | yes |
| `renderer/render_data_orchestrator.h` | 2 | 0 |  |
| `renderer/render_debug_state_orchestrator.cpp` | 2 | 9 |  |
| `renderer/render_debug_state_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_device_orchestrator.cpp` | 1 | 6 |  |
| `renderer/render_device_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_diagnostics_orchestrator.cpp` | 3 | 133 |  |
| `renderer/render_diagnostics_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_instancing_orchestrator.cpp` | 1 | 19 |  |
| `renderer/render_instancing_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_output_orchestrator.cpp` | 1 | 21 |  |
| `renderer/render_output_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_pipeline_stages.cpp` | 16 | 56 |  |
| `renderer/render_pipeline_stages.h` | 2 | 0 | yes |
| `renderer/render_quality_orchestrator.cpp` | 1 | 9 |  |
| `renderer/render_quality_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_resource_orchestrator.cpp` | 1 | 112 |  |
| `renderer/render_resource_orchestrator.h` | 2 | 0 | yes |
| `renderer/render_sorting_orchestrator.cpp` | 3 | 131 | yes |
| `renderer/render_sorting_orchestrator.h` | 2 | 0 |  |
| `renderer/render_streaming_orchestrator.cpp` | 2 | 23 |  |
| `renderer/render_streaming_orchestrator.h` | 2 | 0 | yes |
| `renderer/rendering_diagnostics.cpp` | 6 | 0 | yes |
| `renderer/rendering_diagnostics.h` | 6 | 0 |  |

## Top Renderer State Accessors

| File | Accessors |
| --- | --- |
| `renderer/render_diagnostics_orchestrator.cpp` | `get_debug_state` x24, `get_device_state` x2, `get_frame_state` x21, `get_performance_state` x28, `get_resource_state` x6, `get_scene_state` x3, `get_sorting_state` x15, `get_subsystem_state` x29, `get_view_state` x5 |
| `renderer/render_sorting_orchestrator.cpp` | `get_debug_state` x15, `get_device_state` x26, `get_frame_state` x31, `get_performance_state` x50, `get_resource_state` x1, `get_scene_state` x2, `get_streaming_state` x2, `get_test_data_state` x2, `get_view_state` x2 |
| `renderer/render_resource_orchestrator.cpp` | `get_performance_state` x28, `get_subsystem_state` x23, `get_test_data_state` x48, `get_tile_renderer_state` x13 |
| `renderer/render_pipeline_stages.cpp` | `get_cache_state` x5, `get_debug_state` x5, `get_device_state` x2, `get_frame_state` x11, `get_performance_state` x7, `get_resource_state` x4, `get_scene_state` x2, `get_sorting_state` x2, `get_streaming_state` x3, `get_subsystem_state` x9, `get_tile_renderer_state` x2, `get_view_state` x4 |
| `renderer/render_data_orchestrator.cpp` | `get_device_state` x1, `get_frame_state` x3, `get_performance_state` x9, `get_resource_state` x3, `get_sorting_state` x6, `get_subsystem_state` x6 |
| `renderer/gaussian_splat_renderer.cpp` | `get_cache_state` x1, `get_device_state` x3, `get_frame_state` x2, `get_global_atlas_state` x1, `get_interactive_state` x2, `get_performance_state` x1, `get_pipeline_state` x2, `get_resource_state` x4, `get_scene_state` x2, `get_sorting_state` x2, `get_streaming_state` x2, `get_subsystem_state` x5 |
| `renderer/render_streaming_orchestrator.cpp` | `get_debug_state` x2, `get_device_state` x2, `get_frame_state` x2, `get_global_atlas_state` x1, `get_performance_state` x4, `get_resource_state` x2, `get_scene_state` x4, `get_streaming_state` x4, `get_subsystem_state` x2 |
| `renderer/render_output_orchestrator.cpp` | `get_cache_state` x1, `get_device_state` x8, `get_frame_state` x1, `get_resource_state` x2, `get_scene_state` x1, `get_test_data_state` x1, `get_view_state` x7 |
| `interfaces/painterly_renderer.cpp` | `get_debug_state` x1, `get_device_state` x5, `get_frame_state` x1, `get_performance_state` x1, `get_resource_state` x1, `get_scene_state` x1, `get_sorting_state` x1, `get_streaming_state` x1, `get_subsystem_state` x4, `get_tile_renderer_state` x1, `get_view_state` x3 |
| `renderer/render_instancing_orchestrator.cpp` | `get_cache_state` x1, `get_debug_state` x2, `get_frame_state` x7, `get_performance_state` x2, `get_resource_state` x1, `get_scene_state` x1, `get_sorting_state` x2, `get_streaming_state` x2, `get_subsystem_state` x1 |
| `interfaces/debug_overlay_system.cpp` | `get_debug_state` x4, `get_device_state` x1, `get_frame_state` x1, `get_performance_state` x1, `get_sorting_state` x1, `get_subsystem_state` x1 |
| `interfaces/gpu_sorting_pipeline.cpp` | `get_device_state` x1, `get_frame_state` x2, `get_performance_state` x1, `get_sort_external_buffer_state` x1, `get_sorting_state` x1, `get_subsystem_state` x1, `get_view_state` x2 |
| `renderer/render_debug_state_orchestrator.cpp` | `get_debug_state` x1, `get_device_state` x1, `get_frame_state` x3, `get_streaming_state` x1, `get_subsystem_state` x2, `get_view_state` x1 |
| `renderer/render_quality_orchestrator.cpp` | `get_frame_state` x2, `get_performance_state` x2, `get_scene_state` x1, `get_sorting_state` x2, `get_test_data_state` x2 |
| `core/gaussian_streaming.h` | `get_global_atlas_state` x7 |

## Files Including `gaussian_splat_renderer.h`

| File |
| --- |
| `core/gaussian_splat_merge_utils.h` |
| `core/gaussian_splat_scene_director.h` |
| `core/gaussian_splat_world.h` |
| `core/performance_monitors.cpp` |
| `editor/gaussian_editor_plugin.h` |
| `editor/gaussian_editor_services.cpp` |
| `editor/gaussian_inspector_plugins.cpp` |
| `interfaces/debug_overlay_system.cpp` |
| `interfaces/gpu_sorting_pipeline.cpp` |
| `interfaces/interactive_state_manager.cpp` |
| `interfaces/output_compositor.cpp` |
| `interfaces/painterly_renderer.cpp` |
| `nodes/gaussian_splat_container.h` |
| `nodes/gaussian_splat_debug_hud.cpp` |
| `nodes/gaussian_splat_node_3d.cpp` |
| `nodes/gaussian_splat_node_helpers.cpp` |
| `nodes/gaussian_splat_world_3d.h` |
| `register_types.cpp` |
| `renderer/debug_overlay_methods.cpp` |
| `renderer/gaussian_splat_renderer.cpp` |
| `renderer/gaussian_splat_renderer_bindings.cpp` |
| `renderer/instance_pipeline_contract.h` |
| `renderer/render_config_orchestrator.h` |
| `renderer/render_data_orchestrator.cpp` |
| `renderer/render_debug_state_orchestrator.h` |
| `renderer/render_device_orchestrator.h` |
| `renderer/render_diagnostics_orchestrator.h` |
| `renderer/render_instancing_orchestrator.h` |
| `renderer/render_output_orchestrator.h` |
| `renderer/render_pipeline_stages.h` |
| `renderer/render_quality_orchestrator.h` |
| `renderer/render_resource_orchestrator.h` |
| `renderer/render_sorting_orchestrator.cpp` |
| `renderer/render_streaming_orchestrator.h` |
| `renderer/rendering_diagnostics.cpp` |

## Notes

- Include edges show build-time/module coupling.
- Symbol-reference edges add a rough semantic layer but are heuristic-based.
- Renderer direct-access metrics are the best signal here for renderer-centric architecture leakage.
