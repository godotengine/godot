# Project Settings Reference

Last generated: 2026-02-13

## Purpose
Use this reference to map Gaussian Splatting project setting keys to source definitions and runtime lookup paths.

## Usage
| Task | Action |
| --- | --- |
| Regenerate this reference | Run `python3 scripts/generate_project_settings_reference.py`. |
| Audit key usage in module code | Run `rg -n "rendering/gaussian_splatting/" modules/gaussian_splatting --glob '*.{h,cpp}'`. |

## API

### Registered keys
These settings are registered with `GLOBAL_DEF(...)` and grouped by key prefix.

| Coverage | Count |
| --- | ---: |
| Registered keys | 76 |
| Runtime-only keys | 33 |
| Registered keys without additional literal lookup | 9 |

#### Core

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/gpu_sorting_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:864</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/max_gpu_buffer_count</code></pre></td>
      <td><pre><code>128</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:871</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/shared_submission_device_enabled</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:865</code></pre></td>
    </tr>
  </tbody>
</table>

#### Import

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/import/use_gsplatworld_cache</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:873</code></pre></td>
    </tr>
  </tbody>
</table>

#### Quality

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/quality/tier_apply_pipeline_toggles</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:928</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/quality/tier_apply_streaming_budgets</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:929</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/quality/tier_preset</code></pre></td>
      <td><pre><code>"custom"</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:927</code></pre></td>
    </tr>
  </tbody>
</table>

#### Streaming

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/async_io_enabled</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:955</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/async_pack_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:939</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/auto_regulate_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:961</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:869</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/chunk_frustum_padding</code></pre></td>
      <td><pre><code>1.5f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:870</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:866</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/eviction_hysteresis_frames</code></pre></td>
      <td><pre><code>5</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:951</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/io_source_path</code></pre></td>
      <td><pre><code>String()</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:956</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame</code></pre></td>
      <td><pre><code>6</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:945</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/max_chunks_in_vram</code></pre></td>
      <td><pre><code>128</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:964</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/max_evictions_per_frame</code></pre></td>
      <td><pre><code>4</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:953</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/max_pack_jobs_in_flight</code></pre></td>
      <td><pre><code>4</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:943</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/max_upload_mb_per_frame</code></pre></td>
      <td><pre><code>32</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:947</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/max_upload_mb_per_slice</code></pre></td>
      <td><pre><code>4</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:949</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/min_chunks_in_vram</code></pre></td>
      <td><pre><code>4</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:963</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/pack_worker_threads</code></pre></td>
      <td><pre><code>2</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:941</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/predictive_prefetch_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:935</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/prefetch_lookahead_distance</code></pre></td>
      <td><pre><code>10.0f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:937</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/regulation_step_percent</code></pre></td>
      <td><pre><code>1.0f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:966</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/vram_budget_mb</code></pre></td>
      <td><pre><code>12288</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:960</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/streaming/vram_warning_threshold_percent</code></pre></td>
      <td><pre><code>85</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:962</code></pre></td>
    </tr>
  </tbody>
</table>

#### Culling

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/cluster_culling_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:880</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/cluster_frustum_slack</code></pre></td>
      <td><pre><code>2.0f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:882</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/cluster_target_size</code></pre></td>
      <td><pre><code>128</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:881</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/min_gaussians_per_leaf</code></pre></td>
      <td><pre><code>32</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:877</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/octree_max_depth</code></pre></td>
      <td><pre><code>8</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:876</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/opacity_aware_bounds</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:886</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/culling/visibility_threshold</code></pre></td>
      <td><pre><code>0.01f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:887</code></pre></td>
    </tr>
  </tbody>
</table>

#### Cull

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/cull/overflow_autotune_enabled</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:891</code></pre></td>
    </tr>
  </tbody>
</table>

#### Lod

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/lod/blend_distance</code></pre></td>
      <td><pre><code>5.0f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/lod/lod_config.cpp:361</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/lod/blend_enabled</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/lod/lod_config.cpp:360</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/lod/hysteresis_zone</code></pre></td>
      <td><pre><code>0.5f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/lod/lod_config.cpp:362</code></pre></td>
    </tr>
  </tbody>
</table>

#### Sorting

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/bitonic_max_elements</code></pre></td>
      <td><pre><code>(int)sorting_bitonic_max</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:969</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/force_cpu_sort</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:978</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/history_size</code></pre></td>
      <td><pre><code>(int)sorting_history_size</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:974</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/hybrid_batch_size</code></pre></td>
      <td><pre><code>(int)sorting_hybrid_batch</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:973</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/hybrid_trigger_elements</code></pre></td>
      <td><pre><code>(int)sorting_hybrid_trigger</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:972</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/log_interval_frames</code></pre></td>
      <td><pre><code>(int)sorting_log_interval</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:975</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/log_metrics</code></pre></td>
      <td><pre><code>sorting_log_metrics</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:977</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/onesweep_max_elements</code></pre></td>
      <td><pre><code>(int)sorting_onesweep_max</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:971</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/radix_max_elements</code></pre></td>
      <td><pre><code>(int)sorting_radix_max</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:970</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/sorting/target_sort_time_ms</code></pre></td>
      <td><pre><code>sorting_target_ms</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:976</code></pre></td>
    </tr>
  </tbody>
</table>

#### Debug

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/cull_guardrail_drop_ratio</code></pre></td>
      <td><pre><code>0.75f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:909</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/cull_guardrail_min_visible</code></pre></td>
      <td><pre><code>256</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:910</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/cull_guardrail_position_epsilon</code></pre></td>
      <td><pre><code>0.05f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:907</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/cull_guardrail_rotation_epsilon</code></pre></td>
      <td><pre><code>0.01f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:908</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_all_debug</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:898</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_autotune_logs</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:911</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_binning_counters</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:904</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_cull_counters</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:905</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_cull_guardrails</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:906</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_data_logging</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:912</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_frame_logging</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:894</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_frame_logging_verbose</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:895</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_gpu_counter_logs</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:903</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_mainloop_probes</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:897</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_sort_path_logs</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:899</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_tile_dispatch_logs</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:902</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_tile_logs</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:900</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/enable_tile_pipeline_logs</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:901</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/debug/frame_log_frequency</code></pre></td>
      <td><pre><code>300</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:896</code></pre></td>
    </tr>
  </tbody>
</table>

#### Logging

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/logging/rate_limit_ms</code></pre></td>
      <td><pre><code>1000</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:916</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/logging/verbosity</code></pre></td>
      <td><pre><code>"silent"</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:915</code></pre></td>
    </tr>
  </tbody>
</table>

#### Diagnostics

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>Default</th>
      <th>Defined In</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/diagnostics/perf_gate_budget_ms</code></pre></td>
      <td><pre><code>16.0f</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:924</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/diagnostics/perf_gate_enabled</code></pre></td>
      <td><pre><code>false</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:922</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/diagnostics/perf_gate_splat_threshold</code></pre></td>
      <td><pre><code>100000</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:923</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/diagnostics/summary_history_size</code></pre></td>
      <td><pre><code>60</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:921</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/diagnostics/summary_interval_frames</code></pre></td>
      <td><pre><code>600</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:920</code></pre></td>
    </tr>
    <tr>
      <td><pre><code>rendering/gaussian_splatting/diagnostics/validate_production_metrics</code></pre></td>
      <td><pre><code>true</code></pre></td>
      <td><pre><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:919</code></pre></td>
    </tr>
  </tbody>
</table>

### Runtime-only keys
These keys are used by module code but are not registered with `GLOBAL_DEF(...)`.

| Setting | First reference |
| --- | --- |
| `rendering/gaussian_splatting/composite/depth_test` | `modules/gaussian_splatting/interfaces/output_compositor.cpp:20` |
| `rendering/gaussian_splatting/cull/frustum_plane_slack` | `modules/gaussian_splatting/interfaces/gpu_culler.cpp:256` |
| `rendering/gaussian_splatting/debug/enable_pipeline_trace` | `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp:350` |
| `rendering/gaussian_splatting/debug/enable_splat_audit` | `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp:354` |
| `rendering/gaussian_splatting/debug/enable_state_guardrails` | `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp:352` |
| `rendering/gaussian_splatting/debug/force_unclustered_lights` | `modules/gaussian_splatting/renderer/gpu_debug_utils.h:101` |
| `rendering/gaussian_splatting/debug/show_density_heatmap` | `modules/gaussian_splatting/core/gaussian_splat_settings_manager.cpp:9` |
| `rendering/gaussian_splatting/debug/show_performance_hud` | `modules/gaussian_splatting/core/gaussian_splat_settings_manager.cpp:10` |
| `rendering/gaussian_splatting/debug/show_residency_hud` | `modules/gaussian_splatting/core/gaussian_splat_settings_manager.cpp:11` |
| `rendering/gaussian_splatting/debug/show_tile_grid` | `modules/gaussian_splatting/core/gaussian_splat_settings_manager.cpp:8` |
| `rendering/gaussian_splatting/debug/splat_audit_sample_count` | `modules/gaussian_splatting/renderer/render_debug_state_orchestrator.cpp:395` |
| `rendering/gaussian_splatting/lighting/dc_logit` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:148` |
| `rendering/gaussian_splatting/lighting/direct_light_scale` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:145` |
| `rendering/gaussian_splatting/lighting/indirect_sh_scale` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:146` |
| `rendering/gaussian_splatting/lighting/shadow_receiver_bias_max` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:151` |
| `rendering/gaussian_splatting/lighting/shadow_receiver_bias_min` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:150` |
| `rendering/gaussian_splatting/lighting/shadow_receiver_bias_scale` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:149` |
| `rendering/gaussian_splatting/lighting/shadow_strength` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:147` |
| `rendering/gaussian_splatting/lod/bias` | `modules/gaussian_splatting/interfaces/gpu_culler.cpp:254` |
| `rendering/gaussian_splatting/lod/enabled` | `modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp:561` |
| `rendering/gaussian_splatting/lod/importance_threshold` | `modules/gaussian_splatting/interfaces/gpu_culler.cpp:255` |
| `rendering/gaussian_splatting/lod/max_distance` | `modules/gaussian_splatting/interfaces/gpu_culler.cpp:253` |
| `rendering/gaussian_splatting/lod/min_screen_size_pixels` | `modules/gaussian_splatting/interfaces/gpu_culler.cpp:252` |
| `rendering/gaussian_splatting/logging/command_buffer` | `modules/gaussian_splatting/logger/gs_logger.cpp:46` |
| `rendering/gaussian_splatting/logging/compositor` | `modules/gaussian_splatting/logger/gs_logger.cpp:44` |
| `rendering/gaussian_splatting/logging/general` | `modules/gaussian_splatting/logger/gs_logger.cpp:34` |
| `rendering/gaussian_splatting/logging/gpu_memory` | `modules/gaussian_splatting/logger/gs_logger.cpp:42` |
| `rendering/gaussian_splatting/logging/gpu_sort` | `modules/gaussian_splatting/logger/gs_logger.cpp:40` |
| `rendering/gaussian_splatting/logging/renderer` | `modules/gaussian_splatting/logger/gs_logger.cpp:36` |
| `rendering/gaussian_splatting/logging/streaming` | `modules/gaussian_splatting/logger/gs_logger.cpp:38` |
| `rendering/gaussian_splatting/logging/tests` | `modules/gaussian_splatting/logger/gs_logger.cpp:48` |
| `rendering/gaussian_splatting/renderdoc_compatibility` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:236` |
| `rendering/gaussian_splatting/streaming/sh_progressive_load` | `modules/gaussian_splatting/renderer/sh_config.h:27` |

### Registered keys without additional literal lookup
These registered keys have no additional string-literal references beyond their registration line.

| Setting | Registered in |
| --- | --- |
| `rendering/gaussian_splatting/cull/overflow_autotune_enabled` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:891` |
| `rendering/gaussian_splatting/culling/cluster_frustum_slack` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:882` |
| `rendering/gaussian_splatting/culling/opacity_aware_bounds` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:886` |
| `rendering/gaussian_splatting/culling/visibility_threshold` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:887` |
| `rendering/gaussian_splatting/debug/enable_mainloop_probes` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:897` |
| `rendering/gaussian_splatting/max_gpu_buffer_count` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:871` |
| `rendering/gaussian_splatting/streaming/async_io_enabled` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:955` |
| `rendering/gaussian_splatting/streaming/enabled` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:866` |
| `rendering/gaussian_splatting/streaming/io_source_path` | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:956` |

## Examples
```bash
python3 scripts/generate_project_settings_reference.py
```

```bash
rg -n "rendering/gaussian_splatting/" modules/gaussian_splatting --glob '*.{h,cpp}'
```

## Troubleshooting
| Issue | Cause | Fix |
| --- | --- | --- |
| A key exists in code but not under registered sections | The key is read at runtime without `GLOBAL_DEF(...)`. | Check the `Runtime-only keys` section and decide whether to register it. |
| A registered key appears unconsumed | No extra string-literal lookup path is present in module runtime code. | Verify lookup paths or remove stale registration. |
| Line references are stale | Source moved after docs were generated. | Regenerate this file after code changes. |
