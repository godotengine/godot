#ifndef TILE_RENDER_STAGES_H
#define TILE_RENDER_STAGES_H

// Split TileRenderer stage/helper structs. Included inside TileRenderer's private section.
    struct TileRendererDebugStats {
        explicit TileRendererDebugStats(TileRenderer &p_owner) : owner(p_owner) {}

        void clear_counters(RenderingDevice *p_device);
        void create_buffers(RenderingDevice *p_device);
        void free_buffers(RenderingDevice *p_device);
        void update_splat_audit_buffer(RenderingDevice *p_device, const RenderParams &p_params, uint64_t p_frame_serial);
        void dump_gpu_debug_counters(RenderingDevice *p_device, const RenderParams &p_params, uint64_t p_frame_serial);
        DebugCounterSnapshot get_debug_counters(RenderingDevice *p_device, uint64_t p_frame_serial) const;
        OverflowStatsSnapshot get_overflow_stats(RenderingDevice *p_device, uint64_t p_frame_serial) const;
        SplatAuditSnapshot get_splat_audit_snapshot(RenderingDevice *p_device, uint64_t p_frame_serial) const;
        void on_debug_counters_readback(const Vector<uint8_t> &p_data);
        void on_overflow_stats_readback(const Vector<uint8_t> &p_data);
        void on_splat_audit_readback(const Vector<uint8_t> &p_data);
        void _log_readback_state_if_needed(uint64_t p_frame_serial, bool p_log_enabled) const;
        void _schedule_debug_readbacks(RenderingDevice *p_device, uint64_t p_frame_serial,
                bool p_has_debug_buffer, bool p_has_overflow_buffer);
        void _log_debug_counters(uint64_t p_frame_serial, bool p_log_enabled,
                bool p_has_debug_data, bool p_has_overflow_data);
        void _log_overflow_stats(uint64_t p_frame_serial, bool p_log_enabled, bool p_has_overflow_data);

        TileRenderer &owner;
        RID overflow_statistics_buffer;
        RID debug_counter_buffer;  // Debug counters for tracking rejection reasons
        RID debug_splat_audit_buffer;
        BufferOwnership debug_counter_owner;
        BufferOwnership overflow_stats_owner;
        BufferOwnership debug_splat_audit_owner;

        mutable DebugCounterSnapshot cached_debug_counters;
        mutable OverflowStatsSnapshot cached_overflow_stats;
        mutable SplatAuditSnapshot cached_splat_audit_snapshot;
        mutable uint64_t cached_debug_frame_serial = 0;
        mutable uint64_t cached_overflow_frame_serial = 0;
        mutable uint64_t cached_splat_audit_frame_serial = 0;
        mutable DebugCounterSnapshot last_logged_debug_counters;
        mutable uint32_t last_logged_overlap_records = 0;
        mutable uint64_t last_logged_frame_serial = 0;
        mutable bool last_logged_tighter_bounds = false;
        mutable bool last_logged_valid = false;

        struct AsyncReadbackState {
            bool pending = false;
            uint64_t requested_frame_serial = 0;
        };

        mutable AsyncReadbackState debug_counter_readback;
        mutable AsyncReadbackState overflow_stats_readback;
        mutable AsyncReadbackState splat_audit_readback;
    };

    struct TileRenderParamsBuilder {
        explicit TileRenderParamsBuilder(TileRenderer &p_owner) : owner(p_owner) {}

        TileRenderParamsGPU build_params(const RenderParams &p_params, uint32_t p_overlap_record_count,
                uint32_t p_resolved_total_gaussians, RenderingDevice *p_resource_device,
                float p_overlap_keep_ratio = 1.0f);

    private:
        Vector2i _resolve_param_view(const RenderParams &p_params, RenderingDevice *p_resource_device,
                const RID &p_viewport_texture) const;

        TileRenderer &owner;
    };

    struct TilePrefixScanStage {
        explicit TilePrefixScanStage(TileRenderer &p_owner) : owner(p_owner) {}

        using PrefixParams = GaussianSplatting::TilePrefixParamsLayout;

        enum class PrefixOverflowMode : uint8_t {
            ASYNC_ESTIMATE = 0,
            DETERMINISTIC_SYNC_READBACK = 1,
            CPU_EMERGENCY = 2,
        };

        struct PrefixDispatchContext {
            RenderingDevice *device = nullptr;
            RD::ComputeListID compute_list = RD::INVALID_ID;
            RID buffer_uniform_set;
            RID param_uniform_set;
            uint32_t workgroup_count = 0;
        };

        PrefixParams build_prefix_params() const;
        RID create_prefix_param_uniform_set(RenderingDevice *p_device, const PrefixParams &p_params);
        RID acquire_prefix_uniform_set(RenderingDevice *p_device);
        PrefixOverflowMode decide_overflow_mode(bool p_allow_sync_readback, bool p_used_cpu_fallback) const;
        bool dispatch_pass1(const PrefixDispatchContext &p_context) const;
        bool dispatch_pass2_hierarchical(const PrefixDispatchContext &p_context) const;
        bool dispatch_pass3(const PrefixDispatchContext &p_context) const;
        bool dispatch_pass2_command(const PrefixDispatchContext &p_context,
                const GaussianSplatting::TilePrefixPass2ControlLayout &p_control) const;
        bool run_cpu_prefix_fallback(RenderingDevice *p_device, const PrefixParams &p_params,
                uint32_t &r_record_count, uint32_t &r_raw_record_count);
        uint64_t dispatch_prefix(uint32_t p_dispatch_x, RID p_pipeline, RID p_buffer_uniform_set, RID p_param_uniform_set,
                RenderingDevice *p_submission_device, bool p_requires_sync);
        bool update_global_tile_ranges(const RID &p_gaussian_buffer, const RID &p_sorted_indices, RenderingDevice *p_device,
                uint32_t &r_record_count, uint32_t &r_raw_record_count, bool p_allow_sync_readback);

        TileRenderer &owner;
        RID cached_binning_prefix_uniform_set;
        RID cached_binning_prefix_uniform_set_alt;
        RenderingDevice *cached_binning_prefix_device = nullptr;
        PrefixParams cached_prefix_params;
        bool cached_prefix_params_valid = false;
        RID cached_prefix_param_buffer;
        RID cached_binning_prefix_tile_counts;     // Double-buffer tracking.
        RID cached_binning_prefix_tile_counts_alt;  // Double-buffer tracking.
        uint64_t cached_generation = 0; // Replaces per-RID owner-side dependency checks.
    };

    struct TileBinningStage {
        explicit TileBinningStage(TileRenderer &p_owner) : owner(p_owner) {}

        struct BinningUniformSets {
            RID param_uniform_set;
            RID buffer_uniform_set;
            RID lighting_uniform_set;
        };

        void prepare_count_uniform_sets(RenderingDevice *p_device, const RID &p_gaussian_buffer, const RID &p_sorted_indices,
                const RenderParams &p_params,
                BinningUniformSets &r_sets);
        void prepare_emit_uniform_sets(RenderingDevice *p_device, const RID &p_gaussian_buffer, const RID &p_sorted_indices,
                const RenderParams &p_params,
                BinningUniformSets &r_sets);
        void clear_tile_counts(RenderingDevice *p_device) const;
        RID acquire_binning_buffer_uniform_set(RenderingDevice *p_device, const RID &p_gaussian_buffer, const RID &p_sorted_indices);
        RID acquire_binning_count_uniform_set(RenderingDevice *p_device, const RID &p_gaussian_buffer, const RID &p_sorted_indices);
        RID acquire_binning_param_uniform_set(RenderingDevice *p_device);
        RID create_binning_lighting_uniform_set(RenderingDevice *p_device, const RenderParams &p_params);
        uint64_t dispatch_tile_binning(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
                RID p_lighting_uniform_set,
                RenderingDevice *p_submission_device, bool p_requires_sync);
        uint64_t dispatch_tile_binning_count(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
                RID p_lighting_uniform_set,
                RenderingDevice *p_submission_device, bool p_requires_sync);

        TileRenderer &owner;
        // Cached GPU uniform set objects.
        RID cached_binning_buffer_uniform_set;
        RID cached_binning_buffer_uniform_set_alt;
        RID cached_binning_count_uniform_set;
        RID cached_binning_count_uniform_set_alt;
        RID cached_binning_param_uniform_set;
        RenderingDevice *cached_binning_buffer_device = nullptr;
        RenderingDevice *cached_binning_count_device = nullptr;
        RenderingDevice *cached_binning_param_device = nullptr;
        // Per-call parameter tracking (varies per-frame, not covered by generation).
        RID cached_binning_gaussian_buffer;
        RID cached_binning_sorted_indices;
        RID cached_binning_count_gaussian_buffer;
        RID cached_binning_count_sorted_indices;
        // Double-buffer tracking for tile_counts (flips each frame).
        RID cached_binning_tile_counts;
        RID cached_binning_tile_counts_alt;
        RID cached_binning_count_tile_counts;
        RID cached_binning_count_tile_counts_alt;
        // Generation counter replaces 21 per-RID owner-side dependency checks.
        uint64_t cached_generation = 0;
    };

    struct TileRasterizerStage {
        explicit TileRasterizerStage(TileRenderer &p_owner) : owner(p_owner) {}

        struct RasterUniformSets {
            RID param_uniform_set;
            RID buffer_uniform_set;
            RID image_uniform_set;
        };

        bool prepare_compute_uniform_sets(RenderingDevice *p_device, const RID &p_state_uniform, const RID &p_gaussian_buffer,
                const RID &p_sorted_indices, RasterUniformSets &r_sets);
        bool prepare_fragment_uniform_sets(RenderingDevice *p_device, const RID &p_state_uniform, const RID &p_gaussian_buffer,
                const RID &p_sorted_indices, RasterUniformSets &r_sets);
        RID acquire_raster_buffer_uniform_set(RenderingDevice *p_device, const RID &p_gaussian_buffer, const RID &p_sorted_indices);
        RID acquire_raster_compute_buffer_uniform_set(RenderingDevice *p_device, const RID &p_gaussian_buffer, const RID &p_sorted_indices);
        RID acquire_raster_param_uniform_set(RenderingDevice *p_device, const RID &p_state_uniform);
        RID acquire_raster_compute_param_uniform_set(RenderingDevice *p_device, const RID &p_state_uniform);
        RID acquire_raster_image_uniform_set(RenderingDevice *p_device);
        uint64_t dispatch_tile_rasterizer_compute(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
                RID p_image_uniform_set, RenderingDevice *p_submission_device);
        uint64_t dispatch_tile_rasterizer(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
                RenderingDevice *p_submission_device);

        TileRenderer &owner;
        // Cached GPU uniform set objects.
        RID cached_raster_buffer_uniform_set;
        RID cached_raster_param_uniform_set;
        RID cached_raster_compute_buffer_uniform_set;
        RID cached_raster_compute_param_uniform_set;
        RID cached_raster_image_uniform_set;
        RenderingDevice *cached_raster_buffer_device = nullptr;
        RenderingDevice *cached_raster_param_device = nullptr;
        RenderingDevice *cached_raster_compute_buffer_device = nullptr;
        RenderingDevice *cached_raster_compute_param_device = nullptr;
        RenderingDevice *cached_raster_image_device = nullptr;
        // Per-call parameter tracking (varies per-frame, not covered by generation).
        RID cached_raster_gaussian_buffer;
        RID cached_raster_sorted_indices;
        RID cached_raster_compute_gaussian_buffer;
        RID cached_raster_compute_sorted_indices;
        RID cached_state_uniform;
        RID cached_raster_compute_state_uniform;
        RID cached_raster_image_output;
        RID cached_raster_image_depth;
        RID cached_raster_image_normal;
        // Generation counter replaces 17 per-RID owner-side dependency checks.
        uint64_t cached_generation = 0;
    };

    struct TileResolveStage {
        explicit TileResolveStage(TileRenderer &p_owner) : owner(p_owner) {}

        struct ResolvePushConstants {
            int32_t viewport_width = 0;
            int32_t viewport_height = 0;
            int32_t tile_size_pixels = 0;
            float feather_pixels = 0.0f;
            int32_t tiles_x = 0;
            int32_t tiles_y = 0;
            int32_t last_tile_width = 0;
            int32_t last_tile_height = 0;
            int32_t debug_visualize_tiles = 0;
            int32_t use_texel_fetch_sampling = 0;
            int32_t output_is_premultiplied = 0;
            int32_t padding1 = 0;
        };

        static_assert(sizeof(ResolvePushConstants) == 48, "ResolvePushConstants size must match tile_resolve.glsl layout");

        void destroy_resolve_textures();
        void ensure_resolve_resources(const Vector2i &p_size, RD::DataFormat p_format);
        void ensure_resolve_pipeline(RenderingDevice *p_device, RD::DataFormat p_format);
        bool ensure_resolve_sampler(RenderingDevice *p_device);
        bool ensure_shadow_sampler(RenderingDevice *p_device);
        RID create_resolve_uniform_set(RenderingDevice *p_device);
        RID create_resolve_param_uniform_set(RenderingDevice *p_device);
        RID create_lighting_uniform_set(RenderingDevice *p_device, const RenderParams &p_params);
        bool ensure_fallback_lighting_buffers(RenderingDevice *p_device);
        void free_fallback_lighting_buffers(RenderingDevice *p_device);
        ResolvePushConstants build_push_constants(const Vector2i &p_viewport, int p_tile_size, bool p_output_is_premultiplied) const;
        void dispatch_tile_resolve(const Vector2i &p_viewport, int p_tile_size, bool p_output_is_premultiplied,
                const RenderParams &p_params);

        TileRenderer &owner;
        RID resolve_sampler;
        BufferOwnership resolve_sampler_owner;
        RID shadow_sampler;
        BufferOwnership shadow_sampler_owner;
        RID fallback_scene_uniform_buffer;
        RID fallback_directional_light_buffer;
        RID fallback_omni_light_buffer;
        RID fallback_spot_light_buffer;
        RID fallback_reflection_buffer;
        RID fallback_cluster_buffer;
        RID fallback_decal_texture;
        RID fallback_reflection_texture;
        RID fallback_shadow_texture;
        RID fallback_directional_shadow_texture;
        BufferOwnership fallback_lighting_owner;
    };

#endif
