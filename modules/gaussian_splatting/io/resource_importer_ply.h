#ifndef RESOURCE_IMPORTER_PLY_H
#define RESOURCE_IMPORTER_PLY_H

#ifdef TOOLS_ENABLED

#include "core/io/resource_importer.h"
#include "core/object/ref_counted.h"
#include "../core/gaussian_splat_asset.h"

class ResourceImporterPLY : public ResourceImporter {
    GDCLASS(ResourceImporterPLY, ResourceImporter);

public:
    virtual String get_importer_name() const override;
    virtual String get_visible_name() const override;
    virtual void get_recognized_extensions(List<String> *p_extensions) const override;
    virtual String get_save_extension() const override;
    virtual String get_resource_type() const override;
    virtual int get_preset_count() const override;
    virtual String get_preset_name(int p_idx) const override;
    virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset = 0) const override;
    virtual bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override;

    virtual Error import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) override;

    // Import runs on a WorkerThreadPool thread when this returns true, but our
    // thumbnail path (ResourceSaver::save -> ImageTexture::_get("image") ->
    // RenderingServer::texture_2d_get) is FUNC1RC = synchronous push_and_ret to
    // the render thread. In `--headless --import` the main thread sits in
    // EditorFileSystem's ep->step busy-loop and never pumps the RS command
    // queue, so every worker deadlocks. Force serial import until the
    // thumbnail pipeline no longer takes a sync RS round-trip.
    virtual bool can_import_threaded() const override { return false; }
    virtual bool has_advanced_options() const override;
    virtual void show_advanced_options(const String &p_path) override;

    // Bump whenever importer behavior changes in a way that requires existing
    // .tres caches to be re-imported. Godot's resource scanner compares this
    // against the value stored in each .ply.import file and re-runs import()
    // when they differ, so users do NOT need to manually wipe .godot/imported/
    // after a fix lands in the importer.
    //   v0 (implicit): pre-versioning baseline.
    //   v1: switch to versioned importer.
    //   v2: optional Packed*Array fields are now zero-initialized at import
    //       time (see gaussian_splat_asset.cpp::_ensure_buffer_sizes —
    //       resize_initialized() instead of resize() for POD vectors).
    //       Caches written by v0/v1 may contain 0xC0C0C0C0 poison and must
    //       be re-imported.
    virtual int get_format_version() const override { return 2; }

    // Validation helpers
    Error validate_ply_properties(const Ref<class PLYLoader> &p_loader) const;
    void log_missing_properties(const Ref<class PLYLoader> &p_loader) const;

    ResourceImporterPLY();
};

#endif // TOOLS_ENABLED

#endif // RESOURCE_IMPORTER_PLY_H
