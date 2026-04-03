#include "gaussian_splat_dynamic_instance_3d.h"

#include "../core/gs_project_settings.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../logger/gs_logger.h"

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"

namespace {
static bool _is_data_log_enabled() { return gs::settings::is_data_log_enabled(); }
} // namespace

void GaussianSplatDynamicInstance3D::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_ply_file_path", "path"), &GaussianSplatDynamicInstance3D::set_ply_file_path);
    ClassDB::bind_method(D_METHOD("get_ply_file_path"), &GaussianSplatDynamicInstance3D::get_ply_file_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "ply_file_path"), "set_ply_file_path", "get_ply_file_path");

    ClassDB::bind_method(D_METHOD("set_splat_asset", "asset"), &GaussianSplatDynamicInstance3D::set_splat_asset);
    ClassDB::bind_method(D_METHOD("get_splat_asset"), &GaussianSplatDynamicInstance3D::get_splat_asset);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "splat_asset", PROPERTY_HINT_RESOURCE_TYPE, "GaussianSplatAsset"),
            "set_splat_asset", "get_splat_asset");

    ClassDB::bind_method(D_METHOD("set_gaussian_data", "data"), &GaussianSplatDynamicInstance3D::set_gaussian_data);
    ClassDB::bind_method(D_METHOD("get_gaussian_data"), &GaussianSplatDynamicInstance3D::get_gaussian_data);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gaussian_data", PROPERTY_HINT_RESOURCE_TYPE, "GaussianData"),
            "set_gaussian_data", "get_gaussian_data");

    ClassDB::bind_method(D_METHOD("set_auto_load", "enabled"), &GaussianSplatDynamicInstance3D::set_auto_load);
    ClassDB::bind_method(D_METHOD("is_auto_load_enabled"), &GaussianSplatDynamicInstance3D::is_auto_load_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_load"), "set_auto_load", "is_auto_load_enabled");

    ClassDB::bind_method(D_METHOD("reload_asset"), &GaussianSplatDynamicInstance3D::reload_asset);
    ClassDB::bind_method(D_METHOD("register_instance"), &GaussianSplatDynamicInstance3D::register_instance);
    ClassDB::bind_method(D_METHOD("unregister_instance"), &GaussianSplatDynamicInstance3D::unregister_instance);
    ClassDB::bind_method(D_METHOD("is_registered"), &GaussianSplatDynamicInstance3D::is_registered);
}

GaussianSplatDynamicInstance3D::GaussianSplatDynamicInstance3D() {
    set_notify_transform(true);
}

GaussianSplatDynamicInstance3D::~GaussianSplatDynamicInstance3D() {
    _unregister_instance();
}

void GaussianSplatDynamicInstance3D::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            if (_is_data_log_enabled()) {
                GS_LOG_RENDERER_DEBUG(vformat("[DynamicInstance] ENTER_TREE: auto_load=%s ply=%s", auto_load ? "true" : "false", ply_file_path));
                GS_LOG_RENDERER_DEBUG("[DynamicInstance] ENTER_TREE");
            }
            if (gaussian_data.is_valid()) {
                _register_instance();
            } else if (auto_load && (splat_asset.is_valid() || !ply_file_path.is_empty())) {
                reload_asset();
            }
        } break;

        case NOTIFICATION_ENTER_WORLD: {
            _register_instance();
        } break;

        case NOTIFICATION_VISIBILITY_CHANGED: {
            if (is_visible_in_tree()) {
                _register_instance();
            } else {
                _unregister_instance();
            }
        } break;

        case NOTIFICATION_TRANSFORM_CHANGED: {
            _update_transform();
        } break;

        case NOTIFICATION_EXIT_TREE: {
            _unregister_instance();
        } break;
    }
}

uint32_t GaussianSplatDynamicInstance3D::_get_instance_flags() const {
    bool is_2d = false;
    if (gaussian_data.is_valid()) {
        is_2d = gaussian_data->get_2d_mode();
    } else if (splat_asset.is_valid()) {
        Dictionary import_metadata = splat_asset->get_import_metadata();
        if (import_metadata.has(StringName("gaussian_2d_mode"))) {
            is_2d = (bool)import_metadata[StringName("gaussian_2d_mode")];
        }
    } else if (runtime_asset.is_valid()) {
        Dictionary import_metadata = runtime_asset->get_import_metadata();
        if (import_metadata.has(StringName("gaussian_2d_mode"))) {
            is_2d = (bool)import_metadata[StringName("gaussian_2d_mode")];
        }
    }
    return is_2d ? 1u : 0u;
}

bool GaussianSplatDynamicInstance3D::_can_register_instance() const {
    if (!is_inside_tree() || !is_inside_world() || !is_visible_in_tree()) {
        return false;
    }
    if (gaussian_data.is_valid()) {
        return gaussian_data->get_count() > 0;
    }
    return splat_asset.is_valid() || runtime_asset.is_valid();
}

bool GaussianSplatDynamicInstance3D::_register_instance_registry() {
    if (!_can_register_instance()) {
        return false;
    }
    Ref<GaussianSplatAsset> asset = splat_asset;
    if (asset.is_null() && gaussian_data.is_valid()) {
        if (runtime_asset.is_null()) {
            runtime_asset.instantiate();
            runtime_asset->set_asset_type(GaussianSplatAsset::ASSET_TYPE_DYNAMIC);
        }
        Error asset_err = runtime_asset->populate_from_gaussian_data(gaussian_data);
        if (asset_err != OK) {
            GS_LOG_WARN_DEFAULT(vformat("[GaussianSplatDynamicInstance3D] Failed to build runtime asset for instance pipeline (err=%d).", asset_err));
        }
        asset = runtime_asset;
    } else if (asset.is_null()) {
        asset = runtime_asset;
    }
    if (asset.is_null()) {
        WARN_PRINT_ONCE("[GaussianSplatDynamicInstance3D] Instance pipeline requires a GaussianSplatAsset or GaussianData; skipping registry.");
        return false;
    }
    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    if (!director) {
        return false;
    }
    director->register_instance(get_instance_id(), asset, get_global_transform(),
            1.0f, 1.0f, _get_instance_flags(), false);
    return true;
}

void GaussianSplatDynamicInstance3D::_unregister_instance_registry() {
    if (GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton()) {
        director->unregister_instance(get_instance_id());
    }
}

void GaussianSplatDynamicInstance3D::_update_instance_transform_registry() {
    if (!is_inside_tree()) {
        return;
    }
    if (GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton()) {
        director->update_instance_transform(get_instance_id(), get_global_transform());
    }
}

void GaussianSplatDynamicInstance3D::set_ply_file_path(const String &p_path) {
    if (ply_file_path == p_path) {
        return;
    }
#ifndef DISABLE_DEPRECATED
    WARN_DEPRECATED_MSG("GaussianSplatDynamicInstance3D::ply_file_path is deprecated. Use splat_asset or gaussian_data instead.");
#endif
    ply_file_path = p_path;
    runtime_asset.unref();
    if (auto_load && is_inside_tree()) {
        reload_asset();
    }
}

void GaussianSplatDynamicInstance3D::set_splat_asset(const Ref<GaussianSplatAsset> &p_asset) {
    if (splat_asset == p_asset) {
        return;
    }
    if (splat_asset.is_valid() && splat_asset->is_connected("changed", callable_mp(this, &GaussianSplatDynamicInstance3D::_on_asset_changed))) {
        splat_asset->disconnect("changed", callable_mp(this, &GaussianSplatDynamicInstance3D::_on_asset_changed));
    }
    runtime_asset.unref();
    splat_asset = p_asset;
    if (splat_asset.is_valid()) {
        if (!splat_asset->is_connected("changed", callable_mp(this, &GaussianSplatDynamicInstance3D::_on_asset_changed))) {
            splat_asset->connect("changed", callable_mp(this, &GaussianSplatDynamicInstance3D::_on_asset_changed));
        }
    }
    if (auto_load && is_inside_tree()) {
        reload_asset();
    }
}

void GaussianSplatDynamicInstance3D::set_gaussian_data(const Ref<GaussianData> &p_data) {
    runtime_asset.unref();
    gaussian_data = p_data;
    if (gaussian_data.is_valid() && gaussian_data->get_count() == 0) {
        gaussian_data.unref();
    }
    if (is_inside_tree()) {
        if (_can_register_instance()) {
            _register_instance();
        } else {
            _unregister_instance();
        }
    }
}

void GaussianSplatDynamicInstance3D::set_auto_load(bool p_enabled) {
    auto_load = p_enabled;
}

void GaussianSplatDynamicInstance3D::reload_asset() {
    if (_is_data_log_enabled()) {
        GS_LOG_STREAMING_DEBUG("[DynamicInstance] reload_asset called");
    }
    if (gaussian_data.is_valid()) {
        _register_instance();
        return;
    }

    if (splat_asset.is_valid()) {
        if (!_populate_from_asset()) {
            GS_LOG_RENDERER_WARN("[GaussianSplatDynamicInstance3D] Failed to build GaussianData from asset.");
            _unregister_instance();
            return;
        }
        _register_instance();
        return;
    }

    if (!_load_from_file()) {
        _unregister_instance();
        return;
    }

    _register_instance();
}

void GaussianSplatDynamicInstance3D::register_instance() {
    _register_instance();
}

void GaussianSplatDynamicInstance3D::unregister_instance() {
    _unregister_instance();
}

void GaussianSplatDynamicInstance3D::_register_instance() {
    if (_is_data_log_enabled()) {
        GS_LOG_RENDERER_DEBUG(vformat("[DynamicInstance] _register_instance: in_tree=%s visible=%s data_valid=%s data_count=%d", is_inside_tree() ? "Y" : "N", is_visible_in_tree() ? "Y" : "N", gaussian_data.is_valid() ? "Y" : "N", gaussian_data.is_valid() ? gaussian_data->get_count() : 0));
    }
    if (!_can_register_instance()) {
        if (registered) {
            _unregister_instance_registry();
            registered = false;
        }
        return;
    }

    const bool was_registered = registered;
    registered = _register_instance_registry();
    if (!registered && was_registered) {
        _unregister_instance_registry();
    }
}

void GaussianSplatDynamicInstance3D::_on_asset_changed() {
    if (!is_inside_tree()) {
        return;
    }
    reload_asset();
}

void GaussianSplatDynamicInstance3D::_unregister_instance() {
    _unregister_instance_registry();
    registered = false;
}

void GaussianSplatDynamicInstance3D::_update_transform() {
    _update_instance_transform_registry();
}

bool GaussianSplatDynamicInstance3D::_load_from_file() {
    if (ply_file_path.is_empty()) {
        return false;
    }

    // Compatibility path only. Bucket B keeps it functional but no longer treats it as a
    // preferred workflow; new usage should go through splat_asset or gaussian_data.
    if (gaussian_data.is_null()) {
        gaussian_data.instantiate();
    }

    Error err = gaussian_data->load_from_file(ply_file_path);
    if (err != OK) {
        GS_LOG_RENDERER_WARN(vformat("[GaussianSplatDynamicInstance3D] Failed to load GaussianData: %s (err=%d)",
                ply_file_path, err));
        gaussian_data.unref();
        return false;
    }

    if (gaussian_data->get_count() == 0) {
        GS_LOG_RENDERER_WARN(vformat("[GaussianSplatDynamicInstance3D] Loaded GaussianData has no splats: %s",
                ply_file_path));
        gaussian_data.unref();
        return false;
    }

    return true;
}

bool GaussianSplatDynamicInstance3D::_populate_from_asset() {
    if (splat_asset.is_null() || !splat_asset->populate_gaussian_data(gaussian_data)) {
        return false;
    }

    if (gaussian_data.is_null() || gaussian_data->get_count() == 0) {
        gaussian_data.unref();
        return false;
    }

    return true;
}
