#ifdef TOOLS_ENABLED

#include "gaussian_editor_integration.h"

#include "core/config/project_settings.h"
#include "core/string/translation.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "gaussian_editor_plugin.h"

void GaussianEditorIntegration::_bind_methods() {}

void GaussianEditorIntegration::setup(GaussianEditorPlugin *p_plugin, EditorFileDialog *p_dialog) {
    teardown();
    editor_plugin = p_plugin;
    file_dialog = p_dialog;
    _ensure_file_dialog_filters();

    if (EditorFileSystem *fs = EditorFileSystem::get_singleton()) {
        if (!fs->is_connected("filesystem_changed", callable_mp(this, &GaussianEditorIntegration::_on_filesystem_changed))) {
            fs->connect("filesystem_changed", callable_mp(this, &GaussianEditorIntegration::_on_filesystem_changed));
        }
    }
}

void GaussianEditorIntegration::teardown() {
    if (EditorFileSystem *fs = EditorFileSystem::get_singleton()) {
        if (fs->is_connected("filesystem_changed", callable_mp(this, &GaussianEditorIntegration::_on_filesystem_changed))) {
            fs->disconnect("filesystem_changed", callable_mp(this, &GaussianEditorIntegration::_on_filesystem_changed));
        }
    }

    editor_plugin = nullptr;
    file_dialog = nullptr;
    filters_configured = false;
}

void GaussianEditorIntegration::_ensure_file_dialog_filters() {
    if (!file_dialog || filters_configured) {
        return;
    }
    file_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
    file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
    file_dialog->clear_filters();
    file_dialog->add_filter("*.ply", TTR("Gaussian PLY"));
    file_dialog->add_filter("*.spz", TTR("Gaussian SPZ"));
    filters_configured = true;
}

String GaussianEditorIntegration::normalize_path(const String &p_path) const {
    if (p_path.is_empty()) {
        return p_path;
    }

    if (p_path.begins_with("res://")) {
        return p_path;
    }

    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return ps->localize_path(p_path);
    }

    return p_path;
}

void GaussianEditorIntegration::_on_filesystem_changed() {
    if (editor_plugin) {
        editor_plugin->_internal_refresh_active_asset_metadata();
    }
}

#endif // TOOLS_ENABLED
