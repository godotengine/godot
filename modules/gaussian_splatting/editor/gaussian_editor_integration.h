#ifndef GAUSSIAN_EDITOR_INTEGRATION_H
#define GAUSSIAN_EDITOR_INTEGRATION_H

#ifdef TOOLS_ENABLED

#include "core/object/ref_counted.h"

class EditorFileDialog;
class GaussianEditorPlugin;

class GaussianEditorIntegration : public RefCounted {
    GDCLASS(GaussianEditorIntegration, RefCounted);

    GaussianEditorPlugin *editor_plugin = nullptr;
    EditorFileDialog *file_dialog = nullptr;
    bool filters_configured = false;

    void _ensure_file_dialog_filters();
    void _on_filesystem_changed();

protected:
    static void _bind_methods();

public:
    void setup(GaussianEditorPlugin *p_plugin, EditorFileDialog *p_dialog);
    void teardown();
    String normalize_path(const String &p_path) const;
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_EDITOR_INTEGRATION_H
