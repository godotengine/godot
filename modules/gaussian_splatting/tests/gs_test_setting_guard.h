#pragma once

#include "core/config/project_settings.h"
#include "core/variant/variant.h"

namespace {

class ProjectSettingGuard {
    ProjectSettings *settings = nullptr;
    String setting_path;
    Variant previous_value;
    bool had_previous_value = false;

public:
    ProjectSettingGuard(ProjectSettings *p_settings, const String &p_setting_path) :
            settings(p_settings), setting_path(p_setting_path) {
        if (settings && settings->has_setting(setting_path)) {
            previous_value = settings->get_setting(setting_path);
            had_previous_value = true;
        }
    }

    ~ProjectSettingGuard() {
        if (!settings) {
            return;
        }

        if (had_previous_value) {
            settings->set_setting(setting_path, previous_value);
        } else {
            settings->clear(setting_path);
        }

        settings->save();
    }
};

} // namespace
