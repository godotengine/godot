if get_option('tools')

    # NOTE: do not use files() here
    _exporter_list = [
        'android',
        'iphone',
        'javascript',
        'linuxbsd',
        'osx',
        'uwp',
        'windows'
    ]

    _exporter_target = custom_target(
        'register_exporters',
        output: ['register_exporters.gen.cpp'],
        command: [SCRIPT_REGISTER_EXPORTERS, _exporter_list, '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_exporter_target]

    # need to built the sources
    SOURCES += files([
        '../platform/android/export/export.cpp',
        '../platform/iphone/export/export.cpp',
        '../platform/javascript/export/export.cpp',
        '../platform/linuxbsd/export/export.cpp',
        '../platform/osx/export/export.cpp',
        '../platform/uwp/export/export.cpp',
        '../platform/windows/export/export.cpp',
    ])

    ################################################################################
    # Doc generation
    ################################################################################

    # Currently, this will only build once. If you need to regenerate the doc files,
    # either delete the generated doc files, or reconfigure.
    # This is because we glob the files in the directories.

    # We can fix this by having doc_tool output a class_list.xml file (or the like)
    # that we can depend on, and auto detect doc changes and rebuild the docs.

    # generated version file
    _docs_target = custom_target(
        'docs',
        output: ['doc_data_class_path.gen.h', 'doc_data_compressed.gen.h'],
        input: [MODULES_DB_FILE],
        command: [SCRIPT_DOCS, '@INPUT@', meson.project_source_root(), '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_docs_target]


    # Translations also will run once

    _doc_translations_target = custom_target(
        'doc_translations',
        output: ['doc_translations.gen.h'],
        command: [SCRIPT_TRANSLATIONS, 'doc/translations', meson.project_source_root(), 'doc', '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_doc_translations_target]

    _editor_translations_target = custom_target(
        'editor_translations',
        output: ['editor_translations.gen.h'],
        command: [SCRIPT_TRANSLATIONS, 'editor/translations', meson.project_source_root(), 'editor', '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_editor_translations_target]

    # ...and so will fonts

    _fonts_target = custom_target(
        'fonts',
        output: ['builtin_fonts.gen.h'],
        command: [SCRIPT_FONTS, 'thirdparty/fonts', meson.project_source_root(), '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_fonts_target]

    # ...aaaaaaaaaaaaaaand editor icons
    _editor_icons_target = custom_target(
        'editor_icons',
        output: ['editor_icons.gen.h'],
        command: [SCRIPT_EDITOR_ICONS, 'editor/icons', meson.project_source_root(), '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_editor_icons_target]


    # Add our sources
    SOURCES += files([
        'animation_bezier_editor.cpp',
        'animation_track_editor.cpp',
        'animation_track_editor_plugins.cpp',
        'array_property_edit.cpp',
        'audio_stream_preview.cpp',
        'code_editor.cpp',
        'connections_dialog.cpp',
        'create_dialog.cpp',
        'dependency_editor.cpp',
        'dictionary_property_edit.cpp',
        'doc_tools.cpp',
        'editor_about.cpp',
        'editor_asset_installer.cpp',
        'editor_atlas_packer.cpp',
        'editor_audio_buses.cpp',
        'editor_autoload_settings.cpp',
        'editor_data.cpp',
        'editor_dir_dialog.cpp',
        'editor_export.cpp',
        'editor_feature_profile.cpp',
        'editor_file_dialog.cpp',
        'editor_file_system.cpp',
        'editor_folding.cpp',
        'editor_fonts.cpp',
        'editor_help.cpp',
        'editor_help_search.cpp',
        'editor_inspector.cpp',
        'editor_layouts_dialog.cpp',
        'editor_log.cpp',
        'editor_native_shader_source_visualizer.cpp',
        'editor_node.cpp',
        'editor_path.cpp',
        'editor_plugin.cpp',
        'editor_plugin_settings.cpp',
        'editor_properties.cpp',
        'editor_properties_array_dict.cpp',
        'editor_resource_preview.cpp',
        'editor_run.cpp',
        'editor_run_native.cpp',
        'editor_run_script.cpp',
        'editor_scale.cpp',
        'editor_sectioned_inspector.cpp',
        'editor_settings.cpp',
        'editor_spin_slider.cpp',
        'editor_sub_scene.cpp',
        'editor_themes.cpp',
        'editor_translation_parser.cpp',
        'editor_vcs_interface.cpp',
        'export_template_manager.cpp',
        'filesystem_dock.cpp',
        'find_in_files.cpp',
        'groups_editor.cpp',
        'import_dock.cpp',
        'input_map_editor.cpp',
        'inspector_dock.cpp',
        'localization_editor.cpp',
        'multi_node_edit.cpp',
        'node_3d_editor_gizmos.cpp',
        'node_dock.cpp',
        'plugin_config_dialog.cpp',
        'pot_generator.cpp',
        'progress_dialog.cpp',
        'project_export.cpp',
        'project_manager.cpp',
        'project_settings_editor.cpp',
        'property_editor.cpp',
        'property_selector.cpp',
        'quick_open.cpp',
        'rename_dialog.cpp',
        'reparent_dialog.cpp',
        'scene_tree_dock.cpp',
        'scene_tree_editor.cpp',
        'script_create_dialog.cpp',
        'settings_config_dialog.cpp',
        'shader_globals_editor.cpp',
    ])

    subdir('debugger')
    subdir('fileserver')
    subdir('import')
    subdir('plugins')

endif
