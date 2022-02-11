if get_option('tools')
    editor_builders = find_program('editor_builders.py')

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
    SOURCES += files(run_command(GLOB, '../platform/*/export/*.cpp', check: true).stdout().split('\n'))
    ################################################################################
    # Doc generation
    ################################################################################

    # Currently, this will only build once. If you need to regenerate the doc files,
    # either delete the generated doc files, or reconfigure.
    # This is because we glob the files in the directories.

    # We can fix this by having doc_tool output a class_list.xml file (or the like)
    # that we can depend on, and auto detect doc changes and rebuild the docs.

    # generated version file

    # TODO But we can do better, need some py scripts refactoring.
    #_docs_target_srcs = []
    #foreach mp : MODULE_BUILD_PATHS
    #    _tmp_module_docs = run_command(GLOB, meson.project_source_root() / mp / 'doc_classes' / '*.xml', check: true).stdout()
    #    # If GLOB returns empty meson fails -.- .
    #    if _tmp_module_docs != ''
    #        _docs_target_srcs += files(_tmp_module_docs.split('\n'))
    #    endif
    #endforeach

    #_other_docs_target = custom_target(
    #    'other-docs',
    #    input: _docs_target_srcs,
    #    output: 'other_doc_data_compressed.gen.h',
    #    env: SCRIPTS_ENV,
    #    command: [SCRIPT_COMPAT, 'make_doc_header', SCRIPT_EDITOR_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@']
    #)
    #SOURCES += [_other_docs_target]

    _docs_target = configure_file(
        output: 'doc_data_compressed.gen.h', # 'doc_data_class_path.gen.h',
        command: [editor_builders, 'make_doc_header', '@OUTPUT@'] + MODULES_ENABLED,
    )
    SOURCES += [_docs_target]


    # Translations also will run once

    _doc_translations_target_srcs = files(run_command(GLOB, meson.project_source_root() / 'doc/translations/*.po', check: true).stdout().split('\n'))
    SOURCES += custom_target('doc_translations',
        output: 'doc_translations.gen.h',
        input:  _doc_translations_target_srcs,
        command: [editor_builders, 'make_doc_translations', '@OUTPUT@', '@INPUT@'],
        )

    _editor_translations_target_srcs = files(run_command(GLOB, 'translations/*.po', check: true).stdout().split('\n'))
    SOURCES += custom_target('editor_translations',
        output: 'editor_translations.gen.h',
        input:  _editor_translations_target_srcs,
        command: [editor_builders, 'make_editor_translations_header', '@OUTPUT@', '@INPUT@'],
    )

    # ...and so will fonts

    _fonts_target_srcs = files(run_command(GLOB, meson.project_source_root() / 'thirdparty/fonts/*.ttf', check: true).stdout().split('\n'))
    builtin_fonts_gen = custom_target(
        'fonts',
        input: _fonts_target_srcs,
        output: ['builtin_fonts.gen.h'],
        env: SCRIPTS_ENV,
        command: [SCRIPT_COMPAT, 'make_fonts_header', SCRIPT_EDITOR_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [builtin_fonts_gen]
    MAIN_SOURCES += [builtin_fonts_gen]

    # ...aaaaaaaaaaaaaaand editor icons
    _editor_icons_srcs = files(run_command(GLOB, 'icons/*.svg', check: true).stdout().split('\n'))
    _editor_icons_target = custom_target(
        'editor_icons',
        input: _editor_icons_srcs,
        output: ['editor_icons.gen.h'],
        env: SCRIPTS_ENV,
        command: [SCRIPT_COMPAT, 'make_editor_icons_action', SCRIPT_EDITOR_ICON_BUILDERS, '-i', '@INPUT@', '-o', '@OUTPUT@'],
        build_by_default: true
    )
    SOURCES += [_editor_icons_target]


    # Add our sources
    SOURCES += files(run_command(GLOB, '*.cpp', check: true).stdout().split('\n'))

    subdir('debugger')
    subdir('fileserver')
    subdir('import')
    subdir('plugins')

endif
