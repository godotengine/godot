templates_gen = custom_target('editor_templates',
    input: ['CharacterBody2D/basic_movement.gd',
            'CharacterBody3D/basic_movement.gd',
            'EditorPlugin/plugin.gd',
            'EditorScript/basic_editor_script.gd',
            'Node/default.gd',
            'Object/empty.gd'],
    output: 'templates.gen.h',
    command: [template_builders_bin, '@OUTPUT@', '@INPUT@'])
