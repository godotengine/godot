
_splash_target = configure_file(
    output: 'splash.gen.h',
    input: '../main/splash.png',
    command: [find_program('../main/main_builders.py'),
              'make_splash',
              '@OUTPUT@',
              '@INPUT@',
	      ]
)
SOURCES += [_splash_target]

if not get_option('no_editor_splash')
  _splash_editor_target = custom_target(
      'splash_editor',
      output: ['splash_editor.gen.h'],
      input: ['../main/splash_editor.png'],
      env: SCRIPTS_ENV,
      command: [SCRIPT_COMPAT,
                'make_splash_editor',
                SCRIPT_MAIN_BUILDERS,
                '-i',
                '@INPUT@',
                '-o',
                '@OUTPUT@',
                '-t',
                'main',
                '--fake-stamp',
                'splash/splash_editor.gen.h'],
      build_by_default: true
  )
  SOURCES += [_splash_editor_target]
endif

_app_icon_target = configure_file(
    output: 'app_icon.gen.h',
    input: '../main/app_icon.png',
    command: [
              SCRIPT_MAIN_BUILDERS,
              'make_app_icon',
              '@INPUT@',
              '@OUTPUT@',
	      ],
)
SOURCES += [_app_icon_target]
