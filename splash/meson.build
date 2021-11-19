
_splash_target = custom_target(
    'splash',
    output: ['splash.gen.h'],
    input: ['../main/splash.png'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT,
              'make_splash',
              SCRIPT_MAIN_BUILDERS,
              '-i',
              '@INPUT@',
              '-o',
              '@OUTPUT@',
              '-t',
              'main',
              '--fake-stamp',
              'splash/splash.gen.h'],
    build_by_default: true
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

_app_icon_target = custom_target(
    'app_icon',
    output: ['app_icon.gen.h'],
    input: ['../main/app_icon.png'],
    env: SCRIPTS_ENV,
    command: [SCRIPT_COMPAT,
              'make_app_icon',
              SCRIPT_MAIN_BUILDERS,
              '-i',
              '@INPUT@',
              '-o',
              '@OUTPUT@',
              '-t',
              'main',
              '--fake-stamp',
              'splash/app_icon.gen.h'],
    build_by_default: true
)
SOURCES += [_app_icon_target]
