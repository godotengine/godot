module_dirs = ['basis_universal',
  'bmp',
  'bullet',
  'camera',
  'csg',
  'cvtt',
  'dds',
  'denoise',
  'enet',
  'etcpak',
#  'fbx',
#  'freetype',
#  'gdnative',
  'gdscript',
  'glslang',
  'gltf',
  'gridmap',
  'hdr',
  'jpg',
  'jsonrpc',
  'lightmapper_rd',
  'mbedtls',
  'meshoptimizer',
  'minimp3',
  'mobile_vr',
  'mono',
  'msdfgen',
  'navigation',
  'ogg',
#  'opensimplex',
#  'raycast',
  'regex',
  'squish',
  'svg',
  'text_server_adv',
  'text_server_fb',
# See below
#  'tga',
  'theora',
  'tinyexr',
  'upnp',
  'vhacd',
  'vorbis',
  'webp',
  'webrtc',
  'websocket',
  'webxr',
  'xatlas_unwrap',
]

disabled_modules = get_option('disabled_modules')
all_disabled = get_option('disable_all_modules')

foreach d: module_dirs
  if all_disabled or disabled_modules.contains(d)
    MODULES_DISABLED += d
  else
    subdir(d)
  endif
endforeach

# This also creates the modules_enabled.gen.h header.
MAIN_SOURCES += configure_file(
    output: 'register_module_types.gen.cpp',
    command: [find_program('gen_registration.py'), '@OUTPUT@'] + MODULES_ENABLED,
    )

# Now process external modules.
external_modules = ['tga'] # In the real implementation this would come from a get_option().
foreach extmod : external_modules
    sp = subproject(extmod)
    depobj = sp.get_variable('module_dep')
    MODULE_DEPENDENCIES += depobj
    MODULES_ENABLED += [depobj.get_variable('module_names')]
    # FIXME, add module paths and documentation paths here.
endforeach
