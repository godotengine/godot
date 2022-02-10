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
  'fbx',
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
  'opensimplex',
#  'raycast',
  'regex',
  'squish',
  'svg',
  'text_server_adv',
  'text_server_fb',
  'tga',
  'theora',
  'tinyexr',
  'upnp',
  'vhacd',
  'visual_script',
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

MAIN_SOURCES += configure_file(
    output: 'register_module_types.gen.cpp',
    command: [find_program('gen_registration.py'), '@OUTPUT@'] + MODULES_ENABLED,
    )
