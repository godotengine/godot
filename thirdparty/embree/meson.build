
embree_srcs = files(run_command(GLOB, 'common/*/*.cpp', check: true).stdout().split('\n'))
embree_srcs += files(run_command(GLOB, 'kernels/*/*.cpp', check: true).stdout().split('\n'))

embree_args = ['-DTASKING_INTERNAL', '-DEMBREE_LOWEST_ISA']

if host_machine.system() == 'windows'
  embree_args += ['-DEMBREE_TARGET_SSE2', '-D__SSE2__', '-D__SSE__']
endif

embree_dep = declare_dependency(include_directories: 'include',
  link_with: library('embree',
    embree_srcs,
    cpp_args: embree_args,
    ),
)
