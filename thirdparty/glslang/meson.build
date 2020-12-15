if get_option('builtin_glslang')
    _glslang_incdir = include_directories('.')

    _glslang_srcs = files([
        'glslang/CInterface/glslang_c_interface.cpp',
        'glslang/MachineIndependent/RemoveTree.cpp',
        'glslang/MachineIndependent/ParseHelper.cpp',
        'glslang/MachineIndependent/iomapper.cpp',
        'glslang/MachineIndependent/propagateNoContraction.cpp',
        'glslang/MachineIndependent/Intermediate.cpp',
        'glslang/MachineIndependent/linkValidate.cpp',
        'glslang/MachineIndependent/attribute.cpp',
        'glslang/MachineIndependent/Scan.cpp',
        'glslang/MachineIndependent/Initialize.cpp',
        'glslang/MachineIndependent/Constant.cpp',
        'glslang/MachineIndependent/reflection.cpp',
        'glslang/MachineIndependent/limits.cpp',
        'glslang/MachineIndependent/preprocessor/PpScanner.cpp',
        'glslang/MachineIndependent/preprocessor/PpTokens.cpp',
        'glslang/MachineIndependent/preprocessor/PpAtom.cpp',
        'glslang/MachineIndependent/preprocessor/PpContext.cpp',
        'glslang/MachineIndependent/preprocessor/Pp.cpp',
        'glslang/MachineIndependent/InfoSink.cpp',
        'glslang/MachineIndependent/intermOut.cpp',
        'glslang/MachineIndependent/SymbolTable.cpp',
        'glslang/MachineIndependent/glslang_tab.cpp',
        'glslang/MachineIndependent/Versions.cpp',
        'glslang/MachineIndependent/ShaderLang.cpp',
        'glslang/MachineIndependent/parseConst.cpp',
        'glslang/MachineIndependent/PoolAlloc.cpp',
        'glslang/MachineIndependent/ParseContextBase.cpp',
        'glslang/MachineIndependent/IntermTraverse.cpp',
        'glslang/GenericCodeGen/Link.cpp',
        'glslang/GenericCodeGen/CodeGen.cpp',
        'OGLCompilersDLL/InitializeDll.cpp',
        'SPIRV/CInterface/spirv_c_interface.cpp',
        'SPIRV/InReadableOrder.cpp',
        'SPIRV/GlslangToSpv.cpp',
        'SPIRV/SpvBuilder.cpp',
        'SPIRV/SpvTools.cpp',
        'SPIRV/disassemble.cpp',
        'SPIRV/doc.cpp',
        'SPIRV/SPVRemapper.cpp',
        'SPIRV/SpvPostProcess.cpp',
        'SPIRV/Logger.cpp',
        'StandAlone/ResourceLimits.cpp',
    ])

    _glslang_cpp_flags = ['-DENABLE_OPT=0']

    if PLATFORM == 'windows'
        _glslang_srcs += files(['glslang/OSDependent/Windows/ossource.cpp'])
    else
        _glslang_srcs += files(['glslang/OSDependent/Unix/ossource.cpp'])
    endif

    # TODO: investiage
    # if meson.get_compiler('cpp').get_id() != 'msvc'
    #     _glslang_cpp_flags += ['-isystem']
    # endif

    _lib_glslang = static_library('glslang', _glslang_srcs, include_directories: _glslang_incdir, cpp_args: _glslang_cpp_flags, build_by_default: false)

    DEP_GLSLANG = declare_dependency(link_with: _lib_glslang, include_directories: _glslang_incdir)

else
    DEP_GLSLANG = [
        dependency('glslang'),
        meson.get_compiler('cpp').find_library('SPIRV')
    ]
endif