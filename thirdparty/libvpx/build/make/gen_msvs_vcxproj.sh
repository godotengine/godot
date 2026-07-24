#!/bin/bash
##
##  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

self=$0
self_basename=${self##*/}
self_dirname=$(dirname "$0")

. "$self_dirname/msvs_common.sh"|| exit 127

show_help() {
    cat <<EOF
Usage: ${self_basename} --name=projname [options] file1 [file2 ...]

This script generates a Visual Studio project file from a list of source
code files.

Options:
    --help                      Print this message
    --exe                       Generate a project for building an Application
    --lib                       Generate a project for creating a static library
    --dll                       Generate a project for creating a dll
    --static-crt                Use the static C runtime (/MT)
    --enable-werror             Treat warnings as errors (/WX)
    --target=isa-os-cc          Target specifier (required)
    --out=filename              Write output to a file [stdout]
    --name=project_name         Name of the project (required)
    --proj-guid=GUID            GUID to use for the project
    --module-def=filename       File containing export definitions (for DLLs)
    --ver=version               Version (14-16) of visual studio to generate for
    --src-path-bare=dir         Path to root of source tree
    -Ipath/to/include           Additional include directories
    -DFLAG[=value]              Preprocessor macros to define
    -Lpath/to/lib               Additional library search paths
    -llibname                   Library to link against
EOF
    exit 1
}

tag_content() {
    local tag=$1
    local content=$2
    shift
    shift
    if [ $# -ne 0 ]; then
        echo "${indent}<${tag}"
        indent_push
        tag_attributes "$@"
        echo "${indent}>${content}</${tag}>"
        indent_pop
    else
        echo "${indent}<${tag}>${content}</${tag}>"
    fi
}

generate_filter() {
    local name=$1
    local pats=$2
    local file_list_sz
    local i
    local f
    local saveIFS="$IFS"
    local pack
    echo "generating filter '$name' from ${#file_list[@]} files" >&2
    IFS=*

    file_list_sz=${#file_list[@]}
    for i in ${!file_list[@]}; do
        f=${file_list[i]}
        for pat in ${pats//;/$IFS}; do
            if [ "${f##*.}" == "$pat" ]; then
                unset file_list[i]

                objf=$(echo ${f%.*}.obj \
                       | sed -e "s,$src_path_bare,," \
                             -e 's/^[\./]\+//g' -e 's,[:/ ],_,g')

                if ([ "$pat" == "asm" ] || [ "$pat" == "s" ] || [ "$pat" == "S" ]) && $uses_asm; then
                    # Avoid object file name collisions, i.e. vpx_config.c and
                    # vpx_config.asm produce the same object file without
                    # this additional suffix.
                    objf=${objf%.obj}_asm.obj
                    open_tag CustomBuild \
                        Include="$f"
                    for plat in "${platforms[@]}"; do
                        for cfg in Debug Release; do
                            tag_content Message "Assembling %(Filename)%(Extension)" \
                                Condition="'\$(Configuration)|\$(Platform)'=='$cfg|$plat'"
                            tag_content Command "$(eval echo \$asm_${cfg}_cmdline) -o \$(IntDir)$objf" \
                                Condition="'\$(Configuration)|\$(Platform)'=='$cfg|$plat'"
                            tag_content Outputs "\$(IntDir)$objf" \
                                Condition="'\$(Configuration)|\$(Platform)'=='$cfg|$plat'"
                        done
                    done
                    close_tag CustomBuild
                elif [ "$pat" == "c" ] || \
                     [ "$pat" == "cc" ] || [ "$pat" == "cpp" ]; then
                    open_tag ClCompile \
                        Include="$f"
                    # Separate file names with Condition?
                    tag_content ObjectFileName "\$(IntDir)$objf"
                    # Check for AVX and turn it on to avoid warnings.
                    if [[ $f =~ avx.?\.c$ ]]; then
                        tag_content AdditionalOptions "/arch:AVX"
                    fi
                    close_tag ClCompile
                elif [ "$pat" == "h" ] ; then
                    tag ClInclude \
                        Include="$f"
                elif [ "$pat" == "vcxproj" ] ; then
                    open_tag ProjectReference \
                        Include="$f"
                    depguid=`grep ProjectGuid "$f" | sed 's,.*<.*>\(.*\)</.*>.*,\1,'`
                    tag_content Project "$depguid"
                    tag_content ReferenceOutputAssembly false
                    close_tag ProjectReference
                else
                    tag None \
                        Include="$f"
                fi

                break
            fi
        done
    done

    IFS="$saveIFS"
}

# Process command line
unset target
for opt in "$@"; do
    optval="${opt#*=}"
    case "$opt" in
        --help|-h) show_help
        ;;
        --target=*)
            target="${optval}"
            platform_toolset=$(echo ${target} | awk 'BEGIN{FS="-"}{print $4}')
            case "$platform_toolset" in
                clangcl) platform_toolset="ClangCl"
                ;;
                "")
                ;;
                *) die Unrecognized Visual Studio Platform Toolset in $opt
                ;;
            esac
        ;;
        --out=*) outfile="$optval"
        ;;
        --name=*) name="${optval}"
        ;;
        --proj-guid=*) guid="${optval}"
        ;;
        --module-def=*) module_def="${optval}"
        ;;
        --exe) proj_kind="exe"
        ;;
        --dll) proj_kind="dll"
        ;;
        --lib) proj_kind="lib"
        ;;
        --as=*) as="${optval}"
        ;;
        --src-path-bare=*)
            src_path_bare=$(fix_path "$optval")
            src_path_bare=${src_path_bare%/}
        ;;
        --static-crt) use_static_runtime=true
        ;;
        --enable-werror) werror=true
        ;;
        --ver=*)
            vs_ver="$optval"
            case "$optval" in
                1[4-7])
                ;;
                *) die Unrecognized Visual Studio Version in $opt
                ;;
            esac
        ;;
        -I*)
            opt=${opt##-I}
            opt=$(fix_path "$opt")
            opt="${opt%/}"
            incs="${incs}${incs:+;}&quot;${opt}&quot;"
            yasmincs="${yasmincs} -I&quot;${opt}&quot;"
        ;;
        -D*) defines="${defines}${defines:+;}${opt##-D}"
        ;;
        -L*) # fudge . to $(OutDir)
            if [ "${opt##-L}" == "." ]; then
                libdirs="${libdirs}${libdirs:+;}&quot;\$(OutDir)&quot;"
            else
                 # Also try directories for this platform/configuration
                 opt=${opt##-L}
                 opt=$(fix_path "$opt")
                 libdirs="${libdirs}${libdirs:+;}&quot;${opt}&quot;"
                 libdirs="${libdirs}${libdirs:+;}&quot;${opt}/\$(PlatformName)/\$(Configuration)&quot;"
                 libdirs="${libdirs}${libdirs:+;}&quot;${opt}/\$(PlatformName)&quot;"
            fi
        ;;
        -l*) libs="${libs}${libs:+ }${opt##-l}.lib"
        ;;
        -*) die_unknown $opt
        ;;
        *)
            # The paths in file_list are fixed outside of the loop.
            file_list[${#file_list[@]}]="$opt"
            case "$opt" in
                 *.asm|*.[Ss]) uses_asm=true
                 ;;
            esac
        ;;
    esac
done

# Make one call to fix_path for file_list to improve performance.
fix_file_list file_list

outfile=${outfile:-/dev/stdout}
guid=${guid:-`generate_uuid`}
uses_asm=${uses_asm:-false}

[ -n "$name" ] || die "Project name (--name) must be specified!"
[ -n "$target" ] || die "Target (--target) must be specified!"

if ${use_static_runtime:-false}; then
    release_runtime=MultiThreaded
    debug_runtime=MultiThreadedDebug
    lib_sfx=mt
else
    release_runtime=MultiThreadedDLL
    debug_runtime=MultiThreadedDebugDLL
    lib_sfx=md
fi

# Calculate debug lib names: If a lib ends in ${lib_sfx}.lib, then rename
# it to ${lib_sfx}d.lib. This precludes linking to release libs from a
# debug exe, so this may need to be refactored later.
for lib in ${libs}; do
    if [ "$lib" != "${lib%${lib_sfx}.lib}" ]; then
        lib=${lib%.lib}d.lib
    fi
    debug_libs="${debug_libs}${debug_libs:+ }${lib}"
done
debug_libs=${debug_libs// /;}
libs=${libs// /;}


# List of all platforms supported for this target
case "$target" in
    x86_64*)
        platforms[0]="x64"
        asm_Debug_cmdline="${as} -Xvc -gcv8 -f win64 ${yasmincs} &quot;%(FullPath)&quot;"
        asm_Release_cmdline="${as} -Xvc -f win64 ${yasmincs} &quot;%(FullPath)&quot;"
    ;;
    x86*)
        platforms[0]="Win32"
        asm_Debug_cmdline="${as} -Xvc -gcv8 -f win32 ${yasmincs} &quot;%(FullPath)&quot;"
        asm_Release_cmdline="${as} -Xvc -f win32 ${yasmincs} &quot;%(FullPath)&quot;"
    ;;
    arm64*)
        platforms[0]="ARM64"
        # As of Visual Studio 2022 17.5.5, clang-cl does not support ARM64EC.
        if [ "$vs_ver" -ge 17 -a "$platform_toolset" != "ClangCl" ]; then
            platforms[1]="ARM64EC"
        fi
        asm_Debug_cmdline="armasm64 -nologo -oldit &quot;%(FullPath)&quot;"
        asm_Release_cmdline="armasm64 -nologo -oldit &quot;%(FullPath)&quot;"
    ;;
    arm*)
        platforms[0]="ARM"
        asm_Debug_cmdline="armasm -nologo -oldit &quot;%(FullPath)&quot;"
        asm_Release_cmdline="armasm -nologo -oldit &quot;%(FullPath)&quot;"
    ;;
    *) die "Unsupported target $target!"
    ;;
esac

generate_vcxproj() {
    echo "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
    open_tag Project \
        DefaultTargets="Build" \
        ToolsVersion="4.0" \
        xmlns="http://schemas.microsoft.com/developer/msbuild/2003" \

    open_tag ItemGroup \
        Label="ProjectConfigurations"
    for plat in "${platforms[@]}"; do
        for config in Debug Release; do
            open_tag ProjectConfiguration \
                Include="$config|$plat"
            tag_content Configuration $config
            tag_content Platform $plat
            close_tag ProjectConfiguration
        done
    done
    close_tag ItemGroup

    open_tag PropertyGroup \
        Label="Globals"
        tag_content ProjectGuid "{${guid}}"
        tag_content RootNamespace ${name}
        tag_content Keyword ManagedCProj
        if [ $vs_ver -ge 12 ] && [ "${platforms[0]}" = "ARM" ]; then
            tag_content AppContainerApplication true
            # The application type can be one of "Windows Store",
            # "Windows Phone" or "Windows Phone Silverlight". The
            # actual value doesn't matter from the libvpx point of view,
            # since a static library built for one works on the others.
            # The PlatformToolset field needs to be set in sync with this;
            # for Windows Store and Windows Phone Silverlight it should be
            # v120 while it should be v120_wp81 if the type is Windows Phone.
            tag_content ApplicationType "Windows Store"
            tag_content ApplicationTypeRevision 8.1
        fi
        if [ "${platforms[0]}" = "ARM64" ]; then
            # Require the first Visual Studio version to have ARM64 support.
            tag_content MinimumVisualStudioVersion 15.9
        fi
        if [ $vs_ver -eq 15 ] && [ "${platforms[0]}" = "ARM64" ]; then
            # Since VS 15 does not have a 'use latest SDK version' facility,
            # specifically require the contemporaneous SDK with official ARM64
            # support.
            tag_content WindowsTargetPlatformVersion 10.0.17763.0
        fi
    close_tag PropertyGroup

    tag Import \
        Project="\$(VCTargetsPath)\\Microsoft.Cpp.Default.props"

    for plat in "${platforms[@]}"; do
        for config in Release Debug; do
            open_tag PropertyGroup \
                Condition="'\$(Configuration)|\$(Platform)'=='$config|$plat'" \
                Label="Configuration"
            if [ "$proj_kind" = "exe" ]; then
                tag_content ConfigurationType Application
            elif [ "$proj_kind" = "dll" ]; then
                tag_content ConfigurationType DynamicLibrary
            else
                tag_content ConfigurationType StaticLibrary
            fi
            if [ -n "$platform_toolset" ]; then
                tag_content PlatformToolset "$platform_toolset"
            else
                if [ "$vs_ver" = "14" ]; then
                    tag_content PlatformToolset v140
                fi
                if [ "$vs_ver" = "15" ]; then
                    tag_content PlatformToolset v141
                fi
                if [ "$vs_ver" = "16" ]; then
                    tag_content PlatformToolset v142
                fi
                if [ "$vs_ver" = "17" ]; then
                    tag_content PlatformToolset v143
                fi
            fi
            tag_content CharacterSet Unicode
            if [ "$config" = "Release" ]; then
                tag_content WholeProgramOptimization true
            fi
            close_tag PropertyGroup
        done
    done

    tag Import \
        Project="\$(VCTargetsPath)\\Microsoft.Cpp.props"

    open_tag ImportGroup \
        Label="PropertySheets"
        tag Import \
            Project="\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props" \
            Condition="exists('\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props')" \
            Label="LocalAppDataPlatform"
    close_tag ImportGroup

    tag PropertyGroup \
        Label="UserMacros"

    for plat in "${platforms[@]}"; do
        plat_no_ws=`echo $plat | sed 's/[^A-Za-z0-9_]/_/g'`
        for config in Debug Release; do
            open_tag PropertyGroup \
                Condition="'\$(Configuration)|\$(Platform)'=='$config|$plat'"
            tag_content OutDir "\$(SolutionDir)$plat_no_ws\\\$(Configuration)\\"
            tag_content IntDir "$plat_no_ws\\\$(Configuration)\\${name}\\"
            if [ "$proj_kind" == "lib" ]; then
              if [ "$config" == "Debug" ]; then
                config_suffix=d
              else
                config_suffix=""
              fi
              tag_content TargetName "${name}${lib_sfx}${config_suffix}"
            fi
            close_tag PropertyGroup
        done
    done

    for plat in "${platforms[@]}"; do
        for config in Debug Release; do
            open_tag ItemDefinitionGroup \
                Condition="'\$(Configuration)|\$(Platform)'=='$config|$plat'"
            if [ "$name" == "vpx" ]; then
                hostplat=$plat
                if [ "$hostplat" == "ARM" ]; then
                    hostplat=Win32
                fi
            fi
            open_tag ClCompile
            if [ "$config" = "Debug" ]; then
                opt=Disabled
                runtime=$debug_runtime
                curlibs=$debug_libs
                debug=_DEBUG
            else
                opt=MaxSpeed
                runtime=$release_runtime
                curlibs=$libs
                tag_content FavorSizeOrSpeed Speed
                debug=NDEBUG
            fi
            extradefines=";$defines"
            tag_content Optimization $opt
            tag_content AdditionalIncludeDirectories "$incs;%(AdditionalIncludeDirectories)"
            tag_content PreprocessorDefinitions "WIN32;$debug;_CRT_SECURE_NO_WARNINGS;_CRT_SECURE_NO_DEPRECATE$extradefines;%(PreprocessorDefinitions)"
            tag_content RuntimeLibrary $runtime
            tag_content WarningLevel Level3
            if ${werror:-false}; then
                tag_content TreatWarningAsError true
            fi
            if [ $vs_ver -ge 11 ]; then
                # We need to override the defaults for these settings
                # if AppContainerApplication is set.
                tag_content CompileAsWinRT false
                tag_content PrecompiledHeader NotUsing
                tag_content SDLCheck false
            fi
            close_tag ClCompile
            case "$proj_kind" in
            exe)
                open_tag Link
                tag_content GenerateDebugInformation true
                # Console is the default normally, but if
                # AppContainerApplication is set, we need to override it.
                tag_content SubSystem Console
                close_tag Link
                ;;
            dll)
                open_tag Link
                tag_content GenerateDebugInformation true
                tag_content ModuleDefinitionFile $module_def
                close_tag Link
                ;;
            lib)
                ;;
            esac
            close_tag ItemDefinitionGroup
        done

    done

    open_tag ItemGroup
    generate_filter "Source Files"   "c;cc;cpp;def;odl;idl;hpj;bat;asm;asmx;s;S"
    close_tag ItemGroup
    open_tag ItemGroup
    generate_filter "Header Files"   "h;hm;inl;inc;xsd"
    close_tag ItemGroup
    open_tag ItemGroup
    generate_filter "Build Files"    "mk"
    close_tag ItemGroup
    open_tag ItemGroup
    generate_filter "References"     "vcxproj"
    close_tag ItemGroup

    tag Import \
        Project="\$(VCTargetsPath)\\Microsoft.Cpp.targets"

    open_tag ImportGroup \
        Label="ExtensionTargets"
    close_tag ImportGroup

    close_tag Project

    # This must be done from within the {} subshell
    echo "Ignored files list (${#file_list[@]} items) is:" >&2
    for f in "${file_list[@]}"; do
        echo "    $f" >&2
    done
}

# This regexp doesn't catch most of the strings in the vcxproj format,
# since they're like <tag>path</tag> instead of <tag attr="path" />
# as previously. It still seems to work ok despite this.
generate_vcxproj |
    sed  -e '/"/s;\([^ "]\)/;\1\\;g' |
    sed  -e '/xmlns/s;\\;/;g' > ${outfile}

exit
