#!/bin/bash

# This script validates shaders (if successfully compiled) using spirv-val.
# It is not meant to preclude the possible addition of the validator to
# glslang.

declare -r EXE='../build/install/bin/glslangValidator'

# search common locations for spirv-tools: keep first one
for toolsdir in '../External/spirv-tools/build/tools' '../../SPIRV-Tools/build/tools/bin' '/usr/local/bin'; do
    [[ -z "$VAL" && -x "${toolsdir}/spirv-val" ]] && declare -r VAL="${toolsdir}/spirv-val"
    [[ -z "$DIS" && -x "${toolsdir}/spirv-dis" ]] && declare -r DIS="${toolsdir}/spirv-dis"
done

declare -r gtests='../gtests/Hlsl.FromFile.cpp ../gtests/Spv.FromFile.cpp'

declare -r targetenv='vulkan1.0'

function fatal() { echo "ERROR: $@"; exit 5; }

function usage
{
    echo
    echo "Usage: $(basename $0) [options...] shaders..."
    echo
    echo "   Validates shaders (if successfully compiled) through spirv-val."
    echo
    echo "General options:"
    echo "   --help          prints this text"
    echo "   --no-color      disables output colorization"
    echo "   --dump-asm      dumps all successfully compiled shader assemblies"
    echo "   --dump-val      dumps all validation results"
    echo "   --dump-comp     dumps all compilation logs"
    echo "Spam reduction options:"
    echo "   --no-summary    disables result summaries"
    echo "   --skip-ok       do not print successful validations"
    echo "   --skip-comperr  do not print compilation errors"
    echo "   --skip-valerr   do not print validation errors"
    echo "   --quiet         synonym for --skip-ok --skip-comperr --skip-valerr --no-summary"
    echo "   --terse         print terse single line progress summary"
    echo "Disassembly options:"
    echo "   --raw-id        uses raw ids for disassembly"
    echo
    echo "Usage examples.  Note most non-hlsl tests fail to compile for expected reasons."
    echo "   Exercise all hlsl.* files:"
    echo "       $(basename $0) hlsl.*"
    echo "   Exercise all hlsl.* files, tersely:"
    echo "       $(basename $0) --terse hlsl.*"
    echo "   Print validator output for myfile.frag:"
    echo "       $(basename $0) --quiet --dump-val myfile.frag"
    echo "   Exercise hlsl.* files, only printing validation errors:"
    echo "       $(basename $0) --skip-ok --skip-comperr hlsl.*"

    exit 5
}

function status()
{
    printf "%-40s: %b\n" "$1" "$2"
}

# make sure we can find glslang
[[ -x "$EXE" ]] || fatal "Unable to locate $(basename "$EXE") executable"
[[ -x "$VAL" ]] || fatal "Unable to locate spirv-val executable"
[[ -x "$DIS" ]] || fatal "Unable to locate spirv-dis executable"

for gtest in $gtests; do
    [[ -r "$gtest" ]] || fatal "Unable to locate source file: $(basename $gtest)"
done

# temp files
declare -r spvfile='out.spv' \
        complog='comp.out' \
        vallog='val.out' \
        dislog='dis.out' \

# options
declare opt_vallog=false \
        opt_complog=false \
        opt_dislog=false \
        opt_summary=true \
        opt_stat_comperr=true \
        opt_stat_ok=true \
        opt_stat_valerr=true \
        opt_color=true \
        opt_raw_id=false \
        opt_quiet=false \
        opt_terse=false

# clean up on exit
trap "rm -f ${spvfile} ${complog} ${vallog} ${dislog}" EXIT

# Language guesser: there is no fixed mapping from filenames to language,
# so this examines the file and return one of:
#     hlsl
#     glsl
#     bin
#     unknown
# This is easier WRT future expansion than a big explicit list.
function FindLanguage()
{
    local test="$1"

    # If it starts with hlsl, assume it's hlsl.
    if [[ "$test" == *hlsl.* ]]; then
        echo hlsl
        return
    fi

    if [[ "$test" == *.spv ]]; then
        echo bin
        return;
    fi

    # If it doesn't start with spv., assume it's GLSL.
    if [[ ! "$test" == spv.* && ! "$test" == remap.* ]]; then
        echo glsl
        return
    fi

    # Otherwise, attempt to guess from shader contents, since there's no
    # fixed mapping of filenames to languages.
    local contents="$(cat "$test")"

    if [[ "$contents" == *#version* ]]; then
        echo glsl
        return
    fi

    if [[ "$contents" == *SamplerState* ||
          "$contents" == *cbuffer* ||
          "$contents" == *SV_* ]]; then
        echo hlsl
        return
    fi

    echo unknown
}

# Attempt to discover entry point
function FindEntryPoint()
{
    local test="$1"

    # if it's not hlsl, always use main
    if [[ "$language" != 'hlsl' ]]; then
        echo 'main'
        return
    fi

    # Try to find it in test sources
    awk -F '[ (){",]+' -e "\$2 == \"${test}\" { print \$3; found=1; } END { if (found==0) print \"main\"; } " $gtests
}

# command line options
while [ $# -gt 0 ]
do
    case "$1" in
        # -c) glslang="$2"; shift 2;;
        --help|-?)      usage;;
        --no-color)     opt_color=false;        shift;;
        --no-summary)   opt_summary=false;      shift;;
        --skip-ok)      opt_stat_ok=false;      shift;;
        --skip-comperr) opt_stat_comperr=false; shift;;
        --skip-valerr)  opt_stat_valerr=false;  shift;;
        --dump-asm)     opt_dislog=true;        shift;;
        --dump-val)     opt_vallog=true;        shift;;
        --dump-comp)    opt_complog=true;       shift;;
        --raw-id)       opt_raw_id=true;        shift;;
        --quiet)        opt_quiet=true;         shift;;
        --terse)        opt_quiet=true
                        opt_terse=true
                        shift;;
        --*)            fatal "Unknown command line option: $1";;
        *) break;;
    esac
done

# this is what quiet means
if $opt_quiet; then
    opt_stat_ok=false
    opt_stat_comperr=false
    opt_stat_valerr=false
    $opt_terse || opt_summary=false
fi

if $opt_color; then
    declare -r white="\e[1;37m" cyan="\e[1;36m" red="\e[0;31m" no_color="\e[0m"
else
    declare -r white="" cyan="" red="" no_color=""
fi

# stats
declare -i count_ok=0 count_err=0 count_nocomp=0 count_total=0

declare -r dashsep='------------------------------------------------------------------------'

testfiles=(${@})
# if no shaders given, look for everything in current directory
[[ ${#testfiles[*]} == 0 ]] && testfiles=(*.frag *.vert *.tesc *.tese *.geom *.comp)

$opt_summary && printf "\nValidating: ${#testfiles[*]} shaders\n\n"

# Loop through the shaders we were given, compiling them if we can.
for test in ${testfiles[*]}
do
    if [[ ! -r "$test" ]]; then
        $opt_quiet || status "$test" "${red}FILE NOT FOUND${no_color}"
        continue
    fi

    ((++count_total))

    $opt_terse && printf "\r[%-3d/%-3d : ${white}comperr=%-3d ${red}valerr=%-3d ${cyan}ok=%-3d${no_color}]" \
                         ${count_total} ${#testfiles[*]} ${count_nocomp} ${count_err} ${count_ok}

    language="$(FindLanguage $test)"
    entry="$(FindEntryPoint $test)"
    langops=''

    case "$language" in
        hlsl) langops='-D --hlsl-iomap --hlsl-offsets';;
        glsl) ;;
        bin) continue;;   # skip binaries
        *) $opt_quiet || status "$test" "${red}UNKNOWN LANGUAGE${no_color}"; continue;;
    esac

    # compile the test file
    if compout=$("$EXE" -e "$entry" $langops -V -o "$spvfile" "$test" 2>&1)
    then
        # successful compilation: validate
        if valout=$("$VAL" --target-env ${targetenv} "$spvfile" 2>&1)
        then
            # validated OK
            $opt_stat_ok && status "$test" "${cyan}OK${no_color}"
            ((++count_ok))
        else
            # validation failure
            $opt_stat_valerr && status "$test" "${red}VAL ERROR${no_color}"
            printf "%s\n%s:\n%s\n" "$dashsep" "$test" "$valout" >> "$vallog"
            ((++count_err))
        fi

        if $opt_dislog; then
            printf "%s\n%s:\n" "$dashsep" "$test" >> "$dislog"
            $opt_raw_id && id_opt=--raw-id
            "$DIS" ${id_opt} "$spvfile" >> "$dislog"
        fi
    else
        # compile failure
        $opt_stat_comperr && status "$test" "${white}COMP ERROR${no_color}"
        printf "%s\n%s\n" "$dashsep" "$compout" >> "$complog"
        ((++count_nocomp))
    fi
done

$opt_terse && echo

# summarize
$opt_summary && printf "\nSummary: ${white}${count_nocomp} compile errors${no_color}, ${red}${count_err} validation errors${no_color}, ${cyan}${count_ok} successes${no_color}\n"

# dump logs
$opt_vallog  && [[ -r $vallog ]]  && cat "$vallog"
$opt_complog && [[ -r $complog ]] && cat "$complog"
$opt_dislog  && [[ -r $dislog ]]  && cat "$dislog"

# exit code
[[ ${count_err} -gt 0 ]] && exit 1
exit 0
