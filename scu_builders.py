"""Functions used to generate scu build source files during build time

"""
import glob, os
import math
from pathlib import Path
from os.path import normpath, basename

base_folder = os.path.abspath("./") + "/"
_verbose = False


def clear_out_existing_files(output_folder, extension):
    output_folder = os.path.abspath(output_folder)
    # print("clear_out_existing_files from folder: " + output_folder)

    if not os.path.isdir(output_folder):
        # folder does not exist or has not been created yet,
        # no files to clearout. (this is not an error)
        return

    os.chdir(output_folder)

    for file in glob.glob("*." + extension):
        # print("removed pre-existing file: " + file)
        os.remove(file)


def find_files_in_folder(folder, sub_folder, include_list, extension, sought_exceptions, found_exceptions):
    abs_folder = base_folder + folder + "/" + sub_folder

    if not os.path.isdir(abs_folder):
        print("ERROR " + abs_folder + " not found.")
        return include_list, found_exceptions

    os.chdir(abs_folder)

    sub_folder_slashed = ""
    if sub_folder != "":
        sub_folder_slashed = sub_folder + "/"

    for file in glob.glob("*." + extension):

        simple_name = Path(file).stem

        if file.endswith(".gen.cpp"):
            continue

        li = '#include "../' + sub_folder_slashed + file + '"'

        if not simple_name in sought_exceptions:
            include_list += [li]
        else:
            found_exceptions += [li]

    return include_list, found_exceptions


def write_output_file(file_count, include_list, start_line, end_line, output_folder, output_filename_prefix, extension):

    output_folder = os.path.abspath(output_folder)

    if not os.path.isdir(output_folder):
        # create
        os.mkdir(output_folder)
        if not os.path.isdir(output_folder):
            print("ERROR " + output_folder + " could not be created.")
            return
        print("CREATING folder " + output_folder)

    file_text = ""

    for l in range(start_line, end_line):
        if l < len(include_list):
            line = include_list[l]
            li = line + "\n"
            file_text += li

    # print(file_text)

    num_string = ""
    if file_count > 0:
        num_string = "_" + str(file_count)

    short_filename = output_filename_prefix + num_string + ".gen." + extension
    output_filename = output_folder + "/" + short_filename
    if _verbose:
        print("generating: " + short_filename)

    # return
    output_file = open(output_filename, "w")
    if not output_file.closed:
        output_file.write(file_text)
        output_file.close()


def write_exception_output_file(file_count, exception_string, output_folder, output_filename_prefix, extension):
    output_folder = os.path.abspath(output_folder)
    if not os.path.isdir(output_folder):
        print("ERROR " + output_folder + " does not exist.")
        return

    file_text = exception_string + "\n"

    num_string = ""
    if file_count > 0:
        num_string = "_" + str(file_count)

    short_filename = output_filename_prefix + "_exception" + num_string + ".gen." + extension
    output_filename = output_folder + "/" + short_filename

    if _verbose:
        print("generating: " + short_filename)

    # print("text: " + file_text)
    # return
    output_file = open(output_filename, "w")
    if not output_file.closed:
        output_file.write(file_text)
        output_file.close()


def find_section_name(sub_folder):
    # Construct a useful name for the section from the path for debug logging
    section_path = os.path.abspath(base_folder + sub_folder) + "/"

    folders = []
    folder = ""

    for i in range(8):
        folder = os.path.dirname(section_path)
        folder = os.path.basename(folder)
        if folder == "godot":
            break
        folders += [folder]
        section_path += "../"
        section_path = os.path.abspath(section_path) + "/"

    section_name = ""
    for n in range(len(folders)):
        section_name += folders[len(folders) - n - 1]
        if n != (len(folders) - 1):
            section_name += "_"

    return section_name


# "folders" is a list of folders to add all the files from to add to the SCU
# "section (like a module)". The name of the scu file will be derived from the first folder
# (thus e.g. scene/3d becomes scu_scene_3d.gen.cpp)

# "num_output_files" allows the scu to be built in several translation units instead of just 1.
# This will usually be slower to compile but will use less memory per compiler instance, which
# is most relevant in release builds.

# "sought_exceptions" are a list of files (without extension) that contain
# e.g. naming conflicts, and are therefore not suitable for the scu build.
# These will automatically be placed in their own separate scu file,
# which is slow like a normal build, but prevents the naming conflicts.
# Ideally in these situations, the source code should be changed to prevent naming conflicts.

# "extension" will usually be cpp, but can also be set to c (for e.g. third party libraries that use c)
def process_folder(folders, sought_exceptions=[], num_output_files=1, extension="cpp"):
    if len(folders) == 0:
        return

    # Construct the filename prefix from the FIRST folder name
    # e.g. "scene_3d"
    out_filename = find_section_name(folders[0])

    found_includes = []
    found_exceptions = []

    main_folder = folders[0]
    abs_main_folder = base_folder + main_folder

    # main folder (first)
    found_includes, found_exceptions = find_files_in_folder(
        main_folder, "", found_includes, extension, sought_exceptions, found_exceptions
    )

    # sub folders
    for d in range(1, len(folders)):
        found_includes, found_exceptions = find_files_in_folder(
            main_folder, folders[d], found_includes, extension, sought_exceptions, found_exceptions
        )

    found_includes = sorted(found_includes)

    # calculate how many lines to write in each file
    total_lines = len(found_includes)
    lines_per_file = math.floor(total_lines / num_output_files)
    lines_per_file = max(lines_per_file, 1)

    start_line = 0
    file_number = 0

    # These do not vary throughout the loop
    output_folder = abs_main_folder + "/scu/"
    output_filename_prefix = "scu_" + out_filename

    # Clear out any existing files (usually we will be overwriting,
    # but we want to remove any that are pre-existing that will not be
    # overwritten, so as to not compile anything stale)
    clear_out_existing_files(output_folder, extension)

    for file_count in range(0, num_output_files):
        end_line = start_line + lines_per_file

        # special case to cover rounding error in final file
        if file_count == (num_output_files - 1):
            end_line = len(found_includes)

        write_output_file(
            file_count, found_includes, start_line, end_line, output_folder, output_filename_prefix, extension
        )

        start_line = end_line

    # Write the exceptions each in their own scu gen file,
    # so they can effectively compile in "old style / normal build".
    for exception_count in range(len(found_exceptions)):
        write_exception_output_file(
            exception_count, found_exceptions[exception_count], output_folder, output_filename_prefix, extension
        )


def generate_scu_files(verbose):

    print("=============================")
    print("Single Compilation Unit Build")
    print("=============================")
    print("Generating SCU build files")
    global _verbose
    _verbose = verbose

    # check we are running from the correct folder
    curr_folder = os.path.abspath("./")
    parent_path = basename(normpath(curr_folder))

    if parent_path != "godot":
        print("ERROR - scu_builders.py must be run from the godot folder.")
        return

    process_folder(["core"])
    process_folder(["core/math"])
    process_folder(["core/os"])
    process_folder(["core/io"])
    process_folder(["core/crypto"])

    process_folder(["drivers/gles2"])
    process_folder(["drivers/gles3"])
    process_folder(["drivers/unix"])
    process_folder(["drivers/png"])

    process_folder(["editor"])
    process_folder(["editor/import"])
    process_folder(["editor/plugins"])

    process_folder(["main"])
    process_folder(["main/tests"])

    process_folder(["modules/bullet"])
    process_folder(["modules/gltf"])
    process_folder(["modules/gltf/structures"])
    process_folder(["modules/navigation"])
    process_folder(["modules/visual_script"])
    process_folder(["modules/webrtc"])
    process_folder(["modules/websocket"])
    process_folder(["modules/gridmap"])

    process_folder(["modules/csg"])
    process_folder(["modules/gdscript"])
    process_folder(["modules/gdscript/language_server"])
    process_folder(["modules/fbx", "tools/", "data", "fbx_parser"])
    process_folder(["modules/gdnative", "android", "gdnative", "nativescript"])
    process_folder(["modules/gdnative/arvr"])
    process_folder(["modules/gdnative/pluginscript"])
    process_folder(["modules/gdnative/net"])

    process_folder(["scene"])
    process_folder(["scene/2d"])
    process_folder(["scene/3d"])
    process_folder(["scene/animation"])
    process_folder(["scene/gui"])
    process_folder(["scene/main"])
    process_folder(["scene/resources"])

    process_folder(["servers"])
    process_folder(["servers/visual"])
    process_folder(["servers/visual/portals"])
    process_folder(["servers/physics_2d"])
    process_folder(["servers/physics"])
    process_folder(["servers/physics/joints"])
    process_folder(["servers/audio"])
    process_folder(["servers/audio/effects"])

    # Finally change back the path to the calling folder
    os.chdir(curr_folder)
