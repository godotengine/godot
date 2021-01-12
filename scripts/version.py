#!/usr/bin/python3

import argparse
import json
import os

modules_to_show_in_version = ['mono']


def __make_version_header(json_version_file: str, out_version_file: str, out_hash_file: str, built_modules: [str]):
    version = {}
    with open(json_version_file, 'r') as f:
        version = json.load(f)

    build_name = "custom_build"
    if os.getenv("BUILD_NAME") != None:
        build_name = os.getenv("BUILD_NAME")
        print("Using custom build name: " + build_name)

    version_modules = [
        x for x in built_modules if x in modules_to_show_in_version]

    f = open(out_version_file, "w")
    f.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    f.write("#ifndef VERSION_GENERATED_GEN_H\n")
    f.write("#define VERSION_GENERATED_GEN_H\n")
    f.write('#define VERSION_SHORT_NAME "' +
            str(version['short_name']) + '"\n')
    f.write('#define VERSION_NAME "' + str(version['name']) + '"\n')
    f.write("#define VERSION_MAJOR " + str(version['major']) + "\n")
    f.write("#define VERSION_MINOR " + str(version['minor']) + "\n")
    f.write("#define VERSION_PATCH " + str(version['patch']) + "\n")
    f.write('#define VERSION_STATUS "' + str(version['status']) + '"\n')
    f.write('#define VERSION_BUILD "' + str(build_name) + '"\n')

    version_array = [str(version['module_config'])] + version_modules

    f.write('#define VERSION_MODULE_CONFIG "' +
            '.'.join(version_array) + '"\n')
    f.write("#define VERSION_YEAR " + str(version['year']) + "\n")
    f.write('#define VERSION_WEBSITE "' + str(version['website']) + '"\n')
    f.write("#endif // VERSION_GENERATED_GEN_H\n")
    f.close()

    # NOTE: It is safe to generate this file here, since this is still executed serially
    fhash = open(out_hash_file, "w")
    fhash.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    fhash.write("#ifndef VERSION_HASH_GEN_H\n")
    fhash.write("#define VERSION_HASH_GEN_H\n")
    githash = ""
    gitfolder = ".git"

    if os.path.isfile(".git"):
        module_folder = open(".git", "r").readline().strip()
        if module_folder.startswith("gitdir: "):
            gitfolder = module_folder[8:]

    if os.path.isfile(os.path.join(gitfolder, "HEAD")):
        head = open(os.path.join(gitfolder, "HEAD"), "r",
                    encoding="utf8").readline().strip()
        if head.startswith("ref: "):
            head = os.path.join(gitfolder, head[5:])
            if os.path.isfile(head):
                githash = open(head, "r").readline().strip()
        else:
            githash = head

    fhash.write('#define VERSION_HASH "' + githash + '"\n')
    fhash.write("#endif // VERSION_HASH_GEN_H\n")
    fhash.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate the version header')
    parser.add_argument('json_version_file', type=str)
    parser.add_argument('out_version_file', type=str)
    parser.add_argument('out_hash_file', type=str)
    parser.add_argument('built_modules', type=str, nargs='*')

    args = parser.parse_args()
    __make_version_header(args.json_version_file, args.out_version_file,
                          args.out_hash_file, args.built_modules)
