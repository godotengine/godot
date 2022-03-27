#!/usr/bin/env python3

"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
import os, sys
import os.path
import shutil
import subprocess
import tempfile
import uuid
import zlib
from platform_methods import subprocess_main


from glob import glob

def replace_if_different(output_path_str, new_content_path_str):
    import pathlib

    output_path = pathlib.Path(output_path_str)
    new_content_path = pathlib.Path(new_content_path_str)
    if not output_path.exists():
        new_content_path.replace(output_path)
        return
    if output_path.read_bytes() == new_content_path.read_bytes():
        new_content_path.unlink()
    else:
        new_content_path.replace(output_path)

def make_doc_header(compr_filename, modules):
    path_filename = os.path.join(os.path.split(compr_filename)[0], 'doc_data_class_path.gen.h')
    compr_tmpname = compr_filename + '~'
    path_tmpname = path_filename + '~'
    buf = ""
    docbegin = ""
    docend = ""
    root = os.environ['MESON_SOURCE_ROOT']

    path_entries = []
    for m in modules:
        subdir = os.path.join('modules', m, 'doc_classes')
        for src in sorted(glob(os.path.join(root, subdir, '*.xml'))):
            with open(src, "r", encoding="utf-8") as f:
                content = f.read()
            buf += content
            path_entries += [(os.path.split(src)[-1], subdir)]

    buf = (docbegin + buf + docend).encode("utf-8")
    decomp_size = len(buf)

    # Use maximum zlib compression level to further reduce file size
    # (at the cost of initial build times).
    buf = zlib.compress(buf, zlib.Z_BEST_COMPRESSION)

    with open(compr_tmpname, "w", encoding="utf-8") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef _DOC_DATA_RAW_H\n")
        g.write("#define _DOC_DATA_RAW_H\n")
        g.write('static const char *_doc_data_hash = "' + str(hash(buf)) + '";\n')
        g.write("static const int _doc_data_compressed_size = " + str(len(buf)) + ";\n")
        g.write("static const int _doc_data_uncompressed_size = " + str(decomp_size) + ";\n")
        g.write("static const unsigned char _doc_data_compressed[] = {\n")
        for i in range(len(buf)):
            g.write("\t" + str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")

    replace_if_different(compr_filename, compr_tmpname)

    with open(path_tmpname, "w", encoding="utf-8") as g:
        g.write(f"static const int _doc_data_class_path_count = {len(path_entries)};\n")
        g.write("struct _DocDataClassPath { const char* name; const char* path; };\n")
        g.write(f"static const _DocDataClassPath _doc_data_class_paths[{len(path_entries)+1}] = {{\n")
        for i in path_entries:
            g.write(f'        {{"{i[0]}", "{i[1]}"}},\n')
        g.write("        {nullptr, nullptr}\n};")
    replace_if_different(path_filename, path_tmpname)

def make_fonts_header(dst, sourcedir):
    source = glob(os.path.join(sourcedir, '*.woff2'))
    source.sort()
    tmpfilename = dst + '~'

    g = open(tmpfilename, "w", encoding="utf-8")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_FONTS_H\n")
    g.write("#define _EDITOR_FONTS_H\n")

    # Saving uncompressed, since FreeType will reference from memory pointer.
    for i in range(len(source)):
        with open(source[i], "rb") as f:
            buf = f.read()

        name = os.path.splitext(os.path.basename(source[i]))[0]

        g.write("static const int _font_" + name + "_size = " + str(len(buf)) + ";\n")
        g.write("static const unsigned char _font_" + name + "[] = {\n")
        for j in range(len(buf)):
            g.write("\t" + str(buf[j]) + ",\n")

        g.write("};\n")

    g.write("#endif")

    g.close()
    replace_if_different(dst, tmpfilename)

def make_translations_header(dst, source, category):
    tmpname = dst + '~'
    g = open(tmpname, "w", encoding="utf-8")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _{}_TRANSLATIONS_H\n".format(category.upper()))
    g.write("#define _{}_TRANSLATIONS_H\n".format(category.upper()))

    sorted_paths = sorted(source, key=lambda path: os.path.splitext(os.path.basename(path))[0])

    msgfmt_available = shutil.which("msgfmt") is not None

    if not msgfmt_available:
        print("WARNING: msgfmt is not found, using .po files instead of .mo")

    xl_names = []
    for i in range(len(sorted_paths)):
        if msgfmt_available:
            mo_path = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + ".mo")
            cmd = "msgfmt " + sorted_paths[i] + " --no-hash -o " + mo_path
            try:
                subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE).communicate()
                with open(mo_path, "rb") as f:
                    buf = f.read()
            except OSError as e:
                print(
                    "WARNING: msgfmt execution failed, using .po file instead of .mo: path=%r; [%s] %s"
                    % (sorted_paths[i], e.__class__.__name__, e)
                )
                with open(sorted_paths[i], "rb") as f:
                    buf = f.read()
            finally:
                try:
                    os.remove(mo_path)
                except OSError as e:
                    # Do not fail the entire build if it cannot delete a temporary file
                    print(
                        "WARNING: Could not delete temporary .mo file: path=%r; [%s] %s"
                        % (mo_path, e.__class__.__name__, e)
                    )
        else:
            with open(sorted_paths[i], "rb") as f:
                buf = f.read()

        decomp_size = len(buf)
        # Use maximum zlib compression level to further reduce file size
        # (at the cost of initial build times).
        buf = zlib.compress(buf, zlib.Z_BEST_COMPRESSION)
        name = os.path.splitext(os.path.basename(sorted_paths[i]))[0]

        g.write("static const unsigned char _{}_translation_{}_compressed[] = {{\n".format(category, name))
        for j in range(len(buf)):
            g.write("\t" + str(buf[j]) + ",\n")

        g.write("};\n")

        xl_names.append([name, len(buf), str(decomp_size)])

    g.write("struct {}TranslationList {{\n".format(category.capitalize()))
    g.write("\tconst char* lang;\n")
    g.write("\tint comp_size;\n")
    g.write("\tint uncomp_size;\n")
    g.write("\tconst unsigned char* data;\n")
    g.write("};\n\n")
    g.write("static {}TranslationList _{}_translations[] = {{\n".format(category.capitalize(), category))
    for x in xl_names:
        g.write(
            '\t{{ "{}", {}, {}, _{}_translation_{}_compressed }},\n'.format(x[0], str(x[1]), str(x[2]), category, x[0])
        )
    g.write("\t{nullptr, 0, 0, nullptr}\n")
    g.write("};\n")

    g.write("#endif")

    g.close()
    replace_if_different(dst, tmpname)
    assert(os.path.exists(dst))

def make_editor_translations_header(target, sourcedir):
    source = glob(os.path.join(sourcedir, '*.po'))
    make_translations_header(target, source, "editor")

def make_editor_translations_header(target, source):
    make_translations_header(target, source, "editor")

def make_property_translations_header(target, source):
    make_translations_header(target, source, "property")

def make_doc_translations_header(target, sourcedir):
    source = glob(os.path.join(sourcedir, '*.po'))
    make_translations_header(target, source, "doc")

if __name__ == "__main__":
    type = sys.argv[1]
    if type == 'make_doc_header':
        make_doc_header(sys.argv[2], sys.argv[3:])
    elif type == 'make_editor_translations_header':
        make_editor_translations_header(sys.argv[2], sys.argv[3])
    elif type == 'make_doc_translations':
        make_doc_translations_header(sys.argv[2], sys.argv[3])
    elif type == 'make_fonts_header':
        make_fonts_header(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f'Unknown command {type}.')
