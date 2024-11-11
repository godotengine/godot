"""Functions used to generate source files during build time"""

import os
import os.path
import shutil
import subprocess
import tempfile
import uuid
import zlib

from methods import print_warning


def make_doc_header(target, source, env):
    dst = str(target[0])
    with open(dst, "w", encoding="utf-8", newline="\n") as g:
        buf = ""
        docbegin = ""
        docend = ""
        for src in source:
            src = str(src)
            if not src.endswith(".xml"):
                continue
            with open(src, encoding="utf-8") as f:
                content = f.read()
            buf += content

        buf = (docbegin + buf + docend).encode("utf-8")
        decomp_size = len(buf)

        # Use maximum zlib compression level to further reduce file size
        # (at the cost of initial build times).
        buf = zlib.compress(buf, zlib.Z_BEST_COMPRESSION)

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


def make_translations_header(target, source, env, category):
    dst = str(target[0])

    with open(dst, "w", encoding="utf-8", newline="\n") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write(f"#ifndef _{category.upper()}_TRANSLATIONS_H\n")
        g.write(f"#define _{category.upper()}_TRANSLATIONS_H\n")

        sorted_paths = sorted([str(x) for x in source], key=lambda path: os.path.splitext(os.path.basename(path))[0])

        msgfmt_available = shutil.which("msgfmt") is not None

        if not msgfmt_available:
            print_warning("msgfmt is not found, using .po files instead of .mo")

        xl_names = []
        for i in range(len(sorted_paths)):
            name = os.path.splitext(os.path.basename(sorted_paths[i]))[0]
            # msgfmt erases non-translated messages, so avoid using it if exporting the POT.
            if msgfmt_available and name != category:
                mo_path = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + ".mo")
                cmd = "msgfmt " + sorted_paths[i] + " --no-hash -o " + mo_path
                try:
                    subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE).communicate()
                    with open(mo_path, "rb") as f:
                        buf = f.read()
                except OSError as e:
                    print_warning(
                        f"msgfmt execution failed, using .po file instead of .mo: path={sorted_paths[i]!r}; [{e.__class__.__name__}] {e}"
                    )
                    with open(sorted_paths[i], "rb") as f:
                        buf = f.read()
                finally:
                    try:
                        os.remove(mo_path)
                    except OSError as e:
                        # Do not fail the entire build if it cannot delete a temporary file.
                        print_warning(
                            f"Could not delete temporary .mo file: path={mo_path!r}; [{e.__class__.__name__}] {e}"
                        )
            else:
                with open(sorted_paths[i], "rb") as f:
                    buf = f.read()

                if name == category:
                    name = "source"

            decomp_size = len(buf)
            # Use maximum zlib compression level to further reduce file size
            # (at the cost of initial build times).
            buf = zlib.compress(buf, zlib.Z_BEST_COMPRESSION)

            g.write(f"static const unsigned char _{category}_translation_{name}_compressed[] = {{\n")
            for j in range(len(buf)):
                g.write("\t" + str(buf[j]) + ",\n")

            g.write("};\n")

            xl_names.append([name, len(buf), str(decomp_size)])

        g.write(f"struct {category.capitalize()}TranslationList {{\n")
        g.write("\tconst char* lang;\n")
        g.write("\tint comp_size;\n")
        g.write("\tint uncomp_size;\n")
        g.write("\tconst unsigned char* data;\n")
        g.write("};\n\n")
        g.write(f"static {category.capitalize()}TranslationList _{category}_translations[] = {{\n")
        for x in xl_names:
            g.write(f'\t{{ "{x[0]}", {x[1]}, {x[2]}, _{category}_translation_{x[0]}_compressed }},\n')
        g.write("\t{nullptr, 0, 0, nullptr}\n")
        g.write("};\n")

        g.write("#endif")


def make_editor_translations_header(target, source, env):
    make_translations_header(target, source, env, "editor")


def make_property_translations_header(target, source, env):
    make_translations_header(target, source, env, "property")


def make_doc_translations_header(target, source, env):
    make_translations_header(target, source, env, "doc")


def make_extractable_translations_header(target, source, env):
    make_translations_header(target, source, env, "extractable")
