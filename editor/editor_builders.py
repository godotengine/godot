"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
import os
import os.path
from platform_methods import subprocess_main
from compat import encode_utf8, byte_to_str, open_utf8, escape_string


def make_certs_header(target, source, env):

    src = source[0]
    dst = target[0]
    f = open(src, "rb")
    g = open_utf8(dst, "w")
    buf = f.read()
    decomp_size = len(buf)
    import zlib
    buf = zlib.compress(buf)

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _CERTS_RAW_H\n")
    g.write("#define _CERTS_RAW_H\n")
    g.write("static const int _certs_compressed_size = " + str(len(buf)) + ";\n")
    g.write("static const int _certs_uncompressed_size = " + str(decomp_size) + ";\n")
    g.write("static const unsigned char _certs_compressed[] = {\n")
    for i in range(len(buf)):
        g.write("\t" + byte_to_str(buf[i]) + ",\n")
    g.write("};\n")
    g.write("#endif")

    g.close()
    f.close()


def make_doc_header(target, source, env):

    dst = target[0]
    g = open_utf8(dst, "w")
    buf = ""
    docbegin = ""
    docend = ""
    for src in source:
        if not src.endswith(".xml"):
            continue
        with open_utf8(src, "r") as f:
            content = f.read()
        buf += content

    buf = encode_utf8(docbegin + buf + docend)
    decomp_size = len(buf)
    import zlib
    buf = zlib.compress(buf)

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _DOC_DATA_RAW_H\n")
    g.write("#define _DOC_DATA_RAW_H\n")
    g.write("static const int _doc_data_compressed_size = " + str(len(buf)) + ";\n")
    g.write("static const int _doc_data_uncompressed_size = " + str(decomp_size) + ";\n")
    g.write("static const unsigned char _doc_data_compressed[] = {\n")
    for i in range(len(buf)):
        g.write("\t" + byte_to_str(buf[i]) + ",\n")
    g.write("};\n")

    g.write("#endif")

    g.close()


def make_fonts_header(target, source, env):

    dst = target[0]

    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_FONTS_H\n")
    g.write("#define _EDITOR_FONTS_H\n")

    # saving uncompressed, since freetype will reference from memory pointer
    xl_names = []
    for i in range(len(source)):
        with open(source[i], "rb")as f:
            buf = f.read()

        name = os.path.splitext(os.path.basename(source[i]))[0]

        g.write("static const int _font_" + name + "_size = " + str(len(buf)) + ";\n")
        g.write("static const unsigned char _font_" + name + "[] = {\n")
        for i in range(len(buf)):
            g.write("\t" + byte_to_str(buf[i]) + ",\n")

        g.write("};\n")

    g.write("#endif")

    g.close()


def make_translations_header(target, source, env):

    dst = target[0]

    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_TRANSLATIONS_H\n")
    g.write("#define _EDITOR_TRANSLATIONS_H\n")

    import zlib
    import os.path

    sorted_paths = sorted(source, key=lambda path: os.path.splitext(os.path.basename(path))[0])

    xl_names = []
    for i in range(len(sorted_paths)):
        with open(sorted_paths[i], "rb") as f:
            buf = f.read()
        decomp_size = len(buf)
        buf = zlib.compress(buf)
        name = os.path.splitext(os.path.basename(sorted_paths[i]))[0]

        g.write("static const unsigned char _translation_" + name + "_compressed[] = {\n")
        for i in range(len(buf)):
            g.write("\t" + byte_to_str(buf[i]) + ",\n")

        g.write("};\n")

        xl_names.append([name, len(buf), str(decomp_size)])

    g.write("struct EditorTranslationList {\n")
    g.write("\tconst char* lang;\n")
    g.write("\tint comp_size;\n")
    g.write("\tint uncomp_size;\n")
    g.write("\tconst unsigned char* data;\n")
    g.write("};\n\n")
    g.write("static EditorTranslationList _editor_translations[] = {\n")
    for x in xl_names:
        g.write("\t{ \"" + x[0] + "\", " + str(x[1]) + ", " + str(x[2]) + ", _translation_" + x[0] + "_compressed},\n")
    g.write("\t{NULL, 0, 0, NULL}\n")
    g.write("};\n")

    g.write("#endif")

    g.close()


def make_authors_header(target, source, env):

    sections = ["Project Founders", "Lead Developer", "Project Manager", "Developers"]
    sections_id = ["dev_founders", "dev_lead", "dev_manager", "dev_names"]

    src = source[0]
    dst = target[0]
    f = open_utf8(src, "r")
    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_AUTHORS_H\n")
    g.write("#define _EDITOR_AUTHORS_H\n")

    current_section = ""
    reading = False

    def close_section():
        g.write("\t0\n")
        g.write("};\n")

    for line in f:
        if reading:
            if line.startswith("    "):
                g.write("\t\"" + escape_string(line.strip()) + "\",\n")
                continue
        if line.startswith("## "):
            if reading:
                close_section()
                reading = False
            for i in range(len(sections)):
                if line.strip().endswith(sections[i]):
                    current_section = escape_string(sections_id[i])
                    reading = True
                    g.write("static const char *" + current_section + "[] = {\n")
                    break

    if reading:
        close_section()

    g.write("#endif\n")

    g.close()
    f.close()

def make_donors_header(target, source, env):

    sections = ["Platinum sponsors", "Gold sponsors", "Mini sponsors", "Gold donors", "Silver donors", "Bronze donors"]
    sections_id = ["donor_s_plat", "donor_s_gold", "donor_s_mini", "donor_gold", "donor_silver", "donor_bronze"]

    src = source[0]
    dst = target[0]
    f = open_utf8(src, "r")
    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_DONORS_H\n")
    g.write("#define _EDITOR_DONORS_H\n")

    current_section = ""
    reading = False

    def close_section():
        g.write("\t0\n")
        g.write("};\n")

    for line in f:
        if reading >= 0:
            if line.startswith("    "):
                g.write("\t\"" + escape_string(line.strip()) + "\",\n")
                continue
        if line.startswith("## "):
            if reading:
                close_section()
                reading = False
            for i in range(len(sections)):
                if line.strip().endswith(sections[i]):
                    current_section = escape_string(sections_id[i])
                    reading = True
                    g.write("static const char *" + current_section + "[] = {\n")
                    break

    if reading:
        close_section()

    g.write("#endif\n")

    g.close()
    f.close()


def make_license_header(target, source, env):

    src_copyright = source[0]
    src_license = source[1]
    dst = target[0]
    f = open_utf8(src_license, "r")
    fc = open_utf8(src_copyright, "r")
    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_LICENSE_H\n")
    g.write("#define _EDITOR_LICENSE_H\n")
    g.write("static const char *about_license =")

    for line in f:
        escaped_string = escape_string(line.strip())
        g.write("\n\t\"" + escaped_string + "\\n\"")

    g.write(";\n")

    tp_current = 0
    tp_file = ""
    tp_comment = ""
    tp_copyright = ""
    tp_license = ""

    tp_licensename = ""
    tp_licensebody = ""

    tp = []
    tp_licensetext = []
    for line in fc:
        if line.startswith("#"):
            continue

        if line.startswith("Files:"):
            tp_file = line[6:].strip()
            tp_current = 1
        elif line.startswith("Comment:"):
            tp_comment = line[8:].strip()
            tp_current = 2
        elif line.startswith("Copyright:"):
            tp_copyright = line[10:].strip()
            tp_current = 3
        elif line.startswith("License:"):
            if tp_current != 0:
                tp_license = line[8:].strip()
                tp_current = 4
            else:
                tp_licensename = line[8:].strip()
                tp_current = 5
        elif line.startswith(" "):
            if tp_current == 1:
                tp_file += "\n" + line.strip()
            elif tp_current == 3:
                tp_copyright += "\n" + line.strip()
            elif tp_current == 5:
                if line.strip() == ".":
                    tp_licensebody += "\n"
                else:
                    tp_licensebody += line[1:]
        else:
            if tp_current != 0:
                if tp_current == 5:
                    tp_licensetext.append([tp_licensename, tp_licensebody])

                    tp_licensename = ""
                    tp_licensebody = ""
                else:
                    added = False
                    for i in tp:
                        if i[0] == tp_comment:
                            i[1].append([tp_file, tp_copyright, tp_license])
                            added = True
                            break
                    if not added:
                        tp.append([tp_comment,[[tp_file, tp_copyright, tp_license]]])

                    tp_file = []
                    tp_comment = ""
                    tp_copyright = []
                    tp_license = ""
                tp_current = 0

    tp_licensetext.append([tp_licensename, tp_licensebody])

    about_thirdparty = ""
    about_tp_copyright_count = ""
    about_tp_license = ""
    about_tp_copyright = ""
    about_tp_file = ""

    for i in tp:
        about_thirdparty += "\t\"" + i[0] + "\",\n"
        about_tp_copyright_count += str(len(i[1])) + ", "
        for j in i[1]:
            file_body = ""
            copyright_body = ""
            for k in j[0].split("\n"):
                if file_body != "":
                    file_body += "\\n\"\n"
                escaped_string = escape_string(k.strip())
                file_body += "\t\"" + escaped_string
            for k in j[1].split("\n"):
                if copyright_body != "":
                    copyright_body += "\\n\"\n"
                escaped_string = escape_string(k.strip())
                copyright_body += "\t\"" + escaped_string

            about_tp_file += "\t" + file_body + "\",\n"
            about_tp_copyright += "\t" + copyright_body + "\",\n"
            about_tp_license += "\t\"" + j[2] + "\",\n"

    about_license_name = ""
    about_license_body = ""

    for i in tp_licensetext:
        body = ""
        for j in i[1].split("\n"):
            if body != "":
                body += "\\n\"\n"
            escaped_string = escape_string(j.strip())
            body += "\t\"" + escaped_string

        about_license_name += "\t\"" + i[0] + "\",\n"
        about_license_body += "\t" + body + "\",\n"

    g.write("static const char *about_thirdparty[] = {\n")
    g.write(about_thirdparty)
    g.write("\t0\n")
    g.write("};\n")
    g.write("#define THIRDPARTY_COUNT " + str(len(tp)) + "\n")

    g.write("static const int about_tp_copyright_count[] = {\n\t")
    g.write(about_tp_copyright_count)
    g.write("0\n};\n")

    g.write("static const char *about_tp_file[] = {\n")
    g.write(about_tp_file)
    g.write("\t0\n")
    g.write("};\n")

    g.write("static const char *about_tp_copyright[] = {\n")
    g.write(about_tp_copyright)
    g.write("\t0\n")
    g.write("};\n")

    g.write("static const char *about_tp_license[] = {\n")
    g.write(about_tp_license)
    g.write("\t0\n")
    g.write("};\n")

    g.write("static const char *about_license_name[] = {\n")
    g.write(about_license_name)
    g.write("\t0\n")
    g.write("};\n")
    g.write("#define LICENSE_COUNT " + str(len(tp_licensetext)) + "\n")

    g.write("static const char *about_license_body[] = {\n")
    g.write(about_license_body)
    g.write("\t0\n")
    g.write("};\n")

    g.write("#endif\n")

    g.close()
    fc.close()
    f.close()


if __name__ == '__main__':
    subprocess_main(globals())
