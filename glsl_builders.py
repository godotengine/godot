"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
from platform_methods import subprocess_main


class RDHeaderStruct:
    def __init__(self):
        self.vertex_lines = []
        self.fragment_lines = []
        self.compute_lines = []

        self.vertex_included_files = []
        self.fragment_included_files = []
        self.compute_included_files = []

        self.reading = ""
        self.line_offset = 0
        self.vertex_offset = 0
        self.fragment_offset = 0
        self.compute_offset = 0


def include_file_in_rd_header(filename, header_data, depth):
    fs = open(filename, "r")
    line = fs.readline()

    while line:

        index = line.find("//")
        if index != -1:
            line = line[:index]

        if line.find("#[vertex]") != -1:
            header_data.reading = "vertex"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.vertex_offset = header_data.line_offset
            continue

        if line.find("#[fragment]") != -1:
            header_data.reading = "fragment"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.fragment_offset = header_data.line_offset
            continue

        if line.find("#[compute]") != -1:
            header_data.reading = "compute"
            line = fs.readline()
            header_data.line_offset += 1
            header_data.compute_offset = header_data.line_offset
            continue

        while line.find("#include ") != -1:
            includeline = line.replace("#include ", "").strip()[1:-1]

            import os.path

            included_file = ""

            if includeline.startswith("thirdparty/"):
                included_file = os.path.relpath(includeline)

            else:
                included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline)

            if not included_file in header_data.vertex_included_files and header_data.reading == "vertex":
                header_data.vertex_included_files += [included_file]
                if include_file_in_rd_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")
            elif not included_file in header_data.fragment_included_files and header_data.reading == "fragment":
                header_data.fragment_included_files += [included_file]
                if include_file_in_rd_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")
            elif not included_file in header_data.compute_included_files and header_data.reading == "compute":
                header_data.compute_included_files += [included_file]
                if include_file_in_rd_header(included_file, header_data, depth + 1) is None:
                    print("Error in file '" + filename + "': #include " + includeline + "could not be found!")

            line = fs.readline()

        line = line.replace("\r", "")
        line = line.replace("\n", "")

        if header_data.reading == "vertex":
            header_data.vertex_lines += [line]
        if header_data.reading == "fragment":
            header_data.fragment_lines += [line]
        if header_data.reading == "compute":
            header_data.compute_lines += [line]

        line = fs.readline()
        header_data.line_offset += 1

    fs.close()

    return header_data


def build_rd_header(filename):
    header_data = RDHeaderStruct()
    include_file_in_rd_header(filename, header_data, 0)

    out_file = filename + ".gen.h"
    fd = open(out_file, "w")

    fd.write("/* WARNING, THIS FILE WAS GENERATED, DO NOT EDIT */\n")

    out_file_base = out_file
    out_file_base = out_file_base[out_file_base.rfind("/") + 1 :]
    out_file_base = out_file_base[out_file_base.rfind("\\") + 1 :]
    out_file_ifdef = out_file_base.replace(".", "_").upper()
    fd.write("#ifndef " + out_file_ifdef + "_RD\n")
    fd.write("#define " + out_file_ifdef + "_RD\n")

    out_file_class = out_file_base.replace(".glsl.gen.h", "").title().replace("_", "").replace(".", "") + "ShaderRD"
    fd.write("\n")
    fd.write('#include "servers/rendering/renderer_rd/shader_rd.h"\n\n')
    fd.write("class " + out_file_class + " : public ShaderRD {\n\n")
    fd.write("public:\n\n")

    fd.write("\t" + out_file_class + "() {\n\n")

    if len(header_data.compute_lines):

        fd.write("\t\tstatic const char _compute_code[] = {\n")
        for x in header_data.compute_lines:
            for c in x:
                fd.write(str(ord(c)) + ",")
            fd.write(str(ord("\n")) + ",")
        fd.write("\t\t0};\n\n")

        fd.write('\t\tsetup(nullptr, nullptr, _compute_code, "' + out_file_class + '");\n')
        fd.write("\t}\n")

    else:

        fd.write("\t\tstatic const char _vertex_code[] = {\n")
        for x in header_data.vertex_lines:
            for c in x:
                fd.write(str(ord(c)) + ",")
            fd.write(str(ord("\n")) + ",")
        fd.write("\t\t0};\n\n")

        fd.write("\t\tstatic const char _fragment_code[]={\n")
        for x in header_data.fragment_lines:
            for c in x:
                fd.write(str(ord(c)) + ",")
            fd.write(str(ord("\n")) + ",")
        fd.write("\t\t0};\n\n")

        fd.write('\t\tsetup(_vertex_code, _fragment_code, nullptr, "' + out_file_class + '");\n')
        fd.write("\t}\n")

    fd.write("};\n\n")

    fd.write("#endif\n")
    fd.close()


def build_rd_headers(target, source, env):
    for x in source:
        build_rd_header(str(x))


class RAWHeaderStruct:
    def __init__(self):
        self.code = ""


def include_file_in_raw_header(filename, header_data, depth):
    fs = open(filename, "r")
    line = fs.readline()

    while line:

        while line.find("#include ") != -1:
            includeline = line.replace("#include ", "").strip()[1:-1]

            import os.path

            included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline)
            include_file_in_raw_header(included_file, header_data, depth + 1)

            line = fs.readline()

        header_data.code += line
        line = fs.readline()

    fs.close()


def build_raw_header(filename):
    header_data = RAWHeaderStruct()
    include_file_in_raw_header(filename, header_data, 0)

    out_file = filename + ".gen.h"
    fd = open(out_file, "w")

    fd.write("/* WARNING, THIS FILE WAS GENERATED, DO NOT EDIT */\n")

    out_file_base = out_file.replace(".glsl.gen.h", "_shader_glsl")
    out_file_base = out_file_base[out_file_base.rfind("/") + 1 :]
    out_file_base = out_file_base[out_file_base.rfind("\\") + 1 :]
    out_file_ifdef = out_file_base.replace(".", "_").upper()
    fd.write("#ifndef " + out_file_ifdef + "_RAW_H\n")
    fd.write("#define " + out_file_ifdef + "_RAW_H\n")
    fd.write("\n")
    fd.write("static const char " + out_file_base + "[] = {\n")
    for c in header_data.code:
        fd.write(str(ord(c)) + ",")
    fd.write("\t\t0};\n\n")
    fd.write("#endif\n")
    fd.close()


def build_raw_headers(target, source, env):
    for x in source:
        build_raw_header(str(x))


if __name__ == "__main__":
    subprocess_main(globals())
