"""Functions used to generate source files during build time"""

import os.path

from methods import generated_wrapper, print_error, to_raw_cstring


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


def include_file_in_rd_header(filename: str, header_data: RDHeaderStruct, depth: int) -> RDHeaderStruct:
    with open(filename, "r", encoding="utf-8") as fs:
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

                if includeline.startswith("thirdparty/"):
                    included_file = os.path.relpath(includeline)

                else:
                    included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline)

                if included_file not in header_data.vertex_included_files and header_data.reading == "vertex":
                    header_data.vertex_included_files += [included_file]
                    if include_file_in_rd_header(included_file, header_data, depth + 1) is None:
                        print_error(f'In file "{filename}": #include "{includeline}" could not be found!"')
                elif included_file not in header_data.fragment_included_files and header_data.reading == "fragment":
                    header_data.fragment_included_files += [included_file]
                    if include_file_in_rd_header(included_file, header_data, depth + 1) is None:
                        print_error(f'In file "{filename}": #include "{includeline}" could not be found!"')
                elif included_file not in header_data.compute_included_files and header_data.reading == "compute":
                    header_data.compute_included_files += [included_file]
                    if include_file_in_rd_header(included_file, header_data, depth + 1) is None:
                        print_error(f'In file "{filename}": #include "{includeline}" could not be found!"')

                line = fs.readline()

            line = line.replace("\r", "").replace("\n", "")

            if header_data.reading == "vertex":
                header_data.vertex_lines += [line]
            if header_data.reading == "fragment":
                header_data.fragment_lines += [line]
            if header_data.reading == "compute":
                header_data.compute_lines += [line]

            line = fs.readline()
            header_data.line_offset += 1

    return header_data


def build_rd_header(filename: str, shader: str) -> None:
    include_file_in_rd_header(shader, header_data := RDHeaderStruct(), 0)
    class_name = os.path.basename(shader).replace(".glsl", "").title().replace("_", "").replace(".", "") + "ShaderRD"

    with generated_wrapper(filename) as file:
        file.write(f"""\
#include "servers/rendering/renderer_rd/shader_rd.h"

class {class_name} : public ShaderRD {{
public:
	{class_name}() {{
""")

        if header_data.compute_lines:
            file.write(f"""\
		static const char *_vertex_code = nullptr;
		static const char *_fragment_code = nullptr;
		static const char _compute_code[] = {{
{to_raw_cstring(header_data.compute_lines)}
		}};
""")
        else:
            file.write(f"""\
		static const char _vertex_code[] = {{
{to_raw_cstring(header_data.vertex_lines)}
		}};
		static const char _fragment_code[] = {{
{to_raw_cstring(header_data.fragment_lines)}
		}};
		static const char *_compute_code = nullptr;
""")

        file.write(f"""\
		setup(_vertex_code, _fragment_code, _compute_code, "{class_name}");
	}}
}};
""")


def build_rd_headers(target, source, env):
    env.NoCache(target)
    for src in source:
        build_rd_header(f"{src}.gen.h", str(src))


class RAWHeaderStruct:
    def __init__(self):
        self.code = ""


def include_file_in_raw_header(filename: str, header_data: RAWHeaderStruct, depth: int) -> None:
    with open(filename, "r", encoding="utf-8") as fs:
        line = fs.readline()

        while line:
            while line.find("#include ") != -1:
                includeline = line.replace("#include ", "").strip()[1:-1]

                included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline)
                include_file_in_raw_header(included_file, header_data, depth + 1)

                line = fs.readline()

            header_data.code += line
            line = fs.readline()


def build_raw_header(filename: str, shader: str) -> None:
    include_file_in_raw_header(shader, header_data := RAWHeaderStruct(), 0)

    with generated_wrapper(filename) as file:
        file.write(f"""\
static const char {os.path.basename(shader).replace(".glsl", "_shader_glsl")}[] = {{
{to_raw_cstring(header_data.code)}
}};
""")


def build_raw_headers(target, source, env):
    env.NoCache(target)
    for src in source:
        build_raw_header(f"{src}.gen.h", str(src))
