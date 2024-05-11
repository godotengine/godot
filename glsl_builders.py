"""Functions used to generate source files during build time"""

import os.path
from typing import Iterable, List, Optional

from methods import generated_wrapper, print_error


def generate_inline_code(input_lines: Iterable[str], insert_newline: bool = True) -> str:
    """Take header data and generate inline code

    :param: input_lines: values for shared inline code
    :return: str - generated inline value
    """
    output = []
    for line in input_lines:
        output += [str(ord(char)) for char in line]
        if insert_newline:
            output.append(str(ord("\n")))
    output.append("0")
    return ", ".join(output)


class RDHeaderStruct:
    def __init__(self) -> None:
        self.vertex_lines: List[str] = []
        self.fragment_lines: List[str] = []
        self.compute_lines: List[str] = []

        self.vertex_included_files: List[str] = []
        self.fragment_included_files: List[str] = []
        self.compute_included_files: List[str] = []

        self.reading: str = ""
        self.line_offset: int = 0
        self.vertex_offset: int = 0
        self.fragment_offset: int = 0
        self.compute_offset: int = 0


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
                    included_file = os.path.relpath(includeline).replace("\\", "/")

                else:
                    included_file = os.path.relpath(os.path.dirname(filename) + "/" + includeline).replace("\\", "/")

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


def build_rd_header(
    filename: str, optional_output_filename: Optional[str] = None, header_data: Optional[RDHeaderStruct] = None
) -> None:
    header_data = header_data or RDHeaderStruct()
    include_file_in_rd_header(filename, header_data, 0)

    if optional_output_filename is None:
        out_file = filename + ".gen.h"
    else:
        out_file = optional_output_filename

    out_file_base = os.path.basename(out_file).split(".")[0]
    out_file_class = out_file_base.title().replace("_", "") + "ShaderRD"

    if header_data.compute_lines:
        body_parts = [
            "static const char _compute_code[] = {\n\t\t\t%s\n\t\t};" % generate_inline_code(header_data.compute_lines),
            f'setup(nullptr, nullptr, _compute_code, "{out_file_class}");',
        ]
    else:
        body_parts = [
            "static const char _vertex_code[] = {\n\t\t\t%s\n\t\t};" % generate_inline_code(header_data.vertex_lines),
            "static const char _fragment_code[] = {\n\t\t\t%s\n\t\t};"
            % generate_inline_code(header_data.fragment_lines),
            f'setup(_vertex_code, _fragment_code, nullptr, "{out_file_class}");',
        ]

    body_content = "\n\t\t".join(body_parts)

    with generated_wrapper(out_file, suffix="rd") as file:
        file.write(
            f"""\
#include "servers/rendering/renderer_rd/shader_rd.h"

class {out_file_class} : public ShaderRD {{
public:
	{out_file_class}() {{
		{body_content}
	}}
}};
"""
        )


def build_rd_headers(target, source, env) -> None:
    for x in source:
        build_rd_header(filename=str(x))


class RAWHeaderStruct:
    def __init__(self) -> None:
        self.code: str = ""


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


def build_raw_header(
    filename: str, optional_output_filename: Optional[str] = None, header_data: Optional[RAWHeaderStruct] = None
) -> None:
    header_data = header_data or RAWHeaderStruct()
    include_file_in_raw_header(filename, header_data, 0)

    if optional_output_filename is None:
        out_file = filename + ".gen.h"
    else:
        out_file = optional_output_filename

    out_file_base = os.path.basename(out_file).split(".")[0] + "_shader_glsl"

    with generated_wrapper(out_file, suffix="raw") as file:
        file.write(
            f"""\
static const char {out_file_base}[] = {{
	{generate_inline_code(header_data.code, insert_newline=False)}
}};
"""
        )


def build_raw_headers(target, source, env) -> None:
    for x in source:
        build_raw_header(filename=str(x))
