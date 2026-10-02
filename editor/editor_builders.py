"""Functions used to generate source files during build time"""

import os.path

import methods


def doc_data_class_path_builder(target, source, env):
    paths = dict(sorted(source[0].read().items()))
    data = "\n".join([f'\t{{"{key}", "{value}"}},' for key, value in paths.items()])
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
struct _DocDataClassPath {{
	const char *name;
	const char *path;
}};

inline constexpr int _doc_data_class_path_count = {len(paths)};
inline constexpr _DocDataClassPath _doc_data_class_paths[{len(paths) + 1}] = {{
	{data}
	{{nullptr, nullptr}},
}};
"""
        )


def register_exporters_builder(target, source, env):
    platforms = source[0].read()
    exp_inc = "\n".join([f'#include "platform/{p}/export/export.h"' for p in platforms])
    exp_reg = "\n\t".join([f"register_{p}_exporter();" for p in platforms])
    exp_type = "\n\t".join([f"register_{p}_exporter_types();" for p in platforms])
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
#include "register_exporters.h"

{exp_inc}

void register_exporters() {{
	{exp_reg}
}}

void register_exporter_types() {{
	{exp_type}
}}
"""
        )


def make_doc_header(target, source, env):
    buffer = b"".join([methods.get_buffer(src) for src in map(str, source)])
    decomp_size = len(buffer)
    buffer = methods.compress_buffer(buffer)

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
inline constexpr const char *_doc_data_hash = "{hash(buffer)}";
inline constexpr int _doc_data_compressed_size = {len(buffer)};
inline constexpr int _doc_data_uncompressed_size = {decomp_size};
inline constexpr const unsigned char _doc_data_compressed[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def make_translations(target, source, env):
    target_h, target_cpp = str(target[0]), str(target[1])

    category = os.path.basename(target_h).split("_")[0]
    sorted_paths = sorted([src.abspath for src in source], key=lambda path: os.path.splitext(os.path.basename(path))[0])

    xl_names = []

    with methods.generated_wrapper(target_cpp) as file:
        for path in sorted_paths:
            name = os.path.splitext(os.path.basename(path))[0]

            buffer = methods.get_buffer(path)
            if name == category:
                name = "source"

            decomp_size = len(buffer)
            buffer = methods.compress_buffer(buffer)

            file.write(f"""\
inline constexpr const unsigned char _{category}_translation_{name}_compressed[] = {{
{methods.format_buffer(buffer, 1)}
}};

""")

            xl_names.append([name, len(buffer), decomp_size])

        file.write(f"""\
#include "{target_h}"

const EditorTranslationList _{category}_translations[] = {{
""")

        for x in xl_names:
            file.write(f'\t{{ "{x[0]}", {x[1]}, {x[2]}, _{category}_translation_{x[0]}_compressed }},\n')

        file.write("""\
	{ nullptr, 0, 0, nullptr },
};
""")

    with methods.generated_wrapper(target_h) as file:
        file.write(f"""\

#ifndef EDITOR_TRANSLATION_LIST
#define EDITOR_TRANSLATION_LIST

struct EditorTranslationList {{
	const char* lang;
	int comp_size;
	int uncomp_size;
	const unsigned char* data;
}};

#endif // EDITOR_TRANSLATION_LIST

extern const EditorTranslationList _{category}_translations[];
""")
