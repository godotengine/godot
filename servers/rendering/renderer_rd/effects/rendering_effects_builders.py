"""Functions used to generate source files during build time"""

import methods


def areatex_builder(target, source, env):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
#define AREATEX_WIDTH 160
#define AREATEX_HEIGHT 560
#define AREATEX_PITCH (AREATEX_WIDTH * 2)
#define AREATEX_SIZE (AREATEX_HEIGHT * AREATEX_PITCH)

inline constexpr const unsigned char area_tex_png[] = {{
	{methods.format_buffer(buffer, 1)}
}};
""")


def searchtex_builder(target, source, env):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
#define SEARCHTEX_WIDTH 64
#define SEARCHTEX_HEIGHT 16
#define SEARCHTEX_PITCH SEARCHTEX_WIDTH
#define SEARCHTEX_SIZE (SEARCHTEX_HEIGHT * SEARCHTEX_PITCH)

inline constexpr const unsigned char search_tex_png[] = {{
	{methods.format_buffer(buffer, 1)}
}};
""")
