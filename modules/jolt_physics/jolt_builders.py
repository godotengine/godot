"""Functions used to generate source files during build time."""

import methods


def cap_files_builder(target, source, env):
    # Embed names: test binaries cannot locate the source tree at runtime.
    names = sorted(source[0].read())
    with methods.generated_wrapper(str(target[0])) as file:
        file.write("inline constexpr const char *CAP_SOURCE_FILES[] = {\n")
        for name in names:
            file.write(f'\t"{name.rsplit("/", 1)[-1]}",\n')
        file.write("};\n")
