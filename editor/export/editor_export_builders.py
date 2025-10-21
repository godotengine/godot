"""Functions used to generate source files during build time"""

import methods


def make_keys_header(target, source, env):
    with methods.generated_wrapper(str(target[0])) as file:
        file.write("inline constexpr const char *trusted_public_keys[] = {")
        for src in map(str, source):
            with open(src, encoding="utf-8", newline="\n") as src_file:
                file.write(f"""\
	{methods.to_raw_cstring(src_file.read())},
""")
        file.write("};")
