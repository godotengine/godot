"""Functions used to generate source files during build time"""


def generate_modules_enabled(target, source, env):
    with open(target[0].path, "w", encoding="utf-8", newline="\n") as f:
        for module in env.module_list:
            f.write("#define %s\n" % ("MODULE_" + module.upper() + "_ENABLED"))


def generate_modules_tests(target, source, env):
    import os

    with open(target[0].path, "w", encoding="utf-8", newline="\n") as f:
        for header in source:
            f.write('#include "%s"\n' % (os.path.normpath(header.path)))
