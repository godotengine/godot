import os

from SCons.Util import WhereIs


def run_closure_compiler(target, source, env, for_signature):
    closure_bin = os.path.join(os.path.dirname(WhereIs("emcc")), "node_modules", ".bin", "google-closure-compiler")
    cmd = [WhereIs("node"), closure_bin]
    cmd.extend(["--compilation_level", "ADVANCED_OPTIMIZATIONS"])
    for f in env["JSEXTERNS"]:
        cmd.extend(["--externs", f.get_abspath()])
    for f in source:
        cmd.extend(["--js", f.get_abspath()])
    cmd.extend(["--js_output_file", target[0].get_abspath()])
    return " ".join(cmd)


def get_build_version(env):
    import version

    name = "custom_build"
    if os.getenv("BUILD_NAME") != None:
        name = os.getenv("BUILD_NAME")
    return "%d.%d.%d.%s.%s" % (version.major, version.minor, version.patch, version.status, name)


def create_engine_file(env, target, source, externs):
    if env["use_closure_compiler"]:
        return env.BuildJS(target, source, JSEXTERNS=externs)
    return env.Textfile(target, [env.File(s) for s in source])


def add_js_libraries(env, libraries):
    env.Append(JS_LIBS=env.File(libraries))


def add_js_pre(env, js_pre):
    env.Append(JS_PRE=env.File(js_pre))


def add_js_externs(env, externs):
    env.Append(JS_EXTERNS=env.File(externs))
