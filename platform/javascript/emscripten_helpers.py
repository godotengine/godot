import os


def parse_config():
    em_config_file = os.getenv("EM_CONFIG") or os.path.expanduser("~/.emscripten")
    if not os.path.exists(em_config_file):
        raise RuntimeError("Emscripten configuration file '%s' does not exist" % em_config_file)

    normalized = {}
    em_config = {}
    with open(em_config_file) as f:
        try:
            # Emscripten configuration file is a Python file with simple assignments.
            exec(f.read(), em_config)
        except StandardError as e:
            raise RuntimeError("Emscripten configuration file '%s' is invalid:\n%s" % (em_config_file, e))
    normalized["EMCC_ROOT"] = em_config.get("EMSCRIPTEN_ROOT")
    normalized["NODE_JS"] = em_config.get("NODE_JS")
    normalized["CLOSURE_BIN"] = os.path.join(normalized["EMCC_ROOT"], "node_modules", ".bin", "google-closure-compiler")
    return normalized


def run_closure_compiler(target, source, env, for_signature):
    cfg = parse_config()
    cmd = [cfg["NODE_JS"], cfg["CLOSURE_BIN"]]
    cmd.extend(["--compilation_level", "ADVANCED_OPTIMIZATIONS"])
    for f in env["JSEXTERNS"]:
        cmd.extend(["--externs", f.get_abspath()])
    for f in source:
        cmd.extend(["--js", f.get_abspath()])
    cmd.extend(["--js_output_file", target[0].get_abspath()])
    return " ".join(cmd)


def create_engine_file(env, target, source, externs):
    if env["use_closure_compiler"]:
        return env.BuildJS(target, source, JSEXTERNS=externs)
    return env.Textfile(target, [env.File(s) for s in source])
