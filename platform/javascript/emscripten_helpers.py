import os, json

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


def get_build_version():
    import version

    name = "custom_build"
    if os.getenv("BUILD_NAME") != None:
        name = os.getenv("BUILD_NAME")
    v = "%d.%d" % (version.major, version.minor)
    if version.patch > 0:
        v += ".%d" % version.patch
    v += ".%s.%s" % (version.status, name)
    return v


def create_engine_file(env, target, source, externs):
    if env["use_closure_compiler"]:
        return env.BuildJS(target, source, JSEXTERNS=externs)
    return env.Textfile(target, [env.File(s) for s in source])


def create_template_zip(env, js, wasm, extra):
    binary_name = "godot.tools" if env["tools"] else "godot"
    zip_dir = env.Dir("#bin/.javascript_zip")
    in_files = [
        js,
        wasm,
        "#platform/javascript/js/libs/audio.worklet.js",
    ]
    out_files = [
        zip_dir.File(binary_name + ".js"),
        zip_dir.File(binary_name + ".wasm"),
        zip_dir.File(binary_name + ".audio.worklet.js"),
    ]
    # GDNative/Threads specific
    if env["gdnative_enabled"]:
        in_files.append(extra)  # Runtime
        out_files.append(zip_dir.File(binary_name + ".side.wasm"))
    elif env["threads_enabled"]:
        in_files.append(extra)  # Worker
        out_files.append(zip_dir.File(binary_name + ".worker.js"))

    service_worker = "#misc/dist/html/service-worker.js"
    if env["tools"]:
        # HTML
        html = "#misc/dist/html/editor.html"
        cache = [
            "godot.tools.html",
            "offline.html",
            "godot.tools.js",
            "godot.tools.worker.js",
            "godot.tools.audio.worklet.js",
            "logo.svg",
            "favicon.png",
        ]
        opt_cache = ["godot.tools.wasm"]
        subst_dict = {
            "@GODOT_VERSION@": get_build_version(),
            "@GODOT_NAME@": "GodotEngine",
            "@GODOT_CACHE@": json.dumps(cache),
            "@GODOT_OPT_CACHE@": json.dumps(opt_cache),
            "@GODOT_OFFLINE_PAGE@": "offline.html",
        }
        html = env.Substfile(target="#bin/godot${PROGSUFFIX}.html", source=html, SUBST_DICT=subst_dict)
        in_files.append(html)
        out_files.append(zip_dir.File(binary_name + ".html"))
        # And logo/favicon
        in_files.append("#misc/dist/html/logo.svg")
        out_files.append(zip_dir.File("logo.svg"))
        in_files.append("#icon.png")
        out_files.append(zip_dir.File("favicon.png"))
        # PWA
        service_worker = env.Substfile(
            target="#bin/godot${PROGSUFFIX}.service.worker.js", source=service_worker, SUBST_DICT=subst_dict
        )
        in_files.append(service_worker)
        out_files.append(zip_dir.File("service.worker.js"))
        in_files.append("#misc/dist/html/manifest.json")
        out_files.append(zip_dir.File("manifest.json"))
        in_files.append("#misc/dist/html/offline.html")
        out_files.append(zip_dir.File("offline.html"))
    else:
        # HTML
        in_files.append("#misc/dist/html/full-size.html")
        out_files.append(zip_dir.File(binary_name + ".html"))
        in_files.append(service_worker)
        out_files.append(zip_dir.File(binary_name + ".service.worker.js"))
        in_files.append("#misc/dist/html/offline-export.html")
        out_files.append(zip_dir.File("godot.offline.html"))

    zip_files = env.InstallAs(out_files, in_files)
    env.Zip(
        "#bin/godot",
        zip_files,
        ZIPROOT=zip_dir,
        ZIPSUFFIX="${PROGSUFFIX}${ZIPSUFFIX}",
        ZIPCOMSTR="Archiving $SOURCES as $TARGET",
    )


def add_js_libraries(env, libraries):
    env.Append(JS_LIBS=env.File(libraries))


def add_js_pre(env, js_pre):
    env.Append(JS_PRE=env.File(js_pre))


def add_js_externs(env, externs):
    env.Append(JS_EXTERNS=env.File(externs))
