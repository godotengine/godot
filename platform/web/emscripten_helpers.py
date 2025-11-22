import json
import os

from SCons.Util import WhereIs

from platform_methods import get_build_version


def run_closure_compiler(target, source, env, for_signature):
    closure_bin = os.path.join(
        os.path.dirname(WhereIs("emcc")),
        "node_modules",
        ".bin",
        "google-closure-compiler",
    )
    cmd = [WhereIs("node"), closure_bin]
    cmd.extend(["--compilation_level", "ADVANCED_OPTIMIZATIONS"])
    for f in env["JSEXTERNS"]:
        cmd.extend(["--externs", f.get_abspath()])
    for f in source:
        cmd.extend(["--js", f.get_abspath()])
    cmd.extend(["--js_output_file", target[0].get_abspath()])
    return " ".join(cmd)


def create_engine_file(env, target, source, externs, threads_enabled):
    if env["use_closure_compiler"]:
        return env.BuildJS(target, source, JSEXTERNS=externs)
    subst_dict = {"___GODOT_THREADS_ENABLED": "true" if threads_enabled else "false"}
    return env.Substfile(target=target, source=[env.File(s) for s in source], SUBST_DICT=subst_dict)


def create_template_zip(
    env,
    js,
    main_wasm,
    side_wasm=None,
    main_wasm_dwarf=None,
    main_wasm_dwarf_package=None,
    side_wasm_dwarf=None,
    side_wasm_dwarf_package=None,
):
    binary_name = "godot.editor" if env.editor_build else "godot"
    zip_dir = env.Dir(env.GetTemplateZipPath())

    in_files = []
    out_files = []

    def add_to_template(in_file, zip_file):
        out_file = zip_dir.File(zip_file)
        in_files.append(in_file)
        out_files.append(out_file)

    add_to_template(js, binary_name + ".js")
    add_to_template(main_wasm, binary_name + ".wasm")
    add_to_template("#platform/web/js/libs/audio.worklet.js", binary_name + ".audio.worklet.js")
    add_to_template("#platform/web/js/libs/audio.position.worklet.js", binary_name + ".audio.position.worklet.js")

    # Dynamic linking (extensions) specific.
    if side_wasm is not None:
        add_to_template(side_wasm, binary_name + ".side.wasm")

    # Those files cannot be renamed, as their relative .wasm file has their name baked in the binary.
    # They must also reside besides their original .wasm files.
    if main_wasm_dwarf is not None:
        add_to_template(main_wasm_dwarf, main_wasm_dwarf.name)
        if main_wasm_dwarf_package is not None:
            add_to_template(main_wasm_dwarf_package, main_wasm_dwarf_package.name)
    if side_wasm_dwarf is not None:
        add_to_template(side_wasm_dwarf, side_wasm_dwarf.name)
        if side_wasm_dwarf_package is not None:
            add_to_template(side_wasm_dwarf_package, side_wasm_dwarf_package.name)

    service_worker = "#misc/dist/html/service-worker.js"
    if env.editor_build:
        # HTML
        html = "#misc/dist/html/editor.html"
        cache = [
            "godot.editor.html",
            "offline.html",
            "godot.editor.js",
            "godot.editor.audio.worklet.js",
            "godot.editor.audio.position.worklet.js",
            "logo.svg",
            "favicon.png",
        ]
        opt_cache = ["godot.editor.wasm"]
        subst_dict = {
            "___GODOT_VERSION___": get_build_version(False),
            "___GODOT_NAME___": "GodotEngine",
            "___GODOT_CACHE___": json.dumps(cache),
            "___GODOT_OPT_CACHE___": json.dumps(opt_cache),
            "___GODOT_OFFLINE_PAGE___": "offline.html",
            "___GODOT_THREADS_ENABLED___": "true" if env["threads"] else "false",
            "___GODOT_ENSURE_CROSSORIGIN_ISOLATION_HEADERS___": "true",
        }
        html = env.Substfile(target="#bin/godot${PROGSUFFIX}.html", source=html, SUBST_DICT=subst_dict)
        add_to_template(html, binary_name + ".html")
        # And logo/favicon
        add_to_template("#misc/dist/html/logo.svg", "logo.svg")
        add_to_template("#icon.png", "favicon.svg")
        # PWA
        service_worker = env.Substfile(
            target="#bin/godot${PROGSUFFIX}.service.worker.js",
            source=service_worker,
            SUBST_DICT=subst_dict,
        )
        add_to_template(service_worker, "service.worker.js")
        add_to_template("#misc/dist/html/manifest.json", "manifest.json")
        add_to_template("#misc/dist/html/offline.html", "offline.html")
    else:
        # HTML
        add_to_template("#misc/dist/html/full-size.html", binary_name + ".html")
        add_to_template(service_worker, binary_name + ".service.worker.js")
        add_to_template("#misc/dist/html/offline-export.html", binary_name + ".offline.html")

    zip_files = env.NoCache(env.InstallAs(out_files, in_files))
    env.NoCache(
        env.Zip(
            "#bin/godot",
            zip_files,
            ZIPROOT=zip_dir,
            ZIPSUFFIX="${PROGSUFFIX}${ZIPSUFFIX}",
        )
    )


def get_template_zip_path(env):
    return "#bin/.web_zip"


def add_js_libraries(env, libraries):
    env.Append(JS_LIBS=env.File(libraries))


def add_js_pre(env, js_pre):
    env.Append(JS_PRE=env.File(js_pre))


def add_js_post(env, js_post):
    env.Append(JS_POST=env.File(js_post))


def add_js_externs(env, externs):
    env.Append(JS_EXTERNS=env.File(externs))
