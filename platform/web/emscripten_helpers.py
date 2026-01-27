import json
import os
import typing

if typing.TYPE_CHECKING:
    T = typing.TypeVar("T")

from SCons.Util import WhereIs

from misc.scripts.copyright_headers import process_file_buffer as process_file_buffer_copyright_buffer
from platform_methods import get_build_version


def ensure_list(value):  # type: (typing.Union[T, typing.List[T]]) -> typing.List[T]
    if not isinstance(value, list):
        return [value]
    return value


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


def package_js_module_generator(target, source, env, for_signature):
    if for_signature:
        return source

    target = target[0]
    source = source[0]

    def get_wrapper(filename):
        def wrapper(target, source, env):
            return package_js_module_action_ensure_copyright_buffer(target, source, env, filename)

        return wrapper

    source_filename = os.path.basename(source.get_abspath())
    wrapper = get_wrapper(source_filename)
    return env.Action(wrapper, "Ensuring copyright buffer.")


def package_js_module_action_ensure_copyright_buffer(target, source, env, filename):
    target = target[0]
    source = source[0]

    with open(source, mode="r", encoding="utf-8") as source_file:
        with open(target, mode="w", encoding="utf-8") as target_file:
            new_contents = process_file_buffer_copyright_buffer(filename, source_file)
            CHUNK_SIZE = 1024
            while True:
                chunk = new_contents.read(CHUNK_SIZE)
                if not chunk:
                    break
                target_file.write(chunk)


def create_template_zip(env, js, wasm, side):
    binary_name = "godot.editor" if env.editor_build else "godot"
    zip_dir = env.Dir(env.GetTemplateZipPath())

    in_files = [
        js,
        wasm,
        "#platform/web/js/libs/audio.worklet.js",
        "#platform/web/js/libs/audio.position.worklet.js",
        env.PackageJSModule(
            target="#bin/obj/platform/web/js/modules/utils/concurrency.js",
            source="#platform/web/js/modules/utils/concurrency.js",
        ),
        env.PackageJSModule(
            target="#bin/obj/platform/web/js/modules/utils/wait.js", source="#platform/web/js/modules/utils/wait.js"
        ),
    ]
    out_files = [
        zip_dir.File(binary_name + ".js"),
        zip_dir.File(binary_name + ".wasm"),
        zip_dir.File(binary_name + ".audio.worklet.js"),
        zip_dir.File(binary_name + ".audio.position.worklet.js"),
        zip_dir.File(binary_name + ".utils.concurrency.js"),
        zip_dir.File(binary_name + ".utils.wait.js"),
    ]
    # Dynamic linking (extensions) specific.
    if env["dlink_enabled"]:
        in_files.append(side)  # Side wasm (contains the actual Godot code).
        out_files.append(zip_dir.File(binary_name + ".side.wasm"))

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
        in_files.append(html)
        out_files.append(zip_dir.File(binary_name + ".html"))
        # And logo/favicon
        in_files.append("#misc/dist/html/logo.svg")
        out_files.append(zip_dir.File("logo.svg"))
        in_files.append("#icon.png")
        out_files.append(zip_dir.File("favicon.png"))
        # PWA
        service_worker = env.Substfile(
            target="#bin/godot${PROGSUFFIX}.service.worker.js",
            source=service_worker,
            SUBST_DICT=subst_dict,
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
