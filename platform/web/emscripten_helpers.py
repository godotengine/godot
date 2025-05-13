import json
import os
import subprocess

from SCons.Errors import UserError
from SCons.Util import WhereIs, splitext

from misc.utility.color import print_info, print_warning
from platform_methods import get_build_version

text_file_extensions = [".js", ".html"]


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


def create_template_zip(env, js, wasm, side):
    binary_name = "godot.editor" if env.editor_build else "godot"
    zip_dir = env.Dir(env.GetTemplateZipPath())

    in_files = []
    out_files = []

    def add_gzip_file(in_file, out_file):
        file_gz = env.CompressGZip(in_file)
        in_files.append(file_gz)
        out_files.append(str(out_file) + ".gz")

    def add_brotli_file(in_file, out_file):
        file_br = env.CompressBrotli(in_file)
        in_files.append(file_br)
        out_files.append(str(out_file) + ".br")

    def compress_file(in_file, out_file):
        if "gzip" in env["compress_for_servers"]:
            add_gzip_file(in_file, out_file)
        if "brotli" in env["compress_for_servers"]:
            add_brotli_file(in_file, out_file)

    def add_to_template(in_file, zip_file):
        out_file = zip_dir.File(zip_file)
        in_files.append(in_file)
        out_files.append(out_file)
        compress_file(in_file, out_file)

    add_to_template(js, binary_name + ".js")
    add_to_template(wasm, binary_name + ".wasm")
    add_to_template("#platform/web/js/libs/audio.worklet.js", binary_name + ".audio.worklet.js")
    add_to_template("#platform/web/js/libs/audio.position.worklet.js", binary_name + ".audio.position.worklet.js")

    # Dynamic linking (extensions) specific.
    if env["dlink_enabled"]:
        add_to_template(side, binary_name + "side.wasm")  # Side wasm (contains the actual Godot code).

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
        add_to_template("#icon.png", "favicon.png")
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
        add_to_template("#misc/dist/html/offline-export.html", "godot.offline.html")

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


def add_js_externs(env, externs):
    env.Append(JS_EXTERNS=env.File(externs))


def run_gzip_compression(target, source, env):
    try:
        import gzip
    except ImportError as exc:
        msg = f"Cannot import 'gzip' module.\n{exc}"
        raise UserError(msg) from exc

    for t in target:
        target_path = str(t)
        source_path = splitext(target_path)[0]

        if source_path in [str(p) for p in source]:
            with gzip.open(target_path, "wb") as tf:
                with open(source_path, "rb") as sf:
                    print_info(f'[gzip] Reading "{source_path}"')
                    source_bytes = sf.read()
                    source_bytes_len = len(source_bytes)
                    human_size = human_readable_size(source_bytes_len)
                    print_info(f'[gzip] Compressing "{source_path}" ({human_size}) / Writing "{target_path}"')
                    tf.write(source_bytes)


def run_brotli_compression(target, source, env):
    brotli_path = WhereIs("brotli")
    if brotli_path is None:
        raise UserError("[Brotli] Could not find `brotli` command line utility.")

    for t in target:
        target_path = str(t)
        source_path = splitext(target_path)[0]

        if source_path in [str(p) for p in source]:
            absolute_source_path = env.File(source_path).get_abspath()
            source_stat = os.stat(absolute_source_path)
            one_megabyte = 1024 * 1024
            is_bigger_than_one_megabyte = source_stat.st_size > one_megabyte
            human_size = human_readable_size(source_stat.st_size)

            print_info(f'[brotli] Compressing "{source_path}" ({human_size}) to "{target_path}" ')
            if is_bigger_than_one_megabyte:
                print_warning(
                    f'[brotli] As "{source_path}" is more than 1MiB ({human_size}), it may take a while. Brotli is notoriously slow when compiling important files.'
                )

            brotli_cmd = [brotli_path, source_path, "-o", target_path, "--force"]
            subprocess.run(brotli_cmd)


def human_readable_size(size: float) -> str:
    for unit in ("", "Ki", "Mi", "Gi"):
        if abs(size) < 1024:
            return f"{size:3.1f}{unit}B"
        size = size / 1024
    return f"{size}B"
