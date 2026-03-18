SUPPORTED_PLATFORMS = {"linuxbsd", "windows", "macos"}
DISALLOWED_CPP_STANDARD_LEVELS = {
    "c++11",
    "gnu++11",
    "c++14",
    "gnu++14",
}


def _env_truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _collect_flag_tokens(env):
    tokens = []
    for key in ("CCFLAGS", "CXXFLAGS"):
        value = env.get(key, [])
        if isinstance(value, str):
            tokens.extend(value.split())
        else:
            tokens.extend(str(item) for item in value)
    return tokens


def _extract_cpp_standard_level(flag):
    lowered = flag.strip().lower()
    if lowered.startswith("-std="):
        return lowered.split("=", 1)[1]
    if lowered.startswith("/std:"):
        return lowered.split(":", 1)[1]
    return ""


def _resolve_effective_cpp_standard_flag(tokens):
    effective_flag = None
    index = 0
    while index < len(tokens):
        token = str(tokens[index]).strip()
        lowered = token.lower()
        if lowered == "-std" and index + 1 < len(tokens):
            effective_flag = f"-std={tokens[index + 1].strip()}"
            index += 2
            continue
        if lowered.startswith("-std=") or lowered.startswith("/std:"):
            effective_flag = token
        index += 1
    return effective_flag


def can_build(env, platform):
    if platform not in SUPPORTED_PLATFORMS:
        print(
            "[gaussian_splatting] Disabled for platform '{platform}'. Supported: {supported}.".format(
                platform=platform,
                supported=", ".join(sorted(SUPPORTED_PLATFORMS)),
            )
        )
        return False

    if _env_truthy(env.get("disable_3d", False)):
        print("[gaussian_splatting] Disabled because disable_3d is enabled.")
        return False

    effective_cpp_std_flag = _resolve_effective_cpp_standard_flag(_collect_flag_tokens(env))
    if effective_cpp_std_flag:
        standard_level = _extract_cpp_standard_level(effective_cpp_std_flag)
        if standard_level in DISALLOWED_CPP_STANDARD_LEVELS:
            print(
                "[gaussian_splatting] Disabled because unsupported effective C++ standard flag is set: {flag}. "
                "Use C++17 or newer.".format(flag=effective_cpp_std_flag)
            )
            return False

    return True


def configure(env):
    # Note: MODULE_GAUSSIAN_SPLATTING_ENABLED is automatically defined by Godot's
    # build system in modules_enabled.gen.h - do not define it here to avoid C4005
    pass


def get_doc_classes():
    return [
        "GaussianData",
        "GaussianSplatRenderer",
        "GaussianSplatManager",
    ]


def get_doc_path():
    return "doc_classes"
