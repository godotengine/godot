import os


def is_active():
    return True


def get_name():
    return "3DS"


def can_build():
    return "DEVKITARM" in os.environ


def get_opts():
    return [
        ("3ds_arch", "3DS architecture", "arm11"),
        ("3ds_graphics", "3DS graphics backend", "citro3d"),
    ]


def get_doc_classes():
    return []


def configure(env):
    devkitarm = os.environ.get("DEVKITARM", "")

    if not devkitarm:
        raise RuntimeError("DEVKITARM não encontrado.")

    env["3DS_ENABLED"] = True
    env["3DS"] = True

    env.Append(
        CPPPATH=[
            os.path.join(devkitarm, "include"),
        ]
    )

    env.Append(
        LIBPATH=[
            os.path.join(devkitarm, "lib"),
        ]
    )

    env.Append(
        LIBS=[
            "ctru",
            "citro3d",
            "citro2d",
        ]
    )

    env.Append(
        CCFLAGS=[
            "-march=armv6k",
            "-mtune=mpcore",
            "-mfloat-abi=hard",
            "-mfpu=vfpv2",
        ]
    )

    env.Append(
        LINKFLAGS=[
            "-march=armv6k",
            "-mtune=mpcore",
            "-mfloat-abi=hard",
            "-mfpu=vfpv2",
        ]
    )
