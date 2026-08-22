import os


def is_active():
    return True


def can_build():
    return True


def get_name():
    return "3DS"


def get_opts():
    return [
        ("3ds_sdk", "Nintendo 3DS SDK path", os.environ.get("DEVKITPRO", "")),
        ("3ds_devkitarm", "devkitARM path", os.environ.get("DEVKITARM", "")),
    ]


def configure(env):
    devkitpro = os.environ.get("DEVKITPRO", "")
    devkitarm = os.environ.get("DEVKITARM", "")

    if not devkitpro:
        print("WARNING: DEVKITPRO não está definido.")

    if not devkitarm:
        print("WARNING: DEVKITARM não está definido.")

    env["3DS_ENABLED"] = True
    env["3DS_EXPORT"] = True
