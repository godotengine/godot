#!/usr/bin/python3

import platform, argparse, os, sys

# e.g. linux
system = platform.system().lower()
# e.g. x86_64
machine = platform.machine()


def replace(instr, ndk, host, machine, android_arch, android_abi):
    REPLACES = {
        "@NDK_PATH@": ndk,
        "@HOST_OS@": host,
        "@HOST_ARCH@": machine,
        "@ANDROID_ARCH@": android_arch,
        "@ANDROID_ABI@": android_abi,
        "@ANDROID_ABI_NAME@": "androideabi" if android_arch == "armv7a" else "android",
        "@ANDROID_AR_PREFIX@": "arm" if android_arch == "armv7a" else android_arch,
    }
    for k, v in REPLACES.items():
        instr = instr.replace(k, v)
    return instr


if __name__ == "__main__":
    cross_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "cross", "android-base")
    parser = argparse.ArgumentParser()
    parser.add_argument("ndk_path")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("-a", "--android-abi", default="24")
    parser.add_argument("-m", "--android-arch", default="aarch64", choices=["armv7a", "aarch64", "i686", "x86_64"])
    parser.add_argument("-i", "--input", default=cross_path)

    args = parser.parse_args()
    with open(args.input, "r") as r, (open(args.output, "w") if args.output else sys.stdout) as w:
        for l in r.readlines():
            w.write(replace(l, args.ndk_path, system, machine, args.android_arch, args.android_abi))
