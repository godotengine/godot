"""Functions used to generate source files during build time"""

from platform_methods import generate_bundle_apple_embedded


def generate_bundle(target, source, env):
    generate_bundle_apple_embedded("ios", "ios-arm64", "ios-arm64_x86_64-simulator", True, target, source, env)
