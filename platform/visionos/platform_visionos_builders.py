"""Functions used to generate source files during build time"""

from platform_methods import generate_bundle_apple_embedded


def generate_bundle(target, source, env):
    generate_bundle_apple_embedded("visionos", "xros-arm64", "xros-arm64-simulator", False, target, source, env)
