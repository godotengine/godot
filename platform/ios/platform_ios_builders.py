"""Functions used to generate source files during build time"""

from platform_methods import generate_bundle_apple_embedded


def generate_bundle(target, source, env):
    env = source[1]
    generate_bundle_apple_embedded("ios", "ios-arm64", "ios-arm64_x86_64-simulator", env["vulkan"], env)
