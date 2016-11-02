
def can_build(platform):
    return True


def configure(env):
    # Tools only, disabled for non-tools
    # TODO: Find a cleaner way to achieve that
    if (env["tools"] == "no"):
        env["module_squish_enabled"] = "no"
        env.disabled_modules.append("squish")
