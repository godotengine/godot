def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
<<<<<<< HEAD
    return [
        "NoiseTexture",
        "OpenSimplexNoise",
    ]
=======
    return ["NoiseTexture", "OpenSimplexNoise"]

>>>>>>> audio-bus-effect-fixed


def get_doc_path():
    return "doc_classes"
