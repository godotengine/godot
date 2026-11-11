def can_build(env, platform):
    return not env["disable_2d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "TileMap",
        "TileMapLayer",
        "TileData",
        "TileMapPattern",
        "TileSet",
        "TileSetAtlasSource",
        "TileSetScenesCollectionSource",
        "TileSetSource",
    ]


def get_doc_path():
    return "doc_classes"
