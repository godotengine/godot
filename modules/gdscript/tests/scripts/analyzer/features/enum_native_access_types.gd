func print_enum(e: TileSet.TileShape) -> TileSet.TileShape:
	print(e)
	return e

func test():
	var v: TileSet.TileShape
	v = TileSet.TILE_SHAPE_SQUARE
	v = print_enum(v)
	v = print_enum(TileSet.TILE_SHAPE_SQUARE)
	v = TileSet.TileShape.TILE_SHAPE_SQUARE
	v = print_enum(v)
	v = print_enum(TileSet.TileShape.TILE_SHAPE_SQUARE)

	v = TileSet.TILE_SHAPE_ISOMETRIC
	v = print_enum(v)
	v = print_enum(TileSet.TILE_SHAPE_ISOMETRIC)
	v = TileSet.TileShape.TILE_SHAPE_ISOMETRIC
	v = print_enum(v)
	v = print_enum(TileSet.TileShape.TILE_SHAPE_ISOMETRIC)
