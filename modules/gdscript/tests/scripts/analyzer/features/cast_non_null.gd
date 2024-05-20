# https://github.com/godotengine/godot/issues/69504#issuecomment-1345725988

func test():
	print("cast to Variant == null: ", 1 as Variant == null)
	print("cast to Object == null: ", self as Object == null)
