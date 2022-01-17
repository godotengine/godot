# https://github.com/godotengine/godot/issues/56702

func test():
	# somewhat obscure feature: referencing parameters in defaults, but only earlier ones!
	ref_default("non-optional")


func ref_default(nondefault1, defa=nondefault1, defb=defc, defc=1):
	prints(nondefault1, nondefault2, defa, defb, defc)
