trait Shearable:
	func shear() -> void:
		print("Sheared")

trait Milkable:
	func milk() -> void:
		print("Milked")

class FluffySheep:
	uses Shearable, Milkable

class FluffyRam:
	uses Shearable

class Hen:
	pass

func test():
	print(Hen is Shearable)
	print(Hen.new() is Milkable)
	print(FluffySheep as Shearable)
	print(FluffyRam.new() as Milkable)
