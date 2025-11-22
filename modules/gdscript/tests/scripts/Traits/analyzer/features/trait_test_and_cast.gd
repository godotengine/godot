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
	var my_animals : Array = []
	my_animals.append(FluffySheep.new())
	my_animals.append(FluffyRam.new())
	my_animals.append(Hen.new())

	# 'is' operator tests
	print("FluffySheep is Shearable:", FluffySheep is Shearable)
	print("FluffySheep is Milkable:", FluffySheep is Milkable)

	var Ram = FluffyRam.new()
	print("FluffyRam is Shearable:", Ram is Shearable)
	print("FluffyRam is Milkable:", Ram is Milkable)
	print()

	var count := 1
	for animal in my_animals:
		# test 'as' casts
		print("Animal: " + str(count))
		var cast1 = animal as Shearable
		var cast2 = animal as Milkable
		print("Success Shearable cast : ", cast1 != null)
		print("Success Milkable cast : ", cast2 != null)

		if animal is Shearable:
			animal.shear()
		if animal is Milkable:
			animal.milk()
		count += 1

	print("ok")
