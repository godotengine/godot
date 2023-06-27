func test():
	var state := PhysicsDirectBodyState3DExtension.new()
	state.center_of_mass.x += 1.0
	state.free()
