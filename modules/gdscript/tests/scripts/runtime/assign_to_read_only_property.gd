func test():
	var state = PhysicsDirectBodyState3DExtension.new()
	assign(state)
	state.free()

func assign(state):
	state.center_of_mass.x -= 1.0
