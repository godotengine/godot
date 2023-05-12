func test():
	var state = PhysicsDirectBodyState3DExtension.new()
	var prop = &"center_of_mass"
	assign(state, prop)
	state.free()

func assign(state, prop):
	state[prop].x = 1.0
