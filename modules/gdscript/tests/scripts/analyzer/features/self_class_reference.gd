class_name SelfClassReference


const Constant := Self

class Extended extends Self:
	pass

static func static_func(instance: Self) -> Self:
	return instance

func instance_func() -> Self:
	return self


func test():
	@warning_ignore("assert_always_true")
	assert(Self == Constant)

	@warning_ignore("assert_always_true")
	assert(Self == SelfClassReference)

	var extended := Extended.new()
	assert(extended is Self)
	assert(extended is Constant)
	assert(extended is SelfClassReference)

	var constructed := Self.new()
	assert(constructed is Self)
	assert(constructed is Constant)
	assert(constructed is SelfClassReference)

	var static_funced := Self.static_func(self)
	assert(static_funced is Self)
	assert(static_funced is Constant)
	assert(static_funced is SelfClassReference)

	var instance_funced := instance_func()
	assert(instance_funced is Self)
	assert(instance_funced is Constant)
	assert(instance_funced is SelfClassReference)

	var variable: Self = self
	assert(variable is Self)
	assert(variable is Constant)
	assert(variable is SelfClassReference)

	print('ok')
