const Preloaded := preload( 'preload_script_native_type.notest.gd' )

func test() -> void:
	var inferred := Preloaded.new()
	var inferred_owner := inferred.owner

	var typed: Preloaded
	typed = Preloaded.new()
	var typed_owner := typed.owner

	print(typed_owner == inferred_owner)

	inferred.free()
	typed.free()
	print('ok')
