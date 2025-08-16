func test():
	var inferred_with_variant := return_variant()
	print(inferred_with_variant)

func return_variant() -> Variant:
	return "warn"
