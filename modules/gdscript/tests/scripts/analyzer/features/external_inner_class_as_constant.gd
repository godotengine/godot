const External = preload("external_inner_class_as_constant_external.notest.gd")
const ExternalInnerClass = External.InnerClass

func test():
	var inst_external: ExternalInnerClass = ExternalInnerClass.new()
	inst_external.x = 4.0
	print(inst_external.x)
