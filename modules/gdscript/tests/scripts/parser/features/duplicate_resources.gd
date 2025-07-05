class ResA extends Resource:
	var a = 1

class ResB extends Resource:
	@export var exported_automatic_copy: ResA
	@never_duplicate
	@export var exported_shared_resource: ResA
	@always_duplicate
	@export var exported_unique_resource: ResA
	var automatic_copy: ResA
	@never_duplicate
	var shared_resource: ResA
	@always_duplicate
	var unique_resource: ResA



func test():
	var res_a := ResA.new()
	var res_b := ResB.new()
	res_b.exported_automatic_copy = res_a
	res_b.exported_shared_resource = res_a
	res_b.exported_unique_resource = res_a
	res_b.automatic_copy = res_a
	res_b.shared_resource = res_a
	res_b.unique_resource = res_a

	var shallow_copy := res_b.duplicate(false)
	var with_subresources_copy := res_b.duplicate(true)
	var copy_all := res_b.duplicate_deep(Resource.DEEP_DUPLICATE_ALL)
	var copy_internal := res_b.duplicate_deep(Resource.DEEP_DUPLICATE_INTERNAL)
	var copy_none := res_b.duplicate_deep(Resource.DEEP_DUPLICATE_NONE)

	var to_test = [
		{
			name = "shallow_copy",
			resource = shallow_copy
		}, {
			name = "with_subresources_copy",
			resource = with_subresources_copy
		}, {
			name = "copy_all",
			resource = copy_all
		}, {
			name = "copy_internal",
			resource = copy_internal
		}, {
			name = "copy_none",
			resource = copy_none
		}
	]
	for field in ["exported_automatic_copy", "exported_shared_resource", "exported_unique_resource", "automatic_copy", "shared_resource", "unique_resource"]:
		for case in to_test:
			var member = case.name
			var copy = case.resource
			var copy_field = copy.get(field)
			print(member, ".", field, " duplicated ? ", copy_field != res_a)
		print("---")
