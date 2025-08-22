extends GutTest

var VersionNumbers = load("res://addons/gut/version_numbers.gd")

func test_can_make_one():
	var vn = VersionNumbers.new()
	assert_not_null(vn)

func test_default_version_values():
	var vn = VersionNumbers.new()
	assert_eq(vn.gut_version, '0.0.0', 'gut version')
	assert_eq(vn.required_godot_version, '0.0.0', 'required gut version')

func test_init_sets_gut_version_from_string():
	var vn = VersionNumbers.new('1.2.3')
	assert_eq(vn.gut_version, '1.2.3')

func test_init_sets_required_godot_version_from_string():
	var vn = VersionNumbers.new('1.2.3', '4.5.6')
	assert_eq(vn.required_godot_version, '4.5.6')



class TestVerNumTools:
	extends GutTest

	var Vnt = load("res://addons/gut/version_numbers.gd").VerNumTools

	var mvs_values = ParameterFactory.named_parameters(
		['parts', 's'],
		[
			[[1, 2, 3], '1.2.3'],
			[[1, 2, 3, 4, 5], '1.2.3.4.5'],
			[['a', 'b', 'c'], 'a.b.c'],
			[[9, 'q', 3, 'foo', 10], '9.q.3.foo.10'],

			[{'major':1, 'minor':2, "patch":3}, '1.2.3'],
			[{'major':4, 'minor':5, "patch":6, "status":9, "build":9}, '4.5.6'],

			['1.2.3', '1.2.3'],
			['1.a.b.4.9.z', '1.a.b.4.9.z']
		])
	func test_make_version_string_with_arrays(params = use_parameters(mvs_values)):
		var result = Vnt.make_version_string(params.parts)
		assert_eq(result, params.s)


	var mva_values = ParameterFactory.named_parameters(
		['v', 'expected'],
		[
			['1.2.3', [1, 2, 3]],
			['1.2.3.4.5', [1, 2, 3, 4, 5]],
			['a.b.c', ['a', 'b', 'c']],
			['9.q.3.foo.10', [9, 'q', 3, 'foo', 10]],

			[{'major':1, 'minor':2, "patch":3}, [1, 2, 3]],
			[{'major':4, 'minor':5, "patch":6, "status":9, "build":9}, [4, 5, 6]],

			[[1,2,3], [1,2,3]],
			[[4, 5, 6, 7, 8], [4, 5, 6, 7, 8]]
		])
	func test_make_version_array(params = use_parameters(mva_values)):
		var result = Vnt.make_version_array(params.v)
		assert_eq(result, params.expected)



	var ivg_values = ParameterFactory.named_parameters(
		['v', 'r', 'expected'],
		[
			['1.2.3', '1.2.3', true],
			['2.0.0', '1.0.0', true],
			['1.0.1', '1.0.0', true],
			['1.1.0', '1.0.0', true],
			['1.1.1', '1.0.0', true],
			['1.2.5', '1.0.10', true],
			['3.3.0', '3.2.3', true],
			['4.0.0', '3.2.0', true],
			['4.5.6', '1', true],
			['4.5.6', '4', true],

			['3.0.0', '3.0.1', false],
			['1.2.3', '2.0.0', false],
			['1.2.1', '1.2.3', false],
			['1.2.3', '1.3.0', false],
		])
	func test_is_version_gte(params = use_parameters(ivg_values)):
		assert_eq(Vnt.is_version_gte(params.v, params.r), params.expected,
			str(params.v, ' >= ', params.r, ' = ', params.expected))


	var ive_values = ParameterFactory.named_parameters(
		['v', 'r', 'expected'],
		[
			['1.2.3', '1.2.3', true],
			['1.2.3', '1.2', true],
			['1.2.3', '1', true],

			['1.2.4', '1.2.3', false],
			['1.3.3', '1.2.3', false],
			['2.2.3', '1.2.3', false],

			['1.2.3', '1.2.3.4', false]
		])
	func test_is_version_eq(params = use_parameters(ive_values)):
		assert_eq(Vnt.is_version_eq(params.v, params.r), params.expected,
			str(params.v, ' == ', params.r, ' = ', params.expected))





# class TestVersionCheck:
# 	extends 'res://addons/gut/test.gd'

# 	var Utils = load('res://addons/gut/utils.gd')

# 	func _fake_engine_version(version):
# 		var parsed = version.split('.')
# 		return{'major':parsed[0], 'minor':parsed[1], 'patch':parsed[2]}

# 	var test_ok_versions = ParameterFactory.named_parameters(
# 		['engine_version', 'req_version', 'expected_result'],
# 		[

# 		])
# 	func test_is_version_ok(p=use_parameters(test_ok_versions)):
# 		var utils = autofree(Utils.new())
# 		var engine_info = _fake_engine_version(p.engine_version)
# 		var req_version = p.req_version.split('.')
# 		assert_eq(utils.is_version_ok(engine_info, req_version), p.expected_result,
# 			str(p.engine_version, ' >= ', p.req_version))

# 	var test_is_versions = ParameterFactory.named_parameters(
# 		['engine_version', 'expected_version', 'expected_result'],
# 		[
# 		])

# 	func test_is_godot_version(p=use_parameters(test_is_versions)):
# 		var utils = autofree(Utils.new())
# 		var engine_info = _fake_engine_version(p.engine_version)
# 		assert_eq(utils.is_godot_version(p.expected_version, engine_info), p.expected_result,
# 			str(p.engine_version, ' is ', p.expected_version))
