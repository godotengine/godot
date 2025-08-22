extends "res://addons/gut/test.gd"

var Gut = load('res://addons/gut/gut.gd')
var Test = load('res://addons/gut/test.gd')
var StubParams = load('res://addons/gut/stub_params.gd')

const DOUBLE_ME_SCENE_PATH = 'res://test/resources/doubler_test_objects/double_me_scene.tscn'
var DoubleMeScene = load(DOUBLE_ME_SCENE_PATH)

func test_can_get_stubber():
	var g = autofree(Gut.new())
	assert_ne(g.get_stubber(), null)

# ---------------------------------
# these two tests use the gut instance that is passed to THIS test.  This isn't
# PURE testing but it appears to cover the bases ok.
# ------
func test_stubber_cleared_between_tests_setup():
	var sp = StubParams.new('thing', 'method').to_return(5)
	gut.get_stubber().add_stub(sp)
	pass_test('this sets up for next test')


func test_stubber_cleared_between_tests():
	assert_eq(gut.get_stubber().get_return('thing', 'method'), null)
# ---------------------------------

func test_can_get_doubler():
	var g = autofree(Gut.new())
	assert_ne(g.get_doubler(), null)

func test_doublers_stubber_is_guts_stubber():
	var g = autofree(Gut.new())
	assert_eq(g.get_doubler().get_stubber(), g.get_stubber())

# Since the stubber and doubler are "global" to gut, this is the best place
# to test this so that the _double_count in the doubler isn't reset which
# causes some super confusing side effects.  This test is here because the
# opposite used to be true when things were indexed in the stubber by their
# path.  This is no longer possible to do, but it doesn't seem like a big
# loss, you can still stub instances just fine.  Not sure of a scenario where
# you would want to stub all instances of a scene one way and all instances
# of a script another way.  And if there is a scenrio where this is needed, you
# can just stub all the instances.
func test_scene_and_script_are_the_same_when_stubbing_resource():
	var script_path = DOUBLE_ME_SCENE_PATH.replace('.tscn', '.gd')

	var scene = double(DoubleMeScene).instantiate()
	var script = double(load(script_path)).new()

	# order here matters.  The 2nd will overwrite the first.
	stub(DOUBLE_ME_SCENE_PATH, 'return_hello').to_return('scene')
	stub(script_path, 'return_hello').to_return('script')

	assert_eq(scene.return_hello(), 'script')
	assert_eq(script.return_hello(), 'script')
