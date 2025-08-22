# ------------------------------------------------------------------------------
# Loading directories
# ------------------------------------------------------------------------------
extends GutTest

# Common parts for the other InnerClasses in this script.
class BaseTest:
	extends GutInternalTester

	const TEST_LOAD_DIR = 'res://test/resources/parsing_and_loading_samples'
	const TEST_BASE_DIR = 'user://test_directories/'

# ------------------------------------------------------------------------------
# I chose a more dynamic approach for creating directories and files for new
# tests.  This InnerClass has the older tests that use the
# parsing_and_loading_samples directory.
# ------------------------------------------------------------------------------
class TestUsingResDirs:
	extends BaseTest
	# GlobalReset(gr) variables to be used by tests.
	# The values of these are reset in the setup or
	# teardown methods.
	var gr = {
		gut = null
	}

	func before_each():
		gr.gut = add_child_autofree(new_gut())


	func test_adding_directory_loads_files():
		gr.gut.add_directory(TEST_LOAD_DIR)
		assert_true(gr.gut._test_collector.has_script(TEST_LOAD_DIR + '/test_samples.gd'))

	func test_adding_directory_does_not_load_bad_prefixed_files():
		gr.gut.add_directory(TEST_LOAD_DIR)
		assert_false(gr.gut._test_collector.has_script(TEST_LOAD_DIR + '/bad_prefix.gd'))

	func test_adding_directory_loads_files_for_given_suffix():
		gr.gut.add_directory(TEST_LOAD_DIR, 'test_', 'specific_suffix.gd')
		assert_true(gr.gut.get_test_collector().has_script(TEST_LOAD_DIR + '/test_with_specific_suffix.gd'))
		assert_eq(gr.gut.get_test_collector().scripts.size(), 1, 'Should not find more than one test script with \'specific_suffix.gd\'')

	func test_adding_directory_skips_files_with_wrong_extension():
		gr.gut.add_directory(TEST_LOAD_DIR)
		assert_false(gr.gut._test_collector.has_script(TEST_LOAD_DIR + '/test_bad_extension.txt'))

	func test_if_directory_does_not_exist_it_does_not_die():
		gr.gut.add_directory('res://adsf')
		assert_true(true, 'We should get here')

	func test_adding_same_directory_does_not_add_duplicates():
		gr.gut.add_directory('res://test/unit')
		var orig = gr.gut._test_collector.scripts.size()
		gr.gut.add_directory('res://test/unit')
		assert_eq(gr.gut._test_collector.scripts.size(), orig)

	# We only have 3 directories with tests in them so test 3
	func test_directories123_defined_in_editor_are_loaded_on_ready():
		var g = autofree(Gut.new())
		var t = autofree(Test.new())
		t.gut = g
		add_child(g)
		g.add_directory('res://test/resources/parsing_and_loading_samples')
		g.add_directory('res://test/unit')
		g.add_directory('res://test/integration')

		t.assert_true(g._test_collector.has_script('res://test/resources/parsing_and_loading_samples/test_samples.gd'), 'Should have dir1 script')
		t.assert_true(g._test_collector.has_script('res://test/unit/test_gut.gd'), 'Should have dir2 script')
		t.assert_true(g._test_collector.has_script('res://test/integration/test_sample_all_passed_integration.gd'), 'Should have dir3 script')
		assert_eq(t.get_pass_count(), 3, 'they should have passed')

# ------------------------------------------------------------------------------
# An attempt to make a more dynamic approach for testing directory and file
# traversal.  This will most likely prove to be an exercise in over engineering.
# ------------------------------------------------------------------------------
class TestUsingDynamicDirs:
	extends BaseTest
	# GlobalReset(gr) variables to be used by tests.
	# The values of these are reset in the setup or
	# teardown methods.
	var gr = {
		gut = null
	}

	# holds directories that are created with _create_test_dir.  These directories
	# are deleted after each test in the teardown.
	var _test_dirs = []

	# Create a directory in a test location.  directories are added to _test_dirs
	# in the order they are created so they can be deleted later.
	func _create_test_dir(rel_path):
		var dir = DirAccess.open(TEST_BASE_DIR)
		dir.make_dir(rel_path)
		_test_dirs.append(TEST_BASE_DIR + rel_path)

	func _create_test_script(rel_path):
		var path = TEST_BASE_DIR + rel_path

		var file = FileAccess.open(path, FileAccess.WRITE)
		file.store_string("extends GutTest\n")
		file.store_string("func test_nothing():\n")
		file.store_string("\tpending()\n")

	func _create_all_dirs_and_files():
		var dir = DirAccess.open('user://')
		dir.make_dir('test_directories')

		_create_test_dir('root')
		_create_test_script('root/test_script.gd')

		_create_test_dir('root/one')
		_create_test_script('root/one/test_script.gd')

		_create_test_dir('root/two')
		_create_test_script('root/two/test_script.gd')

		_create_test_dir('other_root')
		_create_test_dir('other_root/three')
		_create_test_script('other_root/three/test_script.gd')

	func before_each():
		gr.gut = add_child_autofree(new_gut())
		_create_all_dirs_and_files()

	func after_each():
		var i = _test_dirs.size() -1
		# delete the directories in reverse order since it is easier than
		# recursively deleting a directory and everything in it.
		while(i > 0):
			gut.directory_delete_files(_test_dirs[i])
			var dir = DirAccess.open(_test_dirs[i])
			if(dir != null):
				dir.remove(_test_dirs[i])
			i -= 1

		_test_dirs.clear()

	func test_test_data_looks_ok():
		gr.gut.add_directory(_test_dirs[0])
		assert_true(gr.gut._test_collector.has_script(TEST_BASE_DIR + 'root/test_script.gd'))


	func test_when_subdir_true_it_finds_subdirectories():
		gr.gut.include_subdirectories = true
		gr.gut.add_directory(TEST_BASE_DIR)
		assert_true(gr.gut._test_collector.has_script(TEST_BASE_DIR + 'other_root/three/test_script.gd'))

	func test_when_subdir_false_it_does_not_find_subdirectories():
		gr.gut.include_subdirectories= false
		gr.gut.add_directory(TEST_BASE_DIR)
		assert_false(gr.gut._test_collector.has_script(TEST_BASE_DIR + 'other_root/three/test_script.gd'))
