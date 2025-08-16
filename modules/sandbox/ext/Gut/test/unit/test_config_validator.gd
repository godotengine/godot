extends GutTest

var VersionConversion = load("res://addons/gut/version_conversion.gd")
var from_file = 'user://test_version_conversion_move_this.cfg'
var to_file = 'user://test_version_conversion_to_here.cfg'


func before_all():
	gut.file_delete(from_file)
	gut.file_delete(to_file)
	register_inner_classes(VersionConversion)

func after_each():
	gut.file_delete(from_file)
	gut.file_delete(to_file)

func test_moved_file_moves_file_to_new_location():
	var updater = VersionConversion.ConfigurationUpdater.new()
	gut.file_touch(from_file)
	updater.moved_file(from_file, to_file)
	assert_file_exists(to_file)

func test_moved_file_does_not_remove_original():
	var updater = VersionConversion.ConfigurationUpdater.new()
	gut.file_touch(from_file)
	updater.moved_file(from_file, to_file)
	assert_file_exists(from_file)

func test_moved_file_issues_warning_when_both_files_exist():
	var updater = partial_double(VersionConversion.ConfigurationUpdater).new()
	gut.file_touch(from_file)
	gut.file_touch(to_file)
	updater.moved_file(from_file, to_file)
	var params = get_call_parameters(updater, 'warn')
	assert_string_contains(params[0], 'You can delete')

func test_move_user_file_moves_the_file():
	var updater = VersionConversion.ConfigurationUpdater.new()
	gut.file_touch(from_file)
	updater.move_user_file(from_file, to_file)
	assert_file_exists(to_file)

func test_move_user_file_deletes_from_file():
	var updater = VersionConversion.ConfigurationUpdater.new()
	gut.file_touch(from_file)
	updater.move_user_file(from_file, to_file)
	assert_file_does_not_exist(from_file)

func test_remove_user_file_deletes_the_file():
	var updater = VersionConversion.ConfigurationUpdater.new()
	gut.file_touch(from_file)
	updater.remove_user_file(from_file)
	assert_file_does_not_exist(from_file)
