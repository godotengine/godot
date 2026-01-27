class_name FileAccessTests
extends BaseTest

const FILE_CONTENT = "This is a test for reading / writing to the "

func run_tests():
	print("FileAccess tests starting...")
	__exec_test(test_obb_dir_access)
	__exec_test(test_internal_app_dir_access)
	__exec_test(test_internal_cache_dir_access)
	__exec_test(test_external_app_dir_access)
	__exec_test(test_downloads_dir_access)
	__exec_test(test_documents_dir_access)

func _test_dir_access(dir_path: String, data_file_content: String) -> void:
	print("Testing access to " + dir_path)
	var data_file_path = dir_path.path_join("data.dat")

	var data_file = FileAccess.open(data_file_path, FileAccess.WRITE)
	assert_true(data_file != null)
	assert_true(data_file.store_string(data_file_content))
	data_file.close()

	data_file = FileAccess.open(data_file_path, FileAccess.READ)
	assert_true(data_file != null)
	var file_content = data_file.get_as_text()
	assert_equal(file_content, data_file_content)
	data_file.close()

	var deletion_result = DirAccess.remove_absolute(data_file_path)
	assert_equal(deletion_result, OK)

func test_obb_dir_access() -> void:
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	assert_true(android_runtime != null)

	var app_context = android_runtime.getApplicationContext()
	var obb_dir: String = app_context.getObbDir().getCanonicalPath()
	_test_dir_access(obb_dir, FILE_CONTENT + "obb dir.")

func test_internal_app_dir_access() -> void:
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	assert_true(android_runtime != null)

	var app_context = android_runtime.getApplicationContext()
	var internal_app_dir: String = app_context.getFilesDir().getCanonicalPath()
	_test_dir_access(internal_app_dir, FILE_CONTENT + "internal app dir.")

func test_internal_cache_dir_access() -> void:
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	assert_true(android_runtime != null)

	var app_context = android_runtime.getApplicationContext()
	var internal_cache_dir: String = app_context.getCacheDir().getCanonicalPath()
	_test_dir_access(internal_cache_dir, FILE_CONTENT + "internal cache dir.")

func test_external_app_dir_access() -> void:
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	assert_true(android_runtime != null)

	var app_context = android_runtime.getApplicationContext()
	var external_app_dir: String = app_context.getExternalFilesDir("").getCanonicalPath()
	_test_dir_access(external_app_dir, FILE_CONTENT + "external app dir.")

func test_downloads_dir_access() -> void:
	var EnvironmentClass = JavaClassWrapper.wrap("android.os.Environment")
	var downloads_dir = EnvironmentClass.getExternalStoragePublicDirectory(EnvironmentClass.DIRECTORY_DOWNLOADS).getCanonicalPath()
	_test_dir_access(downloads_dir, FILE_CONTENT + "downloads dir.")

func test_documents_dir_access() -> void:
	var EnvironmentClass = JavaClassWrapper.wrap("android.os.Environment")
	var documents_dir = EnvironmentClass.getExternalStoragePublicDirectory(EnvironmentClass.DIRECTORY_DOCUMENTS).getCanonicalPath()
	_test_dir_access(documents_dir, FILE_CONTENT + "documents dir.")
