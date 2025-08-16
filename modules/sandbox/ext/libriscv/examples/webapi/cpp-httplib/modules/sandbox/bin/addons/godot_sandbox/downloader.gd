@tool
extends Window

var http_cmake: HTTPRequest
var http_zig: HTTPRequest
var http_ninja: HTTPRequest

@export var cmake_status: RichTextLabel
@export var make_status: RichTextLabel
@export var git_status: RichTextLabel
@export var zig_status: RichTextLabel
@export var os_tab_container: TabContainer

var status = {
	"not_installed": "[color=red]Not installed[/color]",
	"installed": "[color=green]Installed[/color]",
	"installing": "[color=yellow]Installing[/color]"
}

func _ready():
	match OS.get_name():
		"Windows":
			os_tab_container.current_tab = 0
		"Linux":
			os_tab_container.current_tab = 2
		"macOS":
			os_tab_container.current_tab = 1
			
	title = "Godot Sandbox Dependencies" + " ( " + OS.get_name() + " " + Engine.get_architecture_name() + " )"
	http_cmake = HTTPRequest.new()
	http_zig = HTTPRequest.new()
	http_ninja = HTTPRequest.new()
	add_child(http_cmake)
	add_child(http_zig)
	add_child(http_ninja)

	http_cmake.request_completed.connect(_on_download_complete.bind("cmake"))
	http_zig.request_completed.connect(_on_download_complete.bind("zig"))
	http_ninja.request_completed.connect(_on_download_complete.bind("ninja"))

	_update_status()

func _update_status():
	# CMake
	var output := []
	var exit_code := OS.execute(ProjectSettings.get_setting("editor/script/cmake", "cmake"), ["--version"], output, true)
	
	if exit_code == 0:
		cmake_status.text = "CMake: " + status["installed"]
	else:
		print("CMake not installed: ", output)
		cmake_status.text = "CMake: " + status["not_installed"]
		show()
	
	# Make (Linux/macOS) or Ninja (Windows)
	if OS.get_name() == "Windows":
		exit_code = OS.execute(ProjectSettings.get_setting("editor/script/make", "ninja"), ["--version"], output, true)
		if exit_code == 0:
			make_status.text = "Ninja: " + status["installed"]
		else:
			print("Ninja not installed: ", output)
			make_status.text = "Ninja: " + status["not_installed"]
			show()
	else:
		exit_code = OS.execute(ProjectSettings.get_setting("editor/script/make", "make"), ["--version"], output, true)
		if exit_code == 0:
			make_status.text = "Make: " + status["installed"]
		else:
			print("Make not installed: ", output)
			make_status.text = "Make: " + status["not_installed"]
			show()

	
	# Git
	exit_code = OS.execute(ProjectSettings.get_setting("editor/script/git", "git"), ["--version"], output, true)
	
	if exit_code == 0:
		git_status.text = "Git: " + status["installed"]
	else:
		print("Git not installed: ", output)
		git_status.text = "Git: " + status["not_installed"]
		show()
	
	# Zig
	exit_code = OS.execute(ProjectSettings.get_setting("editor/script/zig", "zig"), ["--help"], output, true)
	
	if exit_code == 0:
		zig_status.text = "Zig: " + status["installed"]
	else:
		print("Zig not installed: ", output)
		zig_status.text = "Zig: " + status["not_installed"]
		show()

func _on_label_meta_clicked(meta: Variant) -> void:
	OS.shell_open(str(meta))

func _on_cmake_button_pressed() -> void:
	cmake_status.text = status["installing"]
	var download_url = "None"
	match OS.get_name():
		"Windows":
			if OS.has_feature("x86_64"):
				download_url = "https://github.com/Kitware/CMake/releases/download/v4.1.0-rc3/cmake-4.1.0-rc3-windows-x86_64.zip"
			elif OS.has_feature("x86_32"):
				download_url = "https://github.com/Kitware/CMake/releases/download/v4.1.0-rc3/cmake-4.1.0-rc3-windows-i386.zip"
			elif OS.has_feature("arm64"):
				download_url = "https://github.com/Kitware/CMake/releases/download/v4.1.0-rc3/cmake-4.1.0-rc3-windows-arm64.zip"
		"Linux":
			if OS.has_feature("x86_64"):
				download_url = "https://github.com/Kitware/CMake/releases/download/v4.1.0-rc3/cmake-4.1.0-rc3-linux-x86_64.tar.gz"
			elif OS.has_feature("arm64"):
				download_url = "https://github.com/Kitware/CMake/releases/download/v4.1.0-rc3/cmake-4.1.0-rc3-linux-aarch64.tar.gz"
		"macOS":
			download_url = "https://github.com/Kitware/CMake/releases/download/v4.1.0-rc3/cmake-4.1.0-rc3-macos-universal.tar.gz"
		_:
			print("Cannot find download URL")
			_update_status()
			return
	var err = http_cmake.request(download_url)
	if err != OK:
		print("Failed to start HTTP request: %s" % err)

func extract_all_from_zip(path: String, dest: String):
	DirAccess.make_dir_recursive_absolute(dest)
	# On Linux and Mac, use tar
	var os_name = OS.get_name()
	if os_name == "Linux" or os_name == "macOS":
		var args = ["-xzf", ProjectSettings.globalize_path(path), "-C", ProjectSettings.globalize_path(dest)]
		var output := []
		var exit_code := OS.execute("tar", args, output)
		
		if exit_code == 0:
			print("✅ Extraction succeeded:\n", output)
		else:
			print("❌ Extraction failed (code %d):\n%s" % [exit_code, output])
		return
	# On windows use zipreader
	var reader = ZIPReader.new()
	var err = reader.open(path)
	if err != OK:
		print("Error opening zip: ", path, " ", err)
		return
	DirAccess.make_dir_recursive_absolute(dest)
	var root_dir = DirAccess.open(dest)

	var files = reader.get_files()
	for file_path in files:
		if file_path.ends_with("/"):
			root_dir.make_dir_recursive(file_path)
			continue

		root_dir.make_dir_recursive(root_dir.get_current_dir().path_join(file_path).get_base_dir())
		var file = FileAccess.open(root_dir.get_current_dir().path_join(file_path), FileAccess.WRITE)
		var buffer = reader.read_file(file_path)
		file.store_buffer(buffer)

func _on_download_complete(result, response_code, headers, body, downloaded_name):
	if result != HTTPRequest.RESULT_SUCCESS:
		print("Download failed with code: %d" % response_code)
		_update_status()
		return

	DirAccess.make_dir_recursive_absolute("user://godot-sandbox/tmp")
	var path = "user://godot-sandbox/tmp/%s.zip" % downloaded_name
	var file = FileAccess.open(path, FileAccess.WRITE)
	if not file:
		print("Failed to save downloaded file")
		_update_status()
		return

	file.store_buffer(body)
	file.close()

	print("CMake downloaded to: %s" % path)
	extract_all_from_zip(path, "user://godot-sandbox/tmp/%s" % downloaded_name)
	DirAccess.make_dir_recursive_absolute("user://godot-sandbox/%s" % downloaded_name)
	var first_subfolder = ("user://godot-sandbox/tmp/%s" % downloaded_name) + "/" + DirAccess.open("user://godot-sandbox/tmp/%s" % downloaded_name).get_directories()[0]
	print("Renaming ", first_subfolder, " to ", "user://godot-sandbox/%s" % downloaded_name)
	DirAccess.rename_absolute(first_subfolder, "user://godot-sandbox/%s" % downloaded_name)
	var binary_name = downloaded_name
	match OS.get_name():
		"Windows":
			if downloaded_name == "cmake":
				binary_name = "bin/cmake.exe"
		"Linux":
			if downloaded_name == "cmake":
				binary_name = "bin/cmake"
		"macOS":
			if downloaded_name == "cmake":
				binary_name = "CMake.app/Contents/bin/cmake"
	ProjectSettings.set("editor/script/%s" % downloaded_name, ProjectSettings.globalize_path("user://godot-sandbox/%s" % downloaded_name).path_join(binary_name))
	ProjectSettings.save()
	_update_status()


func _on_about_to_popup() -> void:
	_update_status()


func _on_zig_button_pressed() -> void:
	zig_status.text = status["installing"]
	var download_url = "None"
	match OS.get_name():
		"Windows":
			if OS.has_feature("x86_64"):
				download_url = "https://ziglang.org/download/0.14.1/zig-x86_64-windows-0.14.1.zip"
			elif OS.has_feature("x86_32"):
				download_url = "https://ziglang.org/download/0.14.1/zig-x86-windows-0.14.1.zip"
			elif OS.has_feature("arm64"):
				download_url = "https://ziglang.org/download/0.14.1/zig-aarch64-windows-0.14.1.zip"
		"Linux":
			if OS.has_feature("x86_64"):
				download_url = "https://ziglang.org/download/0.14.1/zig-x86_64-linux-0.14.1.tar.xz"
			elif OS.has_feature("arm64"):
				download_url = "https://ziglang.org/download/0.14.1/zig-aarch64-linux-0.14.1.tar.xz"
		"macOS":
			if OS.has_feature("x86_64"):
				download_url = "https://ziglang.org/download/0.14.1/zig-x86_64-macos-0.14.1.tar.xz"
			elif OS.has_feature("arm64"):
				download_url = "https://ziglang.org/download/0.14.1/zig-aarch64-macos-0.14.1.tar.xz"
		_:
			print("Cannot find download URL")
			_update_status()
			return
	var err = http_zig.request(download_url)
	if err != OK:
		print("Failed to start HTTP request: %s" % err)


func _on_button_pressed() -> void:
	hide()
