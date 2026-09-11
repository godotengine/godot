/**************************************************************************/
/*  test_editor_file_system.cpp                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_editor_file_system)

#ifdef TOOLS_ENABLED

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_importer.h"
#include "core/io/resource_uid.h"
#include "editor/file_system/editor_file_system.h"
#include "tests/test_utils.h"

class EditorFileSystemTestAccessor {
public:
	static bool test_for_reimport(EditorFileSystem *p_efs, const String &p_path) {
		return p_efs->_test_for_reimport(p_path);
	}

	static bool can_import_file(EditorFileSystem *p_efs, const String &p_path) {
		return p_efs->_can_import_file(p_path);
	}

	static void add_file(EditorFileSystem *p_efs, const String &p_file, const String &p_import_md5) {
		EditorFileSystemDirectory::FileInfo *fi = memnew(EditorFileSystemDirectory::FileInfo);
		fi->file = p_file;
		fi->import_md5 = p_import_md5;
		fi->import_valid = true;
		p_efs->filesystem->files.push_back(fi);
	}

	static void add_import_extension(EditorFileSystem *p_efs, const String &p_extension) {
		p_efs->import_extensions.insert(p_extension);
	}

	static void clear_singleton() {
		EditorFileSystem::singleton = nullptr;
	}
};

namespace TestEditorFileSystem {

// Minimal importer so that `_test_for_reimport()` can query its version and settings.
class TestReimportImporter : public ResourceImporter {
	GDCLASS(TestReimportImporter, ResourceImporter);

protected:
	static void _bind_methods() {}

public:
	int format_version = 1;
	bool settings_valid = true;

	String get_importer_name() const override { return "test_reimport"; }
	String get_visible_name() const override { return "Test Reimport"; }
	void get_recognized_extensions(List<String> *p_extensions) const override { p_extensions->push_back("rtest"); }
	String get_save_extension() const override { return "res"; }
	String get_resource_type() const override { return "Resource"; }
	int get_format_version() const override { return format_version; }

	void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset = 0) const override {}
	bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override { return true; }

	Error import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) override {
		return OK;
	}

	bool are_import_settings_valid(const String &p_path, const Dictionary &p_meta) const override { return settings_valid; }
};

struct TestResource {
	String res_path = "res://model.rtest";

	// Which files to write
	bool write_source = true; // source file will exist
	bool write_dest = true; // godot/.imported/file will exist

	// Writing of .import file
	bool write_import = true;
	String importer_name = "test_reimport";
	int importer_version = 1;
	bool valid = true;
	bool write_uid = true;
	String uid;
	String source_file; // Empty means "same as res_path".
	bool dest_exists_in_import = true; // Whether the `path=` entry points at the dest we wrote.

	// Writing of .md5 file
	bool write_md5 = true;
	String source_md5_override;
	String dest_md5_override;
	bool write_source_md5 = true;
	bool write_dest_md5 = true;

	String dest_path() const {
		return ResourceFormatImporter::get_singleton()->get_import_base_path(res_path) + ".res";
	}

	String md5_path() const {
		return ResourceFormatImporter::get_singleton()->get_import_base_path(res_path) + ".md5";
	}
};

static void write_file(const String &p_path, const String &p_contents) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	REQUIRE_MESSAGE(f.is_valid(), vformat("Could not open '%s' for writing.", p_path));
	f->store_string(p_contents);
	f->close();
}

static void write(const TestResource &p_resource) {
	if (p_resource.write_source) {
		write_file(p_resource.res_path, "source-contents");
	}

	const String dest = p_resource.dest_path();
	if (p_resource.write_dest) {
		write_file(dest, "dest-contents");
	}

	if (p_resource.write_import) {
		const String source_file = p_resource.source_file.is_empty() ? p_resource.res_path : p_resource.source_file;
		const String referenced_dest = p_resource.dest_exists_in_import ? dest : (p_resource.dest_path() + ".missing");

		String import_text;
		import_text += "[remap]\n\n";
		import_text += "importer=\"" + p_resource.importer_name + "\"\n";
		import_text += "importer_version=" + itos(p_resource.importer_version) + "\n";
		import_text += "type=\"Resource\"\n";
		if (p_resource.write_uid) {
			import_text += "uid=\"" + p_resource.uid + "\"\n";
		}
		import_text += "path=\"" + referenced_dest + "\"\n";
		import_text += p_resource.valid ? "valid=true\n" : "valid=false\n";
		import_text += "\n[deps]\n\n";
		import_text += "source_file=\"" + source_file + "\"\n";
		import_text += "dest_files=[\"" + dest + "\"]\n";
		import_text += "\n[params]\n\n";
		write_file(p_resource.res_path + ".import", import_text);
	}

	if (p_resource.write_md5) {
		String source_md5 = p_resource.source_md5_override;
		if (source_md5.is_empty() && p_resource.write_source) {
			source_md5 = FileAccess::get_md5(p_resource.res_path);
		}

		String dest_md5 = p_resource.dest_md5_override;
		if (dest_md5.is_empty() && p_resource.write_dest) {
			Vector<String> dests;
			dests.push_back(dest);
			dest_md5 = FileAccess::get_multiple_md5(dests);
		}

		String md5_text;
		if (p_resource.write_source_md5) {
			md5_text += "source_md5=\"" + source_md5 + "\"\n";
		}
		if (p_resource.write_dest_md5) {
			md5_text += "dest_md5=\"" + dest_md5 + "\"\n";
		}
		write_file(p_resource.md5_path(), md5_text);
	}
}

static String get_import_md5(const TestResource &p_resource) {
	return FileAccess::get_md5(p_resource.res_path + ".import");
}

class TestHarness {
public:
	EditorFileSystem *efs = nullptr;
	Ref<TestReimportImporter> importer;
	ResourceUID::ID uid = ResourceUID::INVALID_ID;
	String uid_text;

private:
	String old_resource_path;

public:
	explicit TestHarness(const String &p_name) {
		old_resource_path = TestProjectSettingsInternalsAccessor::resource_path();

		const String test_folder = TestUtils::get_temp_path(p_name);
		ProjectSettings *ps = ProjectSettings::get_singleton();

		// Recreate test folder
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (da.is_valid() && da->change_dir(test_folder) == OK) {
			da->erase_contents_recursive();
		}
		da->make_dir_recursive(test_folder);

		TestProjectSettingsInternalsAccessor::resource_path() = test_folder;
		ERR_PRINT_OFF;
		ps->setup(test_folder, String(), false);
		ERR_PRINT_ON;

		// create .imported folder
		da->make_dir_recursive(ps->globalize_path(ps->get_imported_files_path()));

		// We want to test what happens when the imported files are missing
		GLOBAL_DEF("editor/import/reimport_missing_imported_files", true);

		importer.instantiate();
		ResourceFormatImporter::get_singleton()->add_importer(importer);

		uid = ResourceUID::get_singleton()->create_id();
		uid_text = ResourceUID::get_singleton()->id_to_text(uid);

		efs = memnew(EditorFileSystem);
		EditorFileSystemTestAccessor::add_import_extension(efs, ".rtest");
	}

	~TestHarness() {
		if (efs) {
			memdelete(efs);
			EditorFileSystemTestAccessor::clear_singleton();
		}
		if (importer.is_valid()) {
			ResourceFormatImporter::get_singleton()->remove_importer(importer);
			importer.unref();
		}
		if (uid != ResourceUID::INVALID_ID && ResourceUID::get_singleton()->has_id(uid)) {
			ResourceUID::get_singleton()->remove_id(uid);
		}
		TestProjectSettingsInternalsAccessor::resource_path() = old_resource_path;
	}

	// Builds a resource with UID is registered so `uid://` resolution works.
	TestResource populate_uid() {
		TestResource resource;
		resource.uid = uid_text;
		ResourceUID::get_singleton()->add_id(uid, resource.res_path);
		return resource;
	}

	bool test_for_reimport(const String &p_path) {
		return EditorFileSystemTestAccessor::test_for_reimport(efs, p_path);
	}
};

TEST_CASE("[EditorFileSystem] up-to-date resource are not reimported") {
	TestHarness test("efs_up_to_date");
	TestResource resource = test.populate_uid();
	write(resource);
	const String import_md5 = get_import_md5(resource);

	SUBCASE("Everything is up-to-date -> don't reimport") {
		EditorFileSystemTestAccessor::add_file(test.efs, "model.rtest", import_md5);
		CHECK_FALSE(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("EditorFileSystem has not been updated, but on-disk everything is up-to-date -> don't reimport") {
		CHECK_FALSE(test.test_for_reimport(resource.res_path));
	}
}

TEST_CASE("[EditorFileSystem] _test_for_reimport resolves uid:// paths") {
	TestHarness test("efs_uid_paths");
	TestResource resource = test.populate_uid();
	write(resource);
	const String import_md5 = get_import_md5(resource);
	EditorFileSystemTestAccessor::add_file(test.efs, "model.rtest", import_md5);

	SUBCASE("Everything is up-to-date  -> don't reimport") {
		CHECK_FALSE(test.test_for_reimport(test.uid_text));
	}

	SUBCASE("EditorFileSystem has not been updated, but on-disk everything is up-to-date -> don't reimport") {
		CHECK_FALSE(test.test_for_reimport(resource.res_path));
	}
}

TEST_CASE("[EditorFileSystem] _test_for_reimport reimports when .import file is missing or stale") {
	TestHarness test("efs_import_state");

	SUBCASE("missing .import file -> perform reimport") {
		TestResource resource = test.populate_uid();
		resource.write_import = false;
		resource.write_md5 = false;
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("cached MD5 in EFS does not match .import -> perform reimport") {
		TestResource resource = test.populate_uid();
		write(resource);
		EditorFileSystemTestAccessor::add_file(test.efs, "model.rtest", "md5_mismatch_in_efs");
		CHECK(test.test_for_reimport(resource.res_path));
	}
}

TEST_CASE("[EditorFileSystem] _test_for_reimport does not reimport invalid or imports marked as 'keep'") {
	TestHarness fixture("efs_skip_paths");

	SUBCASE("previous import failed (valid=false) -> don't reimport") {
		TestResource resource = fixture.populate_uid();
		resource.valid = false;
		write(resource);
		CHECK_FALSE(fixture.test_for_reimport(resource.res_path));
	}

	SUBCASE("importer set to 'keep' -> don't reimport") {
		TestResource resource = fixture.populate_uid();
		resource.importer_name = "keep";
		write(resource);
		CHECK_FALSE(fixture.test_for_reimport(resource.res_path));
	}

	SUBCASE("importer set to 'skip' -> don't reimport") {
		TestResource resource = fixture.populate_uid();
		resource.importer_name = "skip";
		write(resource);
		CHECK_FALSE(fixture.test_for_reimport(resource.res_path));
	}
}

TEST_CASE("[EditorFileSystem] _test_for_reimport reimports on format and metadata problems") {
	TestHarness test("efs_format_problems");

	SUBCASE("missing uid (old format) -> reimport") {
		TestResource resource = test.populate_uid();
		resource.write_uid = false;
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("destination file missing -> reimport") {
		TestResource resource = test.populate_uid();
		resource.write_dest = false;
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("importer is no longer registered -> reimport") {
		TestResource resource = test.populate_uid();
		resource.importer_name = "nonexistent_importer";
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("importer version mismatch -> reimport") {
		TestResource resource = test.populate_uid();
		resource.importer_version = 1;
		write(resource);
		test.importer->format_version = 2;
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("import settings are invalid -> reimport") {
		TestResource resource = test.populate_uid();
		write(resource);
		test.importer->settings_valid = false;
		CHECK(test.test_for_reimport(resource.res_path));
	}
}

TEST_CASE("[EditorFileSystem] _test_for_reimport reimports on md5 mismatches") {
	TestHarness test("efs_md5_mismatch");

	SUBCASE("missing .md5 file -> reimport") {
		TestResource resource = test.populate_uid();
		resource.write_md5 = false;
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("source file has been moved -> reimport") {
		TestResource resource = test.populate_uid();
		resource.source_file = "res://moved_away.rtest";
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("source md5 is absent -> reimport") {
		TestResource resource = test.populate_uid();
		resource.write_source_md5 = false;
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("source md5 no longer matches -> reimport") {
		TestResource resource = test.populate_uid();
		resource.source_md5_override = "source_md5_mismatch";
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}

	SUBCASE("destination md5 no longer matches -> reimport") {
		TestResource resource = test.populate_uid();
		resource.dest_md5_override = "destination_md5_mismatch";
		write(resource);
		CHECK(test.test_for_reimport(resource.res_path));
	}
}

} // namespace TestEditorFileSystem

#endif // TOOLS_ENABLED
