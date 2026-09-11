/**************************************************************************/
/*  test_material.cpp                                                     */
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

TEST_FORCE_LINK(test_material)

#ifndef _3D_DISABLED

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "tests/test_utils.h"

namespace TestMaterial {

// Properties that copy_from() is not expected to transfer.
static bool _is_ignored_property(const String &p_name) {
	return p_name == "resource_local_to_scene" ||
			p_name == "resource_name" ||
			p_name == "resource_path" ||
			p_name == "resource_scene_unique_id" ||
			p_name == "script" ||
			p_name.begins_with("metadata/");
}

// Produce a value different from the property's default
static bool _perturb_value(const PropertyInfo &p_prop, const Variant &p_default, const Ref<Texture2D> &p_texture, Variant &r_out) {
	switch (p_prop.type) {
		case Variant::BOOL:
			r_out = !bool(p_default);
			return true;
		case Variant::INT:
			r_out = int64_t(p_default) + 1;
			return true;
		case Variant::FLOAT:
			r_out = double(p_default) + 1.0;
			return true;
		case Variant::COLOR:
			r_out = Color(0.12, 0.34, 0.56, 0.78);
			return true;
		case Variant::VECTOR3:
			r_out = Vector3(p_default) + Vector3(1.5, 2.5, 3.5);
			return true;
		case Variant::OBJECT:
			// Only texture slots can take our dummy texture; other object
			// properties (e.g. next_pass) expect different types.
			if (p_prop.hint_string.containsn("Texture")) {
				r_out = p_texture;
				return true;
			}
			return false;
		default:
			return false;
	}
}

// The resource loaders reload a cached CACHE_MODE_REPLACE resource by loading a
// fresh instance and transplanting it with Resource::copy_from(). This checks the
// mechanism directly: a heavily-modified material must end up matching a
// freshly-constructed one after copy_from(), so no stale value survives a reload.
TEST_CASE("[SceneTree][Material] copy_from restores all storage properties to defaults") {
	Ref<StandardMaterial3D> fresh;
	fresh.instantiate();

	Ref<StandardMaterial3D> mat;
	mat.instantiate();

	// Texture to assign to every texture slot.
	Ref<Image> image = Image::create_empty(2, 2, false, Image::FORMAT_RGBA8);
	image->fill(Color(1, 1, 1, 1));
	Ref<ImageTexture> texture = ImageTexture::create_from_image(image);

	List<PropertyInfo> props;
	mat->get_property_list(&props);

	for (const PropertyInfo &E : props) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE) || _is_ignored_property(E.name)) {
			continue;
		}
		Variant perturbed;
		if (!_perturb_value(E, mat->get(E.name), texture, perturbed)) {
			continue;
		}
		mat->set(E.name, perturbed);
	}

	// At least confirm the perturbation took effect for a representative property.
	CHECK(mat->get_texture(BaseMaterial3D::TEXTURE_ALBEDO).is_valid());
	CHECK(mat->get_metallic() != fresh->get_metallic());

	// This is what the loaders do to a cached resource on reload.
	mat->copy_from(fresh);

	// Every storage property must now match a freshly-constructed material.
	for (const PropertyInfo &E : props) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE) || _is_ignored_property(E.name)) {
			continue;
		}
		const Variant expected = fresh->get(E.name);
		const Variant actual = mat->get(E.name);
		CHECK_MESSAGE(actual == expected,
				vformat("Property '%s' was not reset (expected %s, got %s).", E.name, expected, actual).get_data());
	}

	CHECK(mat->get_texture(BaseMaterial3D::TEXTURE_ALBEDO).is_null());
}

static void _test_replace_resets_stale_subresource(const String &p_extension) {
	const String path = TestUtils::get_temp_path("material_reimport." + p_extension);

	// "Imported" file: a parent material whose next_pass has a non-default culling mode.
	{
		Ref<StandardMaterial3D> child;
		child.instantiate();
		child->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
		Ref<StandardMaterial3D> parent;
		parent.instantiate();
		parent->set_next_pass(child);
		REQUIRE(ResourceSaver::save(parent, path) == OK);
	}

	// First load caches the parent and its sub-resource.
	Ref<StandardMaterial3D> first = ResourceLoader::load(path, "", ResourceFormatLoader::CACHE_MODE_REUSE);
	REQUIRE(first.is_valid());
	Ref<StandardMaterial3D> first_child = first->get_next_pass();
	REQUIRE(first_child.is_valid());
	CHECK(first_child->get_cull_mode() == BaseMaterial3D::CULL_DISABLED);

	Ref<StandardMaterial3D> child;
	child.instantiate();
	REQUIRE(child->get_cull_mode() == BaseMaterial3D::CULL_BACK);
	Ref<StandardMaterial3D> parent;
	parent.instantiate();
	parent->set_next_pass(child);
	REQUIRE(ResourceSaver::save(parent, path) == OK);

	Ref<StandardMaterial3D> second = ResourceLoader::load(path, "", ResourceFormatLoader::CACHE_MODE_REPLACE);
	REQUIRE(second.is_valid());
	Ref<StandardMaterial3D> second_child = second->get_next_pass();
	REQUIRE(second_child.is_valid());

	CHECK_MESSAGE(second_child == first_child,
			"CACHE_MODE_REPLACE must reuse the cached sub-resource in place so existing references see the update.");
	CHECK_MESSAGE(second_child->get_cull_mode() == BaseMaterial3D::CULL_BACK,
			"A sub-resource property that reverted to its default on reimport must be reset, not kept stale.");
}

TEST_CASE("[SceneTree][Material] CACHE_MODE_REPLACE resets stale sub-resource properties (text)") {
	_test_replace_resets_stale_subresource("tres");
}

TEST_CASE("[SceneTree][Material] CACHE_MODE_REPLACE resets stale sub-resource properties (binary)") {
	_test_replace_resets_stale_subresource("res");
}

} // namespace TestMaterial

#endif // _3D_DISABLED
