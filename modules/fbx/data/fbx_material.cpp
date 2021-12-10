/*************************************************************************/
/*  fbx_material.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "fbx_material.h"

// FIXME: Shouldn't depend on core_bind.h! Use DirAccessRef like the rest of
// the engine instead of core_bind::Directory.
#include "core/core_bind.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"
#include "tools/validation_tools.h"

String FBXMaterial::get_material_name() const {
	return material_name;
}

void FBXMaterial::set_imported_material(FBXDocParser::Material *p_material) {
	material = p_material;
}

void FBXMaterial::add_search_string(String p_filename, String p_current_directory, String search_directory, Vector<String> &texture_search_paths) {
	if (search_directory.is_empty()) {
		texture_search_paths.push_back(p_current_directory.get_base_dir().plus_file(p_filename));
	} else {
		texture_search_paths.push_back(p_current_directory.get_base_dir().plus_file(search_directory + "/" + p_filename));
		texture_search_paths.push_back(p_current_directory.get_base_dir().plus_file("../" + search_directory + "/" + p_filename));
	}
}

String find_file(const String &p_base, const String &p_file_to_find) {
	core_bind::Directory dir;
	dir.open(p_base);

	dir.list_dir_begin();
	String n = dir.get_next();
	while (!n.is_empty()) {
		if (n == "." || n == "..") {
			n = dir.get_next();
			continue;
		}
		if (dir.current_is_dir()) {
			// Don't use `path_to` or the returned path will be wrong.
			const String f = find_file(p_base + "/" + n, p_file_to_find);
			if (!f.is_empty()) {
				return f;
			}
		} else if (n == p_file_to_find) {
			return p_base + "/" + n;
		}
		n = dir.get_next();
	}
	dir.list_dir_end();

	return String();
}

// fbx will not give us good path information and let's not regex them to fix them
// no relative paths are in fbx generally they have a rel field but it's populated incorrectly by the SDK.
String FBXMaterial::find_texture_path_by_filename(const String p_filename, const String p_current_directory) {
	core_bind::Directory dir;
	Vector<String> paths;
	add_search_string(p_filename, p_current_directory, "", paths);
	add_search_string(p_filename, p_current_directory, "texture", paths);
	add_search_string(p_filename, p_current_directory, "textures", paths);
	add_search_string(p_filename, p_current_directory, "Textures", paths);
	add_search_string(p_filename, p_current_directory, "materials", paths);
	add_search_string(p_filename, p_current_directory, "mats", paths);
	add_search_string(p_filename, p_current_directory, "pictures", paths);
	add_search_string(p_filename, p_current_directory, "images", paths);

	for (int i = 0; i < paths.size(); i++) {
		if (dir.file_exists(paths[i])) {
			return paths[i];
		}
	}

	// We were not able to find the texture in the common locations,
	// try to find it into the project globally.
	// The common textures can be stored into one of those folders:
	// res://asset
	// res://texture
	// res://material
	// res://mat
	// res://image
	// res://picture
	//
	// Note the folders can also be called with custom names, like:
	// res://my_assets
	// since the keyword `asset` is into the directory name the textures will be
	// searched there too.

	dir.open("res://");
	dir.list_dir_begin();
	String n = dir.get_next();
	while (!n.is_empty()) {
		if (n == "." || n == "..") {
			n = dir.get_next();
			continue;
		}
		if (dir.current_is_dir()) {
			const String lower_n = n.to_lower();
			if (
					// Don't need to use plural.
					lower_n.find("asset") >= 0 ||
					lower_n.find("texture") >= 0 ||
					lower_n.find("material") >= 0 ||
					lower_n.find("mat") >= 0 ||
					lower_n.find("image") >= 0 ||
					lower_n.find("picture") >= 0) {
				// Don't use `path_to` or the returned path will be wrong.
				const String f = find_file(String("res://") + n, p_filename);
				if (!f.is_empty()) {
					return f;
				}
			}
		}
		n = dir.get_next();
	}
	dir.list_dir_end();

	return "";
}

template <class T>
T extract_from_prop(FBXDocParser::PropertyPtr prop, const T &p_default, const std::string &p_name, const String &p_type) {
	ERR_FAIL_COND_V_MSG(prop == nullptr, p_default, "invalid property passed to extractor");
	const FBXDocParser::TypedProperty<T> *val = dynamic_cast<const FBXDocParser::TypedProperty<T> *>(prop);

	ERR_FAIL_COND_V_MSG(val == nullptr, p_default, "The FBX is corrupted, the property `" + String(p_name.c_str()) + "` is a `" + String(typeid(*prop).name()) + "` but should be a " + p_type);
	// Make sure to not lost any eventual opacity.
	return val->Value();
}

Ref<StandardMaterial3D> FBXMaterial::import_material(ImportState &state) {
	ERR_FAIL_COND_V(material == nullptr, nullptr);

	const String p_fbx_current_directory = state.path;

	Ref<StandardMaterial3D> spatial_material;
	spatial_material.instantiate();

	// read the material file
	// is material two sided
	// read material name
	print_verbose("[material] material name: " + ImportUtils::FBXNodeToName(material->Name()));

	material_name = ImportUtils::FBXNodeToName(material->Name());

	for (const std::pair<std::string, const FBXDocParser::Texture *> iter : material->Textures()) {
		const uint64_t texture_id = iter.second->ID();
		const std::string &fbx_mapping_name = iter.first;
		const FBXDocParser::Texture *fbx_texture_data = iter.second;
		const String absolute_texture_path = iter.second->FileName().c_str();
		const String texture_name = absolute_texture_path.get_file();
		const String file_extension = absolute_texture_path.get_extension().to_upper();

		const String debug_string = "texture id: " + itos(texture_id) + " texture name: " + String(iter.second->Name().c_str()) + " mapping name: " + String(fbx_mapping_name.c_str());
		// remember errors STILL need this string at the end for when you aren't in verbose debug mode :) they need context for when you're not verbose-ing.
		print_verbose(debug_string);

		const String file_extension_uppercase = file_extension.to_upper();

		if (fbx_transparency_flags.count(fbx_mapping_name) > 0) {
			// just enable it later let's make this fine-tuned.
			spatial_material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
		}

		ERR_CONTINUE_MSG(file_extension.is_empty(), "your texture has no file extension so we had to ignore it, let us know if you think this is wrong file an issue on github! " + debug_string);
		ERR_CONTINUE_MSG(fbx_texture_map.count(fbx_mapping_name) <= 0, "This material has a texture with mapping name: " + String(fbx_mapping_name.c_str()) + " which is not yet supported by this importer. Consider opening an issue so we can support it.");
		ERR_CONTINUE_MSG(
				file_extension_uppercase != "PNG" &&
						file_extension_uppercase != "JPEG" &&
						file_extension_uppercase != "JPG" &&
						file_extension_uppercase != "TGA" &&
						file_extension_uppercase != "WEBP" &&
						file_extension_uppercase != "DDS",
				"The FBX file contains a texture with an unrecognized extension: " + file_extension_uppercase);

		print_verbose("Getting FBX mapping mode for " + String(fbx_mapping_name.c_str()));
		// get the texture map type
		const StandardMaterial3D::TextureParam mapping_mode = fbx_texture_map.at(fbx_mapping_name);
		print_verbose("Set FBX mapping mode to " + get_texture_param_name(mapping_mode));

		Ref<Texture> texture;
		print_verbose("texture mapping name: " + texture_name);

		if (state.cached_image_searches.has(texture_name)) {
			texture = state.cached_image_searches[texture_name];
		} else {
			String path = find_texture_path_by_filename(texture_name, p_fbx_current_directory);
			if (!path.is_empty()) {
				Ref<Texture2D> image_texture = ResourceLoader::load(path);

				ERR_CONTINUE(image_texture.is_null());

				texture = image_texture;
				state.cached_image_searches.insert(texture_name, texture);
				print_verbose("Created texture from loaded image file.");

			} else if (fbx_texture_data != nullptr && fbx_texture_data->Media() != nullptr && fbx_texture_data->Media()->IsEmbedded()) {
				// This is an embedded texture. Extract it.
				Ref<Image> image;
				//image.instantiate(); // oooo double instance bug? why make Image::_png_blah call

				const String extension = texture_name.get_extension().to_upper();
				if (extension == "PNG") {
					// The stored file is a PNG.
					image = Image::_png_mem_loader_func(fbx_texture_data->Media()->Content(), fbx_texture_data->Media()->ContentLength());
					ERR_CONTINUE_MSG(image.is_valid() == false, "FBX Embedded PNG image load fail.");

				} else if (
						extension == "JPEG" ||
						extension == "JPG") {
					// The stored file is a JPEG.
					image = Image::_jpg_mem_loader_func(fbx_texture_data->Media()->Content(), fbx_texture_data->Media()->ContentLength());
					ERR_CONTINUE_MSG(image.is_valid() == false, "FBX Embedded JPEG image load fail.");

				} else if (extension == "TGA") {
					// The stored file is a TGA.
					image = Image::_tga_mem_loader_func(fbx_texture_data->Media()->Content(), fbx_texture_data->Media()->ContentLength());
					ERR_CONTINUE_MSG(image.is_valid() == false, "FBX Embedded TGA image load fail.");

				} else if (extension == "WEBP") {
					// The stored file is a WEBP.
					image = Image::_webp_mem_loader_func(fbx_texture_data->Media()->Content(), fbx_texture_data->Media()->ContentLength());
					ERR_CONTINUE_MSG(image.is_valid() == false, "FBX Embedded WEBP image load fail.");

					// } else if (extension == "DDS") {
					// 	// In this moment is not possible to extract a DDS from a buffer, TODO consider add it to godot. See `textureloader_dds.cpp::load().
					// 	// The stored file is a DDS.
				} else {
					ERR_CONTINUE_MSG(true, "The embedded image with extension: " + extension + " is not yet supported. Open an issue please.");
				}

				Ref<ImageTexture> image_texture;
				image_texture.instantiate();
				image_texture->create_from_image(image);

				texture = image_texture;

				// TODO: this is potentially making something with the same name have a match incorrectly USE FBX ID as Hash. #fuck it later.
				state.cached_image_searches[texture_name] = texture;
				print_verbose("Created texture from embedded image.");
			} else {
				ERR_CONTINUE_MSG(true, "The FBX texture, with name: `" + texture_name + "`, is not found into the project nor is stored as embedded file. Make sure to insert the texture as embedded file or into the project, then reimport.");
			}
		}

		spatial_material->set_texture(mapping_mode, texture);
	}

	if (spatial_material.is_valid()) {
		spatial_material->set_name(material_name);
	}

	/// ALL below is related to properties
	for (FBXDocParser::LazyPropertyMap::value_type iter : material->GetLazyProperties()) {
		const std::string name = iter.first;

		if (name.empty()) {
			continue;
		}

		PropertyDesc desc = PROPERTY_DESC_NOT_FOUND;
		if (fbx_properties_desc.count(name) > 0) {
			desc = fbx_properties_desc.at(name);
		}

		// check if we can ignore this it will be done at the next phase
		if (desc == PROPERTY_DESC_NOT_FOUND || desc == PROPERTY_DESC_IGNORE) {
			// count the texture mapping references. Skip this one if it's found and we can't look up a property value.
			if (fbx_texture_map.count(name) > 0) {
				continue; // safe to ignore it's a texture mapping.
			}
		}

		if (desc == PROPERTY_DESC_IGNORE) {
			//WARN_PRINT("[Ignored] The FBX material parameter: `" + String(name.c_str()) + "` is ignored.");
			continue;
		} else {
			print_verbose("FBX Material parameter: " + String(name.c_str()));

			// Check for Diffuse material system / lambert materials / legacy basically
			if (name == "Diffuse" && !warning_non_pbr_material) {
				ValidationTracker::get_singleton()->add_validation_error(state.path, "Invalid material settings change to Ai Standard Surface shader, mat name: " + material_name.c_escape());
				warning_non_pbr_material = true;
			}
		}

		// DISABLE when adding support for all weird and wonderful material formats
		if (desc == PROPERTY_DESC_NOT_FOUND) {
			continue;
		}

		ERR_CONTINUE_MSG(desc == PROPERTY_DESC_NOT_FOUND, "The FBX material parameter: `" + String(name.c_str()) + "` was not recognized. Please open an issue so we can add the support to it.");

		const FBXDocParser::PropertyTable *tbl = material;
		FBXDocParser::PropertyPtr prop = tbl->Get(name);

		ERR_CONTINUE_MSG(prop == nullptr, "This file may be corrupted because is not possible to extract the material parameter: " + String(name.c_str()));

		if (spatial_material.is_null()) {
			// Done here so if no data no material is created.
			spatial_material.instantiate();
		}

		const FBXDocParser::TypedProperty<real_t> *real_value = dynamic_cast<const FBXDocParser::TypedProperty<real_t> *>(prop);
		const FBXDocParser::TypedProperty<Vector3> *vector_value = dynamic_cast<const FBXDocParser::TypedProperty<Vector3> *>(prop);

		if (!real_value && !vector_value) {
			//WARN_PRINT("unsupported datatype in property: " + String(name.c_str()));
			continue;
		}

		if (vector_value && !real_value) {
			if (vector_value->Value() == Vector3(0, 0, 0) && !real_value) {
				continue;
			}
		}

		switch (desc) {
			case PROPERTY_DESC_ALBEDO_COLOR: {
				if (vector_value) {
					const Vector3 &color = vector_value->Value();
					// Make sure to not lost any eventual opacity.
					if (color != Vector3(0, 0, 0)) {
						Color c = Color();
						c[0] = color[0];
						c[1] = color[1];
						c[2] = color[2];
						spatial_material->set_albedo(c);
					}

				} else if (real_value) {
					print_error("albedo is unsupported format?");
				}
			} break;
			case PROPERTY_DESC_TRANSPARENT: {
				if (real_value) {
					const real_t opacity = real_value->Value();
					if (opacity < (1.0 - CMP_EPSILON)) {
						Color c = spatial_material->get_albedo();
						c.a = opacity;
						spatial_material->set_albedo(c);

						spatial_material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
						spatial_material->set_depth_draw_mode(BaseMaterial3D::DEPTH_DRAW_OPAQUE_ONLY);
					}
				} else if (vector_value) {
					print_error("unsupported transparent desc type vector!");
				}
			} break;
			case PROPERTY_DESC_SPECULAR: {
				if (real_value) {
					print_verbose("specular real value: " + rtos(real_value->Value()));
					spatial_material->set_specular(MIN(1.0, real_value->Value()));
				}

				if (vector_value) {
					print_error("unsupported specular vector value: " + vector_value->Value());
				}
			} break;

			case PROPERTY_DESC_SPECULAR_COLOR: {
				if (vector_value) {
					print_error("unsupported specular color: " + vector_value->Value());
				}
			} break;
			case PROPERTY_DESC_SHINYNESS: {
				if (real_value) {
					print_error("unsupported shinyness:" + rtos(real_value->Value()));
				}
			} break;
			case PROPERTY_DESC_METALLIC: {
				if (real_value) {
					print_verbose("metallic real value: " + rtos(real_value->Value()));
					spatial_material->set_metallic(MIN(1.0f, real_value->Value()));
				} else {
					print_error("unsupported value type for metallic");
				}
			} break;
			case PROPERTY_DESC_ROUGHNESS: {
				if (real_value) {
					print_verbose("roughness real value: " + rtos(real_value->Value()));
					spatial_material->set_roughness(MIN(1.0f, real_value->Value()));
				} else {
					print_error("unsupported value type for roughness");
				}
			} break;
			case PROPERTY_DESC_COAT: {
				if (real_value) {
					print_verbose("clearcoat real value: " + rtos(real_value->Value()));
					spatial_material->set_clearcoat(MIN(1.0f, real_value->Value()));
				} else {
					print_error("unsupported value type for clearcoat");
				}
			} break;
			case PROPERTY_DESC_COAT_ROUGHNESS: {
				// meaning is that approx equal to zero is disabled not actually zero. ;)
				if (real_value && Math::is_zero_approx(real_value->Value())) {
					print_verbose("clearcoat real value: " + rtos(real_value->Value()));
					spatial_material->set_clearcoat_gloss(1.0 - real_value->Value());
				} else {
					print_error("unsupported value type for clearcoat gloss");
				}
			} break;
			case PROPERTY_DESC_EMISSIVE: {
				if (real_value && Math::is_zero_approx(real_value->Value())) {
					print_verbose("Emissive real value: " + rtos(real_value->Value()));
					spatial_material->set_emission_energy(real_value->Value());
				} else if (vector_value && !vector_value->Value().is_equal_approx(Vector3(0, 0, 0))) {
					const Vector3 &color = vector_value->Value();
					Color c;
					c[0] = color[0];
					c[1] = color[1];
					c[2] = color[2];
					spatial_material->set_emission(c);
				}
			} break;
			case PROPERTY_DESC_EMISSIVE_COLOR: {
				if (vector_value && !vector_value->Value().is_equal_approx(Vector3(0, 0, 0))) {
					const Vector3 &color = vector_value->Value();
					Color c;
					c[0] = color[0];
					c[1] = color[1];
					c[2] = color[2];
					spatial_material->set_emission(c);
				} else {
					print_error("unsupported value type for emissive color");
				}
			} break;
			case PROPERTY_DESC_NOT_FOUND:
			case PROPERTY_DESC_IGNORE:
				break;
			default:
				break;
		}
	}

	return spatial_material;
}
