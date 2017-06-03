/*************************************************************************/
/*  collada.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef TOOLS_ENABLED

#include "collada.h"

#include "stdio.h"

//#define DEBUG_DEFAULT_ANIMATION
//#define DEBUG_COLLADA
#ifdef DEBUG_COLLADA
#define COLLADA_PRINT(m_what) print_line(m_what)
#else
#define COLLADA_PRINT(m_what)
#endif

#define COLLADA_IMPORT_SCALE_SCENE

/* HELPERS */

String Collada::Effect::get_texture_path(const String &p_source, Collada &state) const {

	String image = p_source;
	ERR_FAIL_COND_V(!state.state.image_map.has(image), "");
	return state.state.image_map[image].path;
}

Transform Collada::get_root_transform() const {

	Transform unit_scale_transform;
#ifndef COLLADA_IMPORT_SCALE_SCENE
	unit_scale_transform.scale(Vector3(state.unit_scale, state.unit_scale, state.unit_scale));
#endif
	return unit_scale_transform;
}

void Collada::Vertex::fix_unit_scale(Collada &state) {
#ifdef COLLADA_IMPORT_SCALE_SCENE
	vertex *= state.state.unit_scale;
#endif
}

static String _uri_to_id(const String &p_uri) {

	if (p_uri.begins_with("#"))
		return p_uri.substr(1, p_uri.size() - 1);
	else
		return p_uri;
}

/** HELPER FUNCTIONS **/

Transform Collada::fix_transform(const Transform &p_transform) {

	Transform tr = p_transform;

#ifndef NO_UP_AXIS_SWAP

	if (state.up_axis != Vector3::AXIS_Y) {

		for (int i = 0; i < 3; i++)
			SWAP(tr.basis[1][i], tr.basis[state.up_axis][i]);
		for (int i = 0; i < 3; i++)
			SWAP(tr.basis[i][1], tr.basis[i][state.up_axis]);

		SWAP(tr.origin[1], tr.origin[state.up_axis]);

		tr.basis[state.up_axis][0] = -tr.basis[state.up_axis][0];
		tr.basis[state.up_axis][1] = -tr.basis[state.up_axis][1];
		tr.basis[0][state.up_axis] = -tr.basis[0][state.up_axis];
		tr.basis[1][state.up_axis] = -tr.basis[1][state.up_axis];
		tr.origin[state.up_axis] = -tr.origin[state.up_axis];
	}
#endif

	//tr.scale(Vector3(state.unit_scale.unit_scale.unit_scale));
	return tr;
	//return state.matrix_fix * p_transform;
}

static Transform _read_transform_from_array(const Vector<float> &array, int ofs = 0) {

	Transform tr;
	// i wonder why collada matrices are transposed, given that's opposed to opengl..
	tr.basis.elements[0][0] = array[0 + ofs];
	tr.basis.elements[0][1] = array[1 + ofs];
	tr.basis.elements[0][2] = array[2 + ofs];
	tr.basis.elements[1][0] = array[4 + ofs];
	tr.basis.elements[1][1] = array[5 + ofs];
	tr.basis.elements[1][2] = array[6 + ofs];
	tr.basis.elements[2][0] = array[8 + ofs];
	tr.basis.elements[2][1] = array[9 + ofs];
	tr.basis.elements[2][2] = array[10 + ofs];
	tr.origin.x = array[3 + ofs];
	tr.origin.y = array[7 + ofs];
	tr.origin.z = array[11 + ofs];
	return tr;
}

/* STRUCTURES */

Transform Collada::Node::compute_transform(Collada &state) const {

	Transform xform;

	for (int i = 0; i < xform_list.size(); i++) {

		Transform xform_step;
		const XForm &xf = xform_list[i];
		switch (xf.op) {

			case XForm::OP_ROTATE: {
				if (xf.data.size() >= 4) {

					xform_step.rotate(Vector3(xf.data[0], xf.data[1], xf.data[2]), Math::deg2rad(xf.data[3]));
				}
			} break;
			case XForm::OP_SCALE: {

				if (xf.data.size() >= 3) {

					xform_step.scale(Vector3(xf.data[0], xf.data[1], xf.data[2]));
				}

			} break;
			case XForm::OP_TRANSLATE: {

				if (xf.data.size() >= 3) {

					xform_step.origin = Vector3(xf.data[0], xf.data[1], xf.data[2]);
				}

			} break;
			case XForm::OP_MATRIX: {

				if (xf.data.size() >= 16) {
					xform_step = _read_transform_from_array(xf.data, 0);
				}

			} break;
			default: {}
		}

		xform = xform * xform_step;
	}

#ifdef COLLADA_IMPORT_SCALE_SCENE
	xform.origin *= state.state.unit_scale;
#endif
	return xform;
}

Transform Collada::Node::get_transform() const {

	return default_transform;
}

Transform Collada::Node::get_global_transform() const {

	if (parent)
		return parent->get_global_transform() * default_transform;
	else
		return default_transform;
}

Vector<float> Collada::AnimationTrack::get_value_at_time(float p_time) {

	ERR_FAIL_COND_V(keys.size() == 0, Vector<float>());
	int i = 0;

	for (i = 0; i < keys.size(); i++) {

		if (keys[i].time > p_time)
			break;
	}

	if (i == 0)
		return keys[0].data;
	if (i == keys.size())
		return keys[keys.size() - 1].data;

	switch (keys[i].interp_type) {

		case INTERP_BEZIER: //wait for bezier
		case INTERP_LINEAR: {

			float c = (p_time - keys[i - 1].time) / (keys[i].time - keys[i - 1].time);

			if (keys[i].data.size() == 16) {
				//interpolate a matrix
				Transform src = _read_transform_from_array(keys[i - 1].data);
				Transform dst = _read_transform_from_array(keys[i].data);

				Transform interp = c < 0.001 ? src : src.interpolate_with(dst, c);

				Vector<float> ret;
				ret.resize(16);
				Transform tr;
				// i wonder why collada matrices are transposed, given that's opposed to opengl..
				ret[0] = interp.basis.elements[0][0];
				ret[1] = interp.basis.elements[0][1];
				ret[2] = interp.basis.elements[0][2];
				ret[4] = interp.basis.elements[1][0];
				ret[5] = interp.basis.elements[1][1];
				ret[6] = interp.basis.elements[1][2];
				ret[8] = interp.basis.elements[2][0];
				ret[9] = interp.basis.elements[2][1];
				ret[10] = interp.basis.elements[2][2];
				ret[3] = interp.origin.x;
				ret[7] = interp.origin.y;
				ret[11] = interp.origin.z;
				ret[12] = 0;
				ret[13] = 0;
				ret[14] = 0;
				ret[15] = 1;

				return ret;
			} else {

				Vector<float> dest;
				dest.resize(keys[i].data.size());
				for (int j = 0; j < dest.size(); j++) {

					dest[j] = keys[i].data[j] * c + keys[i - 1].data[j] * (1.0 - c);
				}
				return dest;
				//interpolate one by one
			}
		} break;
	}

	ERR_FAIL_V(Vector<float>());
}

void Collada::_parse_asset(XMLParser &parser) {

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();

			if (name == "up_axis") {

				parser.read();
				if (parser.get_node_data() == "X_UP")
					state.up_axis = Vector3::AXIS_X;
				if (parser.get_node_data() == "Y_UP")
					state.up_axis = Vector3::AXIS_Y;
				if (parser.get_node_data() == "Z_UP")
					state.up_axis = Vector3::AXIS_Z;

				COLLADA_PRINT("up axis: " + parser.get_node_data());
			} else if (name == "unit") {

				state.unit_scale = parser.get_attribute_value("meter").to_double();
				COLLADA_PRINT("unit scale: " + rtos(state.unit_scale));
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "asset")
			break; //end of <asset>
	}
}

void Collada::_parse_image(XMLParser &parser) {

	String id = parser.get_attribute_value("id");

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	Image image;

	if (state.version < State::Version(1, 4, 0)) {
		/* <1.4 */
		String path = parser.get_attribute_value("source").strip_edges();
		if (path.find("://") == -1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			image.path = GlobalConfig::get_singleton()->localize_path(state.local_path.get_base_dir() + "/" + path.percent_decode());
		}
	} else {

		while (parser.read() == OK) {

			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

				String name = parser.get_node_name();

				if (name == "init_from") {

					parser.read();
					String path = parser.get_node_data().strip_edges().percent_decode();

					if (path.find("://") == -1 && path.is_rel_path()) {
						// path is relative to file being loaded, so convert to a resource path
						path = GlobalConfig::get_singleton()->localize_path(state.local_path.get_base_dir() + "/" + path);

					} else if (path.find("file:///") == 0) {
						path = path.replace_first("file:///", "");
						path = GlobalConfig::get_singleton()->localize_path(path);
					}

					image.path = path;

				} else if (name == "data") {

					ERR_PRINT("COLLADA Embedded image data not supported!");

				} else if (name == "extra" && !parser.is_empty())
					parser.skip_section();

			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "image")
				break; //end of <asset>
		}
	}

	state.image_map[id] = image;
}

void Collada::_parse_material(XMLParser &parser) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	Material material;

	String id = parser.get_attribute_value("id");
	if (parser.has_attribute("name"))
		material.name = parser.get_attribute_value("name");

	if (state.version < State::Version(1, 4, 0)) {
		/* <1.4 */
		ERR_PRINT("Collada Materials < 1.4 are not supported (yet)");
	} else {

		while (parser.read() == OK) {

			if (parser.get_node_type() == XMLParser::NODE_ELEMENT && parser.get_node_name() == "instance_effect") {

				material.instance_effect = _uri_to_id(parser.get_attribute_value("url"));
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "material")
				break; //end of <asset>
		}
	}

	state.material_map[id] = material;
}

//! reads floats from inside of xml element until end of xml element
Vector<float> Collada::_read_float_array(XMLParser &parser) {

	if (parser.is_empty())
		return Vector<float>();

	Vector<String> splitters;
	splitters.push_back(" ");
	splitters.push_back("\n");
	splitters.push_back("\r");
	splitters.push_back("\t");

	Vector<float> array;
	while (parser.read() == OK) {
		// TODO: check for comments inside the element
		// and ignore them.

		if (parser.get_node_type() == XMLParser::NODE_TEXT) {
			// parse float data
			String str = parser.get_node_data();
			array = str.split_floats_mk(splitters, false);
			//array=str.split_floats(" ",false);
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END)
			break; // end parsing text
	}

	return array;
}

Vector<String> Collada::_read_string_array(XMLParser &parser) {

	if (parser.is_empty())
		return Vector<String>();

	Vector<String> array;
	while (parser.read() == OK) {
		// TODO: check for comments inside the element
		// and ignore them.

		if (parser.get_node_type() == XMLParser::NODE_TEXT) {
			// parse String data
			String str = parser.get_node_data();
			array = str.split_spaces();
			/*
			for(int i=0;i<array.size();i++) {
				print_line(itos(i)+": "+array[i]);
			}
			*/
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END)
			break; // end parsing text
	}

	return array;
}

Transform Collada::_read_transform(XMLParser &parser) {

	if (parser.is_empty())
		return Transform();

	Vector<String> array;
	while (parser.read() == OK) {
		// TODO: check for comments inside the element
		// and ignore them.

		if (parser.get_node_type() == XMLParser::NODE_TEXT) {
			// parse float data
			String str = parser.get_node_data();
			array = str.split_spaces();
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END)
			break; // end parsing text
	}

	ERR_FAIL_COND_V(array.size() != 16, Transform());
	Vector<float> farr;
	farr.resize(16);
	for (int i = 0; i < 16; i++) {
		farr[i] = array[i].to_double();
	}

	return _read_transform_from_array(farr);
}

String Collada::_read_empty_draw_type(XMLParser &parser) {

	String empty_draw_type = "";

	if (parser.is_empty())
		return empty_draw_type;

	while (parser.read() == OK) {
		if (parser.get_node_type() == XMLParser::NODE_TEXT) {
			empty_draw_type = parser.get_node_data();
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END)
			break; // end parsing text
	}
	return empty_draw_type;
}

Variant Collada::_parse_param(XMLParser &parser) {

	if (parser.is_empty())
		return Variant();

	String from = parser.get_node_name();
	Variant data;

	while (parser.read() == OK) {
		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			if (parser.get_node_name() == "float") {

				parser.read();
				if (parser.get_node_type() == XMLParser::NODE_TEXT) {

					data = parser.get_node_data().to_double();
				}
			} else if (parser.get_node_name() == "float2") {

				Vector<float> v2 = _read_float_array(parser);

				if (v2.size() >= 2) {

					data = Vector2(v2[0], v2[1]);
				}
			} else if (parser.get_node_name() == "float3") {

				Vector<float> v3 = _read_float_array(parser);

				if (v3.size() >= 3) {

					data = Vector3(v3[0], v3[1], v3[2]);
				}
			} else if (parser.get_node_name() == "float4") {

				Vector<float> v4 = _read_float_array(parser);

				if (v4.size() >= 4) {

					data = Color(v4[0], v4[1], v4[2], v4[3]);
				}
			} else if (parser.get_node_name() == "sampler2D") {

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "source") {

							parser.read();

							if (parser.get_node_type() == XMLParser::NODE_TEXT) {

								data = parser.get_node_data();
							}
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "sampler2D")
						break;
				}
			} else if (parser.get_node_name() == "surface") {

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "init_from") {

							parser.read();

							if (parser.get_node_type() == XMLParser::NODE_TEXT) {

								data = parser.get_node_data();
							}
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "surface")
						break;
				}
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == from)
			break;
	}

	COLLADA_PRINT("newparam ending " + parser.get_node_name());
	return data;
}

void Collada::_parse_effect_material(XMLParser &parser, Effect &effect, String &id) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			// first come the tags we descend, but ignore the top-levels

			COLLADA_PRINT("node name: " + parser.get_node_name());

			if (!parser.is_empty() && (parser.get_node_name() == "profile_COMMON" || parser.get_node_name() == "technique" || parser.get_node_name() == "extra")) {

				_parse_effect_material(parser, effect, id); // try again

			} else if (parser.get_node_name() == "newparam") {
				String name = parser.get_attribute_value("sid");
				Variant value = _parse_param(parser);
				effect.params[name] = value;
				COLLADA_PRINT("param: " + name + " value:" + String(value));

			} else if (parser.get_node_name() == "constant" ||
					   parser.get_node_name() == "lambert" ||
					   parser.get_node_name() == "phong" ||
					   parser.get_node_name() == "blinn") {

				COLLADA_PRINT("shade model: " + parser.get_node_name());
				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						String what = parser.get_node_name();

						if (what == "emission" ||
								what == "diffuse" ||
								what == "specular" ||
								what == "reflective") {

							// color or texture types
							while (parser.read() == OK) {

								if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

									if (parser.get_node_name() == "color") {

										Vector<float> colorarr = _read_float_array(parser);
										COLLADA_PRINT("colorarr size: " + rtos(colorarr.size()));

										if (colorarr.size() >= 3) {

											// alpha strangely not allright? maybe it needs to be multiplied by value as a channel intensity
											Color color(colorarr[0], colorarr[1], colorarr[2], 1.0);
											if (what == "diffuse")
												effect.diffuse.color = color;
											if (what == "specular")
												effect.specular.color = color;
											if (what == "emission")
												effect.emission.color = color;

											COLLADA_PRINT(what + " color: " + color);
										}

									} else if (parser.get_node_name() == "texture") {

										String sampler = parser.get_attribute_value("texture");
										if (!effect.params.has(sampler)) {
											ERR_PRINT(String("Couldn't find sampler: " + sampler + " in material:" + id).utf8().get_data());
										} else {
											String surface = effect.params[sampler];

											if (!effect.params.has(surface)) {
												ERR_PRINT(String("Couldn't find surface: " + surface + " in material:" + id).utf8().get_data());
											} else {
												String uri = effect.params[surface];

												if (what == "diffuse") {
													effect.diffuse.texture = uri;
												} else if (what == "specular") {
													effect.specular.texture = uri;
												} else if (what == "emission") {
													effect.emission.texture = uri;
												} else if (what == "bump") {
													if (parser.has_attribute("bumptype") && parser.get_attribute_value("bumptype") != "NORMALMAP") {
														WARN_PRINT("'bump' texture type is not NORMALMAP, only NORMALMAP is supported.")
													}

													effect.bump.texture = uri;
												}

												COLLADA_PRINT(what + " texture: " + uri);
											}
										}
									} else if (!parser.is_empty())
										parser.skip_section();

								} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == what)
									break;
							}

						} else if (what == "shininess") {
#if 1
							effect.shininess = _parse_param(parser);
#else

							parser.read();
							float shininess = parser.get_node_data().to_double();
							effect.shininess = shininess;
							COLLADA_PRINT("shininess: " + rtos(shininess));
#endif
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && (parser.get_node_name() == "constant" ||
																								parser.get_node_name() == "lambert" ||
																								parser.get_node_name() == "phong" ||
																								parser.get_node_name() == "blinn"))
						break;
				}
			} else if (parser.get_node_name() == "double_sided" || parser.get_node_name() == "show_double_sided") { // colladamax / google earth

				// 3DS Max / Google Earth double sided extension
				parser.read();
				effect.found_double_sided = true;
				effect.double_sided = parser.get_node_data().to_int();
				COLLADA_PRINT("double sided: " + itos(parser.get_node_data().to_int()));
			} else if (parser.get_node_name() == "unshaded") {
				parser.read();
				effect.unshaded = parser.get_node_data().to_int();
			} else if (parser.get_node_name() == "bump") {

				// color or texture types
				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "texture") {

							String sampler = parser.get_attribute_value("texture");
							if (!effect.params.has(sampler)) {
								ERR_PRINT(String("Couldn't find sampler: " + sampler + " in material:" + id).utf8().get_data());
							} else {
								String surface = effect.params[sampler];

								if (!effect.params.has(surface)) {
									ERR_PRINT(String("Couldn't find surface: " + surface + " in material:" + id).utf8().get_data());
								} else {
									String uri = effect.params[surface];

									if (parser.has_attribute("bumptype") && parser.get_attribute_value("bumptype") != "NORMALMAP") {
										WARN_PRINT("'bump' texture type is not NORMALMAP, only NORMALMAP is supported.")
									}

									effect.bump.texture = uri;
									COLLADA_PRINT(" bump: " + uri);
								}
							}
						} else if (!parser.is_empty())
							parser.skip_section();

					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "bump")
						break;
				}

			} else if (!parser.is_empty())
				parser.skip_section();
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END &&
				   (parser.get_node_name() == "effect" ||
						   parser.get_node_name() == "profile_COMMON" ||
						   parser.get_node_name() == "technique" ||
						   parser.get_node_name() == "extra"))
			break;
	}
}

void Collada::_parse_effect(XMLParser &parser) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	String id = parser.get_attribute_value("id");

	Effect effect;
	if (parser.has_attribute("name"))
		effect.name = parser.get_attribute_value("name");
	_parse_effect_material(parser, effect, id);

	state.effect_map[id] = effect;

	COLLADA_PRINT("Effect ID:" + id);
}

void Collada::_parse_camera(XMLParser &parser) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	String id = parser.get_attribute_value("id");

	state.camera_data_map[id] = CameraData();
	CameraData &camera = state.camera_data_map[id];

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();

			if (name == "perspective") {

				camera.mode = CameraData::MODE_PERSPECTIVE;
			} else if (name == "orthographic") {

				camera.mode = CameraData::MODE_ORTHOGONAL;
			} else if (name == "xfov") {

				parser.read();
				camera.perspective.x_fov = parser.get_node_data().to_double();

			} else if (name == "yfov") {

				parser.read();
				camera.perspective.y_fov = parser.get_node_data().to_double();
			} else if (name == "xmag") {

				parser.read();
				camera.orthogonal.x_mag = parser.get_node_data().to_double();

			} else if (name == "ymag") {

				parser.read();
				camera.orthogonal.y_mag = parser.get_node_data().to_double();
			} else if (name == "aspect_ratio") {

				parser.read();
				camera.aspect = parser.get_node_data().to_double();

			} else if (name == "znear") {

				parser.read();
				camera.z_near = parser.get_node_data().to_double();

			} else if (name == "zfar") {

				parser.read();
				camera.z_far = parser.get_node_data().to_double();
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "camera")
			break; //end of <asset>
	}

	COLLADA_PRINT("Camera ID:" + id);
}

void Collada::_parse_light(XMLParser &parser) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	String id = parser.get_attribute_value("id");

	state.light_data_map[id] = LightData();
	LightData &light = state.light_data_map[id];

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();

			if (name == "ambient") {

				light.mode = LightData::MODE_AMBIENT;
			} else if (name == "directional") {

				light.mode = LightData::MODE_DIRECTIONAL;
			} else if (name == "point") {

				light.mode = LightData::MODE_OMNI;
			} else if (name == "spot") {

				light.mode = LightData::MODE_SPOT;
			} else if (name == "color") {

				parser.read();
				Vector<float> colorarr = _read_float_array(parser);
				COLLADA_PRINT("colorarr size: " + rtos(colorarr.size()));

				if (colorarr.size() >= 4) {
					// alpha strangely not allright? maybe it needs to be multiplied by value as a channel intensity
					Color color(colorarr[0], colorarr[1], colorarr[2], 1.0);
					light.color = color;
				}

			} else if (name == "constant_attenuation") {

				parser.read();
				light.constant_att = parser.get_node_data().to_double();
			} else if (name == "linear_attenuation") {

				parser.read();
				light.linear_att = parser.get_node_data().to_double();
			} else if (name == "quadratic_attenuation") {

				parser.read();
				light.quad_att = parser.get_node_data().to_double();
			} else if (name == "falloff_angle") {

				parser.read();
				light.spot_angle = parser.get_node_data().to_double();

			} else if (name == "falloff_exponent") {

				parser.read();
				light.spot_exp = parser.get_node_data().to_double();
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "light")
			break; //end of <asset>
	}

	COLLADA_PRINT("Light ID:" + id);
}

void Collada::_parse_curve_geometry(XMLParser &parser, String p_id, String p_name) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	//load everything into a pre dictionary

	state.curve_data_map[p_id] = CurveData();

	CurveData &curvedata = state.curve_data_map[p_id];
	curvedata.name = p_name;

	COLLADA_PRINT("curve name: " + p_name);

	String current_source;
	// handles geometry node and the curve childs in this loop
	// read sources with arrays and accessor for each curve
	if (parser.is_empty()) {
		return;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "source") {

				String id = parser.get_attribute_value("id");
				curvedata.sources[id] = CurveData::Source();
				current_source = id;
				COLLADA_PRINT("source data: " + id);

			} else if (section == "float_array" || section == "array") {
				// create a new array and read it.
				if (curvedata.sources.has(current_source)) {

					curvedata.sources[current_source].array = _read_float_array(parser);
					COLLADA_PRINT("section: " + current_source + " read " + itos(curvedata.sources[current_source].array.size()) + " values.");
				}
			} else if (section == "Name_array") {
				// create a new array and read it.
				if (curvedata.sources.has(current_source)) {

					curvedata.sources[current_source].sarray = _read_string_array(parser);
					COLLADA_PRINT("section: " + current_source + " read " + itos(curvedata.sources[current_source].array.size()) + " values.");
				}

			} else if (section == "technique_common") {
				//skip it
			} else if (section == "accessor") { // child of source (below a technique tag)

				if (curvedata.sources.has(current_source)) {
					curvedata.sources[current_source].stride = parser.get_attribute_value("stride").to_int();
					COLLADA_PRINT("section: " + current_source + " stride " + itos(curvedata.sources[current_source].stride));
				}
			} else if (section == "control_vertices") {

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "input") {

							String semantic = parser.get_attribute_value("semantic");
							String source = _uri_to_id(parser.get_attribute_value("source"));

							curvedata.control_vertices[semantic] = source;

							COLLADA_PRINT(section + " input semantic: " + semantic + " source: " + source);
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == section)
						break;
				}

			} else if (!parser.is_empty()) {

				parser.skip_section();
			}
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "spline")
			break;
	}
}

void Collada::_parse_mesh_geometry(XMLParser &parser, String p_id, String p_name) {

	if (!(state.import_flags & IMPORT_FLAG_SCENE)) {
		if (!parser.is_empty())
			parser.skip_section();
		return;
	}

	//load everything into a pre dictionary

	state.mesh_data_map[p_id] = MeshData();

	MeshData &meshdata = state.mesh_data_map[p_id];
	meshdata.name = p_name;

	COLLADA_PRINT("mesh name: " + p_name);

	String current_source;
	// handles geometry node and the mesh childs in this loop
	// read sources with arrays and accessor for each mesh
	if (parser.is_empty()) {
		return;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "source") {

				String id = parser.get_attribute_value("id");
				meshdata.sources[id] = MeshData::Source();
				current_source = id;
				COLLADA_PRINT("source data: " + id);

			} else if (section == "float_array" || section == "array") {
				// create a new array and read it.
				if (meshdata.sources.has(current_source)) {

					meshdata.sources[current_source].array = _read_float_array(parser);
					COLLADA_PRINT("section: " + current_source + " read " + itos(meshdata.sources[current_source].array.size()) + " values.");
				}
			} else if (section == "technique_common") {
				//skip it
			} else if (section == "accessor") { // child of source (below a technique tag)

				if (meshdata.sources.has(current_source)) {
					meshdata.sources[current_source].stride = parser.get_attribute_value("stride").to_int();
					COLLADA_PRINT("section: " + current_source + " stride " + itos(meshdata.sources[current_source].stride));
				}
			} else if (section == "vertices") {

				MeshData::Vertices vert;
				String id = parser.get_attribute_value("id");

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "input") {

							String semantic = parser.get_attribute_value("semantic");
							String source = _uri_to_id(parser.get_attribute_value("source"));

							vert.sources[semantic] = source;

							COLLADA_PRINT(section + " input semantic: " + semantic + " source: " + source);
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == section)
						break;
				}

				meshdata.vertices[id] = vert;

			} else if (section == "triangles" || section == "polylist" || section == "polygons") {

				bool polygons = (section == "polygons");
				if (polygons) {
					WARN_PRINT("Primitive type \"polygons\" is not well supported (concave shapes may fail). To ensure that the geometry is properly imported, please re-export using \"triangles\" or \"polylist\".");
				}
				MeshData::Primitives prim;

				if (parser.has_attribute("material"))
					prim.material = parser.get_attribute_value("material");
				prim.count = parser.get_attribute_value("count").to_int();
				prim.vertex_size = 0;
				int last_ref = 0;

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "input") {

							String semantic = parser.get_attribute_value("semantic");
							String source = _uri_to_id(parser.get_attribute_value("source"));

							if (semantic == "TEXCOORD") {
								/*
								if (parser.has_attribute("set"))// a texcoord
									semantic+=parser.get_attribute_value("set");
								else
									semantic="TEXCOORD0";*/
								semantic = "TEXCOORD" + itos(last_ref++);
							}
							int offset = parser.get_attribute_value("offset").to_int();

							MeshData::Primitives::SourceRef sref;
							sref.source = source;
							sref.offset = offset;
							prim.sources[semantic] = sref;
							prim.vertex_size = MAX(prim.vertex_size, offset + 1);

							COLLADA_PRINT(section + " input semantic: " + semantic + " source: " + source + " offset: " + itos(offset));

						} else if (parser.get_node_name() == "p") { //indices

							Vector<float> values = _read_float_array(parser);
							if (polygons) {

								prim.polygons.push_back(values.size() / prim.vertex_size);
								int from = prim.indices.size();
								prim.indices.resize(from + values.size());
								for (int i = 0; i < values.size(); i++)
									prim.indices[from + i] = values[i];

							} else if (prim.vertex_size > 0) {
								prim.indices = values;
							}

							COLLADA_PRINT("read " + itos(values.size()) + " index values");

						} else if (parser.get_node_name() == "vcount") { // primitive

							Vector<float> values = _read_float_array(parser);
							prim.polygons = values;
							COLLADA_PRINT("read " + itos(values.size()) + " polygon values");
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == section)
						break;
				}

				meshdata.primitives.push_back(prim);

			} else if (parser.get_node_name() == "double_sided") {

				parser.read();
				meshdata.found_double_sided = true;
				meshdata.double_sided = parser.get_node_data().to_int();

			} else if (parser.get_node_name() == "polygons") {
				ERR_PRINT("Primitive type \"polygons\" not supported, re-export using \"polylist\" or \"triangles\".");
			} else if (!parser.is_empty()) {

				parser.skip_section();
			}
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "mesh")
			break;
	}
}

void Collada::_parse_skin_controller(XMLParser &parser, String p_id) {

	state.skin_controller_data_map[p_id] = SkinControllerData();
	SkinControllerData &skindata = state.skin_controller_data_map[p_id];

	skindata.base = _uri_to_id(parser.get_attribute_value("source"));

	String current_source;

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "bind_shape_matrix") {

				skindata.bind_shape = _read_transform(parser);
#ifdef COLLADA_IMPORT_SCALE_SCENE
				skindata.bind_shape.origin *= state.unit_scale;

#endif
				COLLADA_PRINT("skeleton bind shape transform: " + skindata.bind_shape);

			} else if (section == "source") {

				String id = parser.get_attribute_value("id");
				skindata.sources[id] = SkinControllerData::Source();
				current_source = id;
				COLLADA_PRINT("source data: " + id);

			} else if (section == "float_array" || section == "array") {
				// create a new array and read it.
				if (skindata.sources.has(current_source)) {

					skindata.sources[current_source].array = _read_float_array(parser);
					COLLADA_PRINT("section: " + current_source + " read " + itos(skindata.sources[current_source].array.size()) + " values.");
				}
			} else if (section == "Name_array" || section == "IDREF_array") {
				// create a new array and read it.

				if (section == "IDREF_array")
					skindata.use_idrefs = true;
				if (skindata.sources.has(current_source)) {

					skindata.sources[current_source].sarray = _read_string_array(parser);
					if (section == "IDREF_array") {
						Vector<String> sa = skindata.sources[current_source].sarray;
						for (int i = 0; i < sa.size(); i++)
							state.idref_joints.insert(sa[i]);
					}
					COLLADA_PRINT("section: " + current_source + " read " + itos(skindata.sources[current_source].array.size()) + " values.");
				}
			} else if (section == "technique_common") {
				//skip it
			} else if (section == "accessor") { // child of source (below a technique tag)

				if (skindata.sources.has(current_source)) {

					int stride = 1;
					if (parser.has_attribute("stride"))
						stride = parser.get_attribute_value("stride").to_int();

					skindata.sources[current_source].stride = stride;
					COLLADA_PRINT("section: " + current_source + " stride " + itos(skindata.sources[current_source].stride));
				}

			} else if (section == "joints") {

				SkinControllerData::Joints joint;

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "input") {

							String semantic = parser.get_attribute_value("semantic");
							String source = _uri_to_id(parser.get_attribute_value("source"));

							joint.sources[semantic] = source;

							COLLADA_PRINT(section + " input semantic: " + semantic + " source: " + source);
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == section)
						break;
				}

				skindata.joints = joint;

			} else if (section == "vertex_weights") {

				SkinControllerData::Weights weights;

				weights.count = parser.get_attribute_value("count").to_int();

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "input") {

							String semantic = parser.get_attribute_value("semantic");
							String source = _uri_to_id(parser.get_attribute_value("source"));

							int offset = parser.get_attribute_value("offset").to_int();

							SkinControllerData::Weights::SourceRef sref;
							sref.source = source;
							sref.offset = offset;
							weights.sources[semantic] = sref;

							COLLADA_PRINT(section + " input semantic: " + semantic + " source: " + source + " offset: " + itos(offset));

						} else if (parser.get_node_name() == "v") { //indices

							Vector<float> values = _read_float_array(parser);
							weights.indices = values;
							COLLADA_PRINT("read " + itos(values.size()) + " index values");

						} else if (parser.get_node_name() == "vcount") { // weightsitive

							Vector<float> values = _read_float_array(parser);
							weights.sets = values;
							COLLADA_PRINT("read " + itos(values.size()) + " polygon values");
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == section)
						break;
				}

				skindata.weights = weights;
			}
			/*
			else if (!parser.is_empty())
				parser.skip_section();
			*/

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "skin")
			break;
	}

	/* STORE REST MATRICES */

	Vector<Transform> rests;
	ERR_FAIL_COND(!skindata.joints.sources.has("JOINT"));
	ERR_FAIL_COND(!skindata.joints.sources.has("INV_BIND_MATRIX"));

	String joint_arr = skindata.joints.sources["JOINT"];
	String ibm = skindata.joints.sources["INV_BIND_MATRIX"];

	ERR_FAIL_COND(!skindata.sources.has(joint_arr));
	ERR_FAIL_COND(!skindata.sources.has(ibm));

	SkinControllerData::Source &joint_source = skindata.sources[joint_arr];
	SkinControllerData::Source &ibm_source = skindata.sources[ibm];

	ERR_FAIL_COND(joint_source.sarray.size() != ibm_source.array.size() / 16);

	for (int i = 0; i < joint_source.sarray.size(); i++) {

		String name = joint_source.sarray[i];
		Transform xform = _read_transform_from_array(ibm_source.array, i * 16); //<- this is a mistake, it must be applied to vertices
		xform.affine_invert(); // inverse for rest, because it's an inverse
#ifdef COLLADA_IMPORT_SCALE_SCENE
		xform.origin *= state.unit_scale;
#endif
		skindata.bone_rest_map[name] = xform;
	}
}

void Collada::_parse_morph_controller(XMLParser &parser, String p_id) {

	state.morph_controller_data_map[p_id] = MorphControllerData();
	MorphControllerData &morphdata = state.morph_controller_data_map[p_id];

	print_line("morph source: " + parser.get_attribute_value("source") + " id: " + p_id);
	morphdata.mesh = _uri_to_id(parser.get_attribute_value("source"));
	print_line("morph source2: " + morphdata.mesh);
	morphdata.mode = parser.get_attribute_value("method");
	printf("JJmorph: %p\n", &morphdata);
	String current_source;

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "source") {

				String id = parser.get_attribute_value("id");
				morphdata.sources[id] = MorphControllerData::Source();
				current_source = id;
				COLLADA_PRINT("source data: " + id);

			} else if (section == "float_array" || section == "array") {
				// create a new array and read it.
				if (morphdata.sources.has(current_source)) {

					morphdata.sources[current_source].array = _read_float_array(parser);
					COLLADA_PRINT("section: " + current_source + " read " + itos(morphdata.sources[current_source].array.size()) + " values.");
				}
			} else if (section == "Name_array" || section == "IDREF_array") {
				// create a new array and read it.

				/*
				if (section=="IDREF_array")
					morphdata.use_idrefs=true;
				*/
				if (morphdata.sources.has(current_source)) {

					morphdata.sources[current_source].sarray = _read_string_array(parser);
					/*
					if (section=="IDREF_array") {
						Vector<String> sa = morphdata.sources[current_source].sarray;
						for(int i=0;i<sa.size();i++)
							state.idref_joints.insert(sa[i]);
					}*/
					COLLADA_PRINT("section: " + current_source + " read " + itos(morphdata.sources[current_source].array.size()) + " values.");
				}
			} else if (section == "technique_common") {
				//skip it
			} else if (section == "accessor") { // child of source (below a technique tag)

				if (morphdata.sources.has(current_source)) {

					int stride = 1;
					if (parser.has_attribute("stride"))
						stride = parser.get_attribute_value("stride").to_int();

					morphdata.sources[current_source].stride = stride;
					COLLADA_PRINT("section: " + current_source + " stride " + itos(morphdata.sources[current_source].stride));
				}

			} else if (section == "targets") {

				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "input") {

							String semantic = parser.get_attribute_value("semantic");
							String source = _uri_to_id(parser.get_attribute_value("source"));

							morphdata.targets[semantic] = source;

							COLLADA_PRINT(section + " input semantic: " + semantic + " source: " + source);
						}
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == section)
						break;
				}
			}
			/*
			else if (!parser.is_empty())
				parser.skip_section();
			*/

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "morph")
			break;
	}

	if (morphdata.targets.has("MORPH_WEIGHT")) {

		state.morph_name_map[morphdata.targets["MORPH_WEIGHT"]] = p_id;
	}
}

void Collada::_parse_controller(XMLParser &parser) {

	String id = parser.get_attribute_value("id");

	if (parser.is_empty()) {
		return;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "skin") {
				_parse_skin_controller(parser, id);
			} else if (section == "morph") {
				_parse_morph_controller(parser, id);
			}
		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "controller")
			break;
	}
}

Collada::Node *Collada::_parse_visual_instance_geometry(XMLParser &parser) {

	String type = parser.get_node_name();
	NodeGeometry *geom = memnew(NodeGeometry);
	geom->controller = type == "instance_controller";
	geom->source = _uri_to_id(parser.get_attribute_value_safe("url"));

	if (parser.is_empty()) //nothing else to parse...
		return geom;
	// try to find also many materials and skeletons!
	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			if (parser.get_node_name() == "instance_material") {

				String symbol = parser.get_attribute_value("symbol");
				String target = _uri_to_id(parser.get_attribute_value("target"));

				NodeGeometry::Material mat;
				mat.target = target;
				geom->material_map[symbol] = mat;
				COLLADA_PRINT("uses material: '" + target + "' on primitive'" + symbol + "'");
			} else if (parser.get_node_name() == "skeleton") {

				parser.read();
				String uri = _uri_to_id(parser.get_node_data());
				if (uri != "") {
					geom->skeletons.push_back(uri);
				}
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == type)
			break;
	}

	if (geom->controller) {

		if (geom->skeletons.empty()) {
			//XSI style

			if (state.skin_controller_data_map.has(geom->source)) {
				SkinControllerData *skin = &state.skin_controller_data_map[geom->source];
				//case where skeletons reference bones with IDREF (XSI)
				ERR_FAIL_COND_V(!skin->joints.sources.has("JOINT"), geom);
				String joint_arr = skin->joints.sources["JOINT"];
				ERR_FAIL_COND_V(!skin->sources.has(joint_arr), geom);
				Collada::SkinControllerData::Source &joint_source = skin->sources[joint_arr];
				geom->skeletons = joint_source.sarray; //quite crazy, but should work.
			}
		}
	}

	return geom;
}

Collada::Node *Collada::_parse_visual_instance_camera(XMLParser &parser) {

	String type = parser.get_node_name();
	NodeCamera *cam = memnew(NodeCamera);
	cam->camera = _uri_to_id(parser.get_attribute_value_safe("url"));

	if (state.up_axis == Vector3::AXIS_Z) //collada weirdness
		cam->post_transform.basis.rotate(Vector3(1, 0, 0), -Math_PI * 0.5);

	if (parser.is_empty()) //nothing else to parse...
		return cam;

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "instance_camera")
			break;
	}

	return cam;
}

Collada::Node *Collada::_parse_visual_instance_light(XMLParser &parser) {

	String type = parser.get_node_name();
	NodeLight *cam = memnew(NodeLight);
	cam->light = _uri_to_id(parser.get_attribute_value_safe("url"));

	if (state.up_axis == Vector3::AXIS_Z) //collada weirdness
		cam->post_transform.basis.rotate(Vector3(1, 0, 0), -Math_PI * 0.5);

	if (parser.is_empty()) //nothing else to parse...
		return cam;

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "instance_light")
			break;
	}

	return cam;
}

Collada::Node *Collada::_parse_visual_node_instance_data(XMLParser &parser) {

	String instance_type = parser.get_node_name();

	if (instance_type == "instance_geometry" || instance_type == "instance_controller") {
		return _parse_visual_instance_geometry(parser);
	} else if (instance_type == "instance_camera") {

		return _parse_visual_instance_camera(parser);
	} else if (instance_type == "instance_light") {
		return _parse_visual_instance_light(parser);
	}

	if (parser.is_empty()) //nothing else to parse...
		return NULL;

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == instance_type)
			break;
	}

	return NULL;
}

Collada::Node *Collada::_parse_visual_scene_node(XMLParser &parser) {

	String name;

	String id = parser.get_attribute_value_safe("id");

	bool found_name = false;

	if (id == "") {

		id = "%NODEID%" + itos(Math::rand());

	} else {
		found_name = true;
	}

	Vector<Node::XForm> xform_list;
	Vector<Node *> children;

	String empty_draw_type = "";

	Node *node = NULL;

	name = parser.has_attribute("name") ? parser.get_attribute_value_safe("name") : parser.get_attribute_value_safe("id");
	if (name == "") {

		name = id;
	} else {
		found_name = true;
	}

	if ((parser.has_attribute("type") && parser.get_attribute_value("type") == "JOINT") || state.idref_joints.has(name)) {
		// handle a bone

		NodeJoint *joint = memnew(NodeJoint);

		if (parser.has_attribute("sid")) { //bones may not have sid
			joint->sid = parser.get_attribute_value("sid");
			//state.bone_map[joint->sid]=joint;
		} else if (state.idref_joints.has(name)) {
			joint->sid = name; //kind of a cheat but..
		} else if (parser.has_attribute("name")) {
			joint->sid = parser.get_attribute_value_safe("name");
		}

		if (joint->sid != "") {
			state.sid_to_node_map[joint->sid] = id;
		}

		node = joint;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "translate") {
				Node::XForm xf;
				if (parser.has_attribute("sid")) {
					xf.id = parser.get_attribute_value("sid");
				}
				xf.op = Node::XForm::OP_TRANSLATE;

				Vector<float> xlt = _read_float_array(parser);
				xf.data = xlt;
				xform_list.push_back(xf);

			} else if (section == "rotate") {
				Node::XForm xf;
				if (parser.has_attribute("sid")) {
					xf.id = parser.get_attribute_value("sid");
				}
				xf.op = Node::XForm::OP_ROTATE;

				Vector<float> rot = _read_float_array(parser);
				xf.data = rot;

				xform_list.push_back(xf);

			} else if (section == "scale") {
				Node::XForm xf;
				if (parser.has_attribute("sid")) {
					xf.id = parser.get_attribute_value("sid");
				}

				xf.op = Node::XForm::OP_SCALE;

				Vector<float> scale = _read_float_array(parser);

				xf.data = scale;

				xform_list.push_back(xf);

			} else if (section == "matrix") {
				Node::XForm xf;
				if (parser.has_attribute("sid")) {
					xf.id = parser.get_attribute_value("sid");
				}
				xf.op = Node::XForm::OP_MATRIX;

				Vector<float> matrix = _read_float_array(parser);

				xf.data = matrix;
				String mtx;
				for (int i = 0; i < matrix.size(); i++)
					mtx += " " + rtos(matrix[i]);

				xform_list.push_back(xf);

			} else if (section == "visibility") {
				Node::XForm xf;
				if (parser.has_attribute("sid")) {
					xf.id = parser.get_attribute_value("sid");
				}
				xf.op = Node::XForm::OP_VISIBILITY;

				Vector<float> visible = _read_float_array(parser);

				xf.data = visible;

				xform_list.push_back(xf);

			} else if (section == "empty_draw_type") {
				empty_draw_type = _read_empty_draw_type(parser);
			} else if (section == "technique" || section == "extra") {

			} else if (section != "node") {
				//usually what defines the type of node
				//print_line(" don't know what to do with "+section);
				if (section.begins_with("instance_")) {

					if (!node) {

						node = _parse_visual_node_instance_data(parser);

					} else {
						ERR_PRINT("Multiple instance_* not supported.");
					}
				}

			} else if (section == "node") {

				/* Found a child node!! what to do..*/

				Node *child = _parse_visual_scene_node(parser);
				children.push_back(child);
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "node")
			break;
	}

	if (!node) {

		node = memnew(Node); //generic node, nothing of relevance found
	}

	node->noname = !found_name;
	node->xform_list = xform_list;
	node->children = children;
	for (int i = 0; i < children.size(); i++) {
		node->children[i]->parent = node;
	}

	node->name = name;
	node->id = id;
	node->empty_draw_type = empty_draw_type;

	if (node->children.size() == 1) {
		if (node->children[0]->noname && !node->noname) {
			node->children[0]->name = node->name;
			node->name = node->name + "-base";
		}
	}

	node->default_transform = node->compute_transform(*this);
	state.scene_map[id] = node;

	return node;
}

void Collada::_parse_visual_scene(XMLParser &parser) {

	String id = parser.get_attribute_value("id");

	if (parser.is_empty()) {
		return;
	}

	state.visual_scene_map[id] = VisualScene();
	VisualScene &vscene = state.visual_scene_map[id];

	if (parser.has_attribute("name"))
		vscene.name = parser.get_attribute_value("name");

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String section = parser.get_node_name();

			if (section == "node") {
				vscene.root_nodes.push_back(_parse_visual_scene_node(parser));
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "visual_scene")
			break;
	}

	COLLADA_PRINT("Scene ID:" + id);
}

void Collada::_parse_animation(XMLParser &parser) {

	if (!(state.import_flags & IMPORT_FLAG_ANIMATION)) {
		if (!parser.is_empty())
			parser.skip_section();

		return;
	}

	Map<String, Vector<float> > float_sources;
	Map<String, Vector<String> > string_sources;
	Map<String, int> source_strides;
	Map<String, Map<String, String> > samplers;
	Map<String, Vector<String> > source_param_names;
	Map<String, Vector<String> > source_param_types;

	String id = "";
	if (parser.has_attribute("id"))
		id = parser.get_attribute_value("id");

	String current_source;
	String current_sampler;
	Vector<String> channel_sources;
	Vector<String> channel_targets;

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();
			if (name == "source") {

				current_source = parser.get_attribute_value("id");
				source_param_names[current_source] = Vector<String>();
				source_param_types[current_source] = Vector<String>();

			} else if (name == "float_array") {

				if (current_source != "") {
					float_sources[current_source] = _read_float_array(parser);
				}

			} else if (name == "Name_array") {

				if (current_source != "") {
					string_sources[current_source] = _read_string_array(parser);
				}
			} else if (name == "accessor") {

				if (current_source != "" && parser.has_attribute("stride")) {
					source_strides[current_source] = parser.get_attribute_value("stride").to_int();
				}
			} else if (name == "sampler") {

				current_sampler = parser.get_attribute_value("id");
				samplers[current_sampler] = Map<String, String>();
			} else if (name == "param") {

				if (parser.has_attribute("name"))
					source_param_names[current_source].push_back(parser.get_attribute_value("name"));
				else
					source_param_names[current_source].push_back("");

				if (parser.has_attribute("type"))
					source_param_types[current_source].push_back(parser.get_attribute_value("type"));
				else
					source_param_types[current_source].push_back("");

			} else if (name == "input") {

				if (current_sampler != "") {

					samplers[current_sampler][parser.get_attribute_value("semantic")] = parser.get_attribute_value("source");
				}

			} else if (name == "channel") {

				channel_sources.push_back(parser.get_attribute_value("source"));
				channel_targets.push_back(parser.get_attribute_value("target"));
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "animation")
			break; //end of <asset>
	}

	for (int i = 0; i < channel_sources.size(); i++) {

		String source = _uri_to_id(channel_sources[i]);
		String target = channel_targets[i];
		if (!samplers.has(source)) {
			print_line("channel lacks source: " + source);
		}
		ERR_CONTINUE(!samplers.has(source));
		Map<String, String> &sampler = samplers[source];

		ERR_CONTINUE(!sampler.has("INPUT")); //no input semantic? wtf?
		String input_id = _uri_to_id(sampler["INPUT"]);
		COLLADA_PRINT("input id is " + input_id);
		ERR_CONTINUE(!float_sources.has(input_id));

		ERR_CONTINUE(!sampler.has("OUTPUT"));
		String output_id = _uri_to_id(sampler["OUTPUT"]);
		ERR_CONTINUE(!float_sources.has(output_id));

		ERR_CONTINUE(!source_param_names.has(output_id));

		Vector<String> &names = source_param_names[output_id];

		for (int l = 0; l < names.size(); l++) {

			String name = names[l];

			Vector<float> &time_keys = float_sources[input_id];
			int key_count = time_keys.size();

			AnimationTrack track; //begin crating track
			track.id = id;

			track.keys.resize(key_count);

			for (int j = 0; j < key_count; j++) {
				track.keys[j].time = time_keys[j];
				state.animation_length = MAX(state.animation_length, time_keys[j]);
			}

			//now read actual values

			int stride = 1;

			if (source_strides.has(output_id))
				stride = source_strides[output_id];
			int output_len = stride / names.size();

			ERR_CONTINUE(output_len == 0);
			ERR_CONTINUE(!float_sources.has(output_id));

			Vector<float> &output = float_sources[output_id];

			ERR_EXPLAIN("Wrong number of keys in output");
			ERR_CONTINUE((output.size() / stride) != key_count);

			for (int j = 0; j < key_count; j++) {
				track.keys[j].data.resize(output_len);
				for (int k = 0; k < output_len; k++)
					track.keys[j].data[k] = output[l + j * stride + k]; //super weird but should work
			}

			if (sampler.has("INTERPOLATION")) {

				String interp_id = _uri_to_id(sampler["INTERPOLATION"]);
				ERR_CONTINUE(!string_sources.has(interp_id));
				Vector<String> &interps = string_sources[interp_id];
				ERR_CONTINUE(interps.size() != key_count);

				for (int j = 0; j < key_count; j++) {
					if (interps[j] == "BEZIER")
						track.keys[j].interp_type = AnimationTrack::INTERP_BEZIER;
					else
						track.keys[j].interp_type = AnimationTrack::INTERP_LINEAR;
				}
			}

			if (sampler.has("IN_TANGENT") && sampler.has("OUT_TANGENT")) {
				//bezier control points..
				String intangent_id = _uri_to_id(sampler["IN_TANGENT"]);
				ERR_CONTINUE(!float_sources.has(intangent_id));
				Vector<float> &intangents = float_sources[intangent_id];

				ERR_CONTINUE(intangents.size() != key_count * 2 * names.size());

				String outangent_id = _uri_to_id(sampler["OUT_TANGENT"]);
				ERR_CONTINUE(!float_sources.has(outangent_id));
				Vector<float> &outangents = float_sources[outangent_id];
				ERR_CONTINUE(outangents.size() != key_count * 2 * names.size());

				for (int j = 0; j < key_count; j++) {
					track.keys[j].in_tangent = Vector2(intangents[j * 2 * names.size() + 0 + l * 2], intangents[j * 2 * names.size() + 1 + l * 2]);
					track.keys[j].out_tangent = Vector2(outangents[j * 2 * names.size() + 0 + l * 2], outangents[j * 2 * names.size() + 1 + l * 2]);
				}
			}

			if (target.find("/") != -1) { //transform component
				track.target = target.get_slicec('/', 0);
				track.param = target.get_slicec('/', 1);
				if (track.param.find(".") != -1)
					track.component = track.param.get_slice(".", 1).to_upper();
				track.param = track.param.get_slice(".", 0);
				if (names.size() > 1 && track.component == "") {
					//this is a guess because the collada spec is ambiguous here...
					//i suppose if you have many names (outputs) you can't use a component and i should abide to that.
					track.component = name;
				}
			} else {
				track.target = target;
			}

			print_line("TARGET: " + track.target);

			state.animation_tracks.push_back(track);

			if (!state.referenced_tracks.has(target))
				state.referenced_tracks[target] = Vector<int>();

			state.referenced_tracks[target].push_back(state.animation_tracks.size() - 1);

			if (id != "") {
				if (!state.by_id_tracks.has(id))
					state.by_id_tracks[id] = Vector<int>();

				state.by_id_tracks[id].push_back(state.animation_tracks.size() - 1);
			}

			COLLADA_PRINT("loaded animation with " + itos(key_count) + " keys");
		}
	}
}

void Collada::_parse_animation_clip(XMLParser &parser) {

	if (!(state.import_flags & IMPORT_FLAG_ANIMATION)) {
		if (!parser.is_empty())
			parser.skip_section();

		return;
	}

	AnimationClip clip;

	if (parser.has_attribute("name"))
		clip.name = parser.get_attribute_value("name");
	else if (parser.has_attribute("id"))
		clip.name = parser.get_attribute_value("id");
	if (parser.has_attribute("start"))
		clip.begin = parser.get_attribute_value("start").to_double();
	if (parser.has_attribute("end"))
		clip.end = parser.get_attribute_value("end").to_double();

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();
			if (name == "instance_animation") {

				String url = _uri_to_id(parser.get_attribute_value("url"));
				clip.tracks.push_back(url);
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "animation_clip")
			break; //end of <asset>
	}

	state.animation_clips.push_back(clip);
	print_line("found anim clip: " + clip.name);
}
void Collada::_parse_scene(XMLParser &parser) {

	if (parser.is_empty()) {
		return;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();

			if (name == "instance_visual_scene") {

				state.root_visual_scene = _uri_to_id(parser.get_attribute_value("url"));
				print_line("***ROOT VISUAL SCENE: " + state.root_visual_scene);
			} else if (name == "instance_physics_scene") {

				state.root_physics_scene = _uri_to_id(parser.get_attribute_value("url"));
			}

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "scene")
			break; //end of <asset>
	}
}

void Collada::_parse_library(XMLParser &parser) {

	if (parser.is_empty()) {
		return;
	}

	while (parser.read() == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			String name = parser.get_node_name();
			COLLADA_PRINT("library name is: " + name);
			if (name == "image") {

				_parse_image(parser);
			} else if (name == "material") {

				_parse_material(parser);
			} else if (name == "effect") {

				_parse_effect(parser);
			} else if (name == "camera") {

				_parse_camera(parser);
			} else if (name == "light") {

				_parse_light(parser);
			} else if (name == "geometry") {

				String id = parser.get_attribute_value("id");
				String name = parser.get_attribute_value_safe("name");
				while (parser.read() == OK) {

					if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

						if (parser.get_node_name() == "mesh") {
							state.mesh_name_map[id] = (name != "") ? name : id;
							_parse_mesh_geometry(parser, id, name);
						} else if (parser.get_node_name() == "spline") {
							state.mesh_name_map[id] = (name != "") ? name : id;
							_parse_curve_geometry(parser, id, name);
						} else if (!parser.is_empty())
							parser.skip_section();
					} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "geometry")
						break;
				}

			} else if (name == "controller") {

				_parse_controller(parser);
			} else if (name == "animation") {

				_parse_animation(parser);
			} else if (name == "animation_clip") {

				_parse_animation_clip(parser);
			} else if (name == "visual_scene") {

				COLLADA_PRINT("visual scene");
				_parse_visual_scene(parser);
			} else if (!parser.is_empty())
				parser.skip_section();

		} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name().begins_with("library_"))
			break; //end of <asset>
	}
}

void Collada::_joint_set_owner(Collada::Node *p_node, NodeSkeleton *p_owner) {

	if (p_node->type == Node::TYPE_JOINT) {

		NodeJoint *nj = static_cast<NodeJoint *>(p_node);
		nj->owner = p_owner;

		for (int i = 0; i < nj->children.size(); i++) {

			_joint_set_owner(nj->children[i], p_owner);
		}
	}
}

void Collada::_create_skeletons(Collada::Node **p_node, NodeSkeleton *p_skeleton) {

	Node *node = *p_node;

	if (node->type == Node::TYPE_JOINT) {

		if (!p_skeleton) {

			// ohohohoohoo it's a joint node, time to work!
			NodeSkeleton *sk = memnew(NodeSkeleton);
			*p_node = sk;
			sk->children.push_back(node);
			sk->parent = node->parent;
			node->parent = sk;
			p_skeleton = sk;
		}

		NodeJoint *nj = static_cast<NodeJoint *>(node);
		nj->owner = p_skeleton;
	} else {
		p_skeleton = NULL;
	}

	for (int i = 0; i < node->children.size(); i++) {
		_create_skeletons(&node->children[i], p_skeleton);
	}
}

bool Collada::_remove_node(Node *p_parent, Node *p_node) {

	for (int i = 0; i < p_parent->children.size(); i++) {

		if (p_parent->children[i] == p_node) {
			p_parent->children.remove(i);
			return true;
		}
		if (_remove_node(p_parent->children[i], p_node))
			return true;
	}

	return false;
}

void Collada::_remove_node(VisualScene *p_vscene, Node *p_node) {

	for (int i = 0; i < p_vscene->root_nodes.size(); i++) {
		if (p_vscene->root_nodes[i] == p_node) {

			p_vscene->root_nodes.remove(i);
			return;
		}
		if (_remove_node(p_vscene->root_nodes[i], p_node))
			return;
	}

	ERR_PRINT("ERROR: Not found node to remove?");
}

void Collada::_merge_skeletons(VisualScene *p_vscene, Node *p_node) {

	if (p_node->type == Node::TYPE_GEOMETRY) {

		NodeGeometry *gnode = static_cast<NodeGeometry *>(p_node);
		if (gnode->controller) {

			// recount skeletons used
			Set<NodeSkeleton *> skeletons;

			for (int i = 0; i < gnode->skeletons.size(); i++) {

				String nodeid = gnode->skeletons[i];

				ERR_CONTINUE(!state.scene_map.has(nodeid)); //weird, it should have it...

				NodeJoint *nj = SAFE_CAST<NodeJoint *>(state.scene_map[nodeid]);
				if (!nj->owner) {
					print_line("no owner for: " + String(nodeid));
				}
				ERR_CONTINUE(!nj->owner); //weird, node should have a skeleton owner

				skeletons.insert(nj->owner);
			}

			if (skeletons.size() > 1) {

				//do the merger!!
				Set<NodeSkeleton *>::Element *E = skeletons.front();
				NodeSkeleton *base = E->get();

				for (E = E->next(); E; E = E->next()) {

					NodeSkeleton *merged = E->get();
					_remove_node(p_vscene, merged);
					for (int i = 0; i < merged->children.size(); i++) {

						_joint_set_owner(merged->children[i], base);
						base->children.push_back(merged->children[i]);
						merged->children[i]->parent = base;
					}

					merged->children.clear(); //take children from it
					memdelete(merged);
				}
			}
		}
	}

	for (int i = 0; i < p_node->children.size(); i++) {
		_merge_skeletons(p_vscene, p_node->children[i]);
	}
}

void Collada::_merge_skeletons2(VisualScene *p_vscene) {

	for (Map<String, SkinControllerData>::Element *E = state.skin_controller_data_map.front(); E; E = E->next()) {

		SkinControllerData &cd = E->get();

		NodeSkeleton *skeleton = NULL;

		for (Map<String, Transform>::Element *F = cd.bone_rest_map.front(); F; F = F->next()) {

			String name;

			if (!state.sid_to_node_map.has(F->key())) {
				continue;
			}

			name = state.sid_to_node_map[F->key()];

			if (!state.scene_map.has(name)) {
				print_line("no foundie node for: " + name);
			}

			ERR_CONTINUE(!state.scene_map.has(name));

			Node *node = state.scene_map[name];
			ERR_CONTINUE(node->type != Node::TYPE_JOINT);
			if (node->type != Node::TYPE_JOINT)
				continue;
			NodeSkeleton *sk = NULL;

			while (node && !sk) {

				if (node->type == Node::TYPE_SKELETON) {
					sk = static_cast<NodeSkeleton *>(node);
				}
				node = node->parent;
			}
			ERR_CONTINUE(!sk);

			if (!sk)
				continue; //bleh

			if (!skeleton) {
				skeleton = sk;
				continue;
			}

			if (skeleton != sk) {
				//whoa.. wtf, merge.
				print_line("MERGED BONES!!");

				//NodeSkeleton *merged = E->get();
				_remove_node(p_vscene, sk);
				for (int i = 0; i < sk->children.size(); i++) {

					_joint_set_owner(sk->children[i], skeleton);
					skeleton->children.push_back(sk->children[i]);
					sk->children[i]->parent = skeleton;
				}

				sk->children.clear(); //take children from it
				memdelete(sk);
			}
		}
	}
}

bool Collada::_optimize_skeletons(VisualScene *p_vscene, Node *p_node) {

	Node *node = p_node;

	if (node->type == Node::TYPE_SKELETON && node->parent && node->parent->type == Node::TYPE_NODE && node->parent->children.size() == 1) {
		//replace parent by this...
		Node *parent = node->parent;

		//i wonder if this is allright.. i think it is since created skeleton (first joint) is already animated by bone..
		node->id = parent->id;
		node->name = parent->name;
		node->xform_list = parent->xform_list;
		node->default_transform = parent->default_transform;

		state.scene_map[node->id] = node;
		node->parent = parent->parent;

		if (parent->parent) {
			Node *gp = parent->parent;
			bool found = false;
			for (int i = 0; i < gp->children.size(); i++) {

				if (gp->children[i] == parent) {
					gp->children[i] = node;
					found = true;
					break;
				}
			}
			if (!found) {
				ERR_PRINT("BUG");
			}
		} else {

			bool found = false;

			for (int i = 0; i < p_vscene->root_nodes.size(); i++) {

				if (p_vscene->root_nodes[i] == parent) {

					p_vscene->root_nodes[i] = node;
					found = true;
					break;
				}
			}
			if (!found) {
				ERR_PRINT("BUG");
			}
		}

		parent->children.clear();
		memdelete(parent);
		return true;
	}

	for (int i = 0; i < node->children.size(); i++) {

		if (_optimize_skeletons(p_vscene, node->children[i]))
			return false; //stop processing, go up
	}

	return false;
}

bool Collada::_move_geometry_to_skeletons(VisualScene *p_vscene, Node *p_node, List<Node *> *p_mgeom) {

	// bind shape matrix escala los huesos y los hace gigantes, asi la matriz despues achica
	// al modelo?
	// solucion: aplicarle la bind shape matrix a los VERTICES, y si el objeto viene con escala, se la dejo me parece!

	if (p_node->type == Node::TYPE_GEOMETRY) {

		NodeGeometry *ng = static_cast<NodeGeometry *>(p_node);
		if (ng->ignore_anim)
			return false; //already made child of skeleton and processeg

		if (ng->controller && ng->skeletons.size()) {

			String nodeid = ng->skeletons[0];

			ERR_FAIL_COND_V(!state.scene_map.has(nodeid), false); //weird, it should have it...
			NodeJoint *nj = SAFE_CAST<NodeJoint *>(state.scene_map[nodeid]);
			ERR_FAIL_COND_V(!nj, false);
			if (!nj->owner) {
				print_line("Has no owner: " + nj->name);
			}
			ERR_FAIL_COND_V(!nj->owner, false); //weird, node should have a skeleton owner

			NodeSkeleton *sk = nj->owner;

			Node *p = sk->parent;
			bool node_is_parent_of_skeleton = false;

			while (p) {
				if (p == p_node) {
					node_is_parent_of_skeleton = true;
					break;
				}
				p = p->parent; // try again
			}

			ERR_FAIL_COND_V(node_is_parent_of_skeleton, false);

			//this should be correct
			ERR_FAIL_COND_V(!state.skin_controller_data_map.has(ng->source), false);
			SkinControllerData &skin = state.skin_controller_data_map[ng->source];
			Transform skel_inv = sk->get_global_transform().affine_inverse();
			p_node->default_transform = skel_inv * (skin.bind_shape /* p_node->get_global_transform()*/); // i honestly have no idea what to do with a previous model xform.. most exporters ignore it

			//make rests relative to the skeleton (they seem to be always relative to world)
			for (Map<String, Transform>::Element *E = skin.bone_rest_map.front(); E; E = E->next()) {

				E->get() = skel_inv * E->get(); //make the bone rest local to the skeleton
				state.bone_rest_map[E->key()] = E->get(); // make it remember where the bone is globally, now that it's relative
			}

			//but most exporters seem to work only if i do this..
			//p_node->default_transform = p_node->get_global_transform();

			//p_node->default_transform=Transform(); //this seems to be correct, because bind shape makes the object local to the skeleton
			p_node->ignore_anim = true; // collada may animate this later, if it does, then this is not supported (redo your original asset and don't animate the base mesh)
			p_node->parent = sk;
			//sk->children.push_back(0,p_node); //avoid INFINITE loop
			p_mgeom->push_back(p_node);
			return true;
		}
	}

	for (int i = 0; i < p_node->children.size(); i++) {

		if (_move_geometry_to_skeletons(p_vscene, p_node->children[i], p_mgeom)) {
			p_node->children.remove(i);
			i--;
		}
	}

	return false;
}

void Collada::_find_morph_nodes(VisualScene *p_vscene, Node *p_node) {

	if (p_node->type == Node::TYPE_GEOMETRY) {

		NodeGeometry *nj = static_cast<NodeGeometry *>(p_node);

		if (nj->controller) {

			String base = nj->source;

			while (base != "" && !state.mesh_data_map.has(base)) {

				if (state.skin_controller_data_map.has(base)) {

					SkinControllerData &sk = state.skin_controller_data_map[base];
					base = sk.base;
				} else if (state.morph_controller_data_map.has(base)) {

					state.morph_ownership_map[base] = nj->id;
					break;
				} else {
					ERR_EXPLAIN("Invalid scene");
					ERR_FAIL();
				}
			}
		}
	}

	for (int i = 0; i < p_node->children.size(); i++) {

		_find_morph_nodes(p_vscene, p_node->children[i]);
	}
}

void Collada::_optimize() {

	for (Map<String, VisualScene>::Element *E = state.visual_scene_map.front(); E; E = E->next()) {

		VisualScene &vs = E->get();
		for (int i = 0; i < vs.root_nodes.size(); i++) {
			_create_skeletons(&vs.root_nodes[i]);
		}
#if 1
		for (int i = 0; i < vs.root_nodes.size(); i++) {
			_merge_skeletons(&vs, vs.root_nodes[i]);
		}

		_merge_skeletons2(&vs);

		for (int i = 0; i < vs.root_nodes.size(); i++) {
			_optimize_skeletons(&vs, vs.root_nodes[i]);
		}

		for (int i = 0; i < vs.root_nodes.size(); i++) {

			List<Node *> mgeom;
			if (_move_geometry_to_skeletons(&vs, vs.root_nodes[i], &mgeom)) {
				vs.root_nodes.remove(i);
				i--;
			}

			while (!mgeom.empty()) {

				Node *n = mgeom.front()->get();
				n->parent->children.push_back(n);
				mgeom.pop_front();
			}
		}
#endif
		for (int i = 0; i < vs.root_nodes.size(); i++) {
			_find_morph_nodes(&vs, vs.root_nodes[i]);
		}
	}
}

int Collada::get_uv_channel(String p_name) {

	if (!channel_map.has(p_name)) {

		ERR_FAIL_COND_V(channel_map.size() == 2, 0);

		channel_map[p_name] = channel_map.size();
	}

	return channel_map[p_name];
}

Error Collada::load(const String &p_path, int p_flags) {

	Ref<XMLParser> parserr = memnew(XMLParser);
	XMLParser &parser = *parserr.ptr();
	Error err = parser.open(p_path);
	ERR_FAIL_COND_V(err, err);

	state.local_path = GlobalConfig::get_singleton()->localize_path(p_path);
	state.import_flags = p_flags;
	/* Skip headers */
	err = OK;
	while ((err = parser.read()) == OK) {

		if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {

			if (parser.get_node_name() == "COLLADA") {
				break;
			} else if (!parser.is_empty())
				parser.skip_section(); // unknown section, likely headers
		}
	}

	ERR_FAIL_COND_V(err != OK, ERR_FILE_CORRUPT);

	/* Start loading Collada */

	{
		//version
		String version = parser.get_attribute_value("version");
		state.version.major = version.get_slice(".", 0).to_int();
		state.version.minor = version.get_slice(".", 1).to_int();
		state.version.rev = version.get_slice(".", 2).to_int();
		COLLADA_PRINT("Collada VERSION: " + version);
	}

	while ((err = parser.read()) == OK) {

		/* Read all the main sections.. */

		if (parser.get_node_type() != XMLParser::NODE_ELEMENT)
			continue; //no idea what this may be, but skipping anyway

		String section = parser.get_node_name();

		COLLADA_PRINT("section: " + section);

		if (section == "asset") {
			_parse_asset(parser);

		} else if (section.begins_with("library_")) {

			_parse_library(parser);
		} else if (section == "scene") {

			_parse_scene(parser);
		} else if (!parser.is_empty()) {
			parser.skip_section(); // unknown section, likely headers
		}
	}

	_optimize();
	return OK;
}

Collada::Collada() {
}

#endif
