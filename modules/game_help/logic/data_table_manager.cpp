#include "data_table_manager.h"
#include "../csv/CSV_EditorImportPlugin.h"
#include "../unity/unity_link_server.h"
#include "../yaml/tree.hpp"
#include "../yaml/parse.hpp"
#include <map>
#include <regex>
#include <stdio.h>




std::map<Variant::Type, const char*> TYPE_NAMES =
		{
			{ Variant::VECTOR2, "Vector2" },
			{ Variant::RECT2, "Rect2" },
			{ Variant::VECTOR3, "Vector3" },
			{ Variant::TRANSFORM2D, "Transform2D" },
			{ Variant::PLANE, "Plane" },
			{ Variant::QUATERNION, "Quat" },
			{ Variant::AABB, "AABB" },
			{ Variant::BASIS, "Basis" },
			{ Variant::TRANSFORM3D, "Transform3D" },
			{ Variant::COLOR, "Color" },
			{ Variant::NODE_PATH, "NodePath" },
			{ Variant::PACKED_BYTE_ARRAY, "PoolByteArray" },
			{ Variant::PACKED_FLOAT32_ARRAY, "PoolRealArray" },
			{ Variant::PACKED_INT32_ARRAY, "PoolIntArray" },
			{ Variant::PACKED_STRING_ARRAY, "PoolStringArray" },
			{ Variant::PACKED_VECTOR2_ARRAY, "PoolVector2Array" },
			{ Variant::PACKED_VECTOR3_ARRAY, "PoolVector3Array" },
			{ Variant::PACKED_COLOR_ARRAY, "PoolColorArray" },
		};

namespace c4 {
namespace yml {

void encode_vector_2(yml::NodeRef *node, const Vector2 &vector_2) {
	*node |= yml::MAP;
	node->append_child() << yml::key("x") << vector_2.x;
	node->append_child() << yml::key("y") << vector_2.y;
}

Vector2 decode_vector_2(yml::ConstNodeRef const &node) {
	Vector2 vector_2;
	node["x"] >> vector_2.x;
	node["y"] >> vector_2.y;
	return vector_2;
}

void encode_vector_3(yml::NodeRef *node, const Vector3 &vector_3) {
	*node |= yml::MAP;
	node->append_child() << yml::key("x") << vector_3.x;
	node->append_child() << yml::key("y") << vector_3.y;
	node->append_child() << yml::key("z") << vector_3.z;
}

Vector3 decode_vector_3(yml::ConstNodeRef const &node) {
	Vector3 vector_3;
	node["x"] >> vector_3.x;
	node["y"] >> vector_3.y;
	node["z"] >> vector_3.z;
	return vector_3;
}

void encode_rect_2(yml::NodeRef *node, const Rect2 &rec_2) {
	*node |= yml::MAP;
	auto position = node->append_child();
	position |= yml::MAP;
	position << yml::key("position");
	position.append_child()	<< yml::key("x") << rec_2.position.x;
	position.append_child()	<< yml::key("y") << rec_2.position.y;
	auto size = node->append_child();
	size |= yml::MAP;
	size << yml::key("size");
	size.append_child()	<< yml::key("x") << rec_2.size.x;
	size.append_child()	<< yml::key("y") << rec_2.size.y;
}

Rect2 decode_rect_2(yml::ConstNodeRef const &node) {
	Rect2 rect;
	node["position"]["x"] >> rect.position.x;
	node["position"]["y"] >> rect.position.y;
	node["size"]["x"] >> rect.size.x;
	node["size"]["y"] >> rect.size.y;
	return rect;
}

void encode_aabb(yml::NodeRef *node, const AABB &aabb) {
	*node |= yml::MAP;
	auto position = node->append_child();
	position |= yml::MAP;
	position << yml::key("position");
    encode_vector_3(&position, aabb.position);
	auto size = node->append_child();
	size |= yml::MAP;
	size << yml::key("size");
    encode_vector_3(&size, aabb.size);
}

AABB decode_aabb(yml::ConstNodeRef const &node) {
	AABB aabb;
	aabb.position = decode_vector_3(node["position"]);
	aabb.size = decode_vector_3(node["size"]);
	return aabb;
}

void encode_basis(yml::NodeRef *node, const Basis &basis) {
	*node |= yml::MAP;
	auto axis = node->append_child();
	axis << yml::key("x");
    encode_vector_3(&axis, basis.rows[0]);

	axis = node->append_child();
	axis << yml::key("y");
    encode_vector_3(&axis, basis.rows[1]);

	axis = node->append_child();
	axis << yml::key("z");
    encode_vector_3(&axis, basis.rows[2]);
}

Basis decode_basis(yml::ConstNodeRef const &node) {
    Vector3 x = decode_vector_3(node["x"]);
    Vector3 y = decode_vector_3(node["y"]);
    Vector3 z = decode_vector_3(node["z"]);
    return {x, y, z};
}


void encode_transform(yml::NodeRef *node, const Transform3D &transform) {
	*node |= yml::MAP;
	auto basis = node->append_child();
	basis << yml::key("basis");
	encode_basis(&basis, transform.basis);
	auto origin = node->append_child();
	origin << yml::key("origin");
    encode_vector_3(&origin, transform.origin);
}

Transform3D decode_transform(yml::ConstNodeRef const &node) {
	Transform3D transform;
	transform.basis = decode_basis(node["basis"]);
	transform.origin = decode_vector_3(node["origin"]);
	return transform;
}

void encode_transform_2_d(yml::NodeRef *node, const Transform2D &transform_2_d) {
	*node |= yml::MAP;
	auto x_axis = node->append_child();
    x_axis << yml::key("x");
    encode_vector_2(&x_axis, transform_2_d.columns[0]);
	auto y_axis = node->append_child();
    y_axis << yml::key("y");
    encode_vector_2(&y_axis, transform_2_d.columns[1]);
	auto origin = node->append_child();
	origin << yml::key("origin");
    encode_vector_2(&origin, transform_2_d.columns[2]);
}

Transform2D decode_transform_2_d(yml::ConstNodeRef const &node) {
	Transform2D transform_2_d;
    transform_2_d.columns[0] = decode_vector_2(node["x"]);
    transform_2_d.columns[1] = decode_vector_2(node["y"]);
    transform_2_d.columns[2] = decode_vector_2(node["origin"]);
	return transform_2_d;
}

void encode_plane(yml::NodeRef *node, const Plane &plane) {
	*node |= yml::MAP;
	auto normal = node ->append_child();
	normal << yml::key("normal");
    encode_vector_3(&normal, plane.normal);
	auto distance = node ->append_child();
	distance << yml::key("d") << plane.d;
}

Plane decode_plane(yml::ConstNodeRef const &node) {
	Plane plane;
	plane.normal = decode_vector_3(node["normal"]);
	node["d"] >> plane.d;
	return plane;
}

void encode_quat(yml::NodeRef *node, const Quaternion &quat) {
	*node |= yml::MAP;
	node->append_child() << yml::key("x") << quat.x;
	node->append_child() <<  yml::key("y") << quat.y;
	node->append_child() <<  yml::key("z") << quat.z;
	node->append_child() << yml::key("w") << quat.w;
}

Quaternion decode_quat(yml::ConstNodeRef const &node) {
	Quaternion quat;
	node["x"] >> quat.x;
	node["y"] >> quat.y;
	node["z"] >> quat.z;
	node["w"] >> quat.w;
	return quat;
}

void encode_array(yml::NodeRef *node, const Array &arr) {
	*node |= yml::SEQ;
	for (int i = 0; i < arr.size(); ++i) {
		node->append_child() << arr[i];
	}
}

Array decode_array(yml::ConstNodeRef const &node) {
	Array array;
	for (auto child = node.begin(); child != node.end(); ++child) {
		Variant variant;
		*child >> variant;
		array.push_back(variant);
	}
	return array;
}

void encode_dictionary(yml::NodeRef *node, const Dictionary &dict) {
	Array keys = dict.keys();
	Array values = dict.values();
	*node |= yml::MAP;
	for (int i = 0; i < keys.size(); ++i) {
		Variant key = keys[i];
		Variant value = values[i];		
		csubstr key_string = to_csubstr(key.operator String().ascii().get_data());
		node->append_child() << yml::key(key_string) << value;
	}
}

Dictionary decode_dictionary(yml::ConstNodeRef const &node) {
	Dictionary dict;
	for (auto iterator = node.begin(); iterator != node.end(); ++iterator) {
		auto child_node = *iterator;

		auto key = String(std::string(child_node.key().data(), child_node.key().len).c_str());
		Variant value;
		node[child_node.key()] >> value;
		dict[key] = value;
	}
	return dict;
}

void encode_color(yml::NodeRef *node, const Color &color) {
	*node |= yml::MAP;
	node->append_child() << yml::key("r") << color.r;
	node->append_child() << yml::key("g") << color.g;
	node->append_child() << yml::key("b") << color.b;
	node->append_child() << yml::key("a") << color.a;
}

Color decode_color(yml::ConstNodeRef const &node) {
	Color color;
	node["r"] >> color.r;
	node["g"] >> color.g;
	node["b"] >> color.b;
	if (node.has_child("a")) {
		node["a"] >> color.a;
	}
	return color;
}

void write(yml::NodeRef *node, const Variant &variant) {
		Variant::Type var_type = variant.get_type();
		bool needs_tag = false;
		switch (var_type) {
			case Variant::NIL: {
				node->set_val(nullptr);
				break;
			}
			case Variant::VECTOR2: {
                needs_tag = true;
                encode_vector_2(node, variant);
				break;
			}
			case Variant::VECTOR3: {
                needs_tag = true;
                encode_vector_3(node, variant);
				break;
			}
			case Variant::PACKED_INT32_ARRAY:
			case Variant::PACKED_FLOAT32_ARRAY:
			case Variant::PACKED_STRING_ARRAY:
			case Variant::PACKED_VECTOR2_ARRAY:
			case Variant::PACKED_VECTOR3_ARRAY:
			case Variant::PACKED_COLOR_ARRAY:
			case Variant::PACKED_BYTE_ARRAY:
				// Pool arrays need a tag to correctly decode them as a pool.
                needs_tag = true;
			case Variant::ARRAY: {
				encode_array(node, variant);
				break;
			}
			case Variant::INT: {
				*node << variant.operator int();
				break;
			}
			case Variant::FLOAT: {
				*node << variant.operator double ();
				break;
			}
			case Variant::STRING: {
				*node |= VALQUO;
				String string = variant.operator String();
				csubstr str_val = to_csubstr(string.utf8().get_data());
				*node << str_val;
				break;
			}
			case Variant::BOOL: {
				if (variant.operator bool()) {
					*node << "true";
				} else {
					*node << "false";
				}
				break;
			}
			case Variant::DICTIONARY: {
				encode_dictionary(node, variant);
				break;
			}
			case Variant::RECT2: {
                encode_rect_2(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::AABB: {
				encode_aabb(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::TRANSFORM3D: {
				encode_transform(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::TRANSFORM2D: {
                encode_transform_2_d(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::PLANE: {
				encode_plane(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::QUATERNION: {
				encode_quat(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::BASIS: {
				encode_basis(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::COLOR: {
				encode_color(node, variant);
                needs_tag = true;
				break;
			}
			case Variant::NODE_PATH: {
				*node |= VALQUO;
				String string = variant.operator String();
				csubstr str_val = to_csubstr(string.utf8().get_data());
				*node << str_val;
                needs_tag = true;
				break;
			}
			default:
				*node << Variant(variant.operator String());
				break;
		}
		if (needs_tag) {
			auto buf = new char[256];
			sprintf_s(buf,256, "Godot/%s", TYPE_NAMES[var_type]);
			node->set_val_tag(to_csubstr(buf));
		}

	}

	////	 Tries to convert a node to a Godot Variant. There should be (almost?) no value that is not converted.

	bool read(yml::ConstNodeRef const& node, Variant *variant)
	{
		if (node.has_val_tag()) {
			auto tag = node.val_tag();
			auto tag_string = std::string(tag.data(), tag.len).substr(1);
			std::regex type_expression(R"((?:\/)?(\w+))");
			std::sregex_iterator pos(tag_string.begin(), tag_string.end(), type_expression);
			std::vector<std::string> tokens;
			std::sregex_iterator end;
			for (; pos != end; ++pos) {
				tokens.push_back(pos->str(1));
			}
			if (!tokens.empty()) {
				std::string token_godot = tokens[0];
				std::transform(token_godot.begin(), token_godot.end(), token_godot.begin(), ::tolower);
				if (token_godot == "godot" && tokens.size() == 2) {
					Variant::Type var_type;
					std::string type_value = tokens[1];
					std::transform(type_value.begin(), type_value.end(), type_value.begin(), ::tolower);
					bool found = false;
					for (auto it = TYPE_NAMES.begin(); it != TYPE_NAMES.end(); ++it) {
						std::string value = it->second;
						std::transform(value.begin(), value.end(), value.begin(), ::tolower);
						if (value == type_value) {
							var_type = it->first;
							found = true;
							break;
						}
					}
					if (!found) {
						var_type = static_cast<Variant::Type>(std::stoi(tokens[2]));
					}
					switch (var_type) {
						case Variant::NIL: {
//							Godot::print("Determined: Nil");
							*variant = Variant();
							break;
						}
						case Variant::VECTOR2: {
//							Godot::print("Determined: VECTOR2");
							*variant = decode_vector_2(node);
							break;
						}
						case Variant::VECTOR3: {
//							Godot::print("Determined: VECTOR3");
							*variant = decode_vector_3(node);
							break;
						}
						case Variant::PACKED_BYTE_ARRAY: {
//							Godot::print("Determined: POOL_BYTE_ARRAY");
							*variant = decode_array(node).get_vector<uint8_t>();
							break;
						}
						case Variant::PACKED_INT32_ARRAY: {
//							Godot::print("Determined: POOL_INT_ARRAY");
							*variant = decode_array(node).get_vector<int32_t>();
							break;
						}
						case Variant::PACKED_FLOAT32_ARRAY: {
//							Godot::print("Determined: POOL_REAL_ARRAY");
							*variant = decode_array(node).get_vector<float>();
							break;
						}
						case Variant::PACKED_STRING_ARRAY: {
//							Godot::print("Determined: POOL_STRING_ARRAY");
							*variant = decode_array(node).get_vector<String>();
							break;
						}
						case Variant::PACKED_VECTOR2_ARRAY: {
//							Godot::print("Determined: POOL_VECTOR2_ARRAY");
							*variant = decode_array(node).get_vector<Vector2>();
							break;
						}
						case Variant::PACKED_VECTOR3_ARRAY: {
//							Godot::print("Determined: POOL_VECTOR3_ARRAY");
							*variant = decode_array(node).get_vector<Vector3>();
							break;
						}
						case Variant::PACKED_COLOR_ARRAY: {
//							Godot::print("Determined: POOL_COLOR_ARRAY");
							*variant = decode_array(node).get_vector<Color>();
							break;
						}
						case Variant::INT: {
//							Godot::print("Determined: INT");
							int64_t value;
							node >> value;
							*variant = value;
							break;
						}
						case Variant::FLOAT: {
//							Godot::print("Determined: REAL");
							double value;
							node >> value;
							*variant = value;
							break;
						}
						case Variant::STRING: {
//							Godot::print("Determined: STRING");
							*variant = std::string(node.val().data(), node.val().len).c_str();
							break;
						}
						case Variant::RECT2: {
//							Godot::print("Determined: RECT2");
							*variant = decode_rect_2(node);
							break;
						}
						case Variant::AABB: {
//							Godot::print("Determined: RECT3");
							*variant = decode_aabb(node);
							break;
						}
						case Variant::TRANSFORM3D: {
//							Godot::print("Determined: TRANSFORM");
							*variant = decode_transform(node);
							break;
						}
						case Variant::TRANSFORM2D: {
//							Godot::print("Determined: TRANSFORM2D");
							*variant = decode_transform_2_d(node);
							break;
						}
						case Variant::PLANE: {
//							Godot::print("Determined: PLANE");
							*variant = decode_plane(node);
							break;
						}
						case Variant::QUATERNION: {
//							Godot::print("Determined: QUAT");
							*variant = decode_quat(node);
							break;
						}
						case Variant::BASIS: {
//							Godot::print("Determined: BASIS");
							*variant = decode_basis(node);
							break;
						}
						case Variant::COLOR: {
//							Godot::print("Determined: COLOR");
							*variant = decode_color(node);
							break;
						}
						case Variant::NODE_PATH: {
//							Godot::print("Determined: NODE_PATH");
							*variant = NodePath(std::string(node.val().data(), node.val().len).c_str());
							break;
						}
						default: {
							return false;
						}
					}
				}
			}
		} else {
//			Godot::print("No tag");
			if (node.is_seq()) {
//				Godot::print("Determined: Array");
				*variant = decode_array(node);
				return true;
			}
			if (node.is_map()) {
//				Godot::print("Determined: Dictionary");
				*variant = decode_dictionary(node);
				return true;
			}
			if (node.is_doc())
			{
				if(node.has_key())
					*variant = decode_dictionary(node);
				else
					*variant = decode_array(node);
				return true;
			}
			// Try to determine the type, first match will return, so order will matter.
            if (!node.is_val_quoted()) {
                if (node.val_is_null()) {
                    *variant = "";
//						Godot::print("Determined: Nil");
                    return true;
                }
                int64_t int_val;
                if (c4::atox(node.val(), &int_val)) {
//						Godot::print("Determined: INT");
                    *variant = int_val;
                    return true;
                }

                double float_val;
                if (c4::atod(node.val(), &float_val)) {
//						Godot::print("Determined: float");
                    *variant = float_val;
                    return true;
                }

                bool bool_val;
                if (from_chars(node.val(), &bool_val)) {
//						Godot::print("Determined: bool");
                    *variant = bool_val;
                    return true;
                }
            }

            *variant = std::string(node.val().data(), node.val().len).c_str();
            return true;
		}
		return true;
	}
} // namespace yaml
} // namespace c4











static DataTableManager* singleton = nullptr;
DataTableManager::DataTableManager() {
    singleton = this;
}

DataTableManager::~DataTableManager() {
    if(singleton == this)
    {
        singleton = nullptr;
    }
}
DataTableManager *DataTableManager::get_singleton() {
    return singleton;
}
void DataTableManager::init() {
    if(is_init)
    {
        return;
    }
    ++version;
    Ref<CSVData> db = ResourceLoader::load(data_table_path);
    if(!db.is_valid())
    {
        ERR_FAIL_MSG("data table not found:" + data_table_path);
    }
    Array data = db->get_data().values();

    HashMap<StringName,Ref<DataTableItem>> old_table = data_table;
    data_table.clear();
    for (int i = 0; i < data.size(); i++) {
        Dictionary d = data[i];
        if(d.has("name") && d.has("path")){
            String name = d["name"];
            String path = d["path"];
            Ref<CSVData> table = ResourceLoader::load(path);
            ERR_CONTINUE_MSG(!table.is_valid(),"data table not found:" + path);

            Ref<DataTableItem> item;
            if(old_table.has(name))
            {
                item = old_table[name];
            } else {
                item.instantiate();
            }
            item->version = version;
            item->data = table->get_data();
            data_table[name] = item;
        }
    }
    is_init = true;
}
Ref<JSON> DataTableManager::parse_yaml_file(const String& file_path)
{
    if(FileAccess::exists(file_path))
    {
	    Ref<FileAccess> f = FileAccess::open(file_path, FileAccess::READ);
        return parse_yaml(f->get_as_text());
    }
    return Ref<JSON>();
}
Ref<JSON> DataTableManager::parse_yaml(const String& text)
{
	// NOLINT(performance-unnecessary-value-param)
    auto parser = c4::yml::Parser();
    Ref<JSON> parseResult = memnew(JSON);
    std::string utf8 = text.utf8().get_data();
	int index = utf8.find("---");
	if (index >= 0)
	{
		for (int i = index; i < text.size(); ++i)
		{
			auto code = text[i];
			if (code == '\n')
			{
				break;
			}
			utf8[i] = ' ';
			
		}
		index = utf8.find("---");
	}
    auto text_string = c4::substr(&utf8[0], utf8.size());
    auto tree = parser.parse_in_place("", text_string);
    
        {
        Variant variant;
        tree.rootref() >> variant;
        parseResult->set_data(variant);
    }
    return parseResult;
}
void DataTableManager::set_animation_load_cb(const Callable& cb )
{
	print_line("DataTableManager::set_animation_load_cb");
	on_load_animation = cb;
}

void DataTableManager::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_data_table_path","path"),&DataTableManager::set_data_table_path);
    ClassDB::bind_method(D_METHOD("reload"),&DataTableManager::reload);
    ClassDB::bind_method(D_METHOD("parse_yaml","text"),&DataTableManager::parse_yaml);
    ClassDB::bind_method(D_METHOD("parse_yaml_file","file_path"),&DataTableManager::parse_yaml_file);
    ClassDB::bind_method(D_METHOD("set_animation_load_cb","callback"),&DataTableManager::set_animation_load_cb);



    ClassDB::bind_method(D_METHOD("get_data_table","name"),&DataTableManager::_get_data_table);
    ClassDB::bind_method(D_METHOD("get_data_item","name"),&DataTableManager::_get_data_item);

    ClassDB::bind_method(D_METHOD("set_animation_table_name","name"),&DataTableManager::set_animation_table_name);
    ClassDB::bind_method(D_METHOD("get_animation_table_name"),&DataTableManager::get_animation_table_name);

    ClassDB::bind_method(D_METHOD("set_body_table_name","name"),&DataTableManager::set_body_table_name);
    ClassDB::bind_method(D_METHOD("get_body_table_name"),&DataTableManager::get_body_table_name);

    ClassDB::bind_method(D_METHOD("set_path_table_name","name"),&DataTableManager::set_path_table_name);
    ClassDB::bind_method(D_METHOD("get_path_table_name"),&DataTableManager::get_path_table_name);

    ClassDB::bind_method(D_METHOD("set_mesh_part_table_name","name"),&DataTableManager::set_mesh_part_table_name);
    ClassDB::bind_method(D_METHOD("get_mesh_part_table_name"),&DataTableManager::get_mesh_part_table_name);

    ClassDB::bind_method(D_METHOD("set_charecter_table_name","name"),&DataTableManager::set_charecter_table_name);
    ClassDB::bind_method(D_METHOD("get_charecter_table_name"),&DataTableManager::get_charecter_table_name);

    ClassDB::bind_method(D_METHOD("set_scene_table_name","name"),&DataTableManager::set_scene_table_name);
    ClassDB::bind_method(D_METHOD("get_scene_table_name"),&DataTableManager::get_scene_table_name);

    ClassDB::bind_method(D_METHOD("set_item_table_name","name"),&DataTableManager::set_item_table_name);
    ClassDB::bind_method(D_METHOD("get_item_table_name"),&DataTableManager::get_item_table_name);

    
    ClassDB::bind_method(D_METHOD("set_skill_table_name","name"),&DataTableManager::set_skill_table_name);
    ClassDB::bind_method(D_METHOD("get_skill_table_name"),&DataTableManager::get_skill_table_name);


    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"animation_table_name"), "set_animation_table_name","get_animation_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"body_table_name"), "set_body_table_name","get_body_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"path_table_name"), "set_path_table_name","get_path_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"mesh_part_table_name"), "set_mesh_part_table_name","get_mesh_part_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"charecter_table_name"), "set_charecter_table_name","get_charecter_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"scene_table_name"), "set_scene_table_name","get_scene_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"item_table_name"), "set_item_table_name","get_item_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"skill_table_name"), "set_skill_table_name","get_skill_table_name");

}
