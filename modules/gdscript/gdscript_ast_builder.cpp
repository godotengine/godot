#include "modules/gdscript/gdscript_ast_builder.h"

GDScriptASTBuilder::GDScriptASTBuilder() {
	m_writer = new rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF16<wchar_t> >(m_s);
}

GDScriptASTBuilder::~GDScriptASTBuilder() {
	delete m_writer;
}

bool GDScriptASTBuilder::build(const String& code) {
		Error err = m_parser.parse(code);

		if (err) {
			m_error = "Parse Error:\n" +
				itos(m_parser.get_error_line()) +
				":" +
				itos(m_parser.get_error_column()) +
				":" +
				m_parser.get_error();
			return false;
		}
		serialize(m_parser.get_parse_tree());
		return true;
	}

void GDScriptASTBuilder::serialize(real_t val) {
	if(Math::is_nan(val)) {
		m_writer->String(L"NaN");
	} else {
		m_writer->Double(val);
	}
}

void GDScriptASTBuilder::serialize(GDScriptParser::OperatorNode::Operator op) {
		switch(op) {
			case GDScriptParser::OperatorNode::OP_CALL:
				serialize(L"OP_CALL");
				break;
			case GDScriptParser::OperatorNode::OP_PARENT_CALL:
				serialize(L"OP_PARENT_CALL");
				break;
			case GDScriptParser::OperatorNode::OP_YIELD:
				serialize(L"OP_YIELD");
				break;
			case GDScriptParser::OperatorNode::OP_IS:
				serialize(L"OP_IS");
				break;
			case GDScriptParser::OperatorNode::OP_IS_BUILTIN:
				serialize(L"OP_IS_BUILTIN");
				break;
			case GDScriptParser::OperatorNode::OP_INDEX:
				serialize(L"OP_INDEX");
				break;
			case GDScriptParser::OperatorNode::OP_INDEX_NAMED:
				serialize(L"OP_INDEX_NAMED");
				break;
			case GDScriptParser::OperatorNode::OP_NEG:
				serialize(L"OP_NEG");
				break;
			case GDScriptParser::OperatorNode::OP_POS:
				serialize(L"OP_POS");
				break;
			case GDScriptParser::OperatorNode::OP_NOT:
				serialize(L"OP_NOT");
				break;
			case GDScriptParser::OperatorNode::OP_BIT_INVERT:
				serialize(L"OP_BIT_INVERT");
				break;
			case GDScriptParser::OperatorNode::OP_IN:
				serialize(L"OP_IN");
				break;
			case GDScriptParser::OperatorNode::OP_EQUAL:
				serialize(L"OP_EQUAL");
				break;
			case GDScriptParser::OperatorNode::OP_NOT_EQUAL:
				serialize(L"OP_NOT_EQUAL");
				break;
			case GDScriptParser::OperatorNode::OP_LESS:
				serialize(L"OP_LESS");
				break;
			case GDScriptParser::OperatorNode::OP_LESS_EQUAL:
				serialize(L"OP_LESS_EQUAL");
				break;
			case GDScriptParser::OperatorNode::OP_GREATER:
				serialize(L"OP_GREATER");
				break;
			case GDScriptParser::OperatorNode::OP_GREATER_EQUAL:
				serialize(L"OP_GREATER_EQUAL");
				break;
			case GDScriptParser::OperatorNode::OP_AND:
				serialize(L"OP_AND");
				break;
			case GDScriptParser::OperatorNode::OP_OR:
				serialize(L"OP_OR");
				break;
			case GDScriptParser::OperatorNode::OP_ADD:
				serialize(L"OP_ADD");
				break;
			case GDScriptParser::OperatorNode::OP_SUB:
				serialize(L"OP_SUB");
				break;
			case GDScriptParser::OperatorNode::OP_MUL:
				serialize(L"OP_MUL");
				break;
			case GDScriptParser::OperatorNode::OP_DIV:
				serialize(L"OP_DIV");
				break;
			case GDScriptParser::OperatorNode::OP_MOD:
				serialize(L"OP_MOD");
				break;
			case GDScriptParser::OperatorNode::OP_SHIFT_LEFT:
				serialize(L"OP_SHIFT_LEFT");
				break;
			case GDScriptParser::OperatorNode::OP_SHIFT_RIGHT:
				serialize(L"OP_SHIFT_RIGHT");
				break;
			case GDScriptParser::OperatorNode::OP_INIT_ASSIGN:
				serialize(L"OP_INIT_ASSIGN");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN:
				serialize(L"OP_ASSIGN");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_ADD:
				serialize(L"OP_ASSIGN_ADD");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_SUB:
				serialize(L"OP_ASSIGN_SUB");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_MUL:
				serialize(L"OP_ASSIGN_MUL");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_DIV:
				serialize(L"OP_ASSIGN_DIV");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_MOD:
				serialize(L"OP_ASSIGN_MOD");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_SHIFT_LEFT:
				serialize(L"OP_ASSIGN_SHIFT_LEFT");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_SHIFT_RIGHT:
				serialize(L"OP_ASSIGN_SHIFT_RIGHT");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_BIT_AND:
				serialize(L"OP_ASSIGN_BIT_AND");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_BIT_OR:
				serialize(L"OP_ASSIGN_BIT_OR");
				break;
			case GDScriptParser::OperatorNode::OP_ASSIGN_BIT_XOR:
				serialize(L"OP_ASSIGN_BIT_XOR");
				break;
			case GDScriptParser::OperatorNode::OP_BIT_AND:
				serialize(L"OP_BIT_AND");
				break;
			case GDScriptParser::OperatorNode::OP_BIT_OR:
				serialize(L"OP_BIT_OR");
				break;
			case GDScriptParser::OperatorNode::OP_BIT_XOR:
				serialize(L"OP_BIT_XOR");
				break;
			case GDScriptParser::OperatorNode::OP_TERNARY_IF:
				serialize(L"OP_TERNARY_IF");
				break;
			case GDScriptParser::OperatorNode::OP_TERNARY_ELSE:
				serialize(L"OP_TERNARY_ELSE");
				break;
			default:
				assert(false);
		}
	}

void GDScriptASTBuilder::serialize(GDScriptFunctions::Function func)
 {
		switch(func) {
		case GDScriptFunctions::MATH_SIN:
			serialize(L"MATH_SIN");
			break;
		case GDScriptFunctions::MATH_COS:
			serialize(L"MATH_COS");
			break;
		case GDScriptFunctions::MATH_TAN:
			serialize(L"MATH_TAN");
			break;
		case GDScriptFunctions::MATH_SINH:
			serialize(L"MATH_SINH");
			break;
		case GDScriptFunctions::MATH_COSH:
			serialize(L"MATH_COSH");
			break;
		case GDScriptFunctions::MATH_TANH:
			serialize(L"MATH_TANH");
			break;
		case GDScriptFunctions::MATH_ASIN:
			serialize(L"MATH_ASIN");
			break;
		case GDScriptFunctions::MATH_ACOS:
			serialize(L"MATH_ACOS");
			break;
		case GDScriptFunctions::MATH_ATAN:
			serialize(L"MATH_ATAN");
			break;
		case GDScriptFunctions::MATH_ATAN2:
			serialize(L"MATH_ATAN2");
			break;
		case GDScriptFunctions::MATH_SQRT:
			serialize(L"MATH_SQRT");
			break;
		case GDScriptFunctions::MATH_FMOD:
			serialize(L"MATH_FMOD");
			break;
		case GDScriptFunctions::MATH_FPOSMOD:
			serialize(L"MATH_FPOSMOD");
			break;
		case GDScriptFunctions::MATH_FLOOR:
			serialize(L"MATH_FLOOR");
			break;
		case GDScriptFunctions::MATH_CEIL:
			serialize(L"MATH_CEIL");
			break;
		case GDScriptFunctions::MATH_ROUND:
			serialize(L"MATH_ROUND");
			break;
		case GDScriptFunctions::MATH_ABS:
			serialize(L"MATH_ABS");
			break;
		case GDScriptFunctions::MATH_SIGN:
			serialize(L"MATH_SIGN");
			break;
		case GDScriptFunctions::MATH_POW:
			serialize(L"MATH_POW");
			break;
		case GDScriptFunctions::MATH_LOG:
			serialize(L"MATH_LOG");
			break;
		case GDScriptFunctions::MATH_EXP:
			serialize(L"MATH_EXP");
			break;
		case GDScriptFunctions::MATH_ISNAN:
			serialize(L"MATH_ISNAN");
			break;
		case GDScriptFunctions::MATH_ISINF:
			serialize(L"MATH_ISINF");
			break;
		case GDScriptFunctions::MATH_EASE:
			serialize(L"MATH_EASE");
			break;
		case GDScriptFunctions::MATH_DECIMALS:
			serialize(L"MATH_DECIMALS");
			break;
		case GDScriptFunctions::MATH_STEPIFY:
			serialize(L"MATH_STEPIFY");
			break;
		case GDScriptFunctions::MATH_LERP:
			serialize(L"MATH_LERP");
			break;
		case GDScriptFunctions::MATH_INVERSE_LERP:
			serialize(L"MATH_INVERSE_LERP");
			break;
		case GDScriptFunctions::MATH_RANGE_LERP:
			serialize(L"MATH_RANGE_LERP");
			break;
		case GDScriptFunctions::MATH_DECTIME:
			serialize(L"MATH_DECTIME");
			break;
		case GDScriptFunctions::MATH_RANDOMIZE:
			serialize(L"MATH_RANDOMIZE");
			break;
		case GDScriptFunctions::MATH_RAND:
			serialize(L"MATH_RAND");
			break;
		case GDScriptFunctions::MATH_RANDF:
			serialize(L"MATH_RANDF");
			break;
		case GDScriptFunctions::MATH_RANDOM:
			serialize(L"MATH_RANDOM");
			break;
		case GDScriptFunctions::MATH_SEED:
			serialize(L"MATH_SEED");
			break;
		case GDScriptFunctions::MATH_RANDSEED:
			serialize(L"MATH_RANDSEED");
			break;
		case GDScriptFunctions::MATH_DEG2RAD:
			serialize(L"MATH_DEG2RAD");
			break;
		case GDScriptFunctions::MATH_RAD2DEG:
			serialize(L"MATH_RAD2DEG");
			break;
		case GDScriptFunctions::MATH_LINEAR2DB:
			serialize(L"MATH_LINEAR2DB");
			break;
		case GDScriptFunctions::MATH_DB2LINEAR:
			serialize(L"MATH_DB2LINEAR");
			break;
		case GDScriptFunctions::MATH_POLAR2CARTESIAN:
			serialize(L"MATH_POLAR2CARTESIAN");
			break;
		case GDScriptFunctions::MATH_CARTESIAN2POLAR:
			serialize(L"MATH_CARTESIAN2POLAR");
			break;
		case GDScriptFunctions::MATH_WRAP:
			serialize(L"MATH_WRAP");
			break;
		case GDScriptFunctions::MATH_WRAPF:
			serialize(L"MATH_WRAPF");
			break;
		case GDScriptFunctions::LOGIC_MAX:
			serialize(L"LOGIC_MAX");
			break;
		case GDScriptFunctions::LOGIC_MIN:
			serialize(L"LOGIC_MIN");
			break;
		case GDScriptFunctions::LOGIC_CLAMP:
			serialize(L"LOGIC_CLAMP");
			break;
		case GDScriptFunctions::LOGIC_NEAREST_PO2:
			serialize(L"LOGIC_NEAREST_PO2");
			break;
		case GDScriptFunctions::OBJ_WEAKREF:
			serialize(L"OBJ_WEAKREF");
			break;
		case GDScriptFunctions::FUNC_FUNCREF:
			serialize(L"FUNC_FUNCREF");
			break;
		case GDScriptFunctions::TYPE_CONVERT:
			serialize(L"TYPE_CONVERT");
			break;
		case GDScriptFunctions::TYPE_OF:
			serialize(L"TYPE_OF");
			break;
		case GDScriptFunctions::TYPE_EXISTS:
			serialize(L"TYPE_EXISTS");
			break;
		case GDScriptFunctions::TEXT_CHAR:
			serialize(L"TEXT_CHAR");
			break;
		case GDScriptFunctions::TEXT_STR:
			serialize(L"TEXT_STR");
			break;
		case GDScriptFunctions::TEXT_PRINT:
			serialize(L"TEXT_PRINT");
			break;
		case GDScriptFunctions::TEXT_PRINT_TABBED:
			serialize(L"TEXT_PRINT_TABBED");
			break;
		case GDScriptFunctions::TEXT_PRINT_SPACED:
			serialize(L"TEXT_PRINT_SPACED");
			break;
		case GDScriptFunctions::TEXT_PRINTERR:
			serialize(L"TEXT_PRINTERR");
			break;
		case GDScriptFunctions::TEXT_PRINTRAW:
			serialize(L"TEXT_PRINTRAW");
			break;
		case GDScriptFunctions::TEXT_PRINT_DEBUG:
			serialize(L"TEXT_PRINT_DEBUG");
			break;
		case GDScriptFunctions::PUSH_ERROR:
			serialize(L"PUSH_ERROR");
			break;
		case GDScriptFunctions::PUSH_WARNING:
			serialize(L"PUSH_WARNING");
			break;
		case GDScriptFunctions::VAR_TO_STR:
			serialize(L"VAR_TO_STR");
			break;
		case GDScriptFunctions::STR_TO_VAR:
			serialize(L"STR_TO_VAR");
			break;
		case GDScriptFunctions::VAR_TO_BYTES:
			serialize(L"VAR_TO_BYTES");
			break;
		case GDScriptFunctions::BYTES_TO_VAR:
			serialize(L"BYTES_TO_VAR");
			break;
		case GDScriptFunctions::GEN_RANGE:
			serialize(L"GEN_RANGE");
			break;
		case GDScriptFunctions::RESOURCE_LOAD:
			serialize(L"RESOURCE_LOAD");
			break;
		case GDScriptFunctions::INST2DICT:
			serialize(L"INST2DICT");
			break;
		case GDScriptFunctions::DICT2INST:
			serialize(L"DICT2INST");
			break;
		case GDScriptFunctions::VALIDATE_JSON:
			serialize(L"VALIDATE_JSON");
			break;
		case GDScriptFunctions::PARSE_JSON:
			serialize(L"PARSE_JSON");
			break;
		case GDScriptFunctions::TO_JSON:
			serialize(L"TO_JSON");
			break;
		case GDScriptFunctions::HASH:
			serialize(L"HASH");
			break;
		case GDScriptFunctions::COLOR8:
			serialize(L"COLOR8");
			break;
		case GDScriptFunctions::COLORN:
			serialize(L"COLORN");
			break;
		case GDScriptFunctions::PRINT_STACK:
			serialize(L"PRINT_STACK");
			break;
		case GDScriptFunctions::GET_STACK:
			serialize(L"GET_STACK");
			break;
		case GDScriptFunctions::INSTANCE_FROM_ID:
			serialize(L"INSTANCE_FROM_ID");
			break;
		case GDScriptFunctions::LEN:
			serialize(L"LEN");
			break;
		case GDScriptFunctions::IS_INSTANCE_VALID:
			serialize(L"IS_INSTANCE_VALID");
			break;
		default:
			assert(false);
		}
	}

void GDScriptASTBuilder::serialize(Variant::Type type) {
		switch(type) {
			case Variant::NIL:
				serialize(L"NIL");
				break;
			case Variant::BOOL:
				serialize(L"BOOL");
				break;
			case Variant::INT:
				serialize(L"INT");
				break;
			case Variant::REAL:
				serialize(L"REAL");
				break;
			case Variant::STRING:
				serialize(L"STRING");
				break;
			case Variant::VECTOR2:
				serialize(L"VECTOR2");
				break;
			case Variant::RECT2:
				serialize(L"RECT2");
				break;
			case Variant::VECTOR3:
				serialize(L"VECTOR3");
				break;
			case Variant::TRANSFORM2D:
				serialize(L"TRANSFORM2D");
				break;
			case Variant::PLANE:
				serialize(L"PLANE");
				break;
			case Variant::QUAT:
				serialize(L"QUAT");
				break;
			case Variant::AABB:
				serialize(L"AABB");
				break;
			case Variant::BASIS:
				serialize(L"BASIS");
				break;
			case Variant::TRANSFORM:
				serialize(L"TRANSFORM");
				break;
			case Variant::COLOR:
				serialize(L"COLOR");
				break;
			case Variant::NODE_PATH:
				serialize(L"NODE_PATH");
				break;
			case Variant::_RID:
				serialize(L"_RID");
				break;
			case Variant::OBJECT:
				serialize(L"OBJECT");
				break;
			case Variant::DICTIONARY:
				serialize(L"DICTIONARY");
				break;
			case Variant::ARRAY:
				serialize(L"ARRAY");
				break;
			case Variant::POOL_BYTE_ARRAY:
				serialize(L"POOL_BYTE_ARRAY");
				break;
			case Variant::POOL_INT_ARRAY:
				serialize(L"POOL_INT_ARRAY");
				break;
			case Variant::POOL_REAL_ARRAY:
				serialize(L"POOL_REAL_ARRAY");
				break;
			case Variant::POOL_STRING_ARRAY:
				serialize(L"POOL_STRING_ARRAY");
				break;
			case Variant::POOL_VECTOR2_ARRAY:
				serialize(L"POOL_VECTOR2_ARRAY");
				break;
			case Variant::POOL_VECTOR3_ARRAY:
				serialize(L"POOL_VECTOR3_ARRAY");
				break;
			case Variant::POOL_COLOR_ARRAY:
				serialize(L"POOL_COLOR_ARRAY");
				break;
			default:
				assert(false);
		}
	}

	// TODO: proper Object serialization
	void GDScriptASTBuilder::serialize(const Object* o) {
		startObject();

		key("class_name");
		serialize(o->get_class_name());

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Vector2& vector2) {
		startObject();

		key("x");
		serialize(vector2.x);

		key("y");
		serialize(vector2.y);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Vector3& vector3) {
		startObject();

		key("x");
		serialize(vector3.x);

		key("y");
		serialize(vector3.y);

		key("z");
		serialize(vector3.z);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Rect2& rect2) {
		startObject();

		key("position");
		serialize(rect2.position);

		key("size");
		serialize(rect2.size);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const AABB& rect2) {
		startObject();

		key("position");
		serialize(rect2.position);

		key("size");
		serialize(rect2.size);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Transform2D& t2) {
		startObject();

		key("elements");
		startArray();
		for(int i = 0; i < 3; i++)
			serialize(t2.elements[i]);

		endArray();

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Plane& p) {
		startObject();

		key("normal");
		serialize(p.normal);

		key("d");
		serialize(p.d);

		endObject();
	}

	void GDScriptASTBuilder::serialize_quat(const Quat& p) {
		startObject();

		key("x");
		serialize(p.x);

		key("y");
		serialize(p.y);

		key("z");
		serialize(p.z);

		key("w");
		serialize(p.z);

		endObject();
	}

	void GDScriptASTBuilder::serialize_basis(const Basis& b) {
		startObject();

		key("elements");
		startArray();
		for(int i = 0; i < 3; i++)
			serialize(b.elements[i]);

		endObject();
	}

	void GDScriptASTBuilder::serialize_transform(const Transform& t) {
		startObject();

		key("basis");
		serialize_basis(t.basis);

		key("origin");
		serialize(Vector3(t.origin));

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Color& c) {
		startObject();

		key("r");
		serialize(c.r);

		key("g");
		serialize(c.g);

		key("b");
		serialize(c.b);

		key("a");
		serialize(c.a);

		endObject();
	}

	void GDScriptASTBuilder::serialize_nodepath(const NodePath& n) {
		startObject();

		key("names");
		serialize(n.get_names());

		key("subnames");
		serialize(n.get_subnames());

		endObject();
	}

	void GDScriptASTBuilder::serialize(const RID& r) {
		startObject();

		key("id");
		serialize((int)r.get_id());

		endObject();
	}

	void GDScriptASTBuilder::serialize(const Dictionary& d) {
		startArray();
		for(int i = 0; i < d.keys().size(); i++) {
			startObject();

			key(L"key");
			serialize(d.keys()[i]);

			key(L"value");
			serialize(d.values()[i]);

			endObject();
		}
		endArray();
	}

	void GDScriptASTBuilder::serialize(const Array& a) {
		startArray();
		for(int i = 0; i < a.size(); i++) {
			serialize(a[i]);
		}
		endArray();
	}

	void GDScriptASTBuilder::serialize(const Variant& variant) {
		startObject();

		key(L"type");
		serialize(variant.get_type());

		key(L"value");
		switch(variant.get_type()) {
			case Variant::NIL:
				serialize(L"NIL");
				break;
			case Variant::BOOL:
				serialize(bool(variant));
				break;
			case Variant::INT:
				serialize(int(variant));
				break;
			case Variant::REAL:
				serialize(real_t(variant));
				break;
			case Variant::STRING:
				serialize(String(variant));
				break;
			case Variant::VECTOR2:
				serialize(Vector2(variant));
				break;
			case Variant::RECT2:
				serialize(Rect2(variant));
				break;
			case Variant::VECTOR3:
				serialize(Vector3(variant));
				break;
			case Variant::TRANSFORM2D:
				serialize(Transform2D(variant));
				break;
			case Variant::PLANE:
				serialize(Plane(variant));
				break;
			case Variant::QUAT:
				serialize_quat(variant);
				break;
			case Variant::AABB:
				serialize(AABB(variant));
				break;
			case Variant::BASIS:
				serialize_basis(variant);
				break;
			case Variant::TRANSFORM:
				serialize_transform(variant);
				break;
			case Variant::COLOR:
				serialize(Color(variant));
				break;
			case Variant::NODE_PATH:
				serialize_nodepath(variant);
				break;
			case Variant::_RID:
				serialize(RID(variant));
				break;
			case Variant::OBJECT:
				serialize(static_cast<Object*>(variant));
				break;
			case Variant::DICTIONARY:
				serialize(Dictionary(variant));
				break;
			case Variant::ARRAY:
				serialize(Array(variant));
				break;
			case Variant::POOL_BYTE_ARRAY:
				serialize(PoolVector<uint8_t>(variant));
				break;
			case Variant::POOL_INT_ARRAY:
				serialize(PoolVector<int>(variant));
				break;
			case Variant::POOL_REAL_ARRAY:
				serialize(PoolVector<real_t>(variant));
				break;
			case Variant::POOL_STRING_ARRAY:
				serialize(PoolVector<String>(variant));
				break;
			case Variant::POOL_VECTOR2_ARRAY:
				serialize(PoolVector<Vector2>(variant));
				break;
			case Variant::POOL_VECTOR3_ARRAY:
				serialize(PoolVector<Vector3>(variant));
				break;
			case Variant::POOL_COLOR_ARRAY:
				serialize(PoolVector<Color>(variant));
				break;
			default:
				serialize("unsupported Variant type: " + itos((int)variant.get_type()));
		}
		endObject();
	}

	void GDScriptASTBuilder::serialize(PropertyHint hint) {
		switch(hint) {
			case ::PROPERTY_HINT_NONE:
				serialize(L"PROPERTY_HINT_NONE");
				break;
			case ::PROPERTY_HINT_RANGE:
				serialize(L"PROPERTY_HINT_RANGE");
				break;
			case ::PROPERTY_HINT_EXP_RANGE:
				serialize(L"PROPERTY_HINT_EXP_RANGE");
				break;
			case ::PROPERTY_HINT_ENUM:
				serialize(L"PROPERTY_HINT_ENUM");
				break;
			case ::PROPERTY_HINT_EXP_EASING:
				serialize(L"PROPERTY_HINT_EXP_EASING");
				break;
			case ::PROPERTY_HINT_LENGTH:
				serialize(L"PROPERTY_HINT_LENGTH");
				break;
			case ::PROPERTY_HINT_SPRITE_FRAME:
				serialize(L"PROPERTY_HINT_SPRITE_FRAME");
				break;
			case ::PROPERTY_HINT_KEY_ACCEL:
				serialize(L"PROPERTY_HINT_KEY_ACCEL");
				break;
			case ::PROPERTY_HINT_FLAGS:
				serialize(L"PROPERTY_HINT_FLAGS");
				break;
			case ::PROPERTY_HINT_LAYERS_2D_RENDER:
				serialize(L"PROPERTY_HINT_LAYERS_2D_RENDER");
				break;
			case ::PROPERTY_HINT_LAYERS_2D_PHYSICS:
				serialize(L"PROPERTY_HINT_LAYERS_2D_PHYSICS");
				break;
			case ::PROPERTY_HINT_LAYERS_3D_RENDER:
				serialize(L"PROPERTY_HINT_LAYERS_3D_RENDER");
				break;
			case ::PROPERTY_HINT_LAYERS_3D_PHYSICS:
				serialize(L"PROPERTY_HINT_LAYERS_3D_PHYSICS");
				break;
			case ::PROPERTY_HINT_FILE:
				serialize(L"PROPERTY_HINT_FILE");
				break;
			case ::PROPERTY_HINT_DIR:
				serialize(L"PROPERTY_HINT_DIR");
				break;
			case ::PROPERTY_HINT_GLOBAL_FILE:
				serialize(L"PROPERTY_HINT_GLOBAL_FILE");
				break;
			case ::PROPERTY_HINT_GLOBAL_DIR:
				serialize(L"PROPERTY_HINT_GLOBAL_DIR");
				break;
			case ::PROPERTY_HINT_RESOURCE_TYPE:
				serialize(L"PROPERTY_HINT_RESOURCE_TYPE");
				break;
			case ::PROPERTY_HINT_MULTILINE_TEXT:
				serialize(L"PROPERTY_HINT_MULTILINE_TEXT");
				break;
			case ::PROPERTY_HINT_PLACEHOLDER_TEXT:
				serialize(L"PROPERTY_HINT_PLACEHOLDER_TEXT");
				break;
			case ::PROPERTY_HINT_COLOR_NO_ALPHA:
				serialize(L"PROPERTY_HINT_COLOR_NO_ALPHA");
				break;
			case ::PROPERTY_HINT_IMAGE_COMPRESS_LOSSY:
				serialize(L"PROPERTY_HINT_IMAGE_COMPRESS_LOSSY");
				break;
			case ::PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS:
				serialize(L"PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS");
				break;
			case ::PROPERTY_HINT_OBJECT_ID:
				serialize(L"PROPERTY_HINT_OBJECT_ID");
				break;
			case ::PROPERTY_HINT_TYPE_STRING:
				serialize(L"PROPERTY_HINT_TYPE_STRING");
				break;
			case ::PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE:
				serialize(L"PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE");
				break;
			case ::PROPERTY_HINT_METHOD_OF_VARIANT_TYPE:
				serialize(L"PROPERTY_HINT_METHOD_OF_VARIANT_TYPE");
				break;
			case ::PROPERTY_HINT_METHOD_OF_BASE_TYPE:
				serialize(L"PROPERTY_HINT_METHOD_OF_BASE_TYPE");
				break;
			case ::PROPERTY_HINT_METHOD_OF_INSTANCE:
				serialize(L"PROPERTY_HINT_METHOD_OF_INSTANCE");
				break;
			case ::PROPERTY_HINT_METHOD_OF_SCRIPT:
				serialize(L"PROPERTY_HINT_METHOD_OF_SCRIPT");
				break;
			case ::PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE:
				serialize(L"PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE");
				break;
			case ::PROPERTY_HINT_PROPERTY_OF_BASE_TYPE:
				serialize(L"PROPERTY_HINT_PROPERTY_OF_BASE_TYPE");
				break;
			case ::PROPERTY_HINT_PROPERTY_OF_INSTANCE:
				serialize(L"PROPERTY_HINT_PROPERTY_OF_INSTANCE");
				break;
			case ::PROPERTY_HINT_PROPERTY_OF_SCRIPT:
				serialize(L"PROPERTY_HINT_PROPERTY_OF_SCRIPT");
				break;
			case ::PROPERTY_HINT_OBJECT_TOO_BIG:
				serialize(L"PROPERTY_HINT_OBJECT_TOO_BIG");
				break;
			case ::PROPERTY_HINT_NODE_PATH_VALID_TYPES:
				serialize(L"PROPERTY_HINT_NODE_PATH_VALID_TYPES");
				break;
			default:
				assert(false);
		}
	}

	void GDScriptASTBuilder::serialize(const PropertyInfo& pi) {
		startObject();

		key(L"type");
		serialize(pi.type);

		key(L"name");
		serialize(pi.name);

		key(L"class_name");
		serialize(pi.class_name);

		key(L"property_hint");
		serialize(pi.hint);

		key(L"hint_string");
		serialize(pi.hint_string);

		key(L"usage");
		serialize((int)pi.usage);

		endObject();
	}

	void GDScriptASTBuilder::serialize(MultiplayerAPI::RPCMode mode) {
		switch(mode) {
			case MultiplayerAPI::RPC_MODE_DISABLED:
				serialize(L"RPC_MODE_DISABLED");
				break;
			case MultiplayerAPI::RPC_MODE_REMOTE:
				serialize(L"RPC_MODE_REMOTE");
				break;
			case MultiplayerAPI::RPC_MODE_MASTER:
				serialize(L"RPC_MODE_MASTER");
				break;
			case MultiplayerAPI::RPC_MODE_PUPPET:
				serialize(L"RPC_MODE_PUPPET");
				break;
			case MultiplayerAPI::RPC_MODE_REMOTESYNC:
				serialize(L"RPC_MODE_REMOTESYNC");
				break;
			case MultiplayerAPI::RPC_MODE_MASTERSYNC:
				serialize(L"RPC_MODE_MASTERSYNC");
				break;
			case MultiplayerAPI::RPC_MODE_PUPPETSYNC:
				serialize(L"RPC_MODE_PUPPETSYNC");
				break;
			default:
				assert(false);
		}
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ClassNode::Member& member) {
		startObject();

		key(L"_export");
		serialize(member._export);

		key(L"default_value");
		serialize(member.default_value);

		key(L"identifier");
		serialize(member.identifier);

		key(L"data_type");
		serialize(member.data_type);

		key(L"getter");
		serialize(member.getter);

		key(L"setter");
		serialize(member.setter);

		key(L"line");
		serialize(member.line);

		key(L"expression");
		serialize(member.expression);

		key(L"initial_assignment");
		serialize(member.initial_assignment);

		key(L"rpc_mode");
		serialize(member.rpc_mode);

		key(L"onready");
		serialize(member.onready);

		key(L"usages");
		serialize(member.usages);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ClassNode::Constant& constant) {
		startObject();

		key(L"data_type");
		serialize(constant.type);

		key(L"expression");
		serialize(constant.expression);

		key(L"is_enum");
		serialize(constant.is_enum);

		key(L"line");
		serialize(constant.line);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ClassNode::Signal& signal) {
		startObject();

		key(L"name");
		serialize(signal.name);

		key(L"arguments");
		serialize(signal.arguments);

		key(L"emissions");
		serialize(signal.emissions);

		key(L"line");
		serialize(signal.line);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::DataType& data_type) {
		startObject();

		key(L"kind");
		switch(data_type.kind) {
			case GDScriptParser::DataType::BUILTIN:
				serialize(L"BUILTIN");
				break;
			case GDScriptParser::DataType::NATIVE:
				serialize(L"NATIVE");
				break;
			case GDScriptParser::DataType::SCRIPT:
				serialize(L"SCRIPT");
				break;
			case GDScriptParser::DataType::GDSCRIPT:
				serialize(L"GDSCRIPT");
				break;
			case GDScriptParser::DataType::CLASS:
				serialize(L"CLASS");
				break;
			case GDScriptParser::DataType::UNRESOLVED:
				serialize(L"UNRESOLVED");
				break;
			default:
				assert(false);
		}

		key(L"has_type");
		serialize(data_type.has_type);

		key(L"is_constant");
		serialize(data_type.is_constant);

		key(L"is_meta_type");
		serialize(data_type.is_meta_type);

		key(L"infer_type");
		serialize(data_type.infer_type);

		key(L"may_yield");
		serialize(data_type.may_yield);

		key(L"builtin_type");
		serialize(data_type.builtin_type);

		key(L"native_type");
		serialize(data_type.native_type);

		// circular reference
		// key(L"class_type");
		// serialize(data_type.class_type);
		// Ref<Script> script_type; // TODO: how to serialize this?

		endObject();
	}

	void GDScriptASTBuilder::node_prefix(const GDScriptParser::Node* node) {
		if(!node) {
			null();
			return;
		}
		startObject();

		key(L"type");
		switch(node->type) {
			case GDScriptParser::Node::TYPE_CLASS:
				serialize(L"class");
				break;
			case GDScriptParser::Node::TYPE_FUNCTION:
				serialize(L"function");
				break;
			case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION:
				serialize(L"built-in function");
				break;
			case GDScriptParser::Node::TYPE_BLOCK:
				serialize(L"block");
				break;
			case GDScriptParser::Node::TYPE_IDENTIFIER:
				serialize(L"identifier");
				break;
			case GDScriptParser::Node::TYPE_TYPE:
				serialize(L"type");
				break;
			case GDScriptParser::Node::TYPE_CONSTANT:
				serialize(L"constant");
				break;
			case GDScriptParser::Node::TYPE_ARRAY:
				serialize(L"array");
				break;
			case GDScriptParser::Node::TYPE_DICTIONARY:
				serialize(L"dictionary");
				break;
			case GDScriptParser::Node::TYPE_SELF:
				serialize(L"self");
				break;
			case GDScriptParser::Node::TYPE_OPERATOR:
				serialize(L"operator");
				break;
			case GDScriptParser::Node::TYPE_CONTROL_FLOW:
				serialize(L"control flow");
				break;
			case GDScriptParser::Node::TYPE_PATTERN:
				serialize(L"pattern");
				break;
			case GDScriptParser::Node::TYPE_PATTERN_BRANCH:
				serialize(L"pattern branch");
				break;
			case GDScriptParser::Node::TYPE_MATCH:
				serialize(L"match");
				break;
			case GDScriptParser::Node::TYPE_LOCAL_VAR:
				serialize(L"local var");
				break;
			case GDScriptParser::Node::TYPE_CAST:
				serialize(L"cast");
				break;
			case GDScriptParser::Node::TYPE_ASSERT:
				serialize(L"assert");
				break;
			case GDScriptParser::Node::TYPE_BREAKPOINT:
				serialize(L"breakpoint");
				break;
			case GDScriptParser::Node::TYPE_NEWLINE:
				serialize(L"newline");
				break;
			default:
				assert(false);
		}

		key(L"line");
		serialize(node->line);

		key(L"col");
		serialize(node->column);
	}

	void GDScriptASTBuilder::node_suffix(const GDScriptParser::Node* node) {
		endObject();
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ClassNode* node) {
		if(!node) {
			null();
			return;
		}

		node_prefix(node);

		key(L"name");
		serialize(node->name);

		key(L"subclasses");
		serialize(node->subclasses);

		key(L"variables");
		serialize(node->variables);

		key(L"tool");
		serialize(node->tool);

		key(L"extends_used");
		serialize(node->extends_used);

		key(L"extends_file");
		serialize(node->extends_file);

		key(L"extends_class");
		serialize(node->extends_class);

		key(L"base_type");
		serialize(node->base_type);

		key(L"icon_path");
		serialize(node->icon_path);

		key(L"constant_expressions");
		serialize(node->constant_expressions);

		key(L"functions");
		serialize(node->functions);

		key(L"static_functions");
		serialize(node->static_functions);

		key(L"_signals");
		serialize(node->_signals);

		key(L"end_line");
		serialize(node->end_line);

		// circular reference
		// key(L"owner");
		// serialize(node->owner);

		key(L"initializer");
		serialize(node->initializer);

		key(L"ready");
		serialize(node->ready);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::FunctionNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"name");
		serialize(node->name);

		key(L"_static");
		serialize(node->_static);

		key(L"rpc_mode");
		serialize(node->rpc_mode);

		key(L"has_yield");
		serialize(node->has_yield);

		key(L"has_unreachable_code");
		serialize(node->has_unreachable_code);

		key(L"return_type");
		serialize(node->return_type);

		key(L"arguments");
		startArray();

		for(int i = 0; i < node->arguments.size(); i++) {
			startObject();

			key(L"name");
			serialize(node->arguments[i]);

			key(L"type");
			serialize(node->argument_types[i]);

			if(i < node->default_values.size()) {
				key(L"default_value");
				serialize(node->default_values[i]);
			}
#ifdef DEBUG_ENABLED
			if(i < node->arguments_usage.size()) {
				key(L"usage");
				serialize(node->arguments_usage[i]);
			}
#endif // DEBUG_ENABLED

			endObject();
		}

		endArray();

		key(L"body");
		serialize(node->body);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::BuiltInFunctionNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"function");
		serialize(node->function);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::BlockNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		// circular reference
		// key(L"parent_class");
		// serialize(node->parent_class);

		// key(L"parent_block");
		// serialize(node->parent_block);

		key(L"statements");
		serialize(node->statements);

		key(L"variables");
		serialize(node->variables);

		key(L"has_return");
		serialize(node->has_return);

		key(L"end_line");
		serialize(node->end_line);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::IdentifierNode* node) {
		if(!node) {
			null();
			return;
		}

		node_prefix(node);

		key(L"name");
		serialize(node->name);

		// circular reference
		// key(L"declared_block");
		// serialize(node->declared_block);

		key(L"datatype");
		serialize(node->datatype);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::TypeNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"vtype");
		serialize(node->vtype);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ConstantNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"value");
		serialize(node->value);

		key(L"datatype");
		serialize(node->datatype);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ArrayNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"datatype");
		serialize(node->datatype);

		key(L"elements");
		serialize(node->elements);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::DictionaryNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"datatype");
		serialize(node->datatype);

		key(L"elements");
		serialize(node->elements);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::SelfNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);
		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::OperatorNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"operator");
		serialize(node->op);

		key(L"arguments");
		serialize(node->arguments);

		key(L"datatype");
		serialize(node->datatype);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::PatternNode::PatternType pattern) {
		switch(pattern) {
			case GDScriptParser::PatternNode::PT_CONSTANT:
				serialize(L"PT_CONSTANT");
				break;
			case GDScriptParser::PatternNode::PT_BIND:
				serialize(L"PT_BIND");
				break;
			case GDScriptParser::PatternNode::PT_DICTIONARY:
				serialize(L"PT_DICTIONARY");
				break;
			case GDScriptParser::PatternNode::PT_ARRAY:
				serialize(L"PT_ARRAY");
				break;
			case GDScriptParser::PatternNode::PT_IGNORE_REST:
				serialize(L"PT_IGNORE_REST");
				break;
			case GDScriptParser::PatternNode::PT_WILDCARD:
				serialize(L"PT_WILDCARD");
				break;
			default:
				assert(false);
		}
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::DictionaryNode::Pair& pair) {
		startObject();

		key(L"key");
		serialize(pair.key);

		key(L"value");
		serialize(pair.value);

		endObject();
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::PatternNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);
		key(L"pt_type");
		serialize(node->pt_type);

		key(L"constant");
		serialize(node->constant);

		key(L"bind");
		serialize(node->bind);

		key(L"dictionary");
		serialize(node->dictionary);

		key(L"bind");
		serialize(node->bind);

		key(L"array");
		serialize(node->array);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::PatternBranchNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"patterns");
		serialize(node->patterns);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::MatchNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"val_to_match");
		serialize(node->val_to_match);

		key(L"branches");
		serialize(node->branches);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ControlFlowNode::CFType type) {
		switch(type) {
			case GDScriptParser::ControlFlowNode::CF_IF:
				serialize(L"CF_IF");
				break;
			case GDScriptParser::ControlFlowNode::CF_FOR:
				serialize(L"CF_FOR");
				break;
			case GDScriptParser::ControlFlowNode::CF_WHILE:
				serialize(L"CF_WHILE");
				break;
			case GDScriptParser::ControlFlowNode::CF_SWITCH:
				serialize(L"CF_SWITCH");
				break;
			case GDScriptParser::ControlFlowNode::CF_BREAK:
				serialize(L"CF_BREAK");
				break;
			case GDScriptParser::ControlFlowNode::CF_CONTINUE:
				serialize(L"CF_CONTINUE");
				break;
			case GDScriptParser::ControlFlowNode::CF_RETURN:
				serialize(L"CF_RETURN");
				break;
			case GDScriptParser::ControlFlowNode::CF_MATCH:
				serialize(L"CF_MATCH");
				break;
			default:
				assert(false);
		}
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::ControlFlowNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"cf_type");
		serialize(node->cf_type);

		key(L"arguments");
		serialize(node->arguments);

		key(L"body");
		serialize(node->body);

		key(L"body_else");
		serialize(node->body_else);

		key(L"match");
		serialize(node->match);

		key(L"_else");
		serialize(node->_else);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::LocalVarNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"name");
		serialize(node->name);

		key(L"assign");
		serialize(node->assign);

		key(L"assign_op");
		serialize(node->assign_op);

		key(L"assignments");
		serialize(node->assignments);

		key(L"usages");
		serialize(node->usages);

		key(L"datatype");
		serialize(node->datatype);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::CastNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"source_node");
		serialize(node->source_node);

		key(L"cast_type");
		serialize(node->cast_type);

		key(L"return_type");
		serialize(node->return_type);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::AssertNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);

		key(L"condition");
		serialize(node->condition);

		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::BreakpointNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);
		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::NewLineNode* node) {
		if(!node) {
			null();
			return;
		}
		node_prefix(node);
		node_suffix(node);
	}

	void GDScriptASTBuilder::serialize(const GDScriptParser::Node* node) {
		if(!node) {
			null();
			return;
		}
		switch(node->type) {
			case GDScriptParser::Node::TYPE_CLASS:
				serialize(static_cast<const GDScriptParser::ClassNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_FUNCTION:
				serialize(static_cast<const GDScriptParser::FunctionNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION:
				serialize(static_cast<const GDScriptParser::BuiltInFunctionNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_BLOCK:
				serialize(static_cast<const GDScriptParser::BlockNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_IDENTIFIER:
				serialize(static_cast<const GDScriptParser::IdentifierNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_TYPE:
				serialize(static_cast<const GDScriptParser::TypeNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_CONSTANT:
				serialize(static_cast<const GDScriptParser::ConstantNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_ARRAY:
				serialize(static_cast<const GDScriptParser::ArrayNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_DICTIONARY:
				serialize(static_cast<const GDScriptParser::DictionaryNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_SELF:
				serialize(static_cast<const GDScriptParser::SelfNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_OPERATOR:
				serialize(static_cast<const GDScriptParser::OperatorNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_CONTROL_FLOW:
				serialize(static_cast<const GDScriptParser::ControlFlowNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_LOCAL_VAR:
				serialize(static_cast<const GDScriptParser::LocalVarNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_CAST:
				serialize(static_cast<const GDScriptParser::CastNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_ASSERT:
				serialize(static_cast<const GDScriptParser::AssertNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_BREAKPOINT:
				serialize(static_cast<const GDScriptParser::BreakpointNode *>(node));
				break;
			case GDScriptParser::Node::TYPE_NEWLINE:
				serialize(static_cast<const GDScriptParser::NewLineNode *>(node));
				break;
			default:
				assert(false);
		}
	}
