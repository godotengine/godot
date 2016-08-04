#include "visual_script_builtin_funcs.h"


const char* VisualScriptBuiltinFunc::func_name[VisualScriptBuiltinFunc::FUNC_MAX]={
	"sin",
	"cos",
	"tan",
	"sinh",
	"cosh",
	"tanh",
	"asin",
	"acos",
	"atan",
	"atan2",
	"sqrt",
	"fmod",
	"fposmod",
	"floor",
	"ceil",
	"round",
	"abs",
	"sign",
	"pow",
	"log",
	"exp",
	"is_nan",
	"is_inf",
	"ease",
	"decimals",
	"stepify",
	"lerp",
	"dectime",
	"randomize",
	"randi",
	"randf",
	"rand_range",
	"seed",
	"rand_seed",
	"deg2rad",
	"rad2deg",
	"linear2db",
	"db2linear",
	"max",
	"min",
	"clamp",
	"nearest_po2",
	"weakref",
	"funcref",
	"convert",
	"typeof",
	"type_exists",
	"str",
	"print",
	"printerr",
	"printraw",
	"var2str",
	"str2var",
	"var2bytes",
	"bytes2var",
};



int VisualScriptBuiltinFunc::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptBuiltinFunc::has_input_sequence_port() const{

	return true;
}

int VisualScriptBuiltinFunc::get_input_value_port_count() const{


	switch(func) {

		case MATH_RANDOMIZE:
		case MATH_RAND:
		case MATH_RANDF:
			return 0;
		case MATH_SIN:
		case MATH_COS:
		case MATH_TAN:
		case MATH_SINH:
		case MATH_COSH:
		case MATH_TANH:
		case MATH_ASIN:
		case MATH_ACOS:
		case MATH_ATAN:
		case MATH_SQRT:
		case MATH_FLOOR:
		case MATH_CEIL:
		case MATH_ROUND:
		case MATH_ABS:
		case MATH_SIGN:
		case MATH_LOG:
		case MATH_EXP:
		case MATH_ISNAN:
		case MATH_ISINF:
		case MATH_DECIMALS:
		case MATH_SEED:
		case MATH_RANDSEED:
		case MATH_DEG2RAD:
		case MATH_RAD2DEG:
		case MATH_LINEAR2DB:
		case MATH_DB2LINEAR:
		case LOGIC_NEAREST_PO2:
		case OBJ_WEAKREF:
		case TYPE_OF:
		case TEXT_STR:
		case TEXT_PRINT:
		case TEXT_PRINTERR:
		case TEXT_PRINTRAW:
		case VAR_TO_STR:
		case STR_TO_VAR:
		case VAR_TO_BYTES:
		case BYTES_TO_VAR:
		case TYPE_EXISTS:
			return 1;
		case MATH_ATAN2:
		case MATH_FMOD:
		case MATH_FPOSMOD:
		case MATH_POW:
		case MATH_EASE:
		case MATH_STEPIFY:
		case MATH_RANDOM:
		case LOGIC_MAX:
		case LOGIC_MIN:
		case FUNC_FUNCREF:
		case TYPE_CONVERT:
			return 2;
		case MATH_LERP:
		case MATH_DECTIME:
		case LOGIC_CLAMP:
			return 3;
		case FUNC_MAX:{}

	}
	return 0;
}
int VisualScriptBuiltinFunc::get_output_value_port_count() const{

	switch(func) {
		case MATH_RANDOMIZE:
		case TEXT_PRINT:
		case TEXT_PRINTERR:
		case TEXT_PRINTRAW:
		case MATH_SEED:
			return 0;
		case MATH_RANDSEED:
			return 2;
		default:
			return 1;
	}

	return 1;
}

String VisualScriptBuiltinFunc::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptBuiltinFunc::get_input_value_port_info(int p_idx) const{

	switch(func) {

		case MATH_SIN:
		case MATH_COS:
		case MATH_TAN:
		case MATH_SINH:
		case MATH_COSH:
		case MATH_TANH:
		case MATH_ASIN:
		case MATH_ACOS:
		case MATH_ATAN:
		case MATH_ATAN2:
		case MATH_SQRT: {
			return PropertyInfo(Variant::REAL,"num");
		} break;
		case MATH_FMOD:
		case MATH_FPOSMOD: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"x");
			else
				return PropertyInfo(Variant::REAL,"y");
		} break;
		case MATH_FLOOR:
		case MATH_CEIL:
		case MATH_ROUND:
		case MATH_ABS:
		case MATH_SIGN: {
			return PropertyInfo(Variant::REAL,"num");

		} break;

		case MATH_POW: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"x");
			else
				return PropertyInfo(Variant::REAL,"y");
		} break;
		case MATH_LOG:
		case MATH_EXP:
		case MATH_ISNAN:
		case MATH_ISINF: {
			return PropertyInfo(Variant::REAL,"num");
		} break;
		case MATH_EASE: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"s");
			else
				return PropertyInfo(Variant::REAL,"curve");
		} break;
		case MATH_DECIMALS: {
			return PropertyInfo(Variant::REAL,"step");
		} break;
		case MATH_STEPIFY: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"s");
			else
				return PropertyInfo(Variant::REAL,"steps");
		} break;
		case MATH_LERP: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"from");
			else if (p_idx==1)
				return PropertyInfo(Variant::REAL,"to");
			else
				return PropertyInfo(Variant::REAL,"weight");

		} break;
		case MATH_DECTIME: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"value");
			else if (p_idx==1)
				return PropertyInfo(Variant::REAL,"amount");
			else
				return PropertyInfo(Variant::REAL,"step");
		} break;
		case MATH_RANDOMIZE: {

		} break;
		case MATH_RAND: {

		} break;
		case MATH_RANDF: {

		} break;
		case MATH_RANDOM: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"from");
			else
				return PropertyInfo(Variant::REAL,"to");
		} break;
		case MATH_SEED: {
			return PropertyInfo(Variant::INT,"seed");
		} break;
		case MATH_RANDSEED: {
			return PropertyInfo(Variant::INT,"seed");
		} break;
		case MATH_DEG2RAD: {
			return PropertyInfo(Variant::REAL,"deg");
		} break;
		case MATH_RAD2DEG: {
			return PropertyInfo(Variant::REAL,"rad");
		} break;
		case MATH_LINEAR2DB: {
			return PropertyInfo(Variant::REAL,"nrg");
		} break;
		case MATH_DB2LINEAR: {
			return PropertyInfo(Variant::REAL,"db");
		} break;
		case LOGIC_MAX: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"a");
			else
				return PropertyInfo(Variant::REAL,"b");
		} break;
		case LOGIC_MIN: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"a");
			else
				return PropertyInfo(Variant::REAL,"b");
		} break;
		case LOGIC_CLAMP: {
			if (p_idx==0)
				return PropertyInfo(Variant::REAL,"a");
			else if (p_idx==0)
				return PropertyInfo(Variant::REAL,"min");
			else
				return PropertyInfo(Variant::REAL,"max");
		} break;
		case LOGIC_NEAREST_PO2: {
			return PropertyInfo(Variant::INT,"num");
		} break;
		case OBJ_WEAKREF: {

			return PropertyInfo(Variant::OBJECT,"source");

		} break;
		case FUNC_FUNCREF: {

			if (p_idx==0)
				return PropertyInfo(Variant::OBJECT,"instance");
			else
				return PropertyInfo(Variant::STRING,"funcname");

		} break;
		case TYPE_CONVERT: {

			if (p_idx==0)
				return PropertyInfo(Variant::NIL,"what");
			else
				return PropertyInfo(Variant::STRING,"type");
		} break;
		case TYPE_OF: {
			return PropertyInfo(Variant::NIL,"what");

		} break;
		case TYPE_EXISTS: {

			return PropertyInfo(Variant::STRING,"type");

		} break;
		case TEXT_STR: {

			return PropertyInfo(Variant::NIL,"value");


		} break;
		case TEXT_PRINT: {

			return PropertyInfo(Variant::NIL,"value");

		} break;
		case TEXT_PRINTERR: {
			return PropertyInfo(Variant::NIL,"value");

		} break;
		case TEXT_PRINTRAW: {

			return PropertyInfo(Variant::NIL,"value");

		} break;
		case VAR_TO_STR: {
			return PropertyInfo(Variant::NIL,"var");

		} break;
		case STR_TO_VAR: {

			return PropertyInfo(Variant::STRING,"string");
		} break;
		case VAR_TO_BYTES: {
			return PropertyInfo(Variant::NIL,"var");

		} break;
		case BYTES_TO_VAR: {

			return PropertyInfo(Variant::RAW_ARRAY,"bytes");
		} break;
		case FUNC_MAX:{}
	}

	return PropertyInfo();
}

PropertyInfo VisualScriptBuiltinFunc::get_output_value_port_info(int p_idx) const{

	Variant::Type t=Variant::NIL;
	switch(func) {

		case MATH_SIN:
		case MATH_COS:
		case MATH_TAN:
		case MATH_SINH:
		case MATH_COSH:
		case MATH_TANH:
		case MATH_ASIN:
		case MATH_ACOS:
		case MATH_ATAN:
		case MATH_ATAN2:
		case MATH_SQRT:
		case MATH_FMOD:
		case MATH_FPOSMOD:
		case MATH_FLOOR:
		case MATH_CEIL: {
			t=Variant::REAL;
		} break;
		case MATH_ROUND: {
			t=Variant::INT;
		} break;
		case MATH_ABS: {
			t=Variant::NIL;
		} break;
		case MATH_SIGN: {
			t=Variant::NIL;
		} break;
		case MATH_POW:
		case MATH_LOG:
		case MATH_EXP: {
			t=Variant::REAL;
		} break;
		case MATH_ISNAN:
		case MATH_ISINF: {
			t=Variant::BOOL;
		} break;
		case MATH_EASE: {
			t=Variant::REAL;
		} break;
		case MATH_DECIMALS: {
			t=Variant::INT;
		} break;
		case MATH_STEPIFY:
		case MATH_LERP:
		case MATH_DECTIME: {
			t=Variant::REAL;

		} break;
		case MATH_RANDOMIZE: {

		} break;
		case MATH_RAND: {

			t=Variant::INT;
		} break;
		case MATH_RANDF:
		case MATH_RANDOM: {
			t=Variant::REAL;
		} break;
		case MATH_SEED: {

		} break;
		case MATH_RANDSEED: {

			if (p_idx==0)
				return PropertyInfo(Variant::INT,"rnd");
			else
				return PropertyInfo(Variant::INT,"seed");
		} break;
		case MATH_DEG2RAD:
		case MATH_RAD2DEG:
		case MATH_LINEAR2DB:
		case MATH_DB2LINEAR: {
			t=Variant::REAL;
		} break;
		case LOGIC_MAX:
		case LOGIC_MIN:
		case LOGIC_CLAMP: {


		} break;

		case LOGIC_NEAREST_PO2: {
			t=Variant::NIL;
		} break;
		case OBJ_WEAKREF: {

			t=Variant::OBJECT;

		} break;
		case FUNC_FUNCREF: {

			t=Variant::OBJECT;

		} break;
		case TYPE_CONVERT: {



		} break;
		case TYPE_OF: {
			t=Variant::INT;

		} break;
		case TYPE_EXISTS: {

			t=Variant::BOOL;

		} break;
		case TEXT_STR: {

			t=Variant::STRING;

		} break;
		case TEXT_PRINT: {


		} break;
		case TEXT_PRINTERR: {

		} break;
		case TEXT_PRINTRAW: {

		} break;
		case VAR_TO_STR: {
			t=Variant::STRING;
		} break;
		case STR_TO_VAR: {

		} break;
		case VAR_TO_BYTES: {
			t=Variant::RAW_ARRAY;

		} break;
		case BYTES_TO_VAR: {


		} break;
		case FUNC_MAX:{}
	}

	return PropertyInfo(t,"");
}

String VisualScriptBuiltinFunc::get_caption() const {

	return "BuiltinFunc";
}

String VisualScriptBuiltinFunc::get_text() const {

	return func_name[func];
}

void VisualScriptBuiltinFunc::set_func(BuiltinFunc p_which) {

	ERR_FAIL_INDEX(p_which,FUNC_MAX);
	func=p_which;
	_change_notify();
	ports_changed_notify();
}

VisualScriptBuiltinFunc::BuiltinFunc VisualScriptBuiltinFunc::get_func() {
	return func;
}



VisualScriptNodeInstance* VisualScriptBuiltinFunc::instance(VScriptInstance* p_instance) {

	return NULL;
}

void VisualScriptBuiltinFunc::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_func","which"),&VisualScriptBuiltinFunc::set_func);
	ObjectTypeDB::bind_method(_MD("get_func"),&VisualScriptBuiltinFunc::get_func);

	String cc;

	for(int i=0;i<FUNC_MAX;i++) {

		if (i>0)
			cc+=",";
		cc+=func_name[i];
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT,"function",PROPERTY_HINT_ENUM,cc),_SCS("set_func"),_SCS("get_func"));
}

VisualScriptBuiltinFunc::VisualScriptBuiltinFunc() {

	func=MATH_SIN;
}

template<VisualScriptBuiltinFunc::BuiltinFunc func>
static Ref<VisualScriptNode> create_builtin_func_node(const String& p_name) {

	Ref<VisualScriptBuiltinFunc> node;
	node.instance();
	node->set_func(func);
	return node;
}

void register_visual_script_builtin_func_node() {


	VisualScriptLanguage::singleton->add_register_func("functions/builtin/sin",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SIN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/cos",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_COS>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/tan",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_TAN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/sinh",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SINH>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/cosh",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_COSH>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/tanh",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_TANH>);

	VisualScriptLanguage::singleton->add_register_func("functions/builtin/asin",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ASIN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/acos",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ACOS>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/atan",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ATAN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/atan2",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ATAN2>);

	VisualScriptLanguage::singleton->add_register_func("functions/builtin/sqrt",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SQRT>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/fmod",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_FMOD>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/fposmod",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_FPOSMOD>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/floor",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_FLOOR>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/ceil",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_CEIL>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/round",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ROUND>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/abs",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ABS>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/sign",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SIGN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/pow",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_POW>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/log",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LOG>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/exp",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_EXP>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/isnan",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ISNAN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/isinf",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_ISINF>);

	VisualScriptLanguage::singleton->add_register_func("functions/builtin/ease",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_EASE>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/decimals",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_DECIMALS>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/stepify",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_STEPIFY>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/lerp",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LERP>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/dectime",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_DECTIME>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/randomize",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDOMIZE>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/rand",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RAND>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/randf",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDF>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/random",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDOM>);

	VisualScriptLanguage::singleton->add_register_func("functions/builtin/seed",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_SEED>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/randseed",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RANDSEED>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/deg2rad",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_DEG2RAD>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/rad2deg",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_RAD2DEG>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/linear2db",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_LINEAR2DB>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/db2linear",create_builtin_func_node<VisualScriptBuiltinFunc::MATH_DB2LINEAR>);

	VisualScriptLanguage::singleton->add_register_func("functions/builtin/max",create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_MAX>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/min",create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_MIN>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/clamp",create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_CLAMP>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/nearest_po2",create_builtin_func_node<VisualScriptBuiltinFunc::LOGIC_NEAREST_PO2>);

	VisualScriptLanguage::singleton->add_register_func("functions/builtin/weakref",create_builtin_func_node<VisualScriptBuiltinFunc::OBJ_WEAKREF>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/funcref",create_builtin_func_node<VisualScriptBuiltinFunc::FUNC_FUNCREF>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/convert",create_builtin_func_node<VisualScriptBuiltinFunc::TYPE_CONVERT>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/typeof",create_builtin_func_node<VisualScriptBuiltinFunc::TYPE_OF>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/type_exists",create_builtin_func_node<VisualScriptBuiltinFunc::TYPE_EXISTS>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/str",create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_STR>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/print",create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINT>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/printerr",create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINTERR>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/printraw",create_builtin_func_node<VisualScriptBuiltinFunc::TEXT_PRINTRAW>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/var2str",create_builtin_func_node<VisualScriptBuiltinFunc::VAR_TO_STR>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/str2var",create_builtin_func_node<VisualScriptBuiltinFunc::STR_TO_VAR>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/var2bytes",create_builtin_func_node<VisualScriptBuiltinFunc::VAR_TO_BYTES>);
	VisualScriptLanguage::singleton->add_register_func("functions/builtin/bytes2var",create_builtin_func_node<VisualScriptBuiltinFunc::BYTES_TO_VAR>);

}
