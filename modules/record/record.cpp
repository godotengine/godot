#include "record.h"

#define ENCODE_VMETHOD (StringName)"_encode"
#define DECODE_VMETHOD (StringName)"_decode"
#define READ_VMETHOD   (StringName)"_read"
#define WRITE_VMETHOD  (StringName)"_write"

Record::Record(){
	encoder = (Ref<RecordEncoder>)memnew(RecordEncoder);
	decoder = (Ref<RecordDecoder>)memnew(RecordDecoder);
}

Record::~Record(){

}

void Record::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_track_list", "new_list"), &Record::set_track_list);
	ClassDB::bind_method(D_METHOD("get_track_list"), &Record::get_track_list);

	ClassDB::bind_method(D_METHOD("set_encoder", "new_encoder"), &Record::set_encoder);
	ClassDB::bind_method(D_METHOD("get_encoder"), &Record::get_encoder);

	ClassDB::bind_method(D_METHOD("set_decoder", "new_decoder"), &Record::set_decoder);
	ClassDB::bind_method(D_METHOD("get_decoder"), &Record::get_decoder);

	ClassDB::bind_method(D_METHOD("set_mask", "new_mask"), &Record::set_mask);
	ClassDB::bind_method(D_METHOD("get_mask"), &Record::get_mask);

	ClassDB::bind_method(D_METHOD("read", "what"), &Record::read);
	ClassDB::bind_method(D_METHOD("write", "what", "with"), &Record::write);

	ClassDB::bind_method(D_METHOD("get_track", "track_name"), &Record::get_track);
	ClassDB::bind_method(D_METHOD("get_all_tracks"), &Record::get_all_tracks);

	ClassDB::bind_method(D_METHOD("encode"), &Record::encode);
	ClassDB::bind_method(D_METHOD("decode", "from"), &Record::decode);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_STRING_ARRAY, "track_list"), "set_track_list", "get_track_list");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "encoder"), "set_encoder", "get_encoder");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "decoder"), "set_decoder", "get_decoder");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mask"), "set_mask", "get_mask");
}

// void Record::print_msg(const char* msg){
// 	Hub::get_singleton()->print_custom("Record", msg);
// }
// void Record::print_msg(const std::string& msg){
// 	Hub::get_singleton()->print_custom("Record", msg.c_str());
// }

Variant Record::read(const StringName& what){
	if (mask.is_null()){
		print_msg("No RecordMask assigned");
		return Variant();
	}
	return mask->read(Ref<Record>(this), what);
}
bool Record::write(const StringName& what, const Variant& with){
	if (mask.is_null()){
		print_msg("No RecordMask assigned");
		return false;
	}
	return mask->write(Ref<Record>(this), what, with);
}

Variant Record::get_track(const StringName& track_name) const{
	bool *validity = new bool();
	auto val = get(track_name, validity);
	if (!(*validity))
		print_msg("Invalid track_name");
	return val;
}
Array Record::get_all_tracks() const{
	Array re;
	auto size = track_list.size();
	for (int i = 0; i < size; i++){
		auto track_name = track_list[i];
		auto returned = get_track(track_name);
		if (returned.get_type() == Variant::OBJECT){
			Object* obj_casted = returned;
			if (obj_casted == nullptr) { re.append(Variant()); continue; }
			if (obj_casted->get_instance_id() == 0) { re.append(Variant()); continue; }
		}
		re.append(returned);
	}
	return re;
}

Variant Record::encode() const{
	if (encoder.is_null()){
		print_msg("No RecordEncoder assigned");
		return Variant();
	}
	auto iref = (Ref<Record>)this;
	return encoder.ptr()->encode(iref);
}
bool Record::decode(const Ref<Record>& from){
	if (decoder.is_null()){
		print_msg("No RecordDecoder assigned");
		return false;
	}
	auto iref = (Ref<Record>)this;
	return decoder.ptr()
	->decode(from, iref);
}

RecordEncoder::RecordEncoder(){

}

RecordEncoder::~RecordEncoder(){
}

void RecordEncoder::_bind_methods(){
	ClassDB::bind_method(D_METHOD("encode", "from", "instruction"), &RecordEncoder::encode);
	ClassDB::bind_method(D_METHOD("builtin_encode", "from", "instruction"), &RecordEncoder::builtin_encode);

	auto encode_vmethod = MethodInfo(Variant::Type::NIL, ENCODE_VMETHOD, PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::Type::NIL, "instruction"));
	encode_vmethod.return_val.name = "Variant";
	encode_vmethod.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_VMETHOD(encode_vmethod);
}

Variant RecordEncoder::encode(const Ref<Record>& from, const Variant& instruction) const{
	auto script = get_script_instance();
	if (script && script->has_method(ENCODE_VMETHOD)){
		return script->call(ENCODE_VMETHOD, from, instruction);
	}
	return builtin_encode(from, instruction);
}
Variant RecordEncoder::builtin_encode(const Ref<Record>& from, const Variant& instruction) const{
	return Variant();
}

RecordDecoder::RecordDecoder(){

}

RecordDecoder::~RecordDecoder(){
}

void RecordDecoder::_bind_methods(){
	ClassDB::bind_method(D_METHOD("decode", "from", "to", "instruction"), &RecordDecoder::decode);
	ClassDB::bind_method(D_METHOD("builtin_decode", "from", "to", "instruction"), &RecordDecoder::builtin_decode);

	BIND_VMETHOD(MethodInfo(Variant::BOOL, DECODE_VMETHOD, PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::OBJECT, "to", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::Type::NIL, "instruction")));
}

bool RecordDecoder::decode(const Ref<Record>& from, const Ref<Record>& to, const Variant& instruction){
	auto script = get_script_instance();
	if (script && script->has_method(DECODE_VMETHOD)){
		return script->call(DECODE_VMETHOD, from, instruction);
	}
	return (bool)builtin_decode(from, to, instruction);
}
bool RecordDecoder::builtin_decode(const Ref<Record>& from, const Ref<Record>& to, const Variant& instruction){
	return false;
}

RecordMask::RecordMask(){

}

RecordMask::~RecordMask(){
}

void RecordMask::_bind_methods(){
	ClassDB::bind_method(D_METHOD("read", "from", "what"), &RecordMask::read);
	ClassDB::bind_method(D_METHOD("write", "to", "what", "with"), &RecordMask::write);
	ClassDB::bind_method(D_METHOD("builtin_read", "from", "what"), &RecordMask::builtin_read);
	ClassDB::bind_method(D_METHOD("builtin_write", "to", "what", "with"), &RecordMask::builtin_write);

	auto read_vmethod = MethodInfo(READ_VMETHOD, PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::STRING, "what"));
	read_vmethod.return_val.name = "Variant";
	read_vmethod.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_VMETHOD(read_vmethod);
	BIND_VMETHOD(MethodInfo(Variant::BOOL, WRITE_VMETHOD, PropertyInfo(Variant::OBJECT, "to", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::STRING, "what"), PropertyInfo(Variant::Type::NIL, "with")));
}

Variant RecordMask::read(const Ref<Record>& from, const StringName& what){
	auto script = get_script_instance();
	if (script && script->has_method(READ_VMETHOD)){
		return script->call(READ_VMETHOD, from, what);
	}
	return builtin_read(from, what);
}
bool RecordMask::write(const Ref<Record>& to, const StringName& what, const Variant& with){
	auto script = get_script_instance();
	if (script && script->has_method(WRITE_VMETHOD)){
		return script->call(WRITE_VMETHOD, to, what, with);
	}
	return builtin_write(to, what, with);
}

Variant RecordMask::builtin_read(const Ref<Record>& from, const StringName& what) const{
	return from->get(what);
}
bool RecordMask::builtin_write(Ref<Record> to, const StringName& what, const Variant& with){
	bool validity = false;
	to->set(what, with, &validity);
	return validity;
}

