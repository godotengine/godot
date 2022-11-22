#include "record.h"

#define ENCODE_VMETHOD (StringName)"_encode"
#define DECODE_VMETHOD (StringName)"_decode"
#define READ_VMETHOD   (StringName)"_read"
#define WRITE_VMETHOD  (StringName)"_write"

#define PRS_VMETHOD (StringName)"_pre_serialization"
#define POS_VMETHOD (StringName)"_pre_serialization"
#define PRD_VMETHOD (StringName)"_pre_deserialization"
#define POD_VMETHOD (StringName)"_post_deserialization"

#define ADVANCED_SERIALIZER_VERSION 0x000100

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

	BIND_VMETHOD(MethodInfo(Variant::Type::NIL, PRS_VMETHOD));
	BIND_VMETHOD(MethodInfo(Variant::Type::NIL, POS_VMETHOD));
	BIND_VMETHOD(MethodInfo(Variant::Type::NIL, PRD_VMETHOD));
	BIND_VMETHOD(MethodInfo(Variant::Type::NIL, POD_VMETHOD));
}

void Record::pre_serialization(){
	auto script = get_script_instance();
	if (script && script->has_method(PRS_VMETHOD)){
		script->call(PRS_VMETHOD);
	}
}
void Record::post_serialization(){
	auto script = get_script_instance();
	if (script && script->has_method(POS_VMETHOD)){
		script->call(POS_VMETHOD);
	}
}
void Record::pre_deserialization(){
	auto script = get_script_instance();
	if (script && script->has_method(PRD_VMETHOD)){
		script->call(PRD_VMETHOD);
	}
}
void Record::post_deserialization(){
	auto script = get_script_instance();
	if (script && script->has_method(POD_VMETHOD)){
		script->call(POD_VMETHOD);
	}
}

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
bool Record::decode(const Variant& from){
	if (decoder.is_null()){
		print_msg("No RecordDecoder assigned");
		return false;
	}
	// print_msg("The missile");
	auto iref = (Ref<Record>)this;
	return decoder->decode(from, iref);
}

RecordEncoder::RecordEncoder(){

}

RecordEncoder::~RecordEncoder(){
}

void RecordEncoder::_bind_methods(){
	// ClassDB::bind_method(D_METHOD("encode", "from", "instruction"), &RecordEncoder::encode);
	// ClassDB::bind_method(D_METHOD("builtin_encode", "from", "instruction"), &RecordEncoder::builtin_encode);

	auto encode_vmethod = MethodInfo(Variant::Type::NIL, ENCODE_VMETHOD, PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::Type::NIL, "instruction"));
	encode_vmethod.return_val.name = "Variant";
	encode_vmethod.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_VMETHOD(encode_vmethod);
}

Variant RecordEncoder::encode(Ref<Record> from, const Variant& instruction) const{
	auto script = get_script_instance();
	auto re = Variant();
	from->pre_serialization();
	if (script && script->has_method(ENCODE_VMETHOD)){
		re = script->call(ENCODE_VMETHOD, from, instruction);
	} else re = builtin_encode(from, instruction);
	from->post_serialization();
	return re;
}
Variant RecordEncoder::builtin_encode(const Ref<Record>& from, const Variant& instruction) const{
	return Variant();
}

RecordDecoder::RecordDecoder(){

}

RecordDecoder::~RecordDecoder(){
}

void RecordDecoder::_bind_methods(){
	// ClassDB::bind_method(D_METHOD("decode", "from", "to", "instruction"), &RecordDecoder::decode);
	// ClassDB::bind_method(D_METHOD("builtin_decode", "from", "to", "instruction"), &RecordDecoder::builtin_decode);

	BIND_VMETHOD(MethodInfo(Variant::BOOL, DECODE_VMETHOD, PropertyInfo(Variant::OBJECT, "from", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::OBJECT, "to", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, "Record"), PropertyInfo(Variant::Type::NIL, "instruction")));
}

bool RecordDecoder::decode(const Variant& from, Ref<Record> to, const Variant& instruction){
	auto script = get_script_instance();
	auto re = false;
	// print_msg("know where it is");
	to->pre_deserialization();
	if (script && script->has_method(DECODE_VMETHOD)){
		re = (bool)script->call(DECODE_VMETHOD, from, instruction);
		// print_msg("at all time.");
	}
	// print_msg("It knows this because");
	re = builtin_decode(from, to, instruction);
	to->post_deserialization();
	return re;
}
bool RecordDecoder::builtin_decode(const Variant& from, Ref<Record> to, const Variant& instruction){
	// print_msg("it knows where it isn\'t.");
	return false;
}

RecordMask::RecordMask(){

}

RecordMask::~RecordMask(){
}

void RecordMask::_bind_methods(){
	// ClassDB::bind_method(D_METHOD("read", "from", "what"), &RecordMask::read);
	// ClassDB::bind_method(D_METHOD("write", "to", "what", "with"), &RecordMask::write);
	// ClassDB::bind_method(D_METHOD("builtin_read", "from", "what"), &RecordMask::builtin_read);
	// ClassDB::bind_method(D_METHOD("builtin_write", "to", "what", "with"), &RecordMask::builtin_write);

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
bool RecordMask::write(Ref<Record> to, const StringName& what, const Variant& with){
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

Variant RecordPrimitiveEncoder::builtin_encode(const Ref<Record>& from, const Variant& instruction) const{
	auto tracks_list = from->get_track_list();
	auto size = tracks_list.size();
	Dictionary encoded;
	for (int i = 0; i < size; i++){
		auto track = tracks_list[i];
		auto item = from->get_track(track);
		if (item.get_type() == Variant::OBJECT) encoded[track] = Variant();
		else encoded[track] = item;
	}
	return Variant(encoded);
}

bool RecordPrimitiveDecoder::builtin_decode(const Variant& from, Ref<Record> to, const Variant& instruction){
	auto result = true;
	// print_msg("By subtracting");
	if (from.get_type() == Variant::DICTIONARY) {
		// print_msg("where it is");
		Dictionary d = from;
		List<Variant> keys;
		d.get_key_list(&keys);
		for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
			String key = E->get();
			auto val = d[E->get()];
			auto ires = true;
			to->set((StringName)key, val, &ires);
			result = result && ires;
		}
		// print_msg("from where it isn\'t.");
	} else result = false;
	// print_msg("I forgor...");
	return result;
}

// Variant RecordAdvancedEncoder::builtin_encode(const Ref<Record>& from, const Variant& instruction) const{


// }

// Ref<RawRecord> RecordAdvancedEncoder::internal_encode(const Ref<Record>& from, const Variant& instruction, Ref<RawRecord>& rrec, Ref<RawRecordData>& rdata) {
// 	if (rrec.is_null())  rrec  = memnew(RawRecord);
// 	if (rdata.is_null()){
// 		auto iid = rrec->create_dependency();
// 		rdata = memnew(RawRecordData);
// 		rrec->entail_dependency(iid, rdata);
// 	}
// 	auto tracks_list = from->get_track_list();
// 	auto tracks_size = tracks_list.size();
// 	for (uint32_t i = 0; i < tracks_size; i++){
// 		auto track = tracks_list[i];
// 		recursive_encode(rrec, rdata, track, from->get_track(track));
// 	}
// 	return rrec;
// }

// Ref<RawRecord> RecordAdvancedEncoder::internal_encode_res(const RES& from, const Variant& instruction, Ref<RawRecord>& rrec, Ref<RawRecordData>& rdata) {
// 	List<PropertyInfo> property_list;
// 	from->get_property_list(&property_list);
// 	for (List<PropertyInfo>::Element *PE = property_list.front(); PE; PE = PE->next()) {
// 		auto pi = PE->get();
// 		auto name = pi.name;
// 		auto value = from->get(name);

// 	}
// }

// void RecordAdvancedEncoder::recursive_encode(Ref<RawRecord>& raw_record, Ref<RawRecordData>& rec_data, const StringName& track_name, const Variant& track_data){
// 	switch (track_data.get_type()){
// 		case Variant::OBJECT:
// 			Object* obj = track_data;
// 			auto uid = obj->get_instance_id();
// 			auto obj_iid = raw_record->uid2iid(uid);
// 			if (obj_iid <= 0){
// 				obj_iid = raw_record->create_dependency();
// 				if (obj_iid <= 0) return;
// 				Ref<RawRecordData> new_data = memnew(RawRecordData);
// 				new_data->unique_id = uid;
// 				raw_record->entail_dependency(obj_iid, new_data);
// 				if (obj->is_class("Record")){
// 					internal_encode(obj, nullptr, raw_record, new_data);
// 				} else if (obj->is_class("Resource")){
// 					internal_encode_res(obj, nullptr, raw_record, new_data);
// 				} else {

// 				}
// 			}
// 			rec_data->external_refs.push_back(obj_iid);
// 			rec_data->table.push_back(Pair<StringName, Variant>(track_name, nullptr));
// 			break;
// 		default:
// 			rec_data->table.push_back(Pair<StringName, Variant>(track_name, track_data));
// 	}
// }

// void RecordAdvancedEncoder::recursive_encode_primitive(const Ref<RawRecord>& raw_record, const StringName& track_name, const Variant& track_data){

// }
// void RecordAdvancedEncoder::recursive_encode_object(const Ref<RawRecord>& raw_record, const StringName& track_name, const Object* track_data){

// }

// bool RecordAdvancedDecoder::builtin_decode(const Variant& from, Ref<Record> to, const Variant& instruction){
// 	auto result = true;
// 	if (from.get_type() == Variant::DICTIONARY) {

// 	} else result = false;
// 	return result;
// }

RawRecord::RawRecord(){

}

RawRecord::~RawRecord(){

}

uint64_t RawRecord::create_dependency(){
	lock.write_lock();
	allocated_id += 1;
	data_table[allocated_id] = Ref<RawRecordData>();
	auto re = allocated_id;
	lock.write_unlock();
	return re;
}


void RawRecord::entail_dependency(const uint64_t& internal_id, const Ref<RawRecordData>& data){
	ERR_FAIL_COND(!data_table.has(internal_id));
	lock.write_lock();
	data_table[internal_id] = data;
	lock.write_unlock();
}

uint64_t RawRecord::iid2uid(const uint64_t& internal_id, const StringName& extra){
	return xid2xid(internal_id, false, extra);
}

uint64_t RawRecord::uid2iid(const uint64_t& unique_id, const StringName& extra){
	return xid2xid(unique_id, true, extra);
}
uint64_t RawRecord::xid2xid(const uint64_t& x_id, const bool& reversed, const StringName& extra){
	lock.read_lock();
	for (auto E = data_table.front(); E; E = E->next()){
		auto val = E->get();
		switch (reversed){
			case false:
				if (val->allocated_id == x_id){
					lock.read_unlock();
					return val->unique_id;
				}
			case true:
				if (val->unique_id == x_id){
					lock.read_unlock();
					return val->allocated_id;
				}
		}
	}
	lock.read_unlock();
	return 0;
}

void RawRecord::clear(){
	lock.write_lock();
	data_table.clear();
	allocated_id = 0;
	lock.write_unlock();
}

// void RawRecord::internal_encode(const StringName& key, const Variant& val, Ref<RawRecordData>& rdata){
// 	switch (data.get_type()){
// 		case Variant::OBJECT:
// 			Object* obj = data;
// 			auto obj_id = obj->get_instance_id();
// 			auto iid = uid2iid(obj_id);
// 			if (iid <= 0){
// 				allocated_id += 1;
// 				iid = allocated_id;
// 				auto rdata = memnew(RawRecordData);
// 				rdata->name = key;
// 				data_table[iid] = rdata;
				

// 			} else {

// 			}
// 			break;
// 		default:
// 			break;
// 	}
// }

// void RawRecord::auto_encode(const Map<StringName, Variant>& properties){
// 	lock.write_lock();
// 	allocated_id += 1;
// 	auto rdata = memnew(RawRecordData);
// 	rdata->name = StringName("__main");
// 	data_table[allocated_id] = rdata;
// 	for (auto E = properties.front(); E; E = E->next()){
// 		auto key = E->key();
// 		auto val = E->get();
// 		internal_encode(key, val, rdata);
// 	}
// 	lock.write_unlock();
// }
