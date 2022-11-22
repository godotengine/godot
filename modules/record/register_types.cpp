#include "register_types.h"
#include "core/class_db.h"
#include "record.h"

void register_record_types(){
	ClassDB::register_class<Record>();
	ClassDB::register_class<RecordEncoder>();
	ClassDB::register_class<RecordDecoder>();
	ClassDB::register_class<RecordMask>();
	ClassDB::register_class<RecordPrimitiveEncoder>();
	ClassDB::register_class<RecordPrimitiveDecoder>();
	// ClassDB::register_class<RecordAdvancedEncoder>();
	// ClassDB::register_class<RecordAdvancedDecoder>();
}
void unregister_record_types(){

}