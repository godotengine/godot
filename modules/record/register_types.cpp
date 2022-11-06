#include "register_types.h"
#include "core/class_db.h"
#include "record.h"

void register_record_types(){
	ClassDB::register_class<Record>();
	ClassDB::register_class<RecordEncoder>();
	ClassDB::register_class<RecordDecoder>();
	ClassDB::register_class<RecordMask>();
}
void unregister_record_types(){

}