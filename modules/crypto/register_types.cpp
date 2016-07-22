#include "register_types.h"
#include "object_type_db.h"
#include "crypto.h"

void register_crypto_types() {

    ObjectTypeDB::register_type<Crypto>();
}

void unregister_crypto_types() {
   //nothing to do here
}

