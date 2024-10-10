const type_of_int: int = typeof(1)
const type_of_float: int = typeof(3.14)

const str_int: String = str(18)
const str_plus: String = str(13) + str(29)

const err_ok: String = error_string(OK)
const err_busy: String = error_string(ERR_BUSY)

const v_to_s: String = var_to_str(32)

const v_to_b: PackedByteArray = var_to_bytes(32)

const v_to_b_objs: PackedByteArray = var_to_bytes_with_objects(64)

const const_rid: RID = rid_from_int64(0)

@warning_ignore("assert_always_true")
func test():
	assert(type_of_int == TYPE_INT)
	assert(type_of_float == TYPE_FLOAT)
	
	assert(str_int == '18')
	assert(str_plus == '1329')
	
	assert(err_ok == "OK")
	assert(err_busy == "Busy")
	
	assert(v_to_s == "32")
	
	assert(const_rid.get_id() == 0)

	print('ok')
