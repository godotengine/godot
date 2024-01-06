#include "luaError.h"

void LuaError::_bind_methods() {
	// Binding static GD methods. This allows the use of LuaError.new_error instead of neededing to call LuaError.new and than calling .setInfo
	ClassDB::bind_static_method("LuaError", D_METHOD("new_error", "Message", "Type"), &LuaError::newError, DEFVAL(LuaError::ERR_RUNTIME));

	ClassDB::bind_method(D_METHOD("set_message", "Message"), &LuaError::setMessage);
	ClassDB::bind_method(D_METHOD("get_message"), &LuaError::getMessage);
	ClassDB::bind_method(D_METHOD("set_type", "Type"), &LuaError::setType);
	ClassDB::bind_method(D_METHOD("get_type"), &LuaError::getType);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "message"), "set_message", "get_message");

	BIND_ENUM_CONSTANT(ERR_TYPE);
	BIND_ENUM_CONSTANT(ERR_RUNTIME);
	BIND_ENUM_CONSTANT(ERR_SYNTAX);
	BIND_ENUM_CONSTANT(ERR_MEMORY);
	BIND_ENUM_CONSTANT(ERR_ERR);
	BIND_ENUM_CONSTANT(ERR_FILE);
}

// Create a new error
Ref<LuaError> LuaError::newError(String msg, ErrorType type) {
	Ref<LuaError> err;
	err.instantiate();
	err->setInfo(msg, static_cast<LuaError::ErrorType>(type));
	return err;
}

void LuaError::setInfo(String msg, ErrorType type) {
	errType = type;
	errMsg = msg;
}

bool LuaError::operator==(const ErrorType type) {
	return errType == type;
}

bool LuaError::operator==(const LuaError err) {
	return errType == err.getType();
}

void LuaError::setMessage(String msg) {
	errMsg = msg;
}

String LuaError::getMessage() const {
	return errMsg;
}

void LuaError::setType(ErrorType type) {
	errType = type;
}

LuaError::ErrorType LuaError::getType() const {
	return errType;
}
