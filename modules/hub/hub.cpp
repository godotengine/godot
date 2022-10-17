#include "hub.h"

Hub *Hub::singleton = nullptr;

Hub::Hub(){
	singleton = this;
}

void Hub::_bind_methods(){
	ClassDB::bind_method(D_METHOD("print_fatal"), &Hub::print_fatal);
	ClassDB::bind_method(D_METHOD("print_warning"), &Hub::print_warning);
	ClassDB::bind_method(D_METHOD("print_debug"), &Hub::print_debug);
	ClassDB::bind_method(D_METHOD("print_custom"), &Hub::print_custom);
}

void Hub::push_stack(Array stack){
	if (!trace_stack || stack.empty()) return;
	int size = stack.size();
	const String builder("\t--- Stack trace ({layer}): function: {func}, line: {line}, source: {source}.");
	Dictionary formatter;
	for(int i = 0; i < size; i++){
		Variant curr = stack[i];
		if (curr.get_type() != Variant::Type::DICTIONARY) continue;
		Dictionary stack_info = curr;
		formatter["layer"]		= i;
		formatter["func"]		= stack_info.get("function", "");
		formatter["line"]		= stack_info.get("line", -1);
		formatter["source"]		= stack_info.get("source", "res://");
		print_line(builder.format(formatter));
	}
}

void Hub::print_fatal(String err, Array stack){
	print_custom("Fatal", err, stack);
}
void Hub::print_warning(String err, Array stack){
	print_custom("Warning", err, stack);
}
void Hub::print_debug(String err, Array stack){
	print_custom("Debug", err, stack);
}
void Hub::print_custom(String flag, String err, Array stack){
	const String builder("[{flag}] {message}");
	Dictionary formatter;
	formatter["flag"] = flag;
	formatter["message"] = err;
	print_line(builder.format(formatter));
	push_stack(stack);
}