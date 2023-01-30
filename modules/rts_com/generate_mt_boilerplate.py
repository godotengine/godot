DEBUG_SYNC = True
COMBAT_SERVER_NAME = 'Sentrience'
COMBAT_SERVER_BASE_INC = 'combat_server.h'
COMBAT_SERVER_GENERIC_NAME = 'combat_server'
COMBAT_SERVER_MT_NAME = "SentrienceWrapMT"
COMBAT_SERVER_MT_GENERIC_NAME = 'combat_server_mt'
FREE_RID_METHOD = 'free_rid'
COMMON_FILE_NAME = '''%s_common.gen.h''' % (COMBAT_SERVER_MT_GENERIC_NAME)
GEN_MAX_ARGS = 8
SPACE_PADDING = 3


def define_concat(raw: str, space_padding: int = SPACE_PADDING, tab_len: int = 4) -> str:
	retval = ''
	curr_line = ''
	lines: list[str] = []
	max_char = 0
	curr_char = 0
	for c in raw:
		if c == '\n':
			if curr_line == '':
				continue
			lines.append(curr_line)
			curr_line = ''
			max_char = max(max_char, curr_char)
			curr_char = 0
		elif c == '\t':
			curr_line += c
			curr_char += tab_len
		else:
			curr_line += c
			curr_char += 1
	iter = 0
	max_iter = len(lines)
	total_len = max_char + abs(space_padding)
	for line in lines:
		tab_count = 0
		for c in line:
			if c == '\t':
				tab_count += 1
			else:
				break
		line_len = len(line)
		pads = total_len - (line_len - tab_count) - (tab_count * tab_len)
		retval += line + (' ' * pads)
		if iter == max_iter - 1:
			retval = retval[:-pads]
		else:
			retval += '\\'
			pass
		retval += '\n'
		iter += 1
	return retval + "\n\n"


def generate_include() -> str:
	retval = \
		f'''
#include "core/command_queue_mt.h"
#include "core/os/thread.h"
#include "core/safe_refcount.h"

#include "{COMBAT_SERVER_BASE_INC}"

'''
	return retval


def generate_misc() -> str:
	retval = \
	f'''
#ifdef DEBUG_SYNC
#define SYNC_DEBUG log("sync on: " + String(__FUNCTION__));
#else
#define SYNC_DEBUG
#endif
'''
	if DEBUG_SYNC:
		retval = '#define DEBUG_SYNC\n' + retval
	return retval


def generate_member_variables() -> str:
	retval = \
		f'''
#define _PRIVATE_MEMBERS_
private:
	mutable {COMBAT_SERVER_NAME}* {COMBAT_SERVER_GENERIC_NAME};
	mutable CommandQueueMT command_queue;

	static void _thread_callback(void *_instance);
	void thread_loop();

	Thread::ID server_thread;
	SafeFlag exit;
	Thread thread;
	SafeFlag draw_thread_up;
	bool create_thread;

	SafeNumeric<uint64_t> poll_pending;
	void thread_poll();
	void thread_flush();
	void thread_exit();
	Mutex alloc_mutex;

	uint16 pool_max_size;
'''
	return define_concat(retval)


def generate_funcrid_args(argc: int = 0) -> str:
	decl = ""
	def_decl = "m_type, "
	arg_call = ""
	arg_call_truncated = ""
	for i in range(0, argc):
		decl += f"m_arg{i} p_arg{i}, "
		def_decl += f"m_arg{i}, "
		arg_call += f"p_arg{i}, "
	if argc > 0:
		decl = decl[:-2]
		arg_call_truncated = arg_call[:-2]
	def_decl = def_decl[:-2]
	retval = \
		f'''
#define FUNCRID_{str(argc)}A({def_decl})
private:
	List<RID> m_type##_id_pool;
	int m_type##allocn() {{
		for (int i = 0; i < pool_max_size; i++) {{
			m_type##_id_pool.push_back({COMBAT_SERVER_GENERIC_NAME}->m_type##_create());
		}}
		return 0;
	}}
	void m_type##_free_cached_ids() {{
		while (m_type##_id_pool.size()) {{
			{COMBAT_SERVER_GENERIC_NAME}->{FREE_RID_METHOD}(m_type##_id_pool.front()->get());
			m_type##_id_pool.pop_front();
		}}
	}}
public:
	virtual RID m_type##_create({decl}) {{
		if (Thread::get_caller_id() != server_thread) {{
			RID rid;
			alloc_mutex.lock();
			if (m_type##_id_pool.size() == 0) {{
				int ret;
				command_queue.push_and_ret(this, &{COMBAT_SERVER_MT_NAME}::m_type##allocn, {arg_call} &ret);
				SYNC_DEBUG
			}}
			rid = m_type##_id_pool.front()->get();
			m_type##_id_pool.pop_front();
			alloc_mutex.unlock();
			return rid;
		}} else {{
			return {COMBAT_SERVER_GENERIC_NAME}->m_type##_create({arg_call_truncated});
		}}
	}}
'''
	return define_concat(retval)


def generate_func_call_args(argc: int = 0) -> str:
	decl = ""
	def_decl = "m_type, "
	arg_call = ", "
	arg_call_truncated = ""
	for i in range(0, argc):
		decl += f"m_arg{i} p_arg{i}, "
		def_decl += f"m_arg{i}, "
		arg_call += f"p_arg{i}, "
	if argc > 0:
		decl = decl[:-2]
		arg_call_truncated = arg_call[2:-2]
	arg_call = arg_call[:-2]
	def_decl = def_decl[:-2]
	indiv = \
		f'''
#define FUNC_{str(argc)}A%s(%s {def_decl}) 
public:
virtual %s m_type ({decl}) %s {{
	if (Thread::get_caller_id() != server_thread) {{
		command_queue.%s({COMBAT_SERVER_GENERIC_NAME}, &{COMBAT_SERVER_NAME}::m_type {arg_call});
		%s
	}} else {{
		%s {COMBAT_SERVER_GENERIC_NAME}->m_type({arg_call_truncated});
	}}
}}
'''

	indiv_re = \
		f'''
#define FUNC_{str(argc)}A%s(%s {def_decl}) 
public:
virtual %s m_type ({decl}) %s {{
	if (Thread::get_caller_id() != server_thread) {{
		m_r ret;
		command_queue.%s({COMBAT_SERVER_GENERIC_NAME}, &{COMBAT_SERVER_NAME}::m_type {arg_call}, &ret);
		%s
		return ret;
	}} else {{
		%s {COMBAT_SERVER_GENERIC_NAME}->m_type({arg_call_truncated});
	}}
}}
	'''
	v1 = indiv_re % ("_R", "m_r,", "m_r", "", "push_and_ret", "SYNC_DEBUG", "return")
	v2 = indiv_re % ("_RC", "m_r,", "m_r", "const", "push_and_ret", "SYNC_DEBUG", "return")
	v3 = indiv % ("_S", "", "void", "", "push_and_sync", "SYNC_DEBUG", "")
	v4 = indiv % ("_SC", "", "void", "const", "push_and_sync", "SYNC_DEBUG", "")
	v5 = indiv % ("_C", "", "voi", "const", "push", "", "")
	v6 = indiv % ("", "", "void", "", "push", "", "")

	v1 = define_concat(v1)
	v2 = define_concat(v2)
	v3 = define_concat(v3)
	v4 = define_concat(v4)
	v5 = define_concat(v5)
	v6 = define_concat(v6)

	retval = f'''{v1}{v2}{v3}{v4}{v5}{v6}'''
	return retval

# def functions_convert(fpath: str) -> str:
# 	content = ""
# 	retval = ""
# 	with open(fpath, "r") as file:
# 		content = file.read(file.__sizeof__())
# 	if content == "":
# 		return retval
# 	funcrid = \
# f'''
# #define {COMBAT_SERVER_NAME.upper()}_FUNCRID_AUTOGEN()
# %s
# '''
# 	func_call = \
# f'''
# #define {COMBAT_SERVER_NAME.upper()}_FUNC_AUTOGEN()
# %s
# '''

# 	FIRST_SLASH = 1
# 	SECOND_SLASH = 2
# 	STRING_MODE = 4

# 	config = 0
# 	iterating = False
# 	listening = False
# 	strbuild = ""

# 	fr = []
# 	fc = []

# 	for c in content:
# 		if not iterating:
# 			if c == '/' and not config & STRING_MODE:
# 				if config & SECOND_SLASH:
# 					pass
# 				elif config & FIRST_SLASH:
# 					config += SECOND_SLASH
# 				else:
# 					config += FIRST_SLASH
# 			elif (c == '"' or c == "'") and not config & FIRST_SLASH:
# 				if config & STRING_MODE:
# 					config -= STRING_MODE
# 				else:
# 					config += STRING_MODE
# 			elif c == "!" and config & SECOND_SLASH:
# 				listening = True
# 				continue
# 			elif c == "\n":
# 				if config & FIRST_SLASH:
# 					config -= FIRST_SLASH
# 				elif config & SECOND_SLASH:
# 					config -= SECOND_SLASH
# 				if strbuild == "iterate_start":
# 					strbuild = ""
# 					iterating = True
# 				listening = False
# 			if listening:
# 				strbuild += c
# 		else:
# 			if c == '/' and not config & STRING_MODE:
# 				if config & SECOND_SLASH:
# 					pass
# 				elif config & FIRST_SLASH:
# 					config += SECOND_SLASH
# 				else:
# 					config += FIRST_SLASH
# 			elif (c == '"' or c == "'") and not config & FIRST_SLASH:
# 				if config & STRING_MODE:
# 					config -= STRING_MODE
# 				else:
# 					config += STRING_MODE
# 			elif c == "\n":
# 				pass
			
# 			else:
# 				pass


def generate_common_inc() -> str:
	define_name = f'{COMMON_FILE_NAME.upper().replace(".", "_")}'
	retval = \
		f'''
// {COMMON_FILE_NAME}
// This file is automatically generated
// Do not alternate the content of this file (it would be pointless lololol)

#ifndef {define_name}
#define {define_name}

/*-------------------------------------------------------*/
// Includes
%s
/*-------------------------------------------------------*/
// Misc
%s
/*-------------------------------------------------------*/
// PrivateMembers
%s
/*-------------------------------------------------------*/
// *_create functions
%s
/*-------------------------------------------------------*/
// Generic functions
%s
/*-------------------------------------------------------*/

#endif
'''

	gen_inc = generate_include()
	gen_misc = generate_misc()
	gen_mem = generate_member_variables()
	gen_funcrids = ""
	gen_func_call = ""

	for i in range(0, GEN_MAX_ARGS):
		gen_funcrids += generate_funcrid_args(i)
		gen_func_call += generate_func_call_args(i)

	retval = retval % (gen_inc, gen_misc, gen_mem, gen_funcrids, gen_func_call)

	return retval

def generate_file(path: str):
	with open(path, 'w') as file:
		file.write(generate_common_inc())

generate_file(COMMON_FILE_NAME)
