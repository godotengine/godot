#include "register_types.h"
#include "vfile_access.h"
#include "basic_scheduler.h"
#include "core/print_string.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/file_access_memory.h"
#include "core/os/os.h"

#include <thread>

#include "markers.gen.h"

static BasicScheduler* BasicSchedulerPtr = nullptr;
static SynchronizationPoint* SynchronizationPointPtr = nullptr;

Vector<uint8_t> get_key(){
	auto exec_path = OS::get_singleton()->get_executable_path();
	auto fa = FileAccess::open(exec_path, FileAccess::READ);
	Vector<uint8_t> re; re.resize(32);
	if (!fa) return re;
	auto f_len = fa->get_len();
	for (uint32_t i = 0; i < 32; i++){
		uint64_t pos = (markers_f64[i] * f_len);
		if (pos >= f_len) pos = f_len - 1;
		fa->seek(pos);
		re.write[i] = fa->get_8();
	}
	fa->close();
	memdelete(fa);
	return re;
}

void __test(){
	auto key = get_key();
	String str_key; str_key.resize(32);
	String addr("http://127.0.0.1/UwU.txt");
	String write_addr("http://127.0.0.1/OwO.txt");
	String dummy_addr("http://127.0.0.1/UmU.txt");
	const String content(R"(
		Dolorem assumenda voluptas accusantium commodi ut doloribus debitis.
		Natus ea incidunt recusandae.
		Error temporibus et officiis nulla minus ut.
		Quas harum voluptatibus velit veniam sit animi...
	)");
	auto vfa = VFileAccess::instantiate();
	vfa->_open(addr, FileAccess::WRITE);
	vfa->store_string(content);
	vfa->close();
	vfa = VFileAccess::free(vfa);

	vfa = VFileAccess::instantiate();
	vfa->_open(addr, FileAccess::READ);

	String composite;
	Vector<uint8_t> bin;
	vfa->seek(0);
	while (!vfa->eof_reached()){
		auto fetched = vfa->get_8();
		bin.push_back(fetched);
		composite += uitos(fetched) + String(" ");
	}
	vfa->close();
	vfa = VFileAccess::free(vfa);

	print_line(String("Composited: ") + composite);
	composite.clear();
	composite.parse_utf8((const char*)bin.ptr(), -1);
	print_line(String("To string: ") + composite);
	composite.clear();
	bin.resize(bin.size() - 1);

	auto fae = memnew(FileAccessEncrypted);
	vfa = VFileAccess::instantiate();
	vfa->_open(write_addr, FileAccess::WRITE);
	if (!vfa->is_open()) return;
	fae->open_and_parse(vfa, key, FileAccessEncrypted::MODE_WRITE_AES256);
	fae->store_buffer((const uint8_t*)bin.ptr(), bin.size());
	fae->close();
	memdelete(fae);

	//--------------------------------------------------------------
	// vfa = VFileAccess::instantiate();
	// vfa->_open(dummy_addr, FileAccess::WRITE);
	// fae = memnew(FileAccessEncrypted);
	// fae->open_and_parse_password(vfa, str_key, FileAccessEncrypted::MODE_WRITE_AES256);
	// fae->store_buffer((const uint8_t*)bin.ptr(), bin.size());
	// fae->close();
	// memdelete(fae);
	// bin.clear();
	// composite.clear();
	// vfa = VFileAccess::instantiate();
	// vfa->_open(dummy_addr, FileAccess::READ);
	// vfa->seek(0);
	// while (!vfa->eof_reached()){
	// 	auto fetched = vfa->get_8();
	// 	bin.push_back(fetched);
	// 	composite += uitos(fetched) + String(" ");
	// }
	// vfa->close();
	// vfa = VFileAccess::free(vfa);
	// print_line(String("Dummy (encrypted): ") + composite);
	//--------------------------------------------------------------

	bin.clear();
	composite.clear();
	vfa = VFileAccess::instantiate();
	vfa->_open(write_addr, FileAccess::READ);
	vfa->seek(0);
	while (!vfa->eof_reached()){
		auto fetched = vfa->get_8();
		bin.push_back(fetched);
		composite += uitos(fetched) + String(" ");
	}
	vfa->close();
	vfa = VFileAccess::free(vfa);
	print_line(String("Composited (encrypted): ") + composite);
	composite.clear();
	composite.parse_utf8((const char*)bin.ptr(), bin.size());
	print_line(String("To string (encrypted): ") + composite);
	composite.clear();

	vfa = VFileAccess::instantiate();
	vfa->_open(write_addr, FileAccess::READ);
	if (!vfa->is_open()) return;
	fae = memnew(FileAccessEncrypted);
	fae->open_and_parse(vfa, key, FileAccessEncrypted::MODE_READ);
	if (!fae->is_open()) return;
	bin.clear();
	fae->seek(0);
	while (!fae->eof_reached()){
		auto fetched = fae->get_8();
		bin.push_back(fetched);
		composite += uitos(fetched) + String(" ");
	}
	fae->close();
	memdelete(fae);
	print_line(String("Composited (decrypted): ") + composite);
	composite.clear();
	if (bin[bin.size() - 1] != 0) bin.push_back(0);
	composite.parse_utf8((const char*)bin.ptr());
	bin.clear();
	print_line(String("To string (decrypted): ") + composite);
	composite.clear();
}

// 1024 MiB
#define LARGE_FILE_STATIC_SIZE 5'368'709'120U

void fake_print(String s){
	s += String("UwU");
	// s.format()
}

void __large_file_test(){
	String path("http://127.0.0.1/unga_bunga.bin");
	MemFS::register_file(path);
	auto vfa = VFileAccess::instantiate();
	double percent = 0.0F;
	double curr = 0.0F;
	vfa->_open(path, FileAccess::WRITE);
	vfa->resize_file(LARGE_FILE_STATIC_SIZE);
	vfa->fill_with();
	vfa->seek(0);
	auto write_epoch = OS::get_singleton()->get_ticks_usec();
	// for (uint64_t i = 0, limit = LARGE_FILE_STATIC_SIZE / sizeof(uint64_t); i < limit; i++){
	// 	vfa->store_64(i);
	// 	curr = (double(i) / double(limit)) * 100.0F;
	// 	if (curr - percent >= 1.0){
	// 		percent = curr;
	// 		print_line(String("Process: ") + rtos(percent) + String("%"));
	// 	}
	// }
	for (uint64_t i = 0, limit = LARGE_FILE_STATIC_SIZE; i < limit; i++){
		vfa->store_8(i);
		curr = (double(i) / double(limit)) * 100.0F;
		if (curr - percent >= 1.0){
			percent = curr;
			print_line(String("Process: ") + rtos(percent) + String("%"));
		}
	}
	auto finish_epoch = OS::get_singleton()->get_ticks_usec();
	percent = 0.0F;
	// for (uint64_t i = 0, limit = LARGE_FILE_STATIC_SIZE / sizeof(uint64_t); i < limit; i++){
	// 	curr = (double(i) / double(limit)) * 100.0F;
	// 	if (curr - percent >= 1.0){
	// 		percent = curr;
	// 		fake_print(String("Process: ") + rtos(percent) + String("%"));
	// 	}
	// }
	for (uint64_t i = 0, limit = LARGE_FILE_STATIC_SIZE; i < limit; i++){
		curr = (double(i) / double(limit)) * 100.0F;
		if (curr - percent >= 1.0){
			percent = curr;
			fake_print(String("Process: ") + rtos(percent) + String("%"));
		}
	}
	auto final_epoch = OS::get_singleton()->get_ticks_usec();
	
	if ((finish_epoch - write_epoch) < (final_epoch - finish_epoch)){
		print_line(String("Bugged tf out"));
	} else {
		auto total_write_time = (finish_epoch - write_epoch) - (final_epoch - finish_epoch);
		double write_sec = (total_write_time / 1000.0F) / 1000.0F;
		double data_written_mib = (vfa->get_len() / 1024.0F) / 1024.0F;
		double data_written_mb  = (vfa->get_len() / 1000.0F) / 1000.0F;
		double speed_mib = data_written_mib / write_sec;
		double speed_mb  = data_written_mb  / write_sec;
		print_line(String("Write time in seconds: ") + rtos(write_sec));
		print_line(String("MemFS speed: ") + rtos(speed_mib) + String(" MiB/s | ") +  rtos(speed_mb) + String (" MB/s"));
	}

	if (vfa->get_len() != LARGE_FILE_STATIC_SIZE) {
		print_line(String("Test failed!!!"));
	}
	print_line(String("Current size: ") + String::humanize_size(vfa->get_len()));
	print_line(String("Size in byte: ") + uitos(LARGE_FILE_STATIC_SIZE));
	vfa->close();
	vfa = VFileAccess::free(vfa);
	MemFS::delete_file(path);
}

void break_test_res(const Vector<uint8_t>& key){
	String out;
	for (uint32_t i = 0; i < 32; i++){
		out += String::num_uint64(key[i], 16, true) + String(" ");
	}
	print_line(String("The following key is found: ") + out);
}

void brute_force_worker(const String& virtualized_gde_path, const String& virtualized_exec_path, const uint64_t& index_start, const uint64_t& index_end, SafeNumeric<uint64_t>* progress_tracker, SafeRefCount *completion){
	VFileAccess *vfa = VFileAccess::open_path(virtualized_exec_path, FileAccess::READ);

	vfa->seek(index_start);
	VFileAccess* vgde = VFileAccess::open_path(virtualized_gde_path, FileAccess::READ);
	FileAccessEncrypted *fae = nullptr;

	Vector<uint8_t> keys;
	for (uint64_t i = index_start; i < index_end; i++){
		keys.push_back(vfa->get_8());
		if (vfa->eof_reached()) return;
		if (keys.size() == 32){
			fae = memnew(FileAccessEncrypted);
			fae->open_and_parse(vgde, keys, FileAccessEncrypted::MODE_READ);
			if (!fae->is_open()){
				i -= 31;
				vfa->seek(i + 1);
				fae->close();
				memdelete(fae);
				fae = nullptr;
				vgde = VFileAccess::open_path(virtualized_gde_path, FileAccess::READ);
				vgde->seek(0);
				keys.clear();
				progress_tracker->increment();
				continue;
			} else {
				break_test_res(keys);
				fae->close();
				memdelete(fae);
				VFileAccess::free(vfa);
				completion->ref();
				return;
			}
		} 
	}
}

void break_test(){
	const String real_gde_loc("D:\\doc\\Godot\\RTS_test\\builds\\CameraController.gde");
	const String virtualized_gde_path("http://127.0.0.1/files/CameraController.gde");
	
	const String exec_path = OS::get_singleton()->get_executable_path();
	const String virtualized_exec_path("http://127.0.0.1/files/godot.windows.opt.tools.64.exe");

	const uint32_t thread_count = 12;

	auto fa = FileAccess::open(real_gde_loc, FileAccess::READ);
	ERR_FAIL_COND(!fa);
	MemFS::virtualize_file(virtualized_gde_path, fa);
	fa->close();
	memdelete(fa);

	fa = FileAccess::open(exec_path, FileAccess::READ);
	ERR_FAIL_COND(!fa);
	MemFS::virtualize_file(virtualized_exec_path, fa);
	uint32_t exec_size = fa->get_len();
	fa->close();
	memdelete(fa);

	uint64_t curr_idx = 0;
	uint64_t span = exec_size / thread_count;
	Vector<std::thread*> threads; threads.resize(thread_count);
	SafeNumeric<uint64_t> progress_tracker;
	SafeRefCount completion; completion.init();
	for (uint32_t i = 0; i < thread_count; i++){
		threads.write[i] = new std::thread(&brute_force_worker, virtualized_gde_path, virtualized_exec_path, i * span, (i * span) + span - 1, &progress_tracker, &completion);
	}

	double percentage = 0.0F;
	double curr = 0.0F;
	auto epoch = OS::get_singleton()->get_ticks_msec();
	while (true) {
		curr = (progress_tracker.get() / double(exec_size)) * 100.0F;
		if (curr - percentage >= 1.0F){
			percentage += 1.0F;
			print_line(String("Progress: ") + rtos(percentage) + String("%"));
		}
		if (percentage == 99.0F) break;
		if (completion.get() > 1){
			break;
		}
	}
	auto end = OS::get_singleton()->get_ticks_msec();
	print_line(String("Brute force finished after ") + uitos(end - epoch) + ("ms"));
	for (uint32_t i = 0; i < thread_count; i++){
		threads.write[i]->join();
		delete threads.write[i];
	}
	ERR_FAIL();
}

void register_enhancer_types() {
	ClassDB::register_class<BasicScheduler>();
	BasicSchedulerPtr = memnew(BasicScheduler);
	SynchronizationPointPtr = new SynchronizationPoint();
	Engine::get_singleton()->add_singleton(Engine::Singleton("BasicScheduler", BasicScheduler::get_singleton()));
	// __test();
	// __large_file_test();
	// break_test();
}
void unregister_enhancer_types() {
	memdelete(BasicSchedulerPtr);
	delete SynchronizationPointPtr;
}