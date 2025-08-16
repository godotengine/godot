#include "server.hpp"
using namespace httplib;

static int create_folder(const std::string& folder);
static int python_sanitize_compile(const std::string& pbase, const std::string& pdir, const std::string& met);
static int write_file(const std::string& file, const std::string& text);

static std::string project_base() {
	return "/tmp/programs";
}
static std::string project_dir(const int id) {
	return "program" + std::to_string(id);
}
static std::string project_path(const int id) {
	return project_base() + "/program" + std::to_string(id);
}

void compile(const Request& req, Response& res)
{
	static size_t request_ID = 0;
	const size_t program_id = request_ID++;
	res.set_header("X-Program-Id", std::to_string(program_id));

	// find compiler method
	std::string method = "linux";
	if (req.has_param("method")) {
		method = req.get_param_value("method");
	}
	// ... which can be overriden by X-Method header field
	if (req.has_header("X-Method")) {
		method = req.get_header_value("X-Method");
	}
	res.set_header("X-Method", method);

	const std::string progpath = project_path(program_id);
	// Create temporary project base folder
	if (create_folder(project_base()) < 0) {
		if (errno != EEXIST) {
			res.status = 500;
			res.set_header("X-Error", "Failed to create program base folder");
			return;
		}
	}
	// Create temporary project folder
	if (create_folder(progpath) < 0) {
		if (errno != EEXIST) {
			res.status = 500;
			res.set_header("X-Error", "Failed to create project folder");
			return;
		}
	}

	// write code into project folder
	if (write_file(progpath + "/code.cpp", req.body) != 0) {
		res.status = 500;
		res.set_header("X-Error", "Failed to write codefile");
		return;
	}

	// sanitize + compile code
	asm("" : : : "memory");
	const uint64_t c0 = monotonic_micros_now();
	asm("" : : : "memory");
	const int cc = python_sanitize_compile(project_base(), project_dir(program_id), method);
	asm("" : : : "memory");
	const uint64_t c1 = monotonic_micros_now();
	asm("" : : : "memory");
	res.set_header("X-Compile-Time", std::to_string(c1 - c0));
	res.set_header("X-Time-Unit", "10e-6");
	if (cc != 0) {
		std::vector<uint8_t> vec;
		load_file(progpath + "/status.txt", vec);
		res.status = 200;
		res.set_header("X-Error", "Compilation failed");
		res.set_content((const char*) vec.data(), vec.size(), "text/plain");
		res.set_header("Cache-Control", "s-max-age=86400");
		return;
	}

	// load binary and execute code
	auto* binary = new std::vector<uint8_t> ();
	load_file(progpath + "/binary", *binary);
	res.set_header("X-Binary-Size", std::to_string(binary->size()));
	if (binary->empty()) {
		res.status = 200;
		res.set_header("X-Error", "Failed to open binary");
		return;
	}
	// indicate caching and send it
	res.set_header("Cache-Control", "s-max-age=86400");
	res.set_content_provider(
		binary->size(), // Content length
		[binary] (uint64_t offset, uint64_t length, DataSink &sink) {
			const auto* d = (const char*) binary->data();
			sink.write(&d[offset], length);
		},
		[binary] {
			delete binary;
		});
}

int create_folder(const std::string& folder)
{
	return mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

int python_sanitize_compile(const std::string& pbase, const std::string& pdir, const std::string& method)
{
	auto cmd = "/usr/bin/python3 sanitize.py " + pbase + " " + pdir + " " + method;
	return system(cmd.c_str());
}

int write_file(const std::string& file, const std::string& text)
{
	FILE* fp = fopen(file.c_str(), "wb");
	if (fp == nullptr) return -1;
    fwrite(text.data(), text.size(), 1, fp);
    fclose(fp);
	return 0;
}
