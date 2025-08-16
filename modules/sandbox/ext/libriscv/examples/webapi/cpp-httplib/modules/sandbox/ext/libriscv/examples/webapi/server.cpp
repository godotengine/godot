#include "server.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

static const char* ADDRESS = "127.0.0.1";
static const uint16_t PORT = 1234;
static const uint16_t CACHE_PORT = 8080;

int main(void)
{
    using namespace httplib;
    Server svr;

    svr.Post("/compile", compile);
	svr.Post("/execute", execute);
	svr.Post("/exec",
		[] (const Request& req, Response& res) {
			// take the POST body and send it to a cache
			httplib::Client cli(ADDRESS, CACHE_PORT);
			// find compilation method
			std::string method = "linux";
			if (req.has_param("method")) {
				method = req.get_param_value("method");
			}
			res.set_header("X-Method", method);

			const httplib::Headers headers = {
				{ "X-Method", method }
			};
			// get the source code in the body compiled
			auto cres = cli.Post("/compile", headers,
				req.body, "text/plain");
			if (cres != nullptr)
			{
				// look for failed compilation
				if (cres->has_header("X-Error")) {
					res.status = cres->status;
					res.set_header("X-Error", cres->get_header_value("X-Error"));
					res.set_content(cres->body, "text/plain");
					return;
				}

				res.headers.merge(cres->headers);
				// remove these unnecessary headers
				res.headers.erase("Content-Length");
				res.headers.erase("Content-Type");
				res.headers.erase("Accept-Ranges: bytes");

				if (cres->status == 200) {
					// execute the resulting binary
					auto eres = cli.Post("/execute", headers,
						cres->body, "application/x-riscv");
					if (eres != nullptr)
					{
						// we will get this in the next merge:
						res.headers.erase("X-Binary-Size");
						// merge with the program execution header fields
						res.headers.merge(eres->headers);
						res.headers.erase("Content-Length");
						res.headers.erase("Content-Type");
						res.headers.erase("Accept-Ranges: bytes");

						// return output from execution back to client
						res.status = eres->status;
						res.set_content(eres->body, "text/plain");
					} else {
						res.status = 500;
					}
					return;
				}
				res.status = 200;
				res.set_content(cres->body, "text/plain");
			} else {
				res.status = 500;
				res.set_header("X-Error", "Failed POST to /compile");
			}
		});

	printf("Listening on %s:%u\n", ADDRESS, PORT);
    svr.listen(ADDRESS, PORT);
}

void common_response_fields(httplib::Response& res, int status)
{
	res.status = status;
	res.set_header("Access-Control-Allow-Origin", "*");
	res.set_header("Access-Control-Expose-Headers", "*");
}

void load_file(const std::string& filename, std::vector<uint8_t>& result)
{
    size_t size = 0;
    FILE* f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
		result.clear();
		return;
	}

    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    result.resize(size);
    if (size != fread(result.data(), 1, size, f))
    {
		result.clear();
    }
    fclose(f);
}
