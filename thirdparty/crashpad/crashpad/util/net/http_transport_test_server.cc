// Copyright 2018 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A one-shot testing webserver.
//
// When invoked, this server will write a short integer to stdout, indicating on
// which port the server is listening. It will then read one integer from stdin,
// indicating the response code to be sent in response to a request. It also
// reads 16 characters from stdin, which, after having "\r\n" appended, will
// form the response body in a successful response (one with code 200). The
// server will process one HTTP request, deliver the prearranged response to the
// client, and write the entire request to stdout. It will then terminate.

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "tools/tool_support.h"
#include "util/file/file_io.h"

#if COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable: 4244 4245 4267 4702)
#endif

#if defined(CRASHPAD_USE_BORINGSSL)
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#define CPPHTTPLIB_ZLIB_SUPPORT
#include "third_party/cpp-httplib/cpp-httplib/httplib.h"

#if COMPILER_MSVC
#pragma warning(pop)
#endif

namespace crashpad {
namespace {

int HttpTransportTestServerMain(int argc, char* argv[]) {
  std::unique_ptr<httplib::Server> server;
  if (argc == 1) {
    server.reset(new httplib::Server);
#if defined(CRASHPAD_USE_BORINGSSL)
  } else if (argc == 3) {
    server.reset(new httplib::SSLServer(argv[1], argv[2]));
#endif
  } else {
    LOG(ERROR) << "usage: http_transport_test_server [cert.pem key.pem]";
    return 1;
  }


  if (!server->is_valid()) {
    LOG(ERROR) << "server creation failed";
    return 1;
  }

  server->set_keep_alive_max_count(1);

  uint16_t response_code;
  char response[16];

  std::string to_stdout;

  server->Post("/upload",
               [&response, &response_code, &server, &to_stdout](
                   const httplib::Request& req, httplib::Response& res) {
                 res.status = response_code;
                 if (response_code == 200) {
                   res.set_content(std::string(response, 16) + "\r\n",
                                   "text/plain");
                 } else {
                   res.set_content("error", "text/plain");
                 }

                 to_stdout += "POST /upload HTTP/1.0\r\n";
                 for (const auto& h : req.headers) {
                   to_stdout += base::StringPrintf(
                       "%s: %s\r\n", h.first.c_str(), h.second.c_str());
                 }
                 to_stdout += "\r\n";
                 to_stdout += req.body;

                 server->stop();
               });

  uint16_t port =
      base::checked_cast<uint16_t>(server->bind_to_any_port("localhost"));

  CheckedWriteFile(
      StdioFileHandle(StdioStream::kStandardOutput), &port, sizeof(port));

  CheckedReadFileExactly(StdioFileHandle(StdioStream::kStandardInput),
                         &response_code,
                         sizeof(response_code));

  CheckedReadFileExactly(StdioFileHandle(StdioStream::kStandardInput),
                         &response,
                         sizeof(response));

  server->listen_after_bind();

  LoggingWriteFile(StdioFileHandle(StdioStream::kStandardOutput),
                   to_stdout.data(),
                   to_stdout.size());

  return 0;
}

}  // namespace
}  // namespace crashpad

#if defined(OS_POSIX) || defined(OS_FUCHSIA)
int main(int argc, char* argv[]) {
  return crashpad::HttpTransportTestServerMain(argc, argv);
}
#elif defined(OS_WIN)
int wmain(int argc, wchar_t* argv[]) {
  return crashpad::ToolSupport::Wmain(
      argc, argv, crashpad::HttpTransportTestServerMain);
}
#endif  // OS_POSIX
