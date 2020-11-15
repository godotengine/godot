// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "base/files/file_path.h"
#include "tools/tool_support.h"
#include "util/file/file_reader.h"
#include "util/file/file_writer.h"
#include "util/net/http_body.h"
#include "util/net/http_multipart_builder.h"
#include "util/net/http_transport.h"
#include "util/string/split_string.h"

namespace crashpad {
namespace {

void Usage(const base::FilePath& me) {
  fprintf(stderr,
"Usage: %" PRFilePath " [OPTION]...\n"
"Send an HTTP POST request.\n"
"  -f, --file=KEY=PATH     upload the file at PATH for the HTTP KEY parameter\n"
"      --no-upload-gzip    don't use gzip compression when uploading\n"
"  -o, --output=FILE       write the response body to FILE instead of stdout\n"
"  -s, --string=KEY=VALUE  set the HTTP KEY parameter to VALUE\n"
"  -u, --url=URL           send the request to URL\n"
"      --help              display this help and exit\n"
"      --version           output version information and exit\n",
          me.value().c_str());
  ToolSupport::UsageTail(me);
}

int HTTPUploadMain(int argc, char* argv[]) {
  const base::FilePath argv0(
      ToolSupport::CommandLineArgumentToFilePathStringType(argv[0]));
  const base::FilePath me(argv0.BaseName());

  enum OptionFlags {
    // “Short” (single-character) options.
    kOptionFile = 'f',
    kOptionOutput = 'o',
    kOptionString = 's',
    kOptionURL = 'u',

    // Long options without short equivalents.
    kOptionLastChar = 255,
    kOptionNoUploadGzip,

    // Standard options.
    kOptionHelp = -2,
    kOptionVersion = -3,
  };

  struct {
    std::string url;
    const char* output;
    bool upload_gzip;
  } options = {};
  options.upload_gzip = true;

  static constexpr option long_options[] = {
      {"file", required_argument, nullptr, kOptionFile},
      {"no-upload-gzip", no_argument, nullptr, kOptionNoUploadGzip},
      {"output", required_argument, nullptr, kOptionOutput},
      {"string", required_argument, nullptr, kOptionString},
      {"url", required_argument, nullptr, kOptionURL},
      {"help", no_argument, nullptr, kOptionHelp},
      {"version", no_argument, nullptr, kOptionVersion},
      {nullptr, 0, nullptr, 0},
  };

  std::vector<std::unique_ptr<FileReader>> readers;
  HTTPMultipartBuilder http_multipart_builder;

  int opt;
  while ((opt = getopt_long(argc, argv, "f:o:s:u:", long_options, nullptr)) !=
         -1) {
    switch (opt) {
      case kOptionFile: {
        std::string key;
        std::string path;
        if (!SplitStringFirst(optarg, '=', &key, &path)) {
          ToolSupport::UsageHint(me, "--file requires KEY=STRING");
          return EXIT_FAILURE;
        }
        base::FilePath file_path(
            ToolSupport::CommandLineArgumentToFilePathStringType(path));
        std::string file_name(
            ToolSupport::FilePathToCommandLineArgument(file_path.BaseName()));

        readers.push_back(std::make_unique<FileReader>());
        FileReader* upload_file_reader = readers.back().get();
        if (!upload_file_reader->Open(file_path)) {
          return EXIT_FAILURE;
        }
        http_multipart_builder.SetFileAttachment(
            key, file_name, upload_file_reader, "application/octet-stream");
        break;
      }
      case kOptionNoUploadGzip: {
        options.upload_gzip = false;
        break;
      }
      case kOptionOutput: {
        options.output = optarg;
        break;
      }
      case kOptionString: {
        std::string key;
        std::string value;
        if (!SplitStringFirst(optarg, '=', &key, &value)) {
          ToolSupport::UsageHint(me, "--string requires KEY=VALUE");
          return EXIT_FAILURE;
        }
        http_multipart_builder.SetFormData(key, value);
        break;
      }
      case kOptionURL:
        options.url = optarg;
        break;
      case kOptionHelp:
        Usage(me);
        return EXIT_SUCCESS;
      case kOptionVersion:
        ToolSupport::Version(me);
        return EXIT_SUCCESS;
      default:
        ToolSupport::UsageHint(me, nullptr);
        return EXIT_FAILURE;
    }
  }
  argc -= optind;
  argv += optind;

  if (options.url.empty()) {
    ToolSupport::UsageHint(me, "--url is required");
    return EXIT_FAILURE;
  }

  if (argc) {
    ToolSupport::UsageHint(me, nullptr);
    return EXIT_FAILURE;
  }

  std::unique_ptr<FileWriterInterface> file_writer;
  if (options.output) {
    FileWriter* file_writer_impl = new FileWriter();
    file_writer.reset(file_writer_impl);
    base::FilePath output_path(
        ToolSupport::CommandLineArgumentToFilePathStringType(options.output));
    if (!file_writer_impl->Open(output_path,
                                FileWriteMode::kTruncateOrCreate,
                                FilePermissions::kWorldReadable)) {
      return EXIT_FAILURE;
    }
  } else {
    file_writer.reset(new WeakFileHandleFileWriter(
        StdioFileHandle(StdioStream::kStandardOutput)));
  }

  http_multipart_builder.SetGzipEnabled(options.upload_gzip);

  std::unique_ptr<HTTPTransport> http_transport(HTTPTransport::Create());
  http_transport->SetURL(options.url);

  HTTPHeaders content_headers;
  http_multipart_builder.PopulateContentHeaders(&content_headers);
  for (const auto& content_header : content_headers) {
    http_transport->SetHeader(content_header.first, content_header.second);
  }

  http_transport->SetBodyStream(http_multipart_builder.GetBodyStream());

  std::string response_body;
  if (!http_transport->ExecuteSynchronously(&response_body)) {
    return EXIT_FAILURE;
  }

  if (!response_body.empty() &&
      !file_writer->Write(&response_body[0], response_body.size())) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace crashpad

#if defined(OS_POSIX)
int main(int argc, char* argv[]) {
  return crashpad::HTTPUploadMain(argc, argv);
}
#elif defined(OS_WIN)
int wmain(int argc, wchar_t* argv[]) {
  return crashpad::ToolSupport::Wmain(argc, argv, crashpad::HTTPUploadMain);
}
#endif  // OS_POSIX
