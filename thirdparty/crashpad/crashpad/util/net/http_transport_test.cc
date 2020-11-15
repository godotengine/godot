// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/net/http_transport.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <memory>
#include <utility>
#include <vector>

#include "base/files/file_path.h"
#include "base/format_macros.h"
#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/multiprocess_exec.h"
#include "test/test_paths.h"
#include "util/file/file_io.h"
#include "util/misc/random_string.h"
#include "util/net/http_body.h"
#include "util/net/http_headers.h"
#include "util/net/http_multipart_builder.h"

namespace crashpad {
namespace test {
namespace {

#if defined(OS_WIN)
std::string ToUTF8IfWin(const base::string16& x) {
  return base::UTF16ToUTF8(x);
}
#else
std::string ToUTF8IfWin(const std::string& x) {
  return x;
}
#endif

class HTTPTransportTestFixture : public MultiprocessExec {
 public:
  using RequestValidator =
      void(*)(HTTPTransportTestFixture*, const std::string&);

  HTTPTransportTestFixture(const base::FilePath::StringType& scheme,
                           const HTTPHeaders& headers,
                           std::unique_ptr<HTTPBodyStream> body_stream,
                           uint16_t http_response_code,
                           RequestValidator request_validator)
      : MultiprocessExec(),
        headers_(headers),
        body_stream_(std::move(body_stream)),
        response_code_(http_response_code),
        request_validator_(request_validator),
        cert_(),
        scheme_and_host_() {
    base::FilePath server_path = TestPaths::Executable().DirName().Append(
        FILE_PATH_LITERAL("http_transport_test_server")
#if defined(OS_WIN)
            FILE_PATH_LITERAL(".exe")
#endif
        );

    if (ToUTF8IfWin(scheme) == "http") {
      scheme_and_host_ = "http://localhost";
      SetChildCommand(server_path, nullptr);
    } else {
      std::vector<std::string> args;
      cert_ = TestPaths::BuildArtifact(FILE_PATH_LITERAL("util"),
                                       FILE_PATH_LITERAL("cert"),
                                       TestPaths::FileType::kCertificate);
      args.push_back(ToUTF8IfWin(cert_.value()));
      args.emplace_back(ToUTF8IfWin(
          TestPaths::BuildArtifact(FILE_PATH_LITERAL("util"),
                                   FILE_PATH_LITERAL("key"),
                                   TestPaths::FileType::kCertificate)
              .value()));
      SetChildCommand(server_path, &args);
      scheme_and_host_ = "https://localhost";
    }
  }

  const HTTPHeaders& headers() { return headers_; }

 private:
  void MultiprocessParent() override {
    // Use Logging*File() instead of Checked*File() so that the test can fail
    // gracefully with a gtest assertion if the child does not execute properly.

    // The child will write the HTTP server port number as a packed unsigned
    // short to stdout.
    uint16_t port;
    ASSERT_TRUE(LoggingReadFileExactly(ReadPipeHandle(), &port, sizeof(port)));

    // Then the parent will tell the web server what response code to send
    // for the HTTP request.
    ASSERT_TRUE(LoggingWriteFile(
        WritePipeHandle(), &response_code_, sizeof(response_code_)));

    // The parent will also tell the web server what response body to send back.
    // The web server will only send the response body if the response code is
    // 200.
    const std::string random_string = RandomString();

    ASSERT_TRUE(LoggingWriteFile(WritePipeHandle(),
                                 random_string.c_str(),
                                 random_string.size()));

    // Now execute the HTTP request.
    std::unique_ptr<HTTPTransport> transport(HTTPTransport::Create());
    transport->SetMethod("POST");

    if (!cert_.empty()) {
      transport->SetRootCACertificatePath(cert_);
    }
    transport->SetURL(
        base::StringPrintf("%s:%d/upload", scheme_and_host_.c_str(), port));
    for (const auto& pair : headers_) {
      transport->SetHeader(pair.first, pair.second);
    }
    transport->SetBodyStream(std::move(body_stream_));

    std::string response_body;
    bool success = transport->ExecuteSynchronously(&response_body);
    if (response_code_ >= 200 && response_code_ <= 203) {
      EXPECT_TRUE(success);
      std::string expect_response_body = random_string + "\r\n";
      EXPECT_EQ(response_body, expect_response_body);
    } else {
      EXPECT_FALSE(success);
      EXPECT_TRUE(response_body.empty());
    }

    // Read until the child's stdout closes.
    std::string request;
    char buf[32];
    FileOperationResult bytes_read;
    while ((bytes_read = ReadFile(ReadPipeHandle(), buf, sizeof(buf))) != 0) {
      ASSERT_GE(bytes_read, 0);
      request.append(buf, bytes_read);
    }

    if (request_validator_)
      request_validator_(this, request);
  }

  HTTPHeaders headers_;
  std::unique_ptr<HTTPBodyStream> body_stream_;
  uint16_t response_code_;
  RequestValidator request_validator_;
  base::FilePath cert_;
  std::string scheme_and_host_;
};

constexpr char kMultipartFormData[] = "multipart/form-data";

void GetHeaderField(const std::string& request,
                    const std::string& header,
                    std::string* value) {
  size_t index = request.find(header);
  ASSERT_NE(index, std::string::npos);
  // Since the header is never the first line of the request, it should always
  // be preceded by a CRLF.
  EXPECT_EQ(request[index - 1], '\n');
  EXPECT_EQ(request[index - 2], '\r');

  index += header.length();
  EXPECT_EQ(request[index++], ':');
  // Per RFC 7230 §3.2, there can be one or more spaces or horizontal tabs.
  // For testing purposes, just assume one space.
  EXPECT_EQ(request[index++], ' ');

  size_t header_end = request.find('\r', index);
  ASSERT_NE(header_end, std::string::npos);

  *value = request.substr(index, header_end - index);
}

void GetMultipartBoundary(const std::string& request,
                          std::string* multipart_boundary) {
  std::string content_type;
  GetHeaderField(request, kContentType, &content_type);

  ASSERT_GE(content_type.length(), strlen(kMultipartFormData));
  size_t index = strlen(kMultipartFormData);
  EXPECT_EQ(content_type.substr(0, index), kMultipartFormData);

  EXPECT_EQ(content_type[index++], ';');

  size_t boundary_begin = content_type.find('=', index);
  ASSERT_NE(boundary_begin, std::string::npos);
  EXPECT_EQ(content_type[boundary_begin++], '=');
  if (multipart_boundary) {
    *multipart_boundary = content_type.substr(boundary_begin);
  }
}

constexpr char kBoundaryEq[] = "boundary=";

void ValidFormData(HTTPTransportTestFixture* fixture,
                   const std::string& request) {
  std::string actual_boundary;
  GetMultipartBoundary(request, &actual_boundary);

  const auto& content_type = fixture->headers().find(kContentType);
  ASSERT_NE(content_type, fixture->headers().end());

  size_t boundary = content_type->second.find(kBoundaryEq);
  ASSERT_NE(boundary, std::string::npos);
  std::string expected_boundary =
      content_type->second.substr(boundary + strlen(kBoundaryEq));
  EXPECT_EQ(actual_boundary, expected_boundary);

  size_t body_start = request.find("\r\n\r\n");
  ASSERT_NE(body_start, std::string::npos);
  body_start += 4;

  std::string expected = "--" + expected_boundary + "\r\n";
  expected += "Content-Disposition: form-data; name=\"key1\"\r\n\r\n";
  expected += "test\r\n";
  ASSERT_LT(body_start + expected.length(), request.length());
  EXPECT_EQ(request.substr(body_start, expected.length()), expected);

  body_start += expected.length();

  expected = "--" + expected_boundary + "\r\n";
  expected += "Content-Disposition: form-data; name=\"key2\"\r\n\r\n";
  expected += "--abcdefg123\r\n";
  expected += "--" + expected_boundary + "--\r\n";
  ASSERT_EQ(request.length(), body_start + expected.length());
  EXPECT_EQ(request.substr(body_start), expected);
}

class HTTPTransport
    : public testing::TestWithParam<base::FilePath::StringType> {};

TEST_P(HTTPTransport, ValidFormData) {
  HTTPMultipartBuilder builder;
  builder.SetFormData("key1", "test");
  builder.SetFormData("key2", "--abcdefg123");

  HTTPHeaders headers;
  builder.PopulateContentHeaders(&headers);

  HTTPTransportTestFixture test(GetParam(),
      headers, builder.GetBodyStream(), 200, &ValidFormData);
  test.Run();
}

TEST_P(HTTPTransport, ValidFormData_Gzip) {
  HTTPMultipartBuilder builder;
  builder.SetGzipEnabled(true);
  builder.SetFormData("key1", "test");
  builder.SetFormData("key2", "--abcdefg123");

  HTTPHeaders headers;
  builder.PopulateContentHeaders(&headers);

  HTTPTransportTestFixture test(
      GetParam(), headers, builder.GetBodyStream(), 200, &ValidFormData);
  test.Run();
}

constexpr char kTextPlain[] = "text/plain";

void ErrorResponse(HTTPTransportTestFixture* fixture,
                   const std::string& request) {
  std::string content_type;
  GetHeaderField(request, kContentType, &content_type);
  EXPECT_EQ(content_type, kTextPlain);
}

TEST_P(HTTPTransport, ErrorResponse) {
  HTTPMultipartBuilder builder;
  HTTPHeaders headers;
  headers[kContentType] = kTextPlain;
  HTTPTransportTestFixture test(GetParam(), headers, builder.GetBodyStream(),
      404, &ErrorResponse);
  test.Run();
}

constexpr char kTextBody[] = "hello world";

void UnchunkedPlainText(HTTPTransportTestFixture* fixture,
                        const std::string& request) {
  std::string header_value;
  GetHeaderField(request, kContentType, &header_value);
  EXPECT_EQ(header_value, kTextPlain);

  GetHeaderField(request, kContentLength, &header_value);
  const auto& content_length = fixture->headers().find(kContentLength);
  ASSERT_NE(content_length, fixture->headers().end());
  EXPECT_EQ(header_value, content_length->second);

  size_t body_start = request.rfind("\r\n");
  ASSERT_NE(body_start, std::string::npos);

  EXPECT_EQ(request.substr(body_start + 2), kTextBody);
}

TEST_P(HTTPTransport, UnchunkedPlainText) {
  std::unique_ptr<HTTPBodyStream> body_stream(
      new StringHTTPBodyStream(kTextBody));

  HTTPHeaders headers;
  headers[kContentType] = kTextPlain;
  headers[kContentLength] = base::StringPrintf("%" PRIuS, strlen(kTextBody));

  HTTPTransportTestFixture test(GetParam(),
      headers, std::move(body_stream), 200, &UnchunkedPlainText);
  test.Run();
}

void RunUpload33k(const base::FilePath::StringType& scheme,
                  bool has_content_length) {
  // On macOS, NSMutableURLRequest winds up calling into a CFReadStream’s Read()
  // callback with a 32kB buffer. Make sure that it’s able to get everything
  // when enough is available to fill this buffer, requiring more than one
  // Read().

  std::string request_string(33 * 1024, 'a');
  std::unique_ptr<HTTPBodyStream> body_stream(
      new StringHTTPBodyStream(request_string));

  HTTPHeaders headers;
  headers[kContentType] = "application/octet-stream";
  if (has_content_length) {
    headers[kContentLength] =
        base::StringPrintf("%" PRIuS, request_string.size());
  }
  HTTPTransportTestFixture test(
      scheme,
      headers,
      std::move(body_stream),
      200,
      [](HTTPTransportTestFixture* fixture, const std::string& request) {
        size_t body_start = request.rfind("\r\n");
        EXPECT_EQ(request.size() - body_start, 33 * 1024u + 2);
      });
  test.Run();
}

TEST_P(HTTPTransport, Upload33k) {
  RunUpload33k(GetParam(), true);
}

TEST_P(HTTPTransport, Upload33k_LengthUnknown) {
  // The same as Upload33k, but without declaring Content-Length ahead of time.
  RunUpload33k(GetParam(), false);
}

#if defined(CRASHPAD_USE_BORINGSSL)
// The test server requires BoringSSL or OpenSSL, so https in tests can only be
// enabled where that's readily available. Additionally on Linux, the bots fail
// lacking libcrypto.so.1.1, so disabled there for now. On Mac, they could also
// likely be enabled relatively easily, if HTTPTransportMac learned to respect
// the user-supplied cert.
INSTANTIATE_TEST_CASE_P(HTTPTransport,
                        HTTPTransport,
                        testing::Values(FILE_PATH_LITERAL("http"),
                                        FILE_PATH_LITERAL("https")));
#else
INSTANTIATE_TEST_CASE_P(HTTPTransport,
                        HTTPTransport,
                        testing::Values(FILE_PATH_LITERAL("http")));
#endif

}  // namespace
}  // namespace test
}  // namespace crashpad
