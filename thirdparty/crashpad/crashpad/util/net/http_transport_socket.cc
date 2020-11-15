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

#include "util/net/http_transport.h"

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>

#include "base/logging.h"
#include "base/macros.h"
#include "base/numerics/safe_conversions.h"
#include "base/posix/eintr_wrapper.h"
#include "base/scoped_generic.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/stringprintf.h"
#include "util/file/file_io.h"
#include "util/net/http_body.h"
#include "util/net/url.h"
#include "util/stdlib/string_number_conversion.h"
#include "util/string/split_string.h"

#if defined(CRASHPAD_USE_BORINGSSL)
#include <openssl/ssl.h>
#endif

namespace crashpad {

namespace {

constexpr const char kCRLFTerminator[] = "\r\n";

class HTTPTransportSocket final : public HTTPTransport {
 public:
  HTTPTransportSocket() = default;
  ~HTTPTransportSocket() override = default;

  bool ExecuteSynchronously(std::string* response_body) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(HTTPTransportSocket);
};

struct ScopedAddrinfoTraits {
  static addrinfo* InvalidValue() { return nullptr; }
  static void Free(addrinfo* ai) { freeaddrinfo(ai); }
};
using ScopedAddrinfo = base::ScopedGeneric<addrinfo*, ScopedAddrinfoTraits>;

class Stream {
 public:
  virtual ~Stream() = default;

  virtual bool LoggingWrite(const void* data, size_t size) = 0;
  virtual bool LoggingRead(void* data, size_t size) = 0;
  virtual bool LoggingReadToEOF(std::string* contents) = 0;
};

class FdStream : public Stream {
 public:
  explicit FdStream(int fd) : fd_(fd) { CHECK(fd_ >= 0); }

  bool LoggingWrite(const void* data, size_t size) override {
    return LoggingWriteFile(fd_, data, size);
  }

  bool LoggingRead(void* data, size_t size) override {
    return LoggingReadFileExactly(fd_, data, size);
  }

  bool LoggingReadToEOF(std::string* result) override {
    return crashpad::LoggingReadToEOF(fd_, result);
  }

 private:
  int fd_;

  DISALLOW_COPY_AND_ASSIGN(FdStream);
};

#if defined(CRASHPAD_USE_BORINGSSL)
class SSLStream : public Stream {
 public:
  SSLStream() = default;

  bool Initialize(const base::FilePath& root_cert_path,
                  int sock,
                  const std::string& hostname) {
    SSL_library_init();

    ctx_.reset(SSL_CTX_new(TLS_method()));
    if (!ctx_.is_valid()) {
      LOG(ERROR) << "SSL_CTX_new";
      return false;
    }

    if (SSL_CTX_set_min_proto_version(ctx_.get(), TLS1_2_VERSION) <= 0) {
      LOG(ERROR) << "SSL_CTX_set_min_proto_version";
      return false;
    }

    SSL_CTX_set_verify(ctx_.get(), SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_verify_depth(ctx_.get(), 5);

    if (!root_cert_path.empty()) {
      if (SSL_CTX_load_verify_locations(
              ctx_.get(), root_cert_path.value().c_str(), nullptr) <= 0) {
        LOG(ERROR) << "SSL_CTX_load_verify_locations";
        return false;
      }
    } else {
#if defined(OS_LINUX)
      if (SSL_CTX_load_verify_locations(
              ctx_.get(), nullptr, "/etc/ssl/certs") <= 0) {
        LOG(ERROR) << "SSL_CTX_load_verify_locations";
        return false;
      }
#elif defined(OS_FUCHSIA)
      if (SSL_CTX_load_verify_locations(
              ctx_.get(), "/config/ssl/cert.pem", nullptr) <= 0) {
        LOG(ERROR) << "SSL_CTX_load_verify_locations";
        return false;
      }
#else
#error cert store location
#endif
    }

    ssl_.reset(SSL_new(ctx_.get()));
    if (!ssl_.is_valid()) {
      LOG(ERROR) << "SSL_new";
      return false;
    }

    BIO* bio = BIO_new_socket(sock, BIO_NOCLOSE);
    if (!bio) {
      LOG(ERROR) << "BIO_new_socket";
      return false;
    }

    // SSL_set_bio() takes ownership of |bio|.
    SSL_set_bio(ssl_.get(), bio, bio);

    if (SSL_set_tlsext_host_name(ssl_.get(), hostname.c_str()) == 0) {
      LOG(ERROR) << "SSL_set_tlsext_host_name";
      return false;
    }

    if (SSL_connect(ssl_.get()) <= 0) {
      LOG(ERROR) << "SSL_connect";
      return false;
    }

    return true;
  }

  bool LoggingWrite(const void* data, size_t size) override {
    return SSL_write(ssl_.get(), data, size) != 0;
  }

  bool LoggingRead(void* data, size_t size) override {
    return SSL_read(ssl_.get(), data, size) != 0;
  }

  bool LoggingReadToEOF(std::string* contents) override {
    contents->clear();
    char buffer[4096];
    FileOperationResult rv;
    while ((rv = SSL_read(ssl_.get(), buffer, sizeof(buffer))) > 0) {
      DCHECK_LE(static_cast<size_t>(rv), sizeof(buffer));
      contents->append(buffer, rv);
    }
    if (rv < 0) {
      LOG(ERROR) << "SSL_read";
      contents->clear();
      return false;
    }
    return true;
  }

 private:
  struct ScopedSSLCTXTraits {
    static SSL_CTX* InvalidValue() { return nullptr; }
    static void Free(SSL_CTX* ctx) { SSL_CTX_free(ctx); }
  };
  using ScopedSSLCTX = base::ScopedGeneric<SSL_CTX*, ScopedSSLCTXTraits>;

  struct ScopedSSLTraits {
    static SSL* InvalidValue() { return nullptr; }
    static void Free(SSL* ssl) {
      SSL_shutdown(ssl);
      SSL_free(ssl);
    }
  };
  using ScopedSSL = base::ScopedGeneric<SSL*, ScopedSSLTraits>;

  ScopedSSLCTX ctx_;
  ScopedSSL ssl_;

  DISALLOW_COPY_AND_ASSIGN(SSLStream);
};
#endif

bool WaitUntilSocketIsReady(int sock) {
  pollfd pollfds;
  pollfds.fd = sock;
  pollfds.events = POLLIN | POLLPRI | POLLOUT;
  constexpr int kTimeoutMS = 1000;
  int ret = HANDLE_EINTR(poll(&pollfds, 1, kTimeoutMS));
  if (ret < 0) {
    PLOG(ERROR) << "poll";
    return false;
  } else if (ret == 1) {
    if (pollfds.revents & POLLERR) {
      int err;
      socklen_t err_len = sizeof(err);
      if (getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &err_len) != 0) {
        PLOG(ERROR) << "getsockopt";
      } else {
        errno = err;
        PLOG(ERROR) << "POLLERR";
      }
      return false;
    }
    if (pollfds.revents & POLLHUP) {
      return false;
    }
    return (pollfds.revents & POLLIN) != 0 || (pollfds.revents & POLLOUT) != 0;
  }

  // Timeout.
  return false;
}

class ScopedSetNonblocking {
 public:
  explicit ScopedSetNonblocking(int sock) : sock_(sock) {
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) {
      PLOG(ERROR) << "fcntl";
      sock_ = -1;
      return;
    }

    if (fcntl(sock_, F_SETFL, flags | O_NONBLOCK) < 0) {
      PLOG(ERROR) << "fcntl";
      sock_ = -1;
    }
  }

  ~ScopedSetNonblocking() {
    if (sock_ >= 0) {
      int flags = fcntl(sock_, F_GETFL, 0);
      if (flags < 0) {
        PLOG(ERROR) << "fcntl";
        return;
      }

      if (fcntl(sock_, F_SETFL, flags & (~O_NONBLOCK)) < 0) {
        PLOG(ERROR) << "fcntl";
      }
    }
  }

 private:
  int sock_;

  DISALLOW_COPY_AND_ASSIGN(ScopedSetNonblocking);
};

base::ScopedFD CreateSocket(const std::string& hostname,
                            const std::string& port) {
  addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = 0;
  hints.ai_flags = 0;

  addrinfo* addrinfo_raw;
  if (getaddrinfo(hostname.c_str(), port.c_str(), &hints, &addrinfo_raw) < 0) {
    PLOG(ERROR) << "getaddrinfo";
    return base::ScopedFD();
  }
  ScopedAddrinfo addrinfo(addrinfo_raw);

  for (const auto* ap = addrinfo.get(); ap; ap = ap->ai_next) {
    base::ScopedFD result(
        socket(ap->ai_family, ap->ai_socktype, ap->ai_protocol));
    if (!result.is_valid()) {
      continue;
    }

    {
      // Set socket to non-blocking to avoid hanging for a long time if the
      // network is down.
      ScopedSetNonblocking nonblocking(result.get());

      if (HANDLE_EINTR(connect(result.get(), ap->ai_addr, ap->ai_addrlen)) <
          0) {
        if (errno != EINPROGRESS) {
          PLOG(ERROR) << "connect";
        } else if (WaitUntilSocketIsReady(result.get())) {
          return result;
        }
        return base::ScopedFD();
      }

      return result;
    }
  }

  return base::ScopedFD();
}

bool WriteRequest(Stream* stream,
                  const std::string& method,
                  const std::string& resource,
                  const HTTPHeaders& headers,
                  HTTPBodyStream* body_stream) {
  std::string request_line = base::StringPrintf(
      "%s %s HTTP/1.0\r\n", method.c_str(), resource.c_str());
  if (!stream->LoggingWrite(request_line.data(), request_line.size()))
    return false;

  // Write headers, and determine if Content-Length has been specified.
  bool chunked = true;
  size_t content_length = 0;
  for (const auto& header : headers) {
    std::string header_str = base::StringPrintf(
        "%s: %s\r\n", header.first.c_str(), header.second.c_str());
    if (header.first == kContentLength) {
      chunked = !base::StringToSizeT(header.second, &content_length);
      DCHECK(!chunked);
    }

    if (!stream->LoggingWrite(header_str.data(), header_str.size()))
      return false;
  }

  // If no Content-Length, then encode as chunked, so add that header too.
  if (chunked) {
    static constexpr const char kTransferEncodingChunked[] =
        "Transfer-Encoding: chunked\r\n";
    if (!stream->LoggingWrite(kTransferEncodingChunked,
                              strlen(kTransferEncodingChunked))) {
      return false;
    }
  }

  if (!stream->LoggingWrite(kCRLFTerminator, strlen(kCRLFTerminator))) {
    return false;
  }

  FileOperationResult data_bytes;
  do {
    constexpr size_t kCRLFSize = arraysize(kCRLFTerminator) - 1;
    struct __attribute__((packed)) {
      char size[8];
      char crlf[2];
      uint8_t data[32 * 1024 + kCRLFSize];
    } buf;
    static_assert(
        sizeof(buf) == sizeof(buf.size) + sizeof(buf.crlf) + sizeof(buf.data),
        "buf should not have padding");

    // Read a block of data.
    data_bytes =
        body_stream->GetBytesBuffer(buf.data, sizeof(buf.data) - kCRLFSize);
    if (data_bytes == -1) {
      return false;
    }
    DCHECK_GE(data_bytes, 0);
    DCHECK_LE(static_cast<size_t>(data_bytes), sizeof(buf.data) - kCRLFSize);

    void* write_start;
    size_t write_size;

    if (chunked) {
      // Chunked encoding uses the entirety of buf. buf.size is presented in
      // hexadecimal without any leading "0x". The terminating CR and LF will be
      // placed immediately following the used portion of buf.data, even if
      // buf.data is not full.

      // Not snprintf because non-null terminated is desired.
      int rv = sprintf(
          buf.size, "%08x", base::checked_cast<unsigned int>(data_bytes));
      DCHECK_GE(rv, 0);
      DCHECK_EQ(static_cast<size_t>(rv), sizeof(buf.size));
      DCHECK_NE(buf.size[sizeof(buf.size) - 1], '\0');

      memcpy(&buf.crlf[0], kCRLFTerminator, kCRLFSize);
      memcpy(&buf.data[data_bytes], kCRLFTerminator, kCRLFSize);

      // Skip leading zeroes in the chunk size.
      size_t size_len;
      for (size_len = sizeof(buf.size); size_len > 1; --size_len) {
        if (buf.size[sizeof(buf.size) - size_len] != '0') {
          break;
        }
      }

      write_start = buf.crlf - size_len;
      write_size = size_len + sizeof(buf.crlf) + data_bytes + kCRLFSize;
    } else {
      // When not using chunked encoding, only use buf.data.
      write_start = buf.data;
      write_size = data_bytes;
    }

    // write_size will be 0 at EOF in non-chunked mode. Skip the write in that
    // case. In contrast, at EOF in chunked mode, a zero-length chunk must be
    // sent to signal EOF. This will happen when processing the EOF indicated by
    // a 0 return from body_stream()->GetBytesBuffer() above.
    if (write_size != 0) {
      if (!stream->LoggingWrite(write_start, write_size))
        return false;
    }
  } while (data_bytes > 0);

  return true;
}

bool ReadLine(Stream* stream, std::string* line) {
  line->clear();
  for (;;) {
    char byte;
    if (!stream->LoggingRead(&byte, 1)) {
      return false;
    }

    line->append(&byte, 1);
    if (byte == '\n')
      return true;
  }
}

bool StartsWith(const std::string& str, const char* with, size_t len) {
  return str.compare(0, len, with) == 0;
}

bool ReadResponseLine(Stream* stream) {
  std::string response_line;
  if (!ReadLine(stream, &response_line)) {
    LOG(ERROR) << "ReadLine";
    return false;
  }
  static constexpr const char kHttp10[] = "HTTP/1.0 ";
  static constexpr const char kHttp11[] = "HTTP/1.1 ";
  if (!(StartsWith(response_line, kHttp10, strlen(kHttp10)) ||
        StartsWith(response_line, kHttp11, strlen(kHttp11))) ||
      response_line.size() < strlen(kHttp10) + 3 ||
      response_line.at(strlen(kHttp10) + 3) != ' ') {
    return false;
  }
  unsigned int http_status = 0;
  return base::StringToUint(response_line.substr(strlen(kHttp10), 3),
                            &http_status) &&
         http_status >= 200 && http_status <= 203;
}

bool ReadResponseHeaders(Stream* stream, HTTPHeaders* headers) {
  for (;;) {
    std::string line;
    if (!ReadLine(stream, &line)) {
      return false;
    }

    if (line == kCRLFTerminator) {
      return true;
    }

    std::string left, right;
    if (!SplitStringFirst(line, ':', &left, &right)) {
      LOG(ERROR) << "SplitStringFirst";
      return false;
    }
    DCHECK_EQ(right[right.size() - 1], '\n');
    DCHECK_EQ(right[right.size() - 2], '\r');
    DCHECK_EQ(right[0], ' ');
    DCHECK_NE(right[1], ' ');
    right = right.substr(1, right.size() - 3);
    (*headers)[left] = right;
  }
}

bool ReadContentChunked(Stream* stream, std::string* body) {
  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  LOG(ERROR) << "TODO(scottmg): chunked response read";
  return false;
}

bool ReadResponse(Stream* stream, std::string* response_body) {
  response_body->clear();

  if (!ReadResponseLine(stream)) {
    return false;
  }

  HTTPHeaders response_headers;
  if (!ReadResponseHeaders(stream, &response_headers)) {
    return false;
  }

  auto it = response_headers.find("Content-Length");
  size_t len = 0;
  if (it != response_headers.end()) {
    if (!base::StringToSizeT(it->second, &len)) {
      LOG(ERROR) << "invalid Content-Length";
      return false;
    }
  }

  if (len) {
    response_body->resize(len, 0);
    return stream->LoggingRead(&(*response_body)[0], len);
  }

  it = response_headers.find("Transfer-Encoding");
  bool chunked = false;
  if (it != response_headers.end() && it->second == "chunked") {
    chunked = true;
  }

  return chunked ? ReadContentChunked(stream, response_body)
                 : stream->LoggingReadToEOF(response_body);
}

bool HTTPTransportSocket::ExecuteSynchronously(std::string* response_body) {
  std::string scheme, hostname, port, resource;
  if (!CrackURL(url(), &scheme, &hostname, &port, &resource)) {
    return false;
  }

#if !defined(CRASHPAD_USE_BORINGSSL)
  CHECK(scheme == "http") << "Got " << scheme << " for scheme in '" << url()
                          << "'";
#endif

  base::ScopedFD sock(CreateSocket(hostname, port));
  if (!sock.is_valid()) {
    return false;
  }

#if defined(CRASHPAD_USE_BORINGSSL)
  std::unique_ptr<Stream> stream;
  if (scheme == "https") {
    auto ssl_stream = std::make_unique<SSLStream>();
    if (!ssl_stream->Initialize(
            root_ca_certificate_path(), sock.get(), hostname)) {
      LOG(ERROR) << "SSLStream Initialize";
      return false;
    }
    stream = std::move(ssl_stream);
  } else {
    stream = std::make_unique<FdStream>(sock.get());
  }
#else  // CRASHPAD_USE_BORINGSSL
  std::unique_ptr<Stream> stream(std::make_unique<FdStream>(sock.get()));
#endif  // CRASHPAD_USE_BORINGSSL

  if (!WriteRequest(
          stream.get(), method(), resource, headers(), body_stream())) {
    return false;
  }

  if (!ReadResponse(stream.get(), response_body)) {
    return false;
  }

  return true;
}

}  // namespace

// static
std::unique_ptr<HTTPTransport> HTTPTransport::Create() {
  return std::unique_ptr<HTTPTransportSocket>(new HTTPTransportSocket);
}

}  // namespace crashpad
