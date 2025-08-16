cpp-httplib
===========

[![](https://github.com/yhirose/cpp-httplib/workflows/test/badge.svg)](https://github.com/yhirose/cpp-httplib/actions)

A C++11 single-file header-only cross platform HTTP/HTTPS library.

It's extremely easy to set up. Just include the **httplib.h** file in your code!

> [!IMPORTANT]
> This library uses 'blocking' socket I/O. If you are looking for a library with 'non-blocking' socket I/O, this is not the one that you want.

Simple examples
---------------

#### Server (Multi-threaded)

```c++
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "path/to/httplib.h"

// HTTP
httplib::Server svr;

// HTTPS
httplib::SSLServer svr;

svr.Get("/hi", [](const httplib::Request &, httplib::Response &res) {
  res.set_content("Hello World!", "text/plain");
});

svr.listen("0.0.0.0", 8080);
```

#### Client

```c++
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "path/to/httplib.h"

// HTTP
httplib::Client cli("http://yhirose.github.io");

// HTTPS
httplib::Client cli("https://yhirose.github.io");

auto res = cli.Get("/hi");
res->status;
res->body;
```

SSL Support
-----------

SSL support is available with `CPPHTTPLIB_OPENSSL_SUPPORT`. `libssl` and `libcrypto` should be linked.

> [!NOTE]
> cpp-httplib currently supports only version 3.0 or later. Please see [this page](https://www.openssl.org/policies/releasestrat.html) to get more information.

> [!TIP]
> For macOS: cpp-httplib now can use system certs with `CPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN`. `CoreFoundation` and `Security` should be linked with `-framework`.

```c++
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "path/to/httplib.h"

// Server
httplib::SSLServer svr("./cert.pem", "./key.pem");

// Client
httplib::Client cli("https://localhost:1234"); // scheme + host
httplib::SSLClient cli("localhost:1234"); // host
httplib::SSLClient cli("localhost", 1234); // host, port

// Use your CA bundle
cli.set_ca_cert_path("./ca-bundle.crt");

// Disable cert verification
cli.enable_server_certificate_verification(false);

// Disable host verification
cli.enable_server_hostname_verification(false);
```

> [!NOTE]
> When using SSL, it seems impossible to avoid SIGPIPE in all cases, since on some operating systems, SIGPIPE can only be suppressed on a per-message basis, but there is no way to make the OpenSSL library do so for its internal communications. If your program needs to avoid being terminated on SIGPIPE, the only fully general way might be to set up a signal handler for SIGPIPE to handle or ignore it yourself.

### SSL Error Handling

When SSL operations fail, cpp-httplib provides detailed error information through two separate error fields:

```c++
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "path/to/httplib.h"

httplib::Client cli("https://example.com");

auto res = cli.Get("/");
if (!res) {
  // Check the error type
  const auto err = res.error();

  switch (err) {
    case httplib::Error::SSLConnection:
      std::cout << "SSL connection failed, SSL error: "
                << res.ssl_error() << std::endl;
      break;

    case httplib::Error::SSLLoadingCerts:
      std::cout << "SSL cert loading failed, OpenSSL error: "
                << std::hex << res.ssl_openssl_error() << std::endl;
      break;

    case httplib::Error::SSLServerVerification:
      std::cout << "SSL verification failed, X509 error: "
                << res.ssl_openssl_error() << std::endl;
      break;

    case httplib::Error::SSLServerHostnameVerification:
      std::cout << "SSL hostname verification failed, X509 error: "
                << res.ssl_openssl_error() << std::endl;
      break;

    default:
      std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
  }
}
```

Server
------

```c++
#include <httplib.h>

int main(void)
{
  using namespace httplib;

  Server svr;

  svr.Get("/hi", [](const Request& req, Response& res) {
    res.set_content("Hello World!", "text/plain");
  });

  // Match the request path against a regular expression
  // and extract its captures
  svr.Get(R"(/numbers/(\d+))", [&](const Request& req, Response& res) {
    auto numbers = req.matches[1];
    res.set_content(numbers, "text/plain");
  });

  // Capture the second segment of the request path as "id" path param
  svr.Get("/users/:id", [&](const Request& req, Response& res) {
    auto user_id = req.path_params.at("id");
    res.set_content(user_id, "text/plain");
  });

  // Extract values from HTTP headers and URL query params
  svr.Get("/body-header-param", [](const Request& req, Response& res) {
    if (req.has_header("Content-Length")) {
      auto val = req.get_header_value("Content-Length");
    }
    if (req.has_param("key")) {
      auto val = req.get_param_value("key");
    }
    res.set_content(req.body, "text/plain");
  });

  // If the handler takes time to finish, you can also poll the connection state
  svr.Get("/task", [&](const Request& req, Response& res) {
    const char * result = nullptr;
    process.run(); // for example, starting an external process
    while (result == nullptr) {
      sleep(1);
      if (req.is_connection_closed()) {
        process.kill(); // kill the process
        return;
      }
      result = process.stdout(); // != nullptr if the process finishes
    }
    res.set_content(result, "text/plain");
  });

  svr.Get("/stop", [&](const Request& req, Response& res) {
    svr.stop();
  });

  svr.listen("localhost", 1234);
}
```

`Post`, `Put`, `Delete` and `Options` methods are also supported.

### Bind a socket to multiple interfaces and any available port

```cpp
int port = svr.bind_to_any_port("0.0.0.0");
svr.listen_after_bind();
```

### Static File Server

```cpp
// Mount / to ./www directory
auto ret = svr.set_mount_point("/", "./www");
if (!ret) {
  // The specified base directory doesn't exist...
}

// Mount /public to ./www directory
ret = svr.set_mount_point("/public", "./www");

// Mount /public to ./www1 and ./www2 directories
ret = svr.set_mount_point("/public", "./www1"); // 1st order to search
ret = svr.set_mount_point("/public", "./www2"); // 2nd order to search

// Remove mount /
ret = svr.remove_mount_point("/");

// Remove mount /public
ret = svr.remove_mount_point("/public");
```

```cpp
// User defined file extension and MIME type mappings
svr.set_file_extension_and_mimetype_mapping("cc", "text/x-c");
svr.set_file_extension_and_mimetype_mapping("cpp", "text/x-c");
svr.set_file_extension_and_mimetype_mapping("hh", "text/x-h");
```

The following are built-in mappings:

| Extension  |          MIME Type          | Extension  |          MIME Type          |
| :--------- | :-------------------------- | :--------- | :-------------------------- |
| css        | text/css                    | mpga       | audio/mpeg                  |
| csv        | text/csv                    | weba       | audio/webm                  |
| txt        | text/plain                  | wav        | audio/wave                  |
| vtt        | text/vtt                    | otf        | font/otf                    |
| html, htm  | text/html                   | ttf        | font/ttf                    |
| apng       | image/apng                  | woff       | font/woff                   |
| avif       | image/avif                  | woff2      | font/woff2                  |
| bmp        | image/bmp                   | 7z         | application/x-7z-compressed |
| gif        | image/gif                   | atom       | application/atom+xml        |
| png        | image/png                   | pdf        | application/pdf             |
| svg        | image/svg+xml               | mjs, js    | text/javascript             |
| webp       | image/webp                  | json       | application/json            |
| ico        | image/x-icon                | rss        | application/rss+xml         |
| tif        | image/tiff                  | tar        | application/x-tar           |
| tiff       | image/tiff                  | xhtml, xht | application/xhtml+xml       |
| jpeg, jpg  | image/jpeg                  | xslt       | application/xslt+xml        |
| mp4        | video/mp4                   | xml        | application/xml             |
| mpeg       | video/mpeg                  | gz         | application/gzip            |
| webm       | video/webm                  | zip        | application/zip             |
| mp3        | audio/mp3                   | wasm       | application/wasm            |

> [!WARNING]
> These static file server methods are not thread-safe.

### File request handler

```cpp
// The handler is called right before the response is sent to a client
svr.set_file_request_handler([](const Request &req, Response &res) {
  ...
});
```

### Logging

cpp-httplib provides separate logging capabilities for access logs and error logs, similar to web servers like Nginx and Apache.

#### Access Logging

Access loggers capture successful HTTP requests and responses:

```cpp
svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
  std::cout << req.method << " " << req.path << " -> " << res.status << std::endl;
});
```

#### Pre-compression Logging

You can also set a pre-compression logger to capture request/response data before compression is applied:

```cpp
svr.set_pre_compression_logger([](const httplib::Request& req, const httplib::Response& res) {
  // Log before compression - res.body contains uncompressed content
  // Content-Encoding header is not yet set
  your_pre_compression_logger(req, res);
});
```

The pre-compression logger is only called when compression would be applied. For responses without compression, only the access logger is called.

#### Error Logging

Error loggers capture failed requests and connection issues. Unlike access loggers, error loggers only receive the Error and Request information, as errors typically occur before a meaningful Response can be generated.

```cpp
svr.set_error_logger([](const httplib::Error& err, const httplib::Request* req) {
  std::cerr << httplib::to_string(err) << " while processing request";
  if (req) {
    std::cerr << ", client: " << req->get_header_value("X-Forwarded-For")
              << ", request: '" << req->method << " " << req->path << " " << req->version << "'"
              << ", host: " << req->get_header_value("Host");
  }
  std::cerr << std::endl;
});
```

### Error handler

```cpp
svr.set_error_handler([](const auto& req, auto& res) {
  auto fmt = "<p>Error Status: <span style='color:red;'>%d</span></p>";
  char buf[BUFSIZ];
  snprintf(buf, sizeof(buf), fmt, res.status);
  res.set_content(buf, "text/html");
});
```

### Exception handler
The exception handler gets called if a user routing handler throws an error.

```cpp
svr.set_exception_handler([](const auto& req, auto& res, std::exception_ptr ep) {
  auto fmt = "<h1>Error 500</h1><p>%s</p>";
  char buf[BUFSIZ];
  try {
    std::rethrow_exception(ep);
  } catch (std::exception &e) {
    snprintf(buf, sizeof(buf), fmt, e.what());
  } catch (...) { // See the following NOTE
    snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
  }
  res.set_content(buf, "text/html");
  res.status = StatusCode::InternalServerError_500;
});
```

> [!CAUTION]
> if you don't provide the `catch (...)` block for a rethrown exception pointer, an uncaught exception will end up causing the server crash. Be careful!

### Pre routing handler

```cpp
svr.set_pre_routing_handler([](const auto& req, auto& res) {
  if (req.path == "/hello") {
    res.set_content("world", "text/html");
    return Server::HandlerResponse::Handled;
  }
  return Server::HandlerResponse::Unhandled;
});
```

### Post routing handler

```cpp
svr.set_post_routing_handler([](const auto& req, auto& res) {
  res.set_header("ADDITIONAL_HEADER", "value");
});
```

### Pre request handler

```cpp
svr.set_pre_request_handler([](const auto& req, auto& res) {
  if (req.matched_route == "/user/:user") {
    auto user = req.path_params.at("user");
    if (user != "john") {
      res.status = StatusCode::Forbidden_403;
      res.set_content("error", "text/html");
      return Server::HandlerResponse::Handled;
    }
  }
  return Server::HandlerResponse::Unhandled;
});
```

### Form data handling

#### URL-encoded form data ('application/x-www-form-urlencoded')

```cpp
svr.Post("/form", [&](const auto& req, auto& res) {
  // URL query parameters and form-encoded data are accessible via req.params
  std::string username = req.get_param_value("username");
  std::string password = req.get_param_value("password");

  // Handle multiple values with same name
  auto interests = req.get_param_values("interests");

  // Check existence
  if (req.has_param("newsletter")) {
    // Handle newsletter subscription
  }
});
```

#### 'multipart/form-data' POST data

```cpp
svr.Post("/multipart", [&](const Request& req, Response& res) {
  // Access text fields (from form inputs without files)
  std::string username = req.form.get_field("username");
  std::string bio = req.form.get_field("bio");

  // Access uploaded files
  if (req.form.has_file("avatar")) {
    const auto& file = req.form.get_file("avatar");
    std::cout << "Uploaded file: " << file.filename
              << " (" << file.content_type << ") - "
              << file.content.size() << " bytes" << std::endl;

    // Access additional headers if needed
    for (const auto& header : file.headers) {
      std::cout << "Header: " << header.first << " = " << header.second << std::endl;
    }

    // Save to disk
    std::ofstream ofs(file.filename, std::ios::binary);
    ofs << file.content;
  }

  // Handle multiple values with same name
  auto tags = req.form.get_fields("tags");  // e.g., multiple checkboxes
  for (const auto& tag : tags) {
    std::cout << "Tag: " << tag << std::endl;
  }

  auto documents = req.form.get_files("documents");  // multiple file upload
  for (const auto& doc : documents) {
    std::cout << "Document: " << doc.filename
              << " (" << doc.content.size() << " bytes)" << std::endl;
  }

  // Check existence before accessing
  if (req.form.has_field("newsletter")) {
    std::cout << "Newsletter subscription: " << req.form.get_field("newsletter") << std::endl;
  }

  // Get counts for validation
  if (req.form.get_field_count("tags") > 5) {
    res.status = StatusCode::BadRequest_400;
    res.set_content("Too many tags", "text/plain");
    return;
  }

  // Summary
  std::cout << "Received " << req.form.fields.size() << " text fields and "
            << req.form.files.size() << " files" << std::endl;

  res.set_content("Upload successful", "text/plain");
});
```

### Receive content with a content receiver

```cpp
svr.Post("/content_receiver",
  [&](const Request &req, Response &res, const ContentReader &content_reader) {
    if (req.is_multipart_form_data()) {
      // NOTE: `content_reader` is blocking until every form data field is read
      // This approach allows streaming processing of large files
      std::vector<FormData> items;
      content_reader(
        [&](const FormData &item) {
          items.push_back(item);
          return true;
        },
        [&](const char *data, size_t data_length) {
          items.back().content.append(data, data_length);
          return true;
        });

      // Process the received items
      for (const auto& item : items) {
        if (item.filename.empty()) {
          // Text field
          std::cout << "Field: " << item.name << " = " << item.content << std::endl;
        } else {
          // File
          std::cout << "File: " << item.name << " (" << item.filename << ") - "
                    << item.content.size() << " bytes" << std::endl;
        }
      }
    } else {
      std::string body;
      content_reader([&](const char *data, size_t data_length) {
        body.append(data, data_length);
        return true;
      });
    }
  });
```

### Send content with the content provider

```cpp
const size_t DATA_CHUNK_SIZE = 4;

svr.Get("/stream", [&](const Request &req, Response &res) {
  auto data = new std::string("abcdefg");

  res.set_content_provider(
    data->size(), // Content length
    "text/plain", // Content type
    [&, data](size_t offset, size_t length, DataSink &sink) {
      const auto &d = *data;
      sink.write(&d[offset], std::min(length, DATA_CHUNK_SIZE));
      return true; // return 'false' if you want to cancel the process.
    },
    [data](bool success) { delete data; });
});
```

Without content length:

```cpp
svr.Get("/stream", [&](const Request &req, Response &res) {
  res.set_content_provider(
    "text/plain", // Content type
    [&](size_t offset, DataSink &sink) {
      if (/* there is still data */) {
        std::vector<char> data;
        // prepare data...
        sink.write(data.data(), data.size());
      } else {
        sink.done(); // No more data
      }
      return true; // return 'false' if you want to cancel the process.
    });
});
```

### Chunked transfer encoding

```cpp
svr.Get("/chunked", [&](const Request& req, Response& res) {
  res.set_chunked_content_provider(
    "text/plain",
    [](size_t offset, DataSink &sink) {
      sink.write("123", 3);
      sink.write("345", 3);
      sink.write("789", 3);
      sink.done(); // No more data
      return true; // return 'false' if you want to cancel the process.
    }
  );
});
```

With trailer:

```cpp
svr.Get("/chunked", [&](const Request& req, Response& res) {
  res.set_header("Trailer", "Dummy1, Dummy2");
  res.set_chunked_content_provider(
    "text/plain",
    [](size_t offset, DataSink &sink) {
      sink.write("123", 3);
      sink.write("345", 3);
      sink.write("789", 3);
      sink.done_with_trailer({
        {"Dummy1", "DummyVal1"},
        {"Dummy2", "DummyVal2"}
      });
      return true;
    }
  );
});
```

### Send file content

```cpp
svr.Get("/content", [&](const Request &req, Response &res) {
  res.set_file_content("./path/to/content.html");
});

svr.Get("/content", [&](const Request &req, Response &res) {
  res.set_file_content("./path/to/content", "text/html");
});
```

### 'Expect: 100-continue' handler

By default, the server sends a `100 Continue` response for an `Expect: 100-continue` header.

```cpp
// Send a '417 Expectation Failed' response.
svr.set_expect_100_continue_handler([](const Request &req, Response &res) {
  return StatusCode::ExpectationFailed_417;
});
```

```cpp
// Send a final status without reading the message body.
svr.set_expect_100_continue_handler([](const Request &req, Response &res) {
  return res.status = StatusCode::Unauthorized_401;
});
```

### Keep-Alive connection

```cpp
svr.set_keep_alive_max_count(2); // Default is 100
svr.set_keep_alive_timeout(10);  // Default is 5
```

### Timeout

```c++
svr.set_read_timeout(5, 0); // 5 seconds
svr.set_write_timeout(5, 0); // 5 seconds
svr.set_idle_interval(0, 100000); // 100 milliseconds
```

### Set maximum payload length for reading a request body

```c++
svr.set_payload_max_length(1024 * 1024 * 512); // 512MB
```

> [!NOTE]
> When the request body content type is 'www-form-urlencoded', the actual payload length shouldn't exceed `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH`.

### Server-Sent Events

Please see [Server example](https://github.com/yhirose/cpp-httplib/blob/master/example/ssesvr.cc) and [Client example](https://github.com/yhirose/cpp-httplib/blob/master/example/ssecli.cc).

### Default thread pool support

`ThreadPool` is used as the **default** task queue, with a default thread count of 8 or `std::thread::hardware_concurrency() - 1`, whichever is greater. You can change it with `CPPHTTPLIB_THREAD_POOL_COUNT`.

If you want to set the thread count at runtime, there is no convenient way... But here is how.

```cpp
svr.new_task_queue = [] { return new ThreadPool(12); };
```

You can also provide an optional parameter to limit the maximum number
of pending requests, i.e. requests `accept()`ed by the listener but
still waiting to be serviced by worker threads.

```cpp
svr.new_task_queue = [] { return new ThreadPool(/*num_threads=*/12, /*max_queued_requests=*/18); };
```

Default limit is 0 (unlimited). Once the limit is reached, the listener
will shutdown the client connection.

### Override the default thread pool with yours

You can supply your own thread pool implementation according to your need.

```cpp
class YourThreadPoolTaskQueue : public TaskQueue {
public:
  YourThreadPoolTaskQueue(size_t n) {
    pool_.start_with_thread_count(n);
  }

  virtual bool enqueue(std::function<void()> fn) override {
    /* Return true if the task was actually enqueued, or false
     * if the caller must drop the corresponding connection. */
    return pool_.enqueue(fn);
  }

  virtual void shutdown() override {
    pool_.shutdown_gracefully();
  }

private:
  YourThreadPool pool_;
};

svr.new_task_queue = [] {
  return new YourThreadPoolTaskQueue(12);
};
```

Client
------

```c++
#include <httplib.h>
#include <iostream>

int main(void)
{
  httplib::Client cli("localhost", 1234);

  if (auto res = cli.Get("/hi")) {
    if (res->status == StatusCode::OK_200) {
      std::cout << res->body << std::endl;
    }
  } else {
    auto err = res.error();
    std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
  }
}
```

> [!TIP]
> Constructor with scheme-host-port string is now supported!

```c++
httplib::Client cli("localhost");
httplib::Client cli("localhost:8080");
httplib::Client cli("http://localhost");
httplib::Client cli("http://localhost:8080");
httplib::Client cli("https://localhost");
httplib::SSLClient cli("localhost");
```

### Error code

Here is the list of errors from `Result::error()`.

```c++
enum Error {
  Success = 0,
  Unknown,
  Connection,
  BindIPAddress,
  Read,
  Write,
  ExceedRedirectCount,
  Canceled,
  SSLConnection,
  SSLLoadingCerts,
  SSLServerVerification,
  SSLServerHostnameVerification,
  UnsupportedMultipartBoundaryChars,
  Compression,
  ConnectionTimeout,
  ProxyConnection,
};
```

### Client Logging

#### Access Logging

```cpp
cli.set_logger([](const httplib::Request& req, const httplib::Response& res) {
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now() - start_time).count();
  std::cout << "✓ " << req.method << " " << req.path
            << " -> " << res.status << " (" << res.body.size() << " bytes, "
            << duration << "ms)" << std::endl;
});
```

#### Error Logging

```cpp
cli.set_error_logger([](const httplib::Error& err, const httplib::Request* req) {
  std::cerr << "✗ ";
  if (req) {
    std::cerr << req->method << " " << req->path << " ";
  }
  std::cerr << "failed: " << httplib::to_string(err);

  // Add specific guidance based on error type
  switch (err) {
    case httplib::Error::Connection:
      std::cerr << " (verify server is running and reachable)";
      break;
    case httplib::Error::SSLConnection:
      std::cerr << " (check SSL certificate and TLS configuration)";
      break;
    case httplib::Error::ConnectionTimeout:
      std::cerr << " (increase timeout or check network latency)";
      break;
    case httplib::Error::Read:
      std::cerr << " (server may have closed connection prematurely)";
      break;
    default:
      break;
  }
  std::cerr << std::endl;
});
```

### GET with HTTP headers

```c++
httplib::Headers headers = {
  { "Hello", "World!" }
};
auto res = cli.Get("/hi", headers);
```
or
```c++
auto res = cli.Get("/hi", {{"Hello", "World!"}});
```
or
```c++
cli.set_default_headers({
  { "Hello", "World!" }
});
auto res = cli.Get("/hi");
```

### POST

```c++
res = cli.Post("/post", "text", "text/plain");
res = cli.Post("/person", "name=john1&note=coder", "application/x-www-form-urlencoded");
```

### POST with parameters

```c++
httplib::Params params;
params.emplace("name", "john");
params.emplace("note", "coder");

auto res = cli.Post("/post", params);
```
 or

```c++
httplib::Params params{
  { "name", "john" },
  { "note", "coder" }
};

auto res = cli.Post("/post", params);
```

### POST with Multipart Form Data

```c++
httplib::UploadFormDataItems items = {
  { "text1", "text default", "", "" },
  { "text2", "aωb", "", "" },
  { "file1", "h\ne\n\nl\nl\no\n", "hello.txt", "text/plain" },
  { "file2", "{\n  \"world\", true\n}\n", "world.json", "application/json" },
  { "file3", "", "", "application/octet-stream" },
};

auto res = cli.Post("/multipart", items);
```

### PUT

```c++
res = cli.Put("/resource/foo", "text", "text/plain");
```

### DELETE

```c++
res = cli.Delete("/resource/foo");
```

### OPTIONS

```c++
res = cli.Options("*");
res = cli.Options("/resource/foo");
```

### Timeout

```c++
cli.set_connection_timeout(0, 300000); // 300 milliseconds
cli.set_read_timeout(5, 0); // 5 seconds
cli.set_write_timeout(5, 0); // 5 seconds

// This method works the same as curl's `--max-time` option
cli.set_max_timeout(5000); // 5 seconds
```

### Receive content with a content receiver

```c++
std::string body;

auto res = cli.Get("/large-data",
  [&](const char *data, size_t data_length) {
    body.append(data, data_length);
    return true;
  });
```

```cpp
std::string body;

auto res = cli.Get(
  "/stream", Headers(),
  [&](const Response &response) {
    EXPECT_EQ(StatusCode::OK_200, response.status);
    return true; // return 'false' if you want to cancel the request.
  },
  [&](const char *data, size_t data_length) {
    body.append(data, data_length);
    return true; // return 'false' if you want to cancel the request.
  });
```

### Send content with a content provider

```cpp
std::string body = ...;

auto res = cli.Post(
  "/stream", body.size(),
  [](size_t offset, size_t length, DataSink &sink) {
    sink.write(body.data() + offset, length);
    return true; // return 'false' if you want to cancel the request.
  },
  "text/plain");
```

### Chunked transfer encoding

```cpp
auto res = cli.Post(
  "/stream",
  [](size_t offset, DataSink &sink) {
    sink.os << "chunked data 1";
    sink.os << "chunked data 2";
    sink.os << "chunked data 3";
    sink.done();
    return true; // return 'false' if you want to cancel the request.
  },
  "text/plain");
```

### With Progress Callback

```cpp
httplib::Client cli(url, port);

// prints: 0 / 000 bytes => 50% complete
auto res = cli.Get("/", [](size_t len, size_t total) {
  printf("%lld / %lld bytes => %d%% complete\n",
    len, total,
    (int)(len*100/total));
  return true; // return 'false' if you want to cancel the request.
}
);
```

![progress](https://user-images.githubusercontent.com/236374/33138910-495c4ecc-cf86-11e7-8693-2fc6d09615c4.gif)

### Authentication

```cpp
// Basic Authentication
cli.set_basic_auth("user", "pass");

// Digest Authentication
cli.set_digest_auth("user", "pass");

// Bearer Token Authentication
cli.set_bearer_token_auth("token");
```

> [!NOTE]
> OpenSSL is required for Digest Authentication.

### Proxy server support

```cpp
cli.set_proxy("host", port);

// Basic Authentication
cli.set_proxy_basic_auth("user", "pass");

// Digest Authentication
cli.set_proxy_digest_auth("user", "pass");

// Bearer Token Authentication
cli.set_proxy_bearer_token_auth("pass");
```

> [!NOTE]
> OpenSSL is required for Digest Authentication.

### Range

```cpp
httplib::Client cli("httpbin.org");

auto res = cli.Get("/range/32", {
  httplib::make_range_header({{1, 10}}) // 'Range: bytes=1-10'
});
// res->status should be 206.
// res->body should be "bcdefghijk".
```

```cpp
httplib::make_range_header({{1, 10}, {20, -1}})      // 'Range: bytes=1-10, 20-'
httplib::make_range_header({{100, 199}, {500, 599}}) // 'Range: bytes=100-199, 500-599'
httplib::make_range_header({{0, 0}, {-1, 1}})        // 'Range: bytes=0-0, -1'
```

### Keep-Alive connection

```cpp
httplib::Client cli("localhost", 1234);

cli.Get("/hello");         // with "Connection: close"

cli.set_keep_alive(true);
cli.Get("/world");

cli.set_keep_alive(false);
cli.Get("/last-request");  // with "Connection: close"
```

### Redirect

```cpp
httplib::Client cli("yahoo.com");

auto res = cli.Get("/");
res->status; // 301

cli.set_follow_location(true);
res = cli.Get("/");
res->status; // 200
```

### Use a specific network interface

> [!NOTE]
> This feature is not available on Windows, yet.

```cpp
cli.set_interface("eth0"); // Interface name, IP address or host name
```

### Automatic Path Encoding

The client automatically encodes special characters in URL paths by default:

```cpp
httplib::Client cli("https://example.com");

// Automatic path encoding (default behavior)
cli.set_path_encode(true);
auto res = cli.Get("/path with spaces/file.txt"); // Automatically encodes spaces

// Disable automatic path encoding
cli.set_path_encode(false);
auto res = cli.Get("/already%20encoded/path"); // Use pre-encoded paths
```

- `set_path_encode(bool on)` - Controls automatic encoding of special characters in URL paths
  - `true` (default): Automatically encodes spaces, plus signs, newlines, and other special characters
  - `false`: Sends paths as-is without encoding (useful for pre-encoded URLs)

### Performance Note for Local Connections

> [!WARNING]
> On Windows systems with improperly configured IPv6 settings, using "localhost" as the hostname may cause significant connection delays (up to 2 seconds per request) due to DNS resolution issues. This affects both client and server operations. For better performance when connecting to local services, use "127.0.0.1" instead of "localhost".
> 
> See: https://github.com/yhirose/cpp-httplib/issues/366#issuecomment-593004264

```cpp
// May be slower on Windows due to DNS resolution delays
httplib::Client cli("localhost", 8080);
httplib::Server svr;
svr.listen("localhost", 8080);

// Faster alternative for local connections
httplib::Client cli("127.0.0.1", 8080);
httplib::Server svr;
svr.listen("127.0.0.1", 8080);
```

Compression
-----------

The server can apply compression to the following MIME type contents:

  * all text types except text/event-stream
  * image/svg+xml
  * application/javascript
  * application/json
  * application/xml
  * application/protobuf
  * application/xhtml+xml

### Zlib Support

'gzip' compression is available with `CPPHTTPLIB_ZLIB_SUPPORT`. `libz` should be linked.

### Brotli Support

Brotli compression is available with `CPPHTTPLIB_BROTLI_SUPPORT`. Necessary libraries should be linked.
Please see https://github.com/google/brotli for more detail.

### Zstd Support

Zstd compression is available with `CPPHTTPLIB_ZSTD_SUPPORT`. Necessary libraries should be linked.
Please see https://github.com/facebook/zstd for more detail.

### Default `Accept-Encoding` value

The default `Accept-Encoding` value contains all possible compression types. So, the following two examples are same.

```c++
res = cli.Get("/resource/foo");
res = cli.Get("/resource/foo", {{"Accept-Encoding", "br, gzip, deflate, zstd"}});
```

If we don't want a response without compression, we have to set `Accept-Encoding` to an empty string. This behavior is similar to curl.

```c++
res = cli.Get("/resource/foo", {{"Accept-Encoding", ""}});
```

### Compress request body on client

```c++
cli.set_compress(true);
res = cli.Post("/resource/foo", "...", "text/plain");
```

### Compress response body on client

```c++
cli.set_decompress(false);
res = cli.Get("/resource/foo");
res->body; // Compressed data

```

Unix Domain Socket Support
--------------------------

Unix Domain Socket support is available on Linux and macOS.

```c++
// Server
httplib::Server svr;
svr.set_address_family(AF_UNIX).listen("./my-socket.sock", 80);

// Client
httplib::Client cli("./my-socket.sock");
cli.set_address_family(AF_UNIX);
```

"my-socket.sock" can be a relative path or an absolute path. Your application must have the appropriate permissions for the path. You can also use an abstract socket address on Linux. To use an abstract socket address, prepend a null byte ('\x00') to the path.

This library automatically sets the Host header to "localhost" for Unix socket connections, similar to curl's behavior:


URI Encoding/Decoding Utilities
-------------------------------

cpp-httplib provides utility functions for URI encoding and decoding:

```cpp
#include <httplib.h>

std::string url = "https://example.com/search?q=hello world";
std::string encoded = httplib::encode_uri(url);
std::string decoded = httplib::decode_uri(encoded);

std::string param = "hello world";
std::string encoded_component = httplib::encode_uri_component(param);
std::string decoded_component = httplib::decode_uri_component(encoded_component);
```

### Functions

- `encode_uri(const std::string &value)` - Encodes a full URI, preserving reserved characters like `://`, `?`, `&`, `=`
- `decode_uri(const std::string &value)` - Decodes a URI-encoded string
- `encode_uri_component(const std::string &value)` - Encodes a URI component (query parameter, path segment), encoding all reserved characters
- `decode_uri_component(const std::string &value)` - Decodes a URI component

Use `encode_uri()` for full URLs and `encode_uri_component()` for individual query parameters or path segments.


Split httplib.h into .h and .cc
-------------------------------

```console
$ ./split.py -h
usage: split.py [-h] [-e EXTENSION] [-o OUT]

This script splits httplib.h into .h and .cc parts.

optional arguments:
  -h, --help            show this help message and exit
  -e EXTENSION, --extension EXTENSION
                        extension of the implementation file (default: cc)
  -o OUT, --out OUT     where to write the files (default: out)

$ ./split.py
Wrote out/httplib.h and out/httplib.cc
```

Dockerfile for Static HTTP Server
---------------------------------

Dockerfile for static HTTP server is available. Port number of this HTTP server is 80, and it serves static files from `/html` directory in the container.

```bash
> docker build -t cpp-httplib-server .
...

> docker run --rm -it -p 8080:80 -v ./docker/html:/html cpp-httplib-server
Serving HTTP on 0.0.0.0 port 80 ...
192.168.65.1 - - [31/Aug/2024:21:33:56 +0000] "GET / HTTP/1.1" 200 599 "-" "curl/8.7.1"
192.168.65.1 - - [31/Aug/2024:21:34:26 +0000] "GET / HTTP/1.1" 200 599 "-" "Mozilla/5.0 ..."
192.168.65.1 - - [31/Aug/2024:21:34:26 +0000] "GET /favicon.ico HTTP/1.1" 404 152 "-" "Mozilla/5.0 ..."
```

From Docker Hub

```bash
> docker run --rm -it -p 8080:80 -v ./docker/html:/html yhirose4dockerhub/cpp-httplib-server
Serving HTTP on 0.0.0.0 port 80 ...
192.168.65.1 - - [31/Aug/2024:21:33:56 +0000] "GET / HTTP/1.1" 200 599 "-" "curl/8.7.1"
192.168.65.1 - - [31/Aug/2024:21:34:26 +0000] "GET / HTTP/1.1" 200 599 "-" "Mozilla/5.0 ..."
192.168.65.1 - - [31/Aug/2024:21:34:26 +0000] "GET /favicon.ico HTTP/1.1" 404 152 "-" "Mozilla/5.0 ..."
```

NOTE
----

### Regular Expression Stack Overflow

> [!CAUTION]
> When using complex regex patterns in route handlers, be aware that certain patterns may cause stack overflow during pattern matching. This is a known issue with `std::regex` implementations and affects the `dispatch_request()` method.
> 
> ```cpp
> // This pattern can cause stack overflow with large input
> svr.Get(".*", handler);
> ```
> 
> Consider using simpler patterns or path parameters to avoid this issue:
> 
> ```cpp
> // Safer alternatives
> svr.Get("/users/:id", handler);           // Path parameters
> svr.Get(R"(/api/v\d+/.*)", handler);     // More specific patterns
> ```

### g++

g++ 4.8 and below cannot build this library since `<regex>` in the versions are [broken](https://stackoverflow.com/questions/12530406/is-gcc-4-8-or-earlier-buggy-about-regular-expressions).

### Windows

Include `httplib.h` before `Windows.h` or include `Windows.h` by defining `WIN32_LEAN_AND_MEAN` beforehand.

```cpp
#include <httplib.h>
#include <Windows.h>
```

```cpp
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <httplib.h>
```

> [!NOTE]
> cpp-httplib officially supports only the latest Visual Studio. It might work with former versions of Visual Studio, but I can no longer verify it. Pull requests are always welcome for the older versions of Visual Studio unless they break the C++11 conformance.

> [!NOTE]
> Windows 8 or lower, Visual Studio 2015 or lower, and Cygwin and MSYS2 including MinGW are neither supported nor tested.

License
-------

MIT license (© 2025 Yuji Hirose)

Special Thanks To
-----------------

[These folks](https://github.com/yhirose/cpp-httplib/graphs/contributors) made great contributions to polish this library to totally another level from a simple toy!
