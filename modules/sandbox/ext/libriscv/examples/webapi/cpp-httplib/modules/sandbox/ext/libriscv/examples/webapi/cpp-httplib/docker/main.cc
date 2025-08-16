//
//  main.cc
//
//  Copyright (c) 2025 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <atomic>
#include <chrono>
#include <ctime>
#include <format>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <sstream>

#include <httplib.h>

using namespace httplib;

const auto SERVER_NAME =
    std::format("cpp-httplib-server/{}", CPPHTTPLIB_VERSION);

Server svr;

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\nReceived signal, shutting down gracefully...\n";
    svr.stop();
  }
}

std::string get_time_format() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%d/%b/%Y:%H:%M:%S %z");
  return ss.str();
}

std::string get_error_time_format() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y/%m/%d %H:%M:%S");
  return ss.str();
}

std::string get_client_ip(const Request &req) {
  // Check for X-Forwarded-For header first (common in reverse proxy setups)
  auto forwarded_for = req.get_header_value("X-Forwarded-For");
  if (!forwarded_for.empty()) {
    // Get the first IP if there are multiple
    auto comma_pos = forwarded_for.find(',');
    if (comma_pos != std::string::npos) {
      return forwarded_for.substr(0, comma_pos);
    }
    return forwarded_for;
  }

  // Check for X-Real-IP header
  auto real_ip = req.get_header_value("X-Real-IP");
  if (!real_ip.empty()) { return real_ip; }

  // Fallback to remote address (though cpp-httplib doesn't provide this
  // directly) For demonstration, we'll use a placeholder
  return "127.0.0.1";
}

// NGINX Combined log format:
// $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent
// "$http_referer" "$http_user_agent"
void nginx_access_logger(const Request &req, const Response &res) {
  auto remote_addr = get_client_ip(req);
  std::string remote_user =
      "-"; // cpp-httplib doesn't have built-in auth user tracking
  auto time_local = get_time_format();
  auto request = std::format("{} {} {}", req.method, req.path, req.version);
  auto status = res.status;
  auto body_bytes_sent = res.body.size();
  auto http_referer = req.get_header_value("Referer");
  if (http_referer.empty()) http_referer = "-";
  auto http_user_agent = req.get_header_value("User-Agent");
  if (http_user_agent.empty()) http_user_agent = "-";

  std::cout << std::format("{} - {} [{}] \"{}\" {} {} \"{}\" \"{}\"",
                           remote_addr, remote_user, time_local, request,
                           status, body_bytes_sent, http_referer,
                           http_user_agent)
            << std::endl;
}

// NGINX Error log format:
// YYYY/MM/DD HH:MM:SS [level] message, client: client_ip, request: "request",
// host: "host"
void nginx_error_logger(const Error &err, const Request *req) {
  auto time_local = get_error_time_format();
  std::string level = "error";

  if (req) {
    auto client_ip = get_client_ip(*req);
    auto request =
        std::format("{} {} {}", req->method, req->path, req->version);
    auto host = req->get_header_value("Host");
    if (host.empty()) host = "-";

    std::cerr << std::format("{} [{}] {}, client: {}, request: "
                             "\"{}\", host: \"{}\"",
                             time_local, level, to_string(err), client_ip,
                             request, host)
              << std::endl;
  } else {
    // If no request context, just log the error
    std::cerr << std::format("{} [{}] {}", time_local, level, to_string(err))
              << std::endl;
  }
}

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --host <hostname>        Server hostname (default: localhost)"
            << std::endl;
  std::cout << "  --port <port>            Server port (default: 8080)"
            << std::endl;
  std::cout << "  --mount <mount:path>     Mount point and document root"
            << std::endl;
  std::cout << "                           Format: mount_point:document_root"
            << std::endl;
  std::cout << "                           (default: /:./html)" << std::endl;
  std::cout << "  --version                Show version information"
            << std::endl;
  std::cout << "  --help                   Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  " << program_name
            << " --host localhost --port 8080 --mount /:./html" << std::endl;
  std::cout << "  " << program_name
            << " --host 0.0.0.0 --port 3000 --mount /api:./api" << std::endl;
}

struct ServerConfig {
  std::string hostname = "localhost";
  int port = 8080;
  std::string mount_point = "/";
  std::string document_root = "./html";
};

enum class ParseResult { SUCCESS, HELP_REQUESTED, VERSION_REQUESTED, ERROR };

ParseResult parse_command_line(int argc, char *argv[], ServerConfig &config) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return ParseResult::HELP_REQUESTED;
    } else if (strcmp(argv[i], "--host") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "Error: --host requires a hostname argument" << std::endl;
        print_usage(argv[0]);
        return ParseResult::ERROR;
      }
      config.hostname = argv[++i];
    } else if (strcmp(argv[i], "--port") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "Error: --port requires a port number argument"
                  << std::endl;
        print_usage(argv[0]);
        return ParseResult::ERROR;
      }
      config.port = std::atoi(argv[++i]);
      if (config.port <= 0 || config.port > 65535) {
        std::cerr << "Error: Invalid port number. Must be between 1 and 65535"
                  << std::endl;
        return ParseResult::ERROR;
      }
    } else if (strcmp(argv[i], "--mount") == 0) {
      if (i + 1 >= argc) {
        std::cerr
            << "Error: --mount requires mount_point:document_root argument"
            << std::endl;
        print_usage(argv[0]);
        return ParseResult::ERROR;
      }
      std::string mount_arg = argv[++i];
      auto colon_pos = mount_arg.find(':');
      if (colon_pos == std::string::npos) {
        std::cerr << "Error: --mount argument must be in format "
                     "mount_point:document_root"
                  << std::endl;
        print_usage(argv[0]);
        return ParseResult::ERROR;
      }
      config.mount_point = mount_arg.substr(0, colon_pos);
      config.document_root = mount_arg.substr(colon_pos + 1);

      if (config.mount_point.empty() || config.document_root.empty()) {
        std::cerr
            << "Error: Both mount_point and document_root must be non-empty"
            << std::endl;
        return ParseResult::ERROR;
      }
    } else if (strcmp(argv[i], "--version") == 0) {
      std::cout << CPPHTTPLIB_VERSION << std::endl;
      return ParseResult::VERSION_REQUESTED;
    } else {
      std::cerr << "Error: Unknown option '" << argv[i] << "'" << std::endl;
      print_usage(argv[0]);
      return ParseResult::ERROR;
    }
  }
  return ParseResult::SUCCESS;
}

bool setup_server(Server &svr, const ServerConfig &config) {
  svr.set_logger(nginx_access_logger);
  svr.set_error_logger(nginx_error_logger);

  auto ret = svr.set_mount_point(config.mount_point, config.document_root);
  if (!ret) {
    std::cerr
        << std::format(
               "Error: Cannot mount '{}' to '{}'. Directory may not exist.",
               config.mount_point, config.document_root)
        << std::endl;
    return false;
  }

  svr.set_file_extension_and_mimetype_mapping("html", "text/html");
  svr.set_file_extension_and_mimetype_mapping("htm", "text/html");
  svr.set_file_extension_and_mimetype_mapping("css", "text/css");
  svr.set_file_extension_and_mimetype_mapping("js", "text/javascript");
  svr.set_file_extension_and_mimetype_mapping("json", "application/json");
  svr.set_file_extension_and_mimetype_mapping("xml", "application/xml");
  svr.set_file_extension_and_mimetype_mapping("png", "image/png");
  svr.set_file_extension_and_mimetype_mapping("jpg", "image/jpeg");
  svr.set_file_extension_and_mimetype_mapping("jpeg", "image/jpeg");
  svr.set_file_extension_and_mimetype_mapping("gif", "image/gif");
  svr.set_file_extension_and_mimetype_mapping("svg", "image/svg+xml");
  svr.set_file_extension_and_mimetype_mapping("ico", "image/x-icon");
  svr.set_file_extension_and_mimetype_mapping("pdf", "application/pdf");
  svr.set_file_extension_and_mimetype_mapping("zip", "application/zip");
  svr.set_file_extension_and_mimetype_mapping("txt", "text/plain");

  svr.set_error_handler([](const Request & /*req*/, Response &res) {
    if (res.status == 404) {
      res.set_content(
          std::format(
              "<html><head><title>404 Not Found</title></head>"
              "<body><h1>404 Not Found</h1>"
              "<p>The requested resource was not found on this server.</p>"
              "<hr><p>{}</p></body></html>",
              SERVER_NAME),
          "text/html");
    }
  });

  svr.set_pre_routing_handler([](const Request & /*req*/, Response &res) {
    res.set_header("Server", SERVER_NAME);
    return Server::HandlerResponse::Unhandled;
  });

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  return true;
}

int main(int argc, char *argv[]) {
  ServerConfig config;

  auto result = parse_command_line(argc, argv, config);
  switch (result) {
  case ParseResult::HELP_REQUESTED:
  case ParseResult::VERSION_REQUESTED: return 0;
  case ParseResult::ERROR: return 1;
  case ParseResult::SUCCESS: break;
  }

  if (!setup_server(svr, config)) { return 1; }

  std::cout << "Serving HTTP on " << config.hostname << ":" << config.port
            << std::endl;
  std::cout << "Mount point: " << config.mount_point << " -> "
            << config.document_root << std::endl;
  std::cout << "Press Ctrl+C to shutdown gracefully..." << std::endl;

  auto ret = svr.listen(config.hostname, config.port);

  std::cout << "Server has been shut down." << std::endl;

  return ret ? 0 : 1;
}
