//
//  server_and_client.cc
//
//  Copyright (c) 2025 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <httplib.h>
#include <iostream>
#include <string>

using namespace httplib;

std::string dump_headers(const Headers &headers) {
  std::string s;
  char buf[BUFSIZ];

  for (auto it = headers.begin(); it != headers.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
    s += buf;
  }

  return s;
}

void logger(const Request &req, const Response &res) {
  std::string s;
  char buf[BUFSIZ];

  s += "================================\n";

  snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
           req.version.c_str(), req.path.c_str());
  s += buf;

  std::string query;
  for (auto it = req.params.begin(); it != req.params.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%c%s=%s",
             (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
             x.second.c_str());
    query += buf;
  }
  snprintf(buf, sizeof(buf), "%s\n", query.c_str());
  s += buf;

  s += dump_headers(req.headers);

  s += "--------------------------------\n";

  snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
  s += buf;
  s += dump_headers(res.headers);
  s += "\n";

  if (!res.body.empty()) { s += res.body; }

  s += "\n";

  std::cout << s;
}

int main(void) {
  // Server
  Server svr;
  svr.set_logger(logger);

  svr.Post("/post", [&](const Request & /*req*/, Response &res) {
    res.set_content("POST", "text/plain");
  });

  auto th = std::thread([&]() { svr.listen("localhost", 8080); });

  auto se = detail::scope_exit([&] {
    svr.stop();
    th.join();
  });

  svr.wait_until_ready();

  // Client
  Client cli{"localhost", 8080};

  std::string body = R"({"hello": "world"})";

  auto res = cli.Post("/post", body, "application/json");
  std::cout << "--------------------------------" << std::endl;
  std::cout << to_string(res.error()) << std::endl;
}
