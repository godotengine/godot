#include "crow_all.h"

class CustomLogger : public crow::ILogHandler {
public:
  void log(std::string, crow::LogLevel) {}
};

int main() {
  CustomLogger logger;
  crow::logger::setHandler(&logger);

  crow::SimpleApp app;

  CROW_ROUTE(app, "/")([]() { return "Hello world!"; });

  app.port(8080).multithreaded().run();
}
