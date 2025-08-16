#include "httplib.h"
#include <iostream>

int main() {
    using namespace httplib;
    
    // Example usage of parse_accept_header function
    std::cout << "=== Accept Header Parser Example ===" << std::endl;
    
    // Example 1: Simple Accept header
    std::string accept1 = "text/html,application/json,text/plain";
    std::vector<std::string> result1;
    if (detail::parse_accept_header(accept1, result1)) {
        std::cout << "\nExample 1: " << accept1 << std::endl;
        std::cout << "Parsed order:" << std::endl;
        for (size_t i = 0; i < result1.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << result1[i] << std::endl;
        }
    } else {
        std::cout << "\nExample 1: Failed to parse Accept header" << std::endl;
    }
    
    // Example 2: Accept header with quality values
    std::string accept2 = "text/html;q=0.9,application/json;q=1.0,text/plain;q=0.8";
    std::vector<std::string> result2;
    if (detail::parse_accept_header(accept2, result2)) {
        std::cout << "\nExample 2: " << accept2 << std::endl;
        std::cout << "Parsed order (sorted by priority):" << std::endl;
        for (size_t i = 0; i < result2.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << result2[i] << std::endl;
        }
    } else {
        std::cout << "\nExample 2: Failed to parse Accept header" << std::endl;
    }
    
    // Example 3: Browser-like Accept header
    std::string accept3 = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8";
    std::vector<std::string> result3;
    if (detail::parse_accept_header(accept3, result3)) {
        std::cout << "\nExample 3: " << accept3 << std::endl;
        std::cout << "Parsed order:" << std::endl;
        for (size_t i = 0; i < result3.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << result3[i] << std::endl;
        }
    } else {
        std::cout << "\nExample 3: Failed to parse Accept header" << std::endl;
    }
    
    // Example 4: Invalid Accept header examples
    std::cout << "\n=== Invalid Accept Header Examples ===" << std::endl;
    
    std::vector<std::string> invalid_examples = {
        "text/html;q=1.5,application/json",  // q > 1.0
        "text/html;q=-0.1,application/json", // q < 0.0
        "text/html;q=invalid,application/json", // invalid q value
        "invalidtype,application/json",      // invalid media type
        ",application/json"                  // empty entry
    };
    
    for (const auto& invalid_accept : invalid_examples) {
        std::vector<std::string> temp_result;
        std::cout << "\nTesting invalid: " << invalid_accept << std::endl;
        if (detail::parse_accept_header(invalid_accept, temp_result)) {
            std::cout << "  Unexpectedly succeeded!" << std::endl;
        } else {
            std::cout << "  Correctly rejected as invalid" << std::endl;
        }
    }
    
    // Example 4: Server usage example
    std::cout << "\n=== Server Usage Example ===" << std::endl;
    Server svr;
    
    svr.Get("/api/data", [](const Request& req, Response& res) {
        // Get Accept header
        auto accept_header = req.get_header_value("Accept");
        if (accept_header.empty()) {
            accept_header = "*/*";  // Default if no Accept header
        }
        
        // Parse accept header to get preferred content types
        std::vector<std::string> preferred_types;
        if (!detail::parse_accept_header(accept_header, preferred_types)) {
            // Invalid Accept header
            res.status = 400;  // Bad Request
            res.set_content("Invalid Accept header", "text/plain");
            return;
        }
        
        std::cout << "Client Accept header: " << accept_header << std::endl;
        std::cout << "Preferred types in order:" << std::endl;
        for (size_t i = 0; i < preferred_types.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << preferred_types[i] << std::endl;
        }
        
        // Choose response format based on client preference
        std::string response_content;
        std::string content_type;
        
        for (const auto& type : preferred_types) {
            if (type == "application/json" || type == "application/*" || type == "*/*") {
                response_content = "{\"message\": \"Hello, World!\", \"data\": [1, 2, 3]}";
                content_type = "application/json";
                break;
            } else if (type == "text/html" || type == "text/*") {
                response_content = "<html><body><h1>Hello, World!</h1><p>Data: 1, 2, 3</p></body></html>";
                content_type = "text/html";
                break;
            } else if (type == "text/plain") {
                response_content = "Hello, World!\nData: 1, 2, 3";
                content_type = "text/plain";
                break;
            }
        }
        
        if (response_content.empty()) {
            // No supported content type found
            res.status = 406;  // Not Acceptable
            res.set_content("No acceptable content type found", "text/plain");
            return;
        }
        
        res.set_content(response_content, content_type);
        std::cout << "Responding with: " << content_type << std::endl;
    });
    
    std::cout << "Server configured. You can test it with:" << std::endl;
    std::cout << "  curl -H \"Accept: application/json\" http://localhost:8080/api/data" << std::endl;
    std::cout << "  curl -H \"Accept: text/html\" http://localhost:8080/api/data" << std::endl;
    std::cout << "  curl -H \"Accept: text/plain\" http://localhost:8080/api/data" << std::endl;
    std::cout << "  curl -H \"Accept: text/html;q=0.9,application/json;q=1.0\" http://localhost:8080/api/data" << std::endl;
    
    return 0;
}
