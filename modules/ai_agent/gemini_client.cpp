/**************************************************************************/
/*  gemini_client.cpp                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "gemini_client.h"

#include "core/io/json.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

GeminiClient::GeminiClient() {
    api_endpoint = "generativelanguage.googleapis.com";
    model = GEMINI_PRO;
    request_state = REQUEST_STATE_IDLE;
    temperature = 0.7f;
    max_tokens = 1024;
    system_prompt = "";
    http_client = HTTPClient::create();
}

GeminiClient::~GeminiClient() {
    if (http_client.is_valid()) {
        http_client->close();
    }
}

void GeminiClient::_bind_methods() {
    // Configuration methods
    ClassDB::bind_method(D_METHOD("set_api_key", "api_key"), &GeminiClient::set_api_key);
    ClassDB::bind_method(D_METHOD("get_api_key"), &GeminiClient::get_api_key);
    
    ClassDB::bind_method(D_METHOD("set_model", "model"), &GeminiClient::set_model);
    ClassDB::bind_method(D_METHOD("get_model"), &GeminiClient::get_model);
    
    ClassDB::bind_method(D_METHOD("set_temperature", "temperature"), &GeminiClient::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"), &GeminiClient::get_temperature);
    
    ClassDB::bind_method(D_METHOD("set_max_tokens", "max_tokens"), &GeminiClient::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"), &GeminiClient::get_max_tokens);
    
    ClassDB::bind_method(D_METHOD("set_system_prompt", "system_prompt"), &GeminiClient::set_system_prompt);
    ClassDB::bind_method(D_METHOD("get_system_prompt"), &GeminiClient::get_system_prompt);

    // API methods
    ClassDB::bind_method(D_METHOD("send_message", "message"), &GeminiClient::send_message);
    ClassDB::bind_method(D_METHOD("send_message_with_context", "message", "context"), &GeminiClient::send_message_with_context);
    
    ClassDB::bind_method(D_METHOD("get_request_state"), &GeminiClient::get_request_state);
    ClassDB::bind_method(D_METHOD("get_last_error"), &GeminiClient::get_last_error);
    
    ClassDB::bind_method(D_METHOD("clear_conversation_history"), &GeminiClient::clear_conversation_history);
    ClassDB::bind_method(D_METHOD("get_conversation_history"), &GeminiClient::get_conversation_history);
    
    ClassDB::bind_method(D_METHOD("poll"), &GeminiClient::poll);
    ClassDB::bind_method(D_METHOD("is_request_complete"), &GeminiClient::is_request_complete);
    ClassDB::bind_method(D_METHOD("get_last_response"), &GeminiClient::get_last_response);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "api_key"), "set_api_key", "get_api_key");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "model", PROPERTY_HINT_ENUM, "Gemini Pro,Gemini Pro Vision,Gemini Flash"), "set_model", "get_model");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature", PROPERTY_HINT_RANGE, "0.0,1.0,0.1"), "set_temperature", "get_temperature");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens", PROPERTY_HINT_RANGE, "1,8192,1"), "set_max_tokens", "get_max_tokens");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "system_prompt", PROPERTY_HINT_MULTILINE_TEXT), "set_system_prompt", "get_system_prompt");

    // Enums
    BIND_ENUM_CONSTANT(GEMINI_PRO);
    BIND_ENUM_CONSTANT(GEMINI_PRO_VISION);
    BIND_ENUM_CONSTANT(GEMINI_FLASH);
    
    BIND_ENUM_CONSTANT(REQUEST_STATE_IDLE);
    BIND_ENUM_CONSTANT(REQUEST_STATE_CONNECTING);
    BIND_ENUM_CONSTANT(REQUEST_STATE_REQUESTING);
    BIND_ENUM_CONSTANT(REQUEST_STATE_RECEIVING);
    BIND_ENUM_CONSTANT(REQUEST_STATE_DONE);
    BIND_ENUM_CONSTANT(REQUEST_STATE_ERROR);

    // Signals
    ADD_SIGNAL(MethodInfo("response_received", PropertyInfo(Variant::STRING, "response")));
    ADD_SIGNAL(MethodInfo("error_occurred", PropertyInfo(Variant::STRING, "error")));
}

void GeminiClient::set_api_key(const String &p_api_key) {
    api_key = p_api_key;
}

String GeminiClient::get_api_key() const {
    return api_key;
}

void GeminiClient::set_model(GeminiModel p_model) {
    model = p_model;
}

GeminiClient::GeminiModel GeminiClient::get_model() const {
    return model;
}

void GeminiClient::set_temperature(float p_temperature) {
    temperature = CLAMP(p_temperature, 0.0f, 1.0f);
}

float GeminiClient::get_temperature() const {
    return temperature;
}

void GeminiClient::set_max_tokens(int p_max_tokens) {
    max_tokens = MAX(1, p_max_tokens);
}

int GeminiClient::get_max_tokens() const {
    return max_tokens;
}

void GeminiClient::set_system_prompt(const String &p_system_prompt) {
    system_prompt = p_system_prompt;
}

String GeminiClient::get_system_prompt() const {
    return system_prompt;
}

Error GeminiClient::send_message(const String &message) {
    return send_message_with_context(message, conversation_history);
}

Error GeminiClient::send_message_with_context(const String &message, const Array &context) {
    if (api_key.is_empty()) {
        last_error = "API key not set";
        request_state = REQUEST_STATE_ERROR;
        _emit_error_occurred(last_error);
        return ERR_UNAUTHORIZED;
    }

    if (message.is_empty()) {
        last_error = "Message cannot be empty";
        request_state = REQUEST_STATE_ERROR;
        _emit_error_occurred(last_error);
        return ERR_INVALID_PARAMETER;
    }

    request_state = REQUEST_STATE_CONNECTING;
    last_error = "";
    
    Error err = http_client->connect_to_host(api_endpoint, 443, true);
    if (err != OK) {
        last_error = "Failed to connect to Gemini API";
        request_state = REQUEST_STATE_ERROR;
        _emit_error_occurred(last_error);
        return err;
    }

    // Add message to conversation history
    Dictionary user_message;
    user_message["role"] = "user";
    user_message["content"] = message;
    conversation_history.push_back(user_message);

    return OK;
}

String GeminiClient::_get_model_string() const {
    switch (model) {
        case GEMINI_PRO:
            return "gemini-pro";
        case GEMINI_PRO_VISION:
            return "gemini-pro-vision";
        case GEMINI_FLASH:
            return "gemini-1.5-flash";
        default:
            return "gemini-pro";
    }
}

Dictionary GeminiClient::_create_request_body(const String &prompt, const Array &history) {
    Dictionary body;
    Array contents;

    // Add system prompt if provided
    if (!system_prompt.is_empty()) {
        Dictionary system_message;
        system_message["role"] = "system";
        Dictionary system_parts;
        system_parts["text"] = system_prompt;
        Array system_parts_array;
        system_parts_array.push_back(system_parts);
        system_message["parts"] = system_parts_array;
        contents.push_back(system_message);
    }

    // Add conversation history
    for (int i = 0; i < history.size(); i++) {
        Dictionary msg = history[i];
        if (msg.has("role") && msg.has("content")) {
            Dictionary content_msg;
            content_msg["role"] = msg["role"];
            Dictionary parts;
            parts["text"] = msg["content"];
            Array parts_array;
            parts_array.push_back(parts);
            content_msg["parts"] = parts_array;
            contents.push_back(content_msg);
        }
    }

    // Add current prompt
    Dictionary user_message;
    user_message["role"] = "user";
    Dictionary user_parts;
    user_parts["text"] = prompt;
    Array user_parts_array;
    user_parts_array.push_back(user_parts);
    user_message["parts"] = user_parts_array;
    contents.push_back(user_message);

    body["contents"] = contents;

    // Add generation config
    Dictionary generation_config;
    generation_config["temperature"] = temperature;
    generation_config["maxOutputTokens"] = max_tokens;
    body["generationConfig"] = generation_config;

    return body;
}

void GeminiClient::_setup_headers(Ref<HTTPClient> &client) {
    PackedStringArray headers;
    headers.push_back("Content-Type: application/json");
    headers.push_back("User-Agent: Godot-AI-Agent/1.0");
    client->set_blocking_mode(false);
}

void GeminiClient::poll() {
    if (!http_client.is_valid()) {
        return;
    }

    HTTPClient::Status status = http_client->get_status();

    switch (request_state) {
        case REQUEST_STATE_CONNECTING: {
            if (status == HTTPClient::STATUS_CONNECTED) {
                // Connected, now send the request
                String model_str = _get_model_string();
                String endpoint = "/v1/models/" + model_str + ":generateContent?key=" + api_key;
                
                Dictionary body = _create_request_body(conversation_history.back().operator Dictionary()["content"], conversation_history);
                JSON json;
                String json_string = json.stringify(body);
                
                PackedStringArray headers;
                headers.push_back("Content-Type: application/json");
                headers.push_back("User-Agent: Godot-AI-Agent/1.0");
                
                Error err = http_client->request(HTTPClient::METHOD_POST, endpoint, headers, json_string);
                if (err != OK) {
                    last_error = "Failed to send request";
                    request_state = REQUEST_STATE_ERROR;
                    _emit_error_occurred(last_error);
                    return;
                }
                
                request_state = REQUEST_STATE_REQUESTING;
            } else if (status == HTTPClient::STATUS_CANT_CONNECT || status == HTTPClient::STATUS_CONNECTION_ERROR) {
                last_error = "Connection failed";
                request_state = REQUEST_STATE_ERROR;
                _emit_error_occurred(last_error);
                return;
            }
        } break;

        case REQUEST_STATE_REQUESTING: {
            if (http_client->get_status() == HTTPClient::STATUS_BODY || 
                http_client->get_status() == HTTPClient::STATUS_CONNECTED) {
                request_state = REQUEST_STATE_RECEIVING;
            }
        } break;

        case REQUEST_STATE_RECEIVING: {
            if (http_client->has_response()) {
                PackedByteArray response = http_client->read_response_body_chunk();
                if (response.size() > 0) {
                    _process_response(response);
                    request_state = REQUEST_STATE_DONE;
                }
            }
        } break;

        default:
            break;
    }

    http_client->poll();
}

void GeminiClient::_process_response(const PackedByteArray &response) {
    String response_text = response.get_string_from_utf8();
    
    JSON json;
    Error err = json.parse(response_text);
    
    if (err != OK) {
        last_error = "Failed to parse JSON response";
        request_state = REQUEST_STATE_ERROR;
        _emit_error_occurred(last_error);
        return;
    }

    Dictionary response_dict = json.data;
    
    if (response_dict.has("error")) {
        Dictionary error = response_dict["error"];
        last_error = error.get("message", "Unknown error");
        request_state = REQUEST_STATE_ERROR;
        _emit_error_occurred(last_error);
        return;
    }

    if (response_dict.has("candidates")) {
        Array candidates = response_dict["candidates"];
        if (candidates.size() > 0) {
            Dictionary candidate = candidates[0];
            if (candidate.has("content")) {
                Dictionary content = candidate["content"];
                if (content.has("parts")) {
                    Array parts = content["parts"];
                    if (parts.size() > 0) {
                        Dictionary part = parts[0];
                        if (part.has("text")) {
                            last_response = part["text"];
                            
                            // Add AI response to conversation history
                            Dictionary ai_message;
                            ai_message["role"] = "assistant";
                            ai_message["content"] = last_response;
                            conversation_history.push_back(ai_message);
                            
                            _emit_response_received(last_response);
                            return;
                        }
                    }
                }
            }
        }
    }

    last_error = "Invalid response format";
    request_state = REQUEST_STATE_ERROR;
    _emit_error_occurred(last_error);
}

GeminiClient::RequestState GeminiClient::get_request_state() const {
    return request_state;
}

String GeminiClient::get_last_error() const {
    return last_error;
}

void GeminiClient::clear_conversation_history() {
    conversation_history.clear();
}

Array GeminiClient::get_conversation_history() const {
    return conversation_history;
}

bool GeminiClient::is_request_complete() const {
    return request_state == REQUEST_STATE_DONE || request_state == REQUEST_STATE_ERROR;
}

String GeminiClient::get_last_response() const {
    return last_response;
}

void GeminiClient::_emit_response_received(const String &response) {
    emit_signal("response_received", response);
}

void GeminiClient::_emit_error_occurred(const String &error) {
    emit_signal("error_occurred", error);
}