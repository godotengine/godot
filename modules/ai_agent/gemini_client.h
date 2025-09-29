/**************************************************************************/
/*  gemini_client.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "core/io/http_client.h"
#include "core/io/json.h"
#include "core/object/ref_counted.h"
#include "scene/main/node.h"

class GeminiClient : public RefCounted {
    GDCLASS(GeminiClient, RefCounted);

public:
    enum GeminiModel {
        GEMINI_PRO,
        GEMINI_PRO_VISION,
        GEMINI_FLASH,
    };

    enum RequestState {
        REQUEST_STATE_IDLE,
        REQUEST_STATE_CONNECTING,
        REQUEST_STATE_REQUESTING,
        REQUEST_STATE_RECEIVING,
        REQUEST_STATE_DONE,
        REQUEST_STATE_ERROR,
    };

private:
    String api_key;
    String api_endpoint;
    GeminiModel model;
    Ref<HTTPClient> http_client;
    RequestState request_state;
    String last_error;
    Array conversation_history;
    
    // Request parameters
    float temperature;
    int max_tokens;
    String system_prompt;

    void _setup_headers(Ref<HTTPClient> &client);
    Dictionary _create_request_body(const String &prompt, const Array &history = Array());
    String _get_model_string() const;
    void _process_response(const PackedByteArray &response);

protected:
    static void _bind_methods();

public:
    GeminiClient();
    ~GeminiClient();

    // Configuration methods
    void set_api_key(const String &p_api_key);
    String get_api_key() const;
    
    void set_model(GeminiModel p_model);
    GeminiModel get_model() const;
    
    void set_temperature(float p_temperature);
    float get_temperature() const;
    
    void set_max_tokens(int p_max_tokens);
    int get_max_tokens() const;
    
    void set_system_prompt(const String &p_system_prompt);
    String get_system_prompt() const;

    // API methods
    Error send_message(const String &message);
    Error send_message_with_context(const String &message, const Array &context);
    
    RequestState get_request_state() const;
    String get_last_error() const;
    
    void clear_conversation_history();
    Array get_conversation_history() const;
    
    // Async processing
    void poll();
    bool is_request_complete() const;
    String get_last_response() const;

    // Signals
    void _emit_response_received(const String &response);
    void _emit_error_occurred(const String &error);

private:
    String last_response;
};

VARIANT_ENUM_CAST(GeminiClient::GeminiModel)
VARIANT_ENUM_CAST(GeminiClient::RequestState)