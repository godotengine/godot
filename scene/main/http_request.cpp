#include "http_request.h"

void HTTPRequest::_redirect_request(const String& p_new_url) {


}

Error HTTPRequest::_request() {

	return client->connect(url,port,use_ssl,validate_ssl);
}

Error HTTPRequest::request(const String& p_url, const Vector<String>& p_custom_headers, bool p_ssl_validate_domain) {

	ERR_FAIL_COND_V(!is_inside_tree(),ERR_UNCONFIGURED);
	if ( requesting ) {
		ERR_EXPLAIN("HTTPRequest is processing a request. Wait for completion or cancel it before attempting a new one.");
		ERR_FAIL_V(ERR_BUSY);
	}

	url=p_url;
	use_ssl=false;

	request_string="";
	port=80;
	headers=p_custom_headers;
	request_sent=false;
	got_response=false;
	validate_ssl=p_ssl_validate_domain;
	body_len=-1;
	body.resize(0);
	redirections=0;

	print_line("1 url: "+url);
	if (url.begins_with("http://")) {

		url=url.substr(7,url.length()-7);
		print_line("no SSL");

	} else if (url.begins_with("https://")) {
		url=url.substr(8,url.length()-8);
		use_ssl=true;
		port=443;
		print_line("yes SSL");
	} else {
		ERR_EXPLAIN("Malformed URL");
		ERR_FAIL_V(ERR_INVALID_PARAMETER);
	}

	print_line("2 url: "+url);

	int slash_pos = url.find("/");

	if (slash_pos!=-1) {
		request_string=url.substr(slash_pos,url.length());
		url=url.substr(0,slash_pos);
		print_line("request string: "+request_string);
	} else {
		request_string="/";
		print_line("no request");
	}

	print_line("3 url: "+url);

	int colon_pos = url.find(":");
	if (colon_pos!=-1) {
		port=url.substr(colon_pos+1,url.length()).to_int();
		url=url.substr(0,colon_pos);
		ERR_FAIL_COND_V(port<1 || port > 65535,ERR_INVALID_PARAMETER);
	}

	print_line("4 url: "+url);

	bool has_user_agent=false;
	bool has_accept=false;

	for(int i=0;i<headers.size();i++) {

		if (headers[i].findn("user-agent:")==0)
			has_user_agent=true;
		if (headers[i].findn("Accept:")==0)
			has_accept=true;
	}

	if (!has_user_agent) {
		headers.push_back("User-Agent: GodotEngine/"+String(VERSION_MKSTRING)+" ("+OS::get_singleton()->get_name()+")");
	}

	if (!has_accept) {
		headers.push_back("Accept: */*");
	}


	Error err = _request();

	if (err==OK) {
		set_process(true);
		requesting=true;
	}


	return err;
}


void HTTPRequest::cancel_request() {

	if (!requesting)
		return;

	if (!use_threads) {
		set_process(false);
	}

	client->close();
	body.resize(0);
	got_response=false;
	response_code=-1;
	body_len=-1;
	request_sent=false;
	requesting=false;
}


bool HTTPRequest::_update_connection() {

	switch( client->get_status() ) {
		case HTTPClient::STATUS_DISCONNECTED: {
			return true; //end it, since it's doing something
		} break;
		case HTTPClient::STATUS_RESOLVING: {
			client->poll();
			//must wait
			return false;
		} break;
		case HTTPClient::STATUS_CANT_RESOLVE: {
			call_deferred("emit_signal","request_completed",RESULT_CANT_RESOLVE,0,StringArray(),ByteArray());
			return true;

		} break;
		case HTTPClient::STATUS_CONNECTING: {
			client->poll();
			//must wait
			return false;
		} break; //connecting to ip
		case HTTPClient::STATUS_CANT_CONNECT: {

			call_deferred("emit_signal","request_completed",RESULT_CANT_CONNECT,0,StringArray(),ByteArray());
			return true;

		} break;
		case HTTPClient::STATUS_CONNECTED: {

			if (request_sent) {

				if (!got_response) {

					//no body

					got_response=true;
					response_code=client->get_response_code();
					List<String> rheaders;
					client->get_response_headers(&rheaders);
					response_headers.resize(0);
					for (List<String>::Element *E=rheaders.front();E;E=E->next()) {
						print_line("HEADER: "+E->get());
						response_headers.push_back(E->get());
					}

					if (response_code==301) {
						//redirect
						if (max_redirects>=0 && redirections>=max_redirects) {

							call_deferred("emit_signal","request_completed",RESULT_REDIRECT_LIMIT_REACHED,response_code,response_headers,ByteArray());
							return true;
						}

						String new_request;

						for (List<String>::Element *E=rheaders.front();E;E=E->next()) {
							if (E->get().findn("Location: ")!=-1) {
								new_request=E->get().substr(9,E->get().length()).strip_edges();
							}
						}

						print_line("NEW LOCATION: "+new_request);

						if (new_request!="") {
							//process redirect
							client->close();
							request_string=new_request;
							int new_redirs=redirections+1; //because _request() will clear it
							Error err = _request();
							print_line("new connection: "+itos(err));
							if (err==OK) {
								request_sent=false;
								got_response=false;
								body_len=-1;
								body.resize(0);
								redirections=new_redirs;
								return false;

							}
						}
					}


					call_deferred("emit_signal","request_completed",RESULT_SUCCESS,response_code,response_headers,ByteArray());
					return true;
				}
				if (got_response && body_len<0) {
					//chunked transfer is done
					call_deferred("emit_signal","request_completed",RESULT_SUCCESS,response_code,response_headers,body);
					return true;

				}

				call_deferred("emit_signal","request_completed",RESULT_CHUNKED_BODY_SIZE_MISMATCH,response_code,response_headers,ByteArray());
				return true;
				//request migh have been done
			} else {
				//did not request yet, do request

				Error err = client->request(HTTPClient::METHOD_GET,request_string,headers);
				if (err!=OK) {
					call_deferred("emit_signal","request_completed",RESULT_CONNECTION_ERROR,0,StringArray(),ByteArray());
					return true;
				}

				request_sent=true;
				return false;
			}
		} break; //connected: { } break requests only accepted here
		case HTTPClient::STATUS_REQUESTING: {
			//must wait, it's requesting
			client->poll();
			return false;

		} break; // request in progress
		case HTTPClient::STATUS_BODY: {

			if (!got_response) {
				if (!client->has_response()) {
					call_deferred("emit_signal","request_completed",RESULT_NO_RESPONSE,0,StringArray(),ByteArray());
					return true;
				}

				got_response=true;
				response_code=client->get_response_code();
				List<String> rheaders;
				client->get_response_headers(&rheaders);
				response_headers.resize(0);
				for (List<String>::Element *E=rheaders.front();E;E=E->next()) {
					print_line("HEADER: "+E->get());
					response_headers.push_back(E->get());
				}

				if (!client->is_response_chunked() && client->get_response_body_length()==0) {

					call_deferred("emit_signal","request_completed",RESULT_SUCCESS,response_code,response_headers,ByteArray());
					return true;
				}


				if (client->is_response_chunked()) {
					body_len=-1;
				} else {
					body_len=client->get_response_body_length();

					if (body_size_limit>=0 && body_len>body_size_limit) {
						call_deferred("emit_signal","request_completed",RESULT_BODY_SIZE_LIMIT_EXCEEDED,response_code,response_headers,ByteArray());
						return true;
					}
				}

			}


			//print_line("BODY: "+itos(body.size()));
			client->poll();

			body.append_array(client->read_response_body_chunk());

			if (body_size_limit>=0 && body.size()>body_size_limit) {
				call_deferred("emit_signal","request_completed",RESULT_BODY_SIZE_LIMIT_EXCEEDED,response_code,response_headers,ByteArray());
				return true;
			}

			if (body_len>=0) {

				if (body.size()==body_len) {
					call_deferred("emit_signal","request_completed",RESULT_SUCCESS,response_code,response_headers,body);
					return true;
				}
				/*if (body.size()>=body_len) {
					call_deferred("emit_signal","request_completed",RESULT_BODY_SIZE_MISMATCH,response_code,response_headers,ByteArray());
					return true;
				}*/
			}

			return false;

		} break; // request resulted in body: { } break which must be read
		case HTTPClient::STATUS_CONNECTION_ERROR: {
			call_deferred("emit_signal","request_completed",RESULT_CONNECTION_ERROR,0,StringArray(),ByteArray());
			return true;
		} break;
		case HTTPClient::STATUS_SSL_HANDSHAKE_ERROR: {
			call_deferred("emit_signal","request_completed",RESULT_SSL_HANDSHAKE_ERROR,0,StringArray(),ByteArray());
			return true;
		} break;

	}

	ERR_FAIL_V(false);
}

void HTTPRequest::_notification(int p_what) {

	if (p_what==NOTIFICATION_PROCESS) {

		bool done = _update_connection();
		if (done) {

			set_process(false);
			cancel_request();
		}
	}
}

void HTTPRequest::set_use_threads(bool p_use) {

	ERR_FAIL_COND( status!=HTTPClient::STATUS_DISCONNECTED );
	use_threads=p_use;
}

bool HTTPRequest::is_using_threads() const {

	return use_threads;
}

void HTTPRequest::set_body_size_limit(int p_bytes) {

	ERR_FAIL_COND( status!=HTTPClient::STATUS_DISCONNECTED );

	body_size_limit=p_bytes;
}

int HTTPRequest::get_body_size_limit() const {

	return body_size_limit;
}

HTTPClient::Status HTTPRequest::get_http_client_status() const {
	return client->get_status();
}

void HTTPRequest::set_max_redirects(int p_max) {

	max_redirects=p_max;
}

int HTTPRequest::get_max_redirects() const{

	return max_redirects;
}


void HTTPRequest::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("request","url","custom_headers","ssl_validate_domain"),&HTTPRequest::request,DEFVAL(StringArray()),DEFVAL(true));
	ObjectTypeDB::bind_method(_MD("cancel_request"),&HTTPRequest::cancel_request);

	ObjectTypeDB::bind_method(_MD("get_http_client_status"),&HTTPRequest::get_http_client_status);

	ObjectTypeDB::bind_method(_MD("set_use_threads","enable"),&HTTPRequest::set_use_threads);
	ObjectTypeDB::bind_method(_MD("is_using_threads"),&HTTPRequest::is_using_threads);

	ObjectTypeDB::bind_method(_MD("set_body_size_limit","bytes"),&HTTPRequest::set_body_size_limit);
	ObjectTypeDB::bind_method(_MD("get_body_size_limit"),&HTTPRequest::get_body_size_limit);

	ObjectTypeDB::bind_method(_MD("set_max_redirects","amount"),&HTTPRequest::set_max_redirects);
	ObjectTypeDB::bind_method(_MD("get_max_redirects"),&HTTPRequest::get_max_redirects);

	ObjectTypeDB::bind_method(_MD("_redirect_request"),&HTTPRequest::_redirect_request);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"use_threads"),_SCS("set_use_threads"),_SCS("is_using_threads"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"body_size_limit",PROPERTY_HINT_RANGE,"-1,2000000000"),_SCS("set_body_size_limit"),_SCS("get_body_size_limit"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"max_redirects",PROPERTY_HINT_RANGE,"-1,1024"),_SCS("set_max_redirects"),_SCS("get_max_redirects"));

	ADD_SIGNAL(MethodInfo("request_completed",PropertyInfo(Variant::INT,"result"),PropertyInfo(Variant::INT,"response_code"),PropertyInfo(Variant::STRING_ARRAY,"headers"),PropertyInfo(Variant::RAW_ARRAY,"body")));

	BIND_CONSTANT( RESULT_SUCCESS );
	//BIND_CONSTANT( RESULT_NO_BODY );
	BIND_CONSTANT( RESULT_CHUNKED_BODY_SIZE_MISMATCH );
	BIND_CONSTANT( RESULT_CANT_CONNECT );
	BIND_CONSTANT( RESULT_CANT_RESOLVE );
	BIND_CONSTANT( RESULT_CONNECTION_ERROR );
	BIND_CONSTANT( RESULT_SSL_HANDSHAKE_ERROR );
	BIND_CONSTANT( RESULT_NO_RESPONSE );
	BIND_CONSTANT( RESULT_BODY_SIZE_LIMIT_EXCEEDED );
	BIND_CONSTANT( RESULT_REQUEST_FAILED );
	BIND_CONSTANT( RESULT_REDIRECT_LIMIT_REACHED );

}

HTTPRequest::HTTPRequest()
{


	port=80;
	redirections=0;
	max_redirects=8;
	body_len=-1;
	got_response=false;
	validate_ssl=false;
	use_ssl=false;
	response_code=0;
	request_sent=false;
	client.instance();
	use_threads=false;
	body_size_limit=-1;
	status=HTTPClient::STATUS_DISCONNECTED;

}

