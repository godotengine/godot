package org.godotengine.godot.utils;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import org.apache.http.NameValuePair;
import org.apache.http.message.BasicNameValuePair;

/**
 * 
 * @author Luis Linietsky <luis.linietsky@gmail.com>
 */
public class RequestParams {

	private HashMap<String,String> params;
	private String url;
	
	public RequestParams(){
		params = new HashMap<String,String>();
	}
	
	public void put(String key, String value){
		params.put(key, value);
	}
	
	public String get(String key){
		return params.get(key);
	}
	
	public void remove(Object key){
		params.remove(key);
	}
	
	public boolean has(String key){
		return params.containsKey(key);
	}
	
	public List<NameValuePair> toPairsList(){
		List<NameValuePair>  fields = new ArrayList<NameValuePair>();

		for(String key : params.keySet()){
			fields.add(new BasicNameValuePair(key, this.get(key)));
		}
		return fields;
	}

	public String getUrl() {
		return url;
	}

	public void setUrl(String url) {
		this.url = url;
	}

	
}
