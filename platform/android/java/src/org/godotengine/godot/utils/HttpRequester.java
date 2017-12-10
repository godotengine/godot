/*************************************************************************/
/*  HttpRequester.java                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
package org.godotengine.godot.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.security.KeyStore;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.http.HttpResponse;
import org.apache.http.HttpVersion;
import org.apache.http.NameValuePair;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.conn.ClientConnectionManager;
import org.apache.http.conn.scheme.PlainSocketFactory;
import org.apache.http.conn.scheme.Scheme;
import org.apache.http.conn.scheme.SchemeRegistry;
import org.apache.http.conn.ssl.SSLSocketFactory;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.impl.conn.tsccm.ThreadSafeClientConnManager;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;
import org.apache.http.params.HttpProtocolParams;
import org.apache.http.protocol.HTTP;
import org.apache.http.util.EntityUtils;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * 
 * @author Luis Linietsky <luis.linietsky@gmail.com>
 */
public class HttpRequester {

	private Context context;
	private static final int TTL = 600000; // 10 minutos
	private long cttl = 0;

	public HttpRequester() {
		//Log.d("XXX", "Creando http request sin contexto");
	}

	public HttpRequester(Context context) {
		this.context = context;
		//Log.d("XXX", "Creando http request con contexto");
	}

	public String post(RequestParams params) {
		HttpPost httppost = new HttpPost(params.getUrl());
		try {
			httppost.setEntity(new UrlEncodedFormEntity(params.toPairsList()));
			return request(httppost);
		} catch (UnsupportedEncodingException e) {
			return null;
		}
	}

	public String get(RequestParams params) {
		String response = getResponseFromCache(params.getUrl());
		if (response == null) {
			//Log.d("XXX", "Cache miss!");
			HttpGet httpget = new HttpGet(params.getUrl());
			long timeInit = new Date().getTime();
			response = request(httpget);
			long delay = new Date().getTime() - timeInit;
			Log.d("com.app11tt.android.utils.HttpRequest::get(url)", "Url: " + params.getUrl() + " downloaded in " + String.format("%.03f", delay / 1000.0f) + " seconds");
			if (response == null || response.length() == 0) {
				response = "";
			} else {
				saveResponseIntoCache(params.getUrl(), response);
			}
		}
		Log.d("XXX", "Req: " + params.getUrl());
		Log.d("XXX", "Resp: " + response);
		return response;
	}

	private String request(HttpUriRequest request) {
		//Log.d("XXX", "Haciendo request a: " + request.getURI() );
		Log.d("PPP", "Haciendo request a: " + request.getURI());
		long init = new Date().getTime();
		HttpClient httpclient = getNewHttpClient();
		HttpParams httpParameters = httpclient.getParams();
		HttpConnectionParams.setConnectionTimeout(httpParameters, 0);
		HttpConnectionParams.setSoTimeout(httpParameters, 0);
		HttpConnectionParams.setTcpNoDelay(httpParameters, true);
		try {
			HttpResponse response = httpclient.execute(request);
			Log.d("PPP", "Fin de request (" + (new Date().getTime() - init) + ") a: " + request.getURI());
			//Log.d("XXX1", "Status:" + response.getStatusLine().toString());
			if (response.getStatusLine().getStatusCode() == 200) {
				String strResponse = EntityUtils.toString(response.getEntity());
				//Log.d("XXX2", strResponse);
				return strResponse;
			} else {
				Log.d("XXX3", "Response status code:" + response.getStatusLine().getStatusCode() + "\n" + EntityUtils.toString(response.getEntity()));
				return null;
			}

		} catch (ClientProtocolException e) {
			Log.d("XXX3", e.getMessage());
		} catch (IOException e) {
			Log.d("XXX4", e.getMessage());
		}
		return null;
	}

	private HttpClient getNewHttpClient() {
		try {
			KeyStore trustStore = KeyStore.getInstance(KeyStore.getDefaultType());
			trustStore.load(null, null);

			SSLSocketFactory sf = new CustomSSLSocketFactory(trustStore);
			sf.setHostnameVerifier(SSLSocketFactory.ALLOW_ALL_HOSTNAME_VERIFIER);

			HttpParams params = new BasicHttpParams();
			HttpProtocolParams.setVersion(params, HttpVersion.HTTP_1_1);
			HttpProtocolParams.setContentCharset(params, HTTP.UTF_8);

			SchemeRegistry registry = new SchemeRegistry();
			registry.register(new Scheme("http", PlainSocketFactory.getSocketFactory(), 80));
			registry.register(new Scheme("https", sf, 443));

			ClientConnectionManager ccm = new ThreadSafeClientConnManager(params, registry);

			return new DefaultHttpClient(ccm, params);
		} catch (Exception e) {
			return new DefaultHttpClient();
		}
	}

	private static String convertStreamToString(InputStream is) {
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		StringBuilder sb = new StringBuilder();
		String line = null;
		try {
			while ((line = reader.readLine()) != null) {
				sb.append((line + "\n"));
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				is.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return sb.toString();
	}

	public void saveResponseIntoCache(String request, String response) {
		if (context == null) {
			//Log.d("XXX", "No context, cache failed!");
			return;
		}
		SharedPreferences sharedPref = context.getSharedPreferences("http_get_cache", Context.MODE_PRIVATE);
		SharedPreferences.Editor editor = sharedPref.edit();
		editor.putString("request_" + Crypt.md5(request), response);
		editor.putLong("request_" + Crypt.md5(request) + "_ttl", new Date().getTime() + getTtl());
		editor.commit();
	}

	public String getResponseFromCache(String request) {
		if (context == null) {
			Log.d("XXX", "No context, cache miss");
			return null;
		}
		SharedPreferences sharedPref = context.getSharedPreferences("http_get_cache", Context.MODE_PRIVATE);
		long ttl = getResponseTtl(request);
		if (ttl == 0l || (new Date().getTime() - ttl) > 0l) {
			Log.d("XXX", "Cache invalid ttl:" + ttl + " vs now:" + new Date().getTime());
			return null;
		}
		return sharedPref.getString("request_" + Crypt.md5(request), null);
	}

	public long getResponseTtl(String request) {
		SharedPreferences sharedPref = context.getSharedPreferences(
				"http_get_cache", Context.MODE_PRIVATE);
		return sharedPref.getLong("request_" + Crypt.md5(request) + "_ttl", 0l);
	}

	public long getTtl() {
		return cttl > 0 ? cttl : TTL;
	}

	public void setTtl(long ttl) {
		this.cttl = (ttl * 1000) + new Date().getTime();
	}
}
