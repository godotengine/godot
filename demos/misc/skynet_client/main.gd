
extends Control

var TCPClient = preload("res://tcp_client.gd")

func _ready():
	pass
	
func _on_login_pressed():
	var token = {
		server = "sample",
		user = "hello",
		pswd = "password",
	}
	
	var conn = TCPClient.new()
	# login
	if conn.connect("127.0.0.1", 8001) != OK:
		return false
		
	var challenge = Crypt.base64decode(conn.read_line())
	
	var clientkey = Crypt.randomkey()
	conn.write_line(Crypt.base64encode(Crypt.dhexchange(clientkey)))
	var secret = Crypt.dhsecret(Crypt.base64decode(conn.read_line()), clientkey)
	
	print("secret is ", Crypt.hexencode(secret).get_string_from_utf8())
	
	var hmac = Crypt.hmac64(challenge, secret)
	conn.write_line(Crypt.base64encode(hmac))
	
	var etoken = Crypt.desencode(secret, ("%s@%s:%s" % [
		Crypt.base64encode(token.user.to_utf8()).get_string_from_utf8(),
		Crypt.base64encode(token.server.to_utf8()).get_string_from_utf8(),
		Crypt.base64encode(token.pswd.to_utf8()).get_string_from_utf8(),
	]).to_utf8())
	var b = Crypt.base64encode(etoken)
	conn.write_line(b)
	
	var line = conn.read_line()
	if line == null:
		return
	var result = line.get_string_from_utf8()
	print(result)
	
	var code = int(result.substr(0, 3))
	assert(code == 200)
	conn.disconnect()
	
	var subid = Crypt.base64decode(result.substr(4, result.length() - 4).to_utf8())
	print("login ok, subid=", subid.get_string_from_utf8())
		
	# connect to game server
	var text = "echo"
	var index = 1
	print("connect")
	if conn.connect("127.0.0.1", 8888) != OK:
		return false
		
	var handshake = "%s@%s#%s:%d" % [
		Crypt.base64encode(token.user.to_utf8()).get_string_from_utf8(),
		Crypt.base64encode(token.server.to_utf8()).get_string_from_utf8(),
		Crypt.base64encode(subid).get_string_from_ascii(),
		index
	]
	var hmac = Crypt.hmac64(Crypt.hashkey(handshake.to_utf8()), secret)
	var hs = "%s:%s" % [handshake, Crypt.base64encode(hmac).get_string_from_utf8()]
	print("handshake = ", handshake)
	print("hs = ", hs)
	if conn.send_package(hs.to_utf8()) != OK:
		return false
	print(conn.recv_package().get_string_from_utf8())
	print("===>", conn.send_request(text.to_utf8(), 0))
	print("disconnect")
	conn.disconnect()
	
	index += 1
	print("connect again")
	if conn.connect("127.0.0.1", 8888) != OK:
		return false
	
	var handshake = "%s@%s#%s:%d" % [
		Crypt.base64encode(token.user.to_utf8()).get_string_from_utf8(),
		Crypt.base64encode(token.server.to_utf8()).get_string_from_utf8(),
		Crypt.base64encode(subid).get_string_from_ascii(),
		index
	]
	var hmac = Crypt.hmac64(Crypt.hashkey(handshake.to_utf8()), secret)
	var hs = "%s:%s" % [handshake, Crypt.base64encode(hmac).get_string_from_utf8()]
	print("handshake = ", handshake)
	print("hs = ", hs)
	if conn.send_package(hs.to_utf8()) != OK:
		return false
	print(conn.recv_package().get_string_from_utf8())
	print("===>", conn.send_request("fake".to_utf8(), 0))
	print("===>", conn.send_request("again".to_utf8(), 1))
	print(conn.recv_request())
	print(conn.recv_request())
	print("disconnect")
	conn.disconnect()
	
	


func _on_sproto_codec_pressed():
	var sproto = preload("addressbook.spb")
	print("Default:")
	print(sproto.get_default("AddressBook"))
	
	var f = File.new()
	if f.open("res://addressbook.json", File.READ) == OK:
		var text = f.get_as_text()
		var d = {}
		f.close()
		if d.parse_json(text) == OK:
			var stream = sproto.encode("AddressBook", d)
			print(stream)
			print(stream.size())
			var dec = sproto.decode("AddressBook", stream, true)
			print(dec)
	
	