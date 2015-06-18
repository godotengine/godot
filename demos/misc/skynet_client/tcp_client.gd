extends StreamPeerTCP

# received buffer
var buffer

func _init():
	buffer = RawArray()

func connect(host, port):
	buffer.resize(0)
	return .connect(host, port)

func _try_recv():
	var res = self.get_partial_data(10240)
	# if recv failed(connection disconnected), return false
	if res.empty() or res[0] != OK:
		return false
	var data = res[1]
	# received size > 0, append to to recv buffer
	if data.size() > 0:
		# push back to buffer
		for i in range(data.size()):
			buffer.push_back(data.get(i))
	return true

func _pop(size):
	var remain = buffer.size() - size
	for x in range(size, buffer.size()):
		buffer.set(x - size, buffer.get(x))
	buffer.resize(remain)
	
func write_line(line):
	line.push_back(10)
	return self.put_data(line)

func read_line():
	while true:
		if !_try_recv():
			return null
		
		var result = RawArray()
		var size = 0
		var found = false
		
		for i in range(buffer.size()):
			size += 1
			if buffer.get(i) == 10: # '\n'
				result.resize(size)
				found = true
				break
		
		if not found:
			continue
		
		var idx = 0
		for i in range(size):
			result.set(idx, buffer.get(i))
			idx += 1
		# pop processed size
		_pop(size)
		
		# 删除最后的换行符'\n'
		result.resize(result.size() - 1)
		return result
		
func send_package(pack):
	var size = pack.size()
	var package = RawArray()
	# word size - in big endian
	package.push_back(size >> 8 & 0xFF)
	package.push_back(size & 0xFF)
	# string context
	for i in range(pack.size()):
		package.push_back(pack.get(i))
	return self.put_data(package)

func recv_package():
	while true:
		if !_try_recv():
			return null
		if buffer.size() < 2:
			continue
		# pack size
		var s = buffer.get(0) * 256 + buffer.get(1)
		if buffer.size() < s + 2:
			continue
		# read context
		var result = RawArray()
		for i in range(s):
			result.push_back(buffer.get(2 + i))
		# pop size
		_pop(s + 2)
		return result

func send_request(v, session):
	var size = v.size() + 4
	var package = RawArray()
	# word size
	package.push_back(size >> 8 & 0xff)
	package.push_back(size & 0xFF)
	# string context
	for i in range(v.size()):
		package.push_back(v.get(i))
	# int session
	package.push_back(session >> 24 & 0xFF)
	package.push_back(session >> 16 & 0xFF)
	package.push_back(session >> 8 & 0xFF)
	package.push_back(session & 0xFF)
	
	return [
		self.put_data(package),
		v,
		session
	]
	
func recv_request():
	var pack = recv_package()
	var size = pack.size() - 5
	var context = RawArray()
	for i in range(size):
		context.push_back(pack.get(i))
	var ok = pack.get(size)
	var session = (pack.get(size + 1) << 24) + (pack.get(size + 2) << 16) + (pack.get(size + 3) << 8) + pack.get(size + 4)
	
	return [
		ok != 0,
		context,
		session,
	]
