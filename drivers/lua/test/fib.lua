-- fibonacci function with cache

-- very inefficient fibonacci function
function fib(n)
	N=N+1
	if n<2 then
		return n
	else
		return fib(n-1)+fib(n-2)
	end
end

-- a general-purpose value cache
function cache(f)
	local c={}
	return function (x)
		local y=c[x]
		if not y then
			y=f(x)
			c[x]=y
		end
		return y
	end
end

-- run and time it
function test(s,f)
	N=0
	local c=os.clock()
	local v=f(n)
	local t=os.clock()-c
	print(s,n,v,t,N)
end

n=arg[1] or 24		-- for other values, do lua fib.lua XX
n=tonumber(n)
print("","n","value","time","evals")
test("plain",fib)
fib=cache(fib)
test("cached",fib)
