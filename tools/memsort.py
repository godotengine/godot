
import sys

arg="memdump.txt"

if (len(sys.argv)>1):
	arg=sys.argv[1]

f = open(arg,"rb")


l=f.readline()


sum = {}
cnt={}


while(l!=""):

	s=l.split("-")
	amount = int(s[1])
	what=s[2]
	if (what in sum):
		sum[what]+=amount
		cnt[what]+=1
	else:
		sum[what]=amount
		cnt[what]=1

	l=f.readline()


for x in sum:
	print(x.strip()+"("+str(cnt[x])+"):\n: "+str(sum[x]))
