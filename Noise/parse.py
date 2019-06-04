

f = open("res/cf_res31.txt")
line = f.readline()
fo = open("res/cf31.txt","w")
while line!="":
    fo.write(line.split(",")[1])
    line = f.readline()

f.close()
fo.close()
