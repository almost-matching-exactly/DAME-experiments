

f = open("ps.txt")
line = f.readline()
fo = open("data/ps_out.txt","w")
while line!="":
    fo.write(line.split(",")[1])
    line = f.readline()

f.close()
fo.close()
