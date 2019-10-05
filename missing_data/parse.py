

f = open("res/cf_res34.csv")
line = f.readline()
fo = open("res/cf_out34.txt","w")
while line!="":
    fo.write(line.split(",")[1])
    line = f.readline()

f.close()
fo.close()
