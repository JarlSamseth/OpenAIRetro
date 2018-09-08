from memory import RingBuf

d = RingBuf(1000)
for i in range(100):
    d.append(i)

d
