l=[]
l2=[1,2,3,9,5,6,4,3,3,56,43,332,2,2,2,22,2,2,2,2,2,2,2,2,2,55,5]

n=len(l2)
k=2
t=[]

for i in range(0, n-k, k):
    print(i,"-->", l2[i:i+k])
    t.append(l2[i:i+k])

print(t)