import matplotlib.pyplot as plt
x = []
for line in open('plot', 'r'):
	x.append(float(line))
y = [i for i in range(0,len(x))]
plt.plot(y,x, label='Loaded from file!')
plt.show()
exit()