import random
import subprocess
import pickle

print("Experiments for ML")
a = open("out.txt","r").read()
out = open("result.csv","w")
out.write("ENo,Lambda,Cost,Descent,1Accuracy,3Accuracy,5Accuracy\n");
b = a.split("\n")
b.pop()
exp_count = 3
print("Number of experiments is %d"%(exp_count))
for i in range(0,exp_count):
	print("======|Experiment %d|====="%(i+1))
	random.shuffle(b)
	with open('train_data', 'w') as f:
		for item in b[0:6000]:
			f.write("%s\n" % item)
	with open('heldout_data', 'w') as f:
		for item in b[6000:8000]:
			f.write("%s\n" % item)
	with open('test_data', 'w') as f:
		for item in b[8000:10000]:
			f.write("%s\n" % item)
	print("Files created.")
	# Train model
	lambdas = [0.05,0.1,0.15]
	cost = [0.5,1.0,1,1.5]
	descs = [0,1]
	for lamb in lambdas:
		for c in cost:
			for desc in descs:
				subprocess.check_output("../../multiTrain -l %f -c %f -h heldout_data train_data puru.model %d 1>/dev/null 2>/dev/null"%(lamb,c,desc),shell = True)
				accuracy1 = subprocess.check_output("../../multiPred test_data puru.model 1 2>&1 | grep Acc | sed -E 's/.*Acc=(.*)/\\1/g'",shell = True)
				print("Lambda = %f Cost = %f Descent = %d Top 1 Accuracy = %f"%(lamb,c,desc,float(accuracy1)))
				accuracy3 = subprocess.check_output("../../multiPred test_data puru.model 3 2>&1 | grep Acc | sed -E 's/.*Acc=(.*)/\\1/g'",shell = True)
				print("Lambda = %f Cost = %f Descent = %d Top 3 Accuracy = %f"%(lamb,c,desc,float(accuracy3)))
				accuracy5 = subprocess.check_output("../../multiPred test_data puru.model 5 2>&1 | grep Acc | sed -E 's/.*Acc=(.*)/\\1/g'",shell = True)
				print("Lambda = %f Cost = %f Descent = %d Top 5 Accuracy = %f"%(lamb,c,desc,float(accuracy5)))
				out.write("%d,%f,%f,%d,%f,%f,%f\n"%(i+1,lamb,c,desc,float(accuracy1),float(accuracy3),float(accuracy5)));
# print("Check out.txt for data analysis")
