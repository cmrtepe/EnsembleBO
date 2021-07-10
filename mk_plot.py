import pickle
from matplotlib import pyplot as plt
import torch
import numpy as np

def main():

	nls = []	
	yls = []
	b1 = [0.0005, 0.001, 0.004]
	b2 = [0.7, 0.75, 0.85, 0.9]
	b3 = [0.95, 0.999]
	for j in b2:
		for k in b3:
			n_list = "res_lists/n_list" + str(0.004) + "-" + str(j) + "-" + str(k)  + "-" + "trial" + ".txt"
			y_list = "res_lists/y_list" + str(0.004) + "-" + str(j) + "-" + str(k)  + "-" + "trial3" + ".txt"
			with open(n_list, "rb") as fb:
				nl = pickle.load(fb)
				nls.append(nl)
			with open(y_list, "rb") as fb:
				yl = pickle.load(fb)
				yls.append(yl)
	plt.figure(1, figsize=(9,6))
	for i, (nl, yl) in enumerate(zip(nls, yls)):
		plt.plot(nl, yl, label="beta 1={} beta 2={}".format(b1[int(i%3)], b2[int(i/3)]))
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylim(top=0.1, bottom=0)
	plt.ylabel("Validation loss")
	plt.savefig("losses_lr0.004.png")
	# plt.savefig("losses_grid1.png")

	# plt.figure(figsize=(9,6))
	# for i, ll in enumerate(losses):
	# 	plt.plot([i for i in range(len(ll))], ll, label={"beta 1={} beta 2={}".format(b1[int(i%3)], b2[int(i/3)])})
	# plt.legend()
	# plt.xlabel("Epochs")
	# plt.ylim(top=0.1, bottom=0)
	# plt.ylabel("Validation loss")
	
	# plt.savefig("losses_grid1.png")

	# return 0
    


	# model_dict = torch.load("run1.pt")
	# weights = []
	# for i in range(len(model_dict)):
	# 	model = model_dict["check" + str(i*25)]
	# 	params = []
	# 	for param in model.parameters():
	# 		params.append(param.view(-1))
	# 	params = torch.cat(params)
	# 	weights.append(params)
	# x, y = np.linspace(0, 250, 10), np.linspace(0, 250, 10)
	# z = np.array([torch.dot(a, b).item()/(torch.linalg.norm(a).item()*torch.linalg.norm(b).item()) for a in weights for b in weights])
	# Z = z.reshape(10, 10)
	# plt.imshow(Z, interpolation="nearest")
	# plt.colorbar()
	# plt.savefig("cosine.png")
	

	

if __name__=="__main__":
	
	main()
