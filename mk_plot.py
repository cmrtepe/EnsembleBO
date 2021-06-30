import pickle
from matplotlib import pyplot as plt


def main():

	with open("n_list_newtnewlr.txt", "rb") as fb:
		n_list = pickle.load(fb)

	with open("y_list_newtnewlr.txt", "rb") as fb:
		y_best = pickle.load(fb)

	plt.figure(figsize=(9,6))

	plt.plot(n_list, y_best)
	print(len(n_list), len(y_best))
	plt.ylim(top=0.17, bottom=0.13)
	plt.xlabel("N")
	plt.ylabel("Y best")
	
	plt.savefig("t_max50lrhalfmh.png")

	return 0

if __name__=="__main__":
	
	main()
