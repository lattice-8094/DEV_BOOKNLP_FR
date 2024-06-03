import sys
from collections import Counter

def proc(filename):
	counts={}
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			if cols[0] == "POS":
				pos=cols[1]
				text=cols[3]
				if text not in counts:
					counts[text]=Counter()
				counts[text][pos]+=1

	for phrase in counts:
		print("%s\t%s\t%s\t%.3f" % (sum(counts[phrase].values()), phrase, counts[phrase].most_common()[0][0], (counts[phrase].most_common()[0][1])/sum(counts[phrase].values())))

proc(sys.argv[1])		