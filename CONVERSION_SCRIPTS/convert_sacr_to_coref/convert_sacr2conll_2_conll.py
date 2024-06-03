import sys

def proc(filename):
	with open(filename) as file:
		docsize=0
		newd=0
		maxdoc=2000
		for line in file:
			if line.startswith("#begin document"):
				docsize=0


			cols=line.rstrip().split("\t")
			if len(cols) < 3:
				print(line.rstrip())
				if docsize > maxdoc:
					sys.stderr.write("breaking doc\n")
					print("#end document")
					print()
					print("#begin document (extra-%s); part 000" % newd)
					newd+=1
					docsize=0
				continue
			print("%s\t%s\t%s\t%s\t_\t_\t_\t_\t_\t_\t_\t%s" % ("doc", "0", cols[0], cols[1], cols[-1]))
			docsize+=1


proc(sys.argv[1])			
