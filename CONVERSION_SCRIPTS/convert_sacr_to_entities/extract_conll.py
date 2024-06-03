import sys, re

props={}
def read_prop(filename):
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			text=cols[1]
			cat=cols[2]
			props[text]=cat


valid=set(["PROP", "NOM", "PRON"])

def proc(filename):
	with open(filename) as file:
		toks=[]
		open_ents={}
		ents={}
		open_count=0
		for line in file:
			if line.startswith("#begin"):
				continue
			cols=line.rstrip().split("\t")
			# print(cols)

			if len(cols) < 3:
	

				# print(toks)
				if len(toks) > 0:
					validFlag=False
					entstr=[]
					for start, end in ents:
						cat=ents[start,end].split(":")[-1].split("_")[-1]
						if len(cat.lstrip().rstrip())> 0:
							# entstr.append("%s %s %s %s %s %s %s" % (cat, None, None, start, end+1, start, end+1))
							# val=get_head_pos(start, end, toks)
							phrase=' '.join(toks[int(start):int(end)+1])
							prop=props[phrase]
							if prop in valid:
								validFlag=True
								entstr.append("%s,%s,%s,%s %s_%s" % (start, end+1, start, end+1, prop, cat))

					if validFlag:
						print (" ".join(toks))
						print()
						print("|".join(entstr))
						print()

						# print(start, end)
						# print(toks[int(start):int(end)+1])

				toks=[]
				open_ents={}
				ents={}
				open_count=0

				continue

			tid=int(cols[0])
			tok=cols[1]
			toks.append(tok)
			# print(toks)
			coref=cols[-1].split("|")
			# print(coref, cols[-1])

			for c in coref:
				if c.startswith("(") and c.endswith(")"):
					c=re.sub("\(", "", c)
					c=(re.sub("\)", "", c))

					ents[(tid,tid)]=c

					# print(ents)
					
				elif c.startswith("("):
					c=(re.sub("\(", "", c))

					if c not in open_ents:
						open_ents[c]=[]
					open_ents[c].append((tid))
					open_count+=1

				elif c.endswith(")"):
					c=(re.sub("\)", "", c))

					if c not in open_ents:
						print("NOT IN OPEN ENTS", c)
					else:
					# assert c in open_ents

						if len(open_ents[c]) == 0:
							print("EMPTY STACK", line)
							continue

						start_tid=open_ents[c].pop()
						open_count-=1

						ents[(start_tid,tid)]=c

read_prop(sys.argv[2])
proc(sys.argv[1])						