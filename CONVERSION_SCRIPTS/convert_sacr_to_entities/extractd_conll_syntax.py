import sys, re
import spacy
from spacy.tokens import Doc
nlp = spacy.load("fr_dep_news_trf")

def get_head_pos(start, end, toks):
	doc = nlp(Doc(nlp.vocab, toks))
	for token in doc[start:end+1]:
		if token.head.i == token.i or token.head.i < start or token.head.i > end:
			return token.pos_, token.text
	return None


def proc(filename):
	print(filename)	
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
					print (" ".join(toks))
					entstr=[]
					for start, end in ents:
						cat=ents[start,end].split(":")[-1].split("_")[-1]
						if len(cat.lstrip().rstrip())> 0:
							# entstr.append("%s %s %s %s %s %s %s" % (cat, None, None, start, end+1, start, end+1))
							entstr.append("%s,%s,%s,%s %s" % (start, end+1, start, end+1, cat))
							val=get_head_pos(start, end, toks)
							print("POS\t%s\t%s\t%s" % (val[0], val[1], ' '.join(toks[start:end+1])))
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


proc(sys.argv[1])						
