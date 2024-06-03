import re
from collections import Counter

class QuoteTagger:
	

	def tag(self, toks):

		predictions=[]
		currentQuote=[]
		curStartTok=None
		lastPar=None

		quote_symbols=Counter()

		for tok in toks:
			if tok.text == "«" or tok.text == "»":
				quote_symbols["GUILLEMETS_DOUBLES"]+=1
			elif tok.text == "‹" or tok.text == "›":
				quote_symbols["GUILLEMET_SEUL"]+=1
			elif tok.text == "“" or tok.text == "”" or tok.text == "\"" or tok.text == "“":
				quote_symbols["GUILLEMETS_ANGLAIS"]+=1
			elif tok.text == "—" or tok.text == "–": #one is a dash, one is emdash but i'm not sure they're any different
				quote_symbols["DASH"]+=1


		quote_symbol="DOUBLE_QUOTE"
		if len(quote_symbols) > 0:
			quote_symbol=quote_symbols.most_common()[0][0]

		for tok in toks:

			w=tok.text

			for w_idx, w_char in enumerate(w):
				if w_char== "«" or w_char == "»":
					w="GUILLEMETS_DOUBLES"
				elif w_char == "‹" or w_char == "›":
					#in the english version, we need to distinguish between apostrophe showing possession/contraction, but in french
					#guillemets should only be used for quotation and nothing else
					w="GUILLEMET_SEUL"

			# start over at each new paragraph
			if tok.paragraph_id != lastPar and lastPar is not None:

				if len(currentQuote) > 0:
					predictions.append((curStartTok, tok.token_id-1))
				curStartTok=None
				currentQuote=[]

			if w == quote_symbol:

				if curStartTok is not None:

					if len(currentQuote) > 0:
						predictions.append((curStartTok, tok.token_id))
						currentQuote.append(tok.text)

					curStartTok=None
					currentQuote=[]
				else:
					curStartTok=tok.token_id

			
			if curStartTok is not None:
				currentQuote.append(tok.text)

			lastPar=tok.paragraph_id

		for start, end in predictions:
			for i in range(start, end+1):
				toks[i].inQuote=True

		return predictions



