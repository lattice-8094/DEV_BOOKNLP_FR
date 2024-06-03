from multilingualbooknlp.english.bert_coref_quote_pronouns import BERTCorefTagger
from multilingualbooknlp.english.bert_qa import QuotationAttribution
from torch.optim.lr_scheduler import ExponentialLR
from multilingualbooknlp.english.name_coref import NameCoref

import torch
import torch.optim as optim

from os import listdir
from os.path import isfile, join
import argparse
import re
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_quotes(folder):

	all_quotes={}

	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	for filename in onlyfiles:

		if not filename.endswith("quotes"):
			continue
		idd=re.sub("\.quotes", "", filename.split("/")[-1])

		quotes={}
		attrib={}

		with open(join(folder, filename)) as file:
			for line in file:
				cols=line.rstrip().split("\t")
				if cols[0] == "QUOTE":
					qid=cols[1]
					start_sid=int(cols[2])
					start_wid=int(cols[3])
					end_sid=int(cols[4])
					end_wid=int(cols[5])

					quotes[qid]=(start_sid, start_wid, end_sid, end_wid)

				elif cols[0] == "ATTRIB":
					qid=cols[1]
					eid=int(cols[2])

					attrib[qid]=eid

		quotes_by_sent={}
		for qid in quotes:
			eid=attrib[qid]
			start_sid, start_wid, end_sid, end_wid=quotes[qid]

			if end_sid not in quotes_by_sent:
				quotes_by_sent[end_sid]={}
				quotes_by_sent[end_sid]["START"]=[]
				quotes_by_sent[end_sid]["END"]=[]
				
			if start_sid not in quotes_by_sent:
				quotes_by_sent[start_sid]={}
				quotes_by_sent[start_sid]["START"]=[]
				quotes_by_sent[start_sid]["END"]=[]

			quotes_by_sent[start_sid]["START"].append((start_sid, start_wid, end_sid, end_wid, eid))
			quotes_by_sent[end_sid]["END"].append((start_sid, start_wid, end_sid, end_wid, eid))

		all_quotes[idd]=quotes_by_sent

	return all_quotes

# If a quotation attribution module or name coref module exist, they can be incorporated here (default is not)
def test(model, quote_attrib, tokensDir, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_names, outfile, iterr, goldFile, path_to_scorer, orig_quotes, doTest=False, iter_idx=0):

	print("testing")
	doNameCoref=False

	out=open(outfile, "w", encoding="utf-8")

	# for each document
	for idx in range(len(test_all_docs)):

		d,p=test_doc_names[idx]
		d=re.sub("/", "_", d)
		test_doc=test_all_docs[idx]
		test_ents=test_all_ents[idx]

		if quote_attrib is not None or doNameCoref:
			toks=model.read_toks("%s/%s.txt" % (tokensDir, re.sub("_brat", "", d)))
		max_words=test_all_max_words[idx]
		max_ents=test_all_max_ents[idx]

		names=[]
		is_named=[]

		for n_idx, sent_ents in enumerate(test_ents):
			for entity in sent_ents:
				name=test_doc[n_idx][entity.start:entity.end+1]
				names.append(name)
				if entity.proper == "PROP":
					is_named.append(1)
				else:
					is_named.append(0)



		test_matrix, test_index, test_token_positions, test_ent_spans, test_starts, test_ends, test_widths, test_data, test_masks, test_transforms, test_quotes=model.get_data(test_doc, test_ents, max_ents, max_words)

		if quote_attrib is not None:
			q_tokens=[]
			q_ents=[]
			q_quotes=[]
			cur=0
			tid=0
			sent_cur=[]

			for sent_idx, sent_w in enumerate(test_doc):

				sent_cur.append(cur)

				ents=test_ents[sent_idx]

				for word in sent_w[1:-1]:
					q_tokens.append(toks[tid])
					tid+=1

				for ent in ents:
					q_ents.append((cur+ent.start-1, cur+ent.end-1, "%s_%s" % (ent.proper, ent.ner_cat), ent.text))

				quotes=orig_quotes[idx][sent_idx]
				for s_sid, s_wid, e_sid, e_wid in quotes:
					q_quotes.append((sent_cur[s_sid]+s_wid-1, sent_cur[e_sid]+e_wid-1))
				cur+=len(sent_w)-2


			attributed=quote_attrib.tag(q_quotes, q_ents, q_tokens)

		entities_by_sentence=test_ents

		global_entities=[]
		for sentence_entities in entities_by_sentence:
			global_entities.extend(sentence_entities)

		for gidx, ent in enumerate(global_entities):
			if ent.in_quote:
				assert attributed[ent.quote_id] is not None
				ent.quote_mention=attributed[ent.quote_id]


		e_list=[]
		in_quotes=[]
		is_named=[]
		entity_names=[]
		for ent in global_entities:
			nextCap=False
			ent_tokens=ent.text.split(" ")
			new_ent_toks=[]
			for tok in ent_tokens:
				if tok == "[CAP]":
					nextCap=True
					continue
				if nextCap:
					tok=''.join(tok[0].upper() + tok[1:])

				nextCap=False

				new_ent_toks.append(tok)
			ent.text=' '.join(new_ent_toks)

			e_list.append((ent.global_start, ent.global_end, "%s_%s" % (ent.proper, ent.ner_cat), ent.text))
			entity_names.append(ent.text.split(" "))
			if ent.in_quote:
				in_quotes.append(1)
			else:
				in_quotes.append(0)

			# print(ent.proper)
			if ent.proper == "PROP":
				is_named.append(1)
			else:
				is_named.append(0)

		refs=None
		doNameCoref=False
		if doNameCoref:
			nameCoref=NameCoref("training/aliases.txt")

			# Create entity for first-person narrator, if present
			refs=nameCoref.cluster_narrator(e_list, in_quotes, toks)

			# Cluster non-PER PROP mentions that are identical
			refs=nameCoref.cluster_identical_propers(e_list, refs)

			# Cluster mentions of named people

			# hon_mapper={"mister":"mr.", "mr.":"mr.", "mr":"mr.", "mistah":"mr.", "mastah":"mr.", "master":"mr.",
			# "miss":"miss", "ms.": "miss", "ms":"miss","missus":"miss","mistress":"miss",
			# "mrs.":"mrs.", "mrs":"mrs."
			# }

			# def map_honorifics(term):
			# 	term=term.lower()
			# 	if term in hon_mapper:
			# 		return hon_mapper[term]
			# 	return None

			# for tok in toks:
			# 	if tok.pos.startswith("N"):
			# 		tok.pos="NOUN"

			# for start, end, cat, text in e_list:
			# 	ner_prop=cat.split("_")[0]
			# 	ner_type=cat.split("_")[1]

			# 	new_text=[]
			# 	for i in range(start,end+1):
			# 		hon_mapped=map_honorifics(toks[i].text)

			# 		# print(text, toks[i].text, toks[i].pos)
			# 		if (hon_mapped is not None or (toks[i].pos == "NOUN" or toks[i].pos == "PROPN")) and toks[i].text.lower()[0] != toks[i].text[0]:
			# 			val=toks[i].text
			# 			if hon_mapped is not None:
			# 				val=hon_mapped
			# 			new_text.append(val)


			# refs=nameCoref.cluster(entity_names, is_named, refs)
			# refs=nameCoref.cluster_only_nouns(e_list, refs, toks)


		assignments=model.forward(test_matrix, test_index, existing=refs, token_positions=test_token_positions, starts=test_starts, ends=test_ends, widths=test_widths, input_ids=test_data, attention_mask=test_masks, transforms=test_transforms, entities=global_entities)

		if doNameCoref:
			assignments=nameCoref.cluster_noms(e_list, assignments)

		for ass in assignments:
			if ass == -1:
				print(assignments)
				sys.exit(1)

		model.print_conll(test_doc_names[idx], test_doc, test_ents, assignments, out, test_token_maps)

	out.close()

	if doTest:
		import multilingualbooknlp.common.calc_coref_metrics as calc_coref_metrics

		print("Goldfile: %s" % goldFile)
		print("Predfile: %s" % outfile)
		
		bcub_f, avg=calc_coref_metrics.get_conll(path_to_scorer, gold=goldFile, preds=outfile)
		print("Iter %s, idx %s, Average F1: %.3f, bcub F1: %s" % (iterr, iter_idx, avg, bcub_f))
		sys.stdout.flush()
		return avg



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-t','--trainData', help='Folder containing train data', required=False)
	parser.add_argument('-v','--valData', help='Folder containing test data', required=False)
	parser.add_argument('-m','--mode', help='mode {train, predict}', required=False)
	parser.add_argument('-w','--model', help='modelFile (to write to or read from)', required=False)
	parser.add_argument('-o','--outFile', help='outFile', required=False)
	parser.add_argument('-s','--path_to_scorer', help='Path to coreference scorer', required=False)
	parser.add_argument('-p','--precoData', help='Folder containing preco train data', required=False)
	parser.add_argument('-e','--base_model', help='Base BERT model', required=False)
	parser.add_argument('-k','--tokens_dir', help='tokens directory', required=False)

	parser.add_argument('-q','--quoteFolder', help='Path to folder containing quotation data', required=False)

	args = vars(parser.parse_args())

	# read quotes if we have them
	quotes={}
	doQuotes=False
	if args["quoteFolder"] is not None:
		quotes=read_quotes(args["quoteFolder"])
		doQuotes=True

	mode=args["mode"]
	modelFile=args["model"]
	valData=args["valData"]
	outfile=args["outFile"]
	tokens_dir=args["tokens_dir"]

	precoData=args["precoData"]
	path_to_scorer=args["path_to_scorer"]
	base_model=args["base_model"]


	quote_attrib=None
	if doQuotes:
		# only used for incorporated attribution into evaluation (coreference has logic for quotation constraints on reference)
		# default is that no text is in quotes so attribution is not needed
		quote_attrib=QuotationAttribution("training/trained_models/speaker/speaker_google_bert_uncased_L-12_H-768_A-12.model")

	freeze_bert=False
	lr=0.001

	if not freeze_bert:
		lr=1e-5	

	# this defines a contrast set to disallow coreference *between* these sets (during inference, not training)
	gender_cats=[ ["he", "him", "his"], ["she", "her"], ["they", "them", "their"] ] 
	model = BERTCorefTagger(gender_cats, base_model=base_model, freeze_bert=freeze_bert, pronominalCorefOnly=False)

	print("freezing bert parameters: %s" % freeze_bert)
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=lr)
	lr_scheduler=ExponentialLR(optimizer, gamma=0.999)

	if mode == "train":

		trainData=args["trainData"]

		# Pretrain on generic data if available
		if precoData is not None:
			preco_all_docs, preco_all_ents, preco_all_named_ents, preco_all_truth, preco_all_max_words, preco_all_max_ents, preco_doc_ids, preco_token_maps, preco_quotes, preco_ids=model.read_conll(precoData)

		all_docs, all_ents, all_named_ents, all_truth, all_max_words, all_max_ents, doc_ids, token_maps, train_quotes, train_ids=model.read_conll(trainData, quotes)
		test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids, test_token_maps, test_quotes, test_ids=model.read_conll(valData, quotes)

		best_f1=0.
		cur_steps=0

		best_idx=0
		patience=10

		# pretrain on PreCo

		if precoData is not None:

			print("pretraining with %s" % precoData)
			for i in range(1):

				bigloss=0.
				for idx in range(len(preco_all_docs)):
					model.train()

					if idx % 10 == 0:
						print(idx, "/", len(preco_all_docs))
						sys.stdout.flush()
					preco_max_words=preco_all_max_words[idx]
					preco_max_ents=preco_all_max_ents[idx]

					matrix, index, token_positions, ent_spans, starts, ends, widths, input_ids, masks, transforms, _=model.get_data(preco_all_docs[idx], preco_all_ents[idx], preco_max_ents, preco_max_words)

					entities_by_sentence=preco_all_ents[idx]
					global_entities=[]
					for sentence_entities in entities_by_sentence:
						global_entities.extend(sentence_entities)
					if preco_max_ents > 1:
						model.zero_grad()

						loss=model.forward(matrix, index, truth=preco_all_truth[idx], token_positions=token_positions, starts=starts, ends=ends, widths=widths, input_ids=input_ids, attention_mask=masks, transforms=transforms, entities=global_entities)
						loss.backward()
						optimizer.step()
						cur_steps+=1
						if cur_steps % 100 == 0:
							lr_scheduler.step()
					bigloss+=loss.item()

					model.eval()
					doTest=True

					if idx % 1000 == 0:
					
						avg_f1=test(model, quote_attrib, tokens_dir, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_ids, outfile, i, valData, path_to_scorer, test_quotes, doTest=doTest, iter_idx=idx)

						if doTest:
							if avg_f1 > best_f1:
								torch.save(model.state_dict(), modelFile)
								print("Saving model ... %.3f is better than %.3f" % (avg_f1, best_f1))
								best_f1=avg_f1
								best_idx=i

							if i-best_idx > patience:
								print ("Stopping pre-training at epoch %s" % i)
								break



				print(bigloss)

		# train on LitBank
		optimizer = optim.Adam(model.parameters(), lr=lr)
		lr_scheduler=ExponentialLR(optimizer, gamma=0.999)
		for i in range(100):

			model.train()
			bigloss=0.
			# for idx in range(10):
			for idx in range(len(all_docs)):

				if idx % 1 == 0:
					print(idx, "/", len(all_docs))
					sys.stdout.flush()
				max_words=all_max_words[idx]
				max_ents=all_max_ents[idx]

				matrix, index, token_positions, ent_spans, starts, ends, widths, input_ids, masks, transforms, _=model.get_data(all_docs[idx], all_ents[idx], max_ents, max_words)
				
				entities_by_sentence=all_ents[idx]
				global_entities=[]
				for sentence_entities in entities_by_sentence:
					global_entities.extend(sentence_entities)
				model.assign_quotes_to_entity(global_entities)

				if max_ents > 1:
					model.zero_grad()
					loss=model.forward(matrix, index, truth=all_truth[idx], token_positions=token_positions, starts=starts, ends=ends, widths=widths, input_ids=input_ids, attention_mask=masks, transforms=transforms, entities=global_entities)
					if loss != 0:
						loss.backward()
						optimizer.step()
						cur_steps+=1
						if cur_steps % 100 == 0:
							lr_scheduler.step()
						
						bigloss+=loss.item()

			print(bigloss)

			model.eval()
			doTest=False

			if precoData is not None:
				doTest=True

			if i >= 2:
				doTest=True
			
			avg_f1=test(model, quote_attrib, tokens_dir, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_ids, outfile, i, valData, path_to_scorer, test_quotes, doTest=doTest)

			if doTest:
				if avg_f1 > best_f1:
					torch.save(model.state_dict(), modelFile)
					print("Saving model ... %.3f is better than %.3f" % (avg_f1, best_f1))
					best_f1=avg_f1
					best_idx=i

				if i-best_idx > patience:
					print ("Stopping training at epoch %s" % i)
					break

	elif mode == "test":

		model.load_state_dict(torch.load(modelFile, map_location=device), strict=False)
		model.eval()
		test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids, test_token_maps, test_quotes, test_ids=model.read_conll(valData, quotes)

		avg_f1=test(model, quote_attrib, tokens_dir, test_all_docs, test_all_ents, test_all_named_ents, test_all_max_words, test_all_max_ents, test_doc_ids, outfile, 0, valData, path_to_scorer, test_quotes, doTest=True)
