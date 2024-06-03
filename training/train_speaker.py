from multilingualbooknlp.english.speaker_attribution import BERTSpeakerID
import torch.nn as nn
import torch
import argparse
import json
from random import shuffle
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_speaker_data(filename):

	with open(filename) as file:
		
		data={}

		for line in file:
			cols=line.rstrip().split("\t")

			sid=cols[0]
			eid=cols[1]
			cands=json.loads(cols[3])
			quote=int(cols[2])
			text=cols[4].split(" ")

			for s,e,_,_ in cands:
				
				if s > len(text) or e > len(text):
					print("reading problem", s, e, len(text))
					sys.exit(1)

			if sid not in data:
				data[sid]=[]
			data[sid].append((eid, cands, quote, text))

		x=[]
		m=[]

		sids=list(data.keys())

		shuffle(sids)

		for sid in sids:
			for eid, cands, quote, text in data[sid]:
				x.append(text)
				m.append((eid, cands, quote))

		return x, m

if __name__ == "__main__":
	

	parser = argparse.ArgumentParser()
	parser.add_argument('--trainData', help='Filename containing training data', required=False)
	parser.add_argument('--devData', help='Filename containing dev data', required=False)
	parser.add_argument('--testData', help='Filename containing test data', required=False)
	parser.add_argument('--base_model', help='Base BERT model', required=False)
	parser.add_argument('--model_name', help='Filename to save model to', required=False)

	args = vars(parser.parse_args())

	trainData=args["trainData"]
	devData=args["devData"]
	testData=args["testData"]
	base_model=args["base_model"]
	model_name=args["model_name"]

	train_x, train_m=read_speaker_data(trainData)
	dev_x, dev_m=read_speaker_data(devData)
	test_x, test_m=read_speaker_data(testData)

	metric="accuracy"
	
	bertSpeaker=BERTSpeakerID(base_model=base_model)
	bertSpeaker.to(device)

	train_x_batches, train_m_batches, train_y_batches, train_o_batches=bertSpeaker.get_batches(train_x, train_m)
	dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches=bertSpeaker.get_batches(dev_x, dev_m)
	test_x_batches, test_m_batches, test_y_batches, test_o_batches=bertSpeaker.get_batches(test_x, test_m)
	
	optimizer = torch.optim.Adam(bertSpeaker.parameters(), lr=1e-5)
	cross_entropy=nn.CrossEntropyLoss()

	best_dev_acc = 0.

	num_epochs=20

	for epoch in range(num_epochs):

		bertSpeaker.train()
		bigLoss=0

		for x1, m1, y1 in zip(train_x_batches, train_m_batches, train_y_batches):
			y_pred = bertSpeaker.forward(x1, m1)

			batch_y=y1["y"].unsqueeze(-1)
			batch_y=torch.abs(batch_y-1)*-100

			true_preds=y_pred+batch_y

			golds_sum=torch.logsumexp(true_preds, 1)
			all_sum=torch.logsumexp(y_pred, 1)

			loss=torch.sum(all_sum-golds_sum)
			bigLoss+=loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print("\t\t\tEpoch %s loss: %.3f" % (epoch, bigLoss))
		
		# Evaluate; save the model that performs best on the dev data
		dev_F1, dev_acc=bertSpeaker.evaluate(dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, epoch)
		sys.stdout.flush()
		if epoch % 1 == 0:
			if dev_F1 > best_dev_acc:
				torch.save(bertSpeaker.state_dict(), model_name)
				best_dev_acc = dev_F1
		
	# Test with best performing model on dev
	bertSpeaker.load_state_dict(torch.load(model_name, map_location=device))
	bertSpeaker.eval()

	test_F1, test_acc=bertSpeaker.evaluate(test_x_batches, test_m_batches, test_y_batches, test_o_batches, "test")
	print("Test F1:\t%.3f\t, accuracy:\t%.3f" % (test_F1, test_acc))