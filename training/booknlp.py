import argparse
from transformers import logging
logging.set_verbosity_error()
from pathlib import Path
import os
from multilingualbooknlp.english.english_booknlp import EnglishBookNLP
from multilingualbooknlp.fr.fr_booknlp import FrBookNLP

class BookNLP():

	def __init__(self, language, model_params):

		if language == "en":
			self.booknlp=EnglishBookNLP(model_params)

		elif language == "fr":
			self.booknlp=FrBookNLP(model_params)

		elif language == "es":
			self.booknlp=EsBookNLP(model_params)

		elif language == "ru":
			self.booknlp=RuBookNLP(model_params)

	def process(self, inputFile, outputFolder, idd):
		self.booknlp.process(inputFile, outputFolder, idd)


def proc():

	parser = argparse.ArgumentParser()
	parser.add_argument('-l','--language', help='Currently on {en}', required=True)
	parser.add_argument('-i','--inputFile', help='Filename to run BookNLP on', required=True)
	parser.add_argument('-o','--outputFolder', help='Folder to write results to', required=True)
	parser.add_argument('--modelPath', help='Path to trained models', required=True)
	parser.add_argument('--id', help='ID of text (for creating filenames within output folder)', required=True)

	args = vars(parser.parse_args())

	language=args["language"]
	inputFile=args["inputFile"]
	outputFolder=args["outputFolder"]
	modelPath=args["modelPath"]
	idd=args["id"]

	print("tagging %s" % inputFile)
	
	valid_languages=set(["en", "fr", "es", "ru"])
	if language not in valid_languages:
		print("%s not recognized; supported languages: %s" % (language, valid_languages))
		sys.exit(1)


	model_params={
		"pipeline":"entity,coref", "model":"big", 
		"model_path":modelPath,
	}

	booknlp=BookNLP(language, model_params)
	booknlp.process(inputFile, outputFolder, idd)
		
if __name__ == "__main__":
	proc()
