top_coref_dir=TRAINED_MODELS/ALL_CAT
top=fr

base_model=camembert-base

trainData=/data/booknlp/JEAN_DEV_BOOKNLP_FR/training/TRAINING_DATA/COREF/train.num
valData=/data/booknlp/JEAN_DEV_BOOKNLP_FR/training/TRAINING_DATA/COREF/dev.num
testData=/data/booknlp/JEAN_DEV_BOOKNLP_FR/training/TRAINING_DATA/COREF/test.num

outfile=$top_coref_dir/$top.$base_model.model
predFile=$top_coref_dir/$top.$base_model.conll

python3.9 train_coref.py --base_model $base_model --trainData $trainData --valData $valData -m train -w $outfile -o $predFile -s reference-coreference-scorers/scorer.pl  

python3.9 train_coref.py --base_model $base_model --valData $testData -m test -w $outfile -o $predFile -s reference-coreference-scorers/scorer.pl -o scores_coref.txt
