top_coref_dir=/data0/dbamman/booknlp-training/training/trained_models/coref
top=fr

base_model=bert-base-multilingual-cased

trainData=/data0/dbamman/mcoref/CorefUD-1.1-public/data/CorefUD_French-Democrat/fr2K/fr_democrat-corefud-train.conll2012.2K
valData=/data0/dbamman/mcoref/CorefUD-1.1-public/data/CorefUD_French-Democrat/fr2K/fr_democrat-corefud-dev.conll2012.2K

outfile=$top_coref_dir/models/$top.model
predFile=$top_coref_dir/preds/$top.conll

python train_coref.py --base_model $base_model --trainData $trainData --valData $valData -m train -w $outfile -o $predFile -s reference-coreference-scorers/scorer.pl  
