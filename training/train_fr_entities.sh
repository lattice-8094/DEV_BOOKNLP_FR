base_model=camembert-base

model=`echo $base_model | sed 's#/#_#g' `
echo $model
outdir="TRAINED_MODELS/PER"

top=/data/booknlp/JEAN_DEV_BOOKNLP_FR/training/TRAINING_DATA/ENTITIES/SACR_V4

python3.9 train_entity_tagger.py --base_model $base_model --mode train --trainFolder_layered $top/train --testFolder_layered $top/test --devFolder_layered $top/dev --tagFile_layered $top/fr.entities.tagset --modelFile $outdir/fr_catprop_${model}.model

python3.9 train_entity_tagger.py --base_model $base_model --mode test --trainFolder_layered $top/train --testFolder_layered $top/test --devFolder_layered $top/dev --tagFile_layered $top/fr.entities.tagset  --modelFile $outdir/fr_catprop_${model}.model > $outdir/fr_catprop_${model}.eval.log 2>&1

