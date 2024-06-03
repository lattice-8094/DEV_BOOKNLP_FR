base_model=$1

model=`echo $base_model | sed 's#/#_#g' `
echo $model

python training/train_speaker.py --trainData training/data/speaker_attribution/quotes.train.txt --devData training/data/speaker_attribution/quotes.dev.txt --testData training/data/speaker_attribution/quotes.test.txt --base_model $base_model --model_name training/trained_models/speaker/speaker_${model}.model > training/trained_models/speaker/${model}.log 2>&1
