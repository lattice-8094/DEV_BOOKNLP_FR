#### Remove the properties info from .sacr files (which seems to mess up the corefconversion script)

```
cd /sacr_V2/CORRIGE_sacr
mkdir ../adjusted
for i in `ls *sacr`
do
grep -v "^#" $i > ../adjusted/$i
done
```

####  Convert to CoNLL

```
cd ../adjusted
mkdir ../conll1
for i in `ls *sacr`
do
python ../../sacr2conll_ent.py -o ../conll1/$i.conll $i
done
```

#### Add PROP/NOM/PRON

The following instructions describe how to add PROP/NOM/PRON tags to the existing annotations, resulting in the creation of the `phrase.pos.txt` file.  This is merely documentary; you can skip this since `phrase.pos.txt` exists in this directory.

* Extract headword for each phrase, along with its POS tag

```
cd ../conll1
for i in `ls *conll`
do
python ../../extractd_conll_syntax.py $i >> ../pos.data.txt
done
```

* Get the most common POS tag for each unique entity string (based on its headword)

```
python ../../gather_pos.py ../pos.data.txt > ../pos.counts.txt
```

* MANUAL: edit `pos.counts.txt` to map each unique string to {PROP,NOM,PRON}. E.g., by replacing all "PROPN" with "PROP", all "NOUN" with "NOM", and manually going through the rest.  The final version after my own manual processing is `phrase.pos.txt`.

#### Convert to ACE format

```
cd conll1
mkdir ../ace
for i in `ls *conll`
do
python ../../extract_conll.py $i ../../phrase.pos.txt > ../ace/$i
done
```
#### Convert to TSV

This creates a name.data file for each name.conll file

```
cd ../ace
for i in `ls *conll`
do
python ../../convert_txt_to_tsv_one.py -i $i
done

```

{train,dev,test}.ids contains the names of the files that belong to those splits.  This then creates a train.data, dev.data, test.data file from that information

```
cd ../
python ../create_splits.py -i . -o tsv -t ace
```

#### The final data is in tsv/
