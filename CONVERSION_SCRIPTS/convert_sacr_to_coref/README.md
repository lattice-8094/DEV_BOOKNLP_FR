
#### Remove the properties info from .sacr files (which seems to mess up the corefconversion script)

```
cd sacr_V2/CORRIGE_sacr
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
python ../../sacr2conll.py -o ../conll1/$i.conll $i
done
```

```
cd ../conll1
mkdir ../splits
for split in train dev test
do
	for i in `cat ../../$split.ids`
		do
			cat $i.sacr.conll >> ../splits/$split.conll
		done
done
```

```
cd ../splits
mkdir ../final
for split in train dev test
do
python ../../convert_sacr2conll_2_conll.py $split.conll > ../final/$split.conll
done
```