# CONVERSION SCRIPTS


On créé un dossier sacr_VX où X est la version du des annotations. Ici X=3 version pas imbriquée. 

Au début je commence avec seulement le dossier annot_noNested du github

Suivre les instructions de Bamman dans le readme du dossier de conversion

/!\ Les chemins de fichiers peuvent changer en fonction de notre arborescence

/!\ Changer les commandes python en python3

## Pour les ENTITIES : 

On se met dans '/data/booknlp/JEAN_DEV_BOOKNLP_FR/CONVERSION_SCRIPTS/convert_sacr_to_entities'
Les deux premières étapes 

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
python3 ../../sacr2conll_ent.py -o ../conll1/$i.conll $i
done
```



se passent bien, mais 


#### Add PROP/NOM/PRON non:
 
```
cd ../conll1
for i in `ls *conll`
do
python3.9 ../../extractd_conll_syntax.py $i >> ../pos.data.txt
done
```

 
Des problemes : 
ici - for i in `ls *conll`; do python3 ../../extractd_conll_syntax.py $i >> ../pos.data.txt; done
le script extractd_conll_syntax est en ImportError

ImportError: /home/jbarre/.local/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: cusparseSpSM_analysis, version libcusparse.so.11

Des problèmes avec Spacy.. Surement que j'utilise la mauvaise version de spacy, à check.

après quelques essais, je passe le script, il n'y en a pas besoin indique Bamman (doute pour les nouvelles données que l'on a là, mais avançons)


Ensuite 

```
python3 ../../gather_pos.py ../pos.data.txt > ../pos.counts.txt
```
OK mais ne compte rien car pos.data.txt n'existe pas 


## UPDATE 

Après résintstallation de spacy et du modèle trf, les deux dernières étapes fonctionnent ! (c'est long)
 

#### Convert to ACE format


```
cd conll1
mkdir ../ace
for i in `ls *conll`
do
python3 ../../extract_conll.py $i ../../phrase.pos.txt > ../ace/$i
done
```

OK pour Sacr V2 reproduction

POUR SACR V3 : ERRORS (cf après)

#### Convert to TSV

```
cd ../ace
for i in `ls *conll`
do
python3 ../../convert_txt_to_tsv_one.py -i $i
done
```

Il faut créer à la main les {train,dev,test}.ids > Je reprends ceux de Bamman

```
cd ../
python3 ../create_splits.py -i . -o tsv -t ace
```

C'est good on obtient nos fichiers d'entraînement {train,dev,test}.data




# POUR VERSION SANS IMBRICATION - convert to ACE pose problème : 

A 3 erreurs, 3 fichiers qui seront pris en compte seulement en partie ? Erreurs bizarres, key not in index, pour plus tard...

 Traceback (most recent call last):
  File "../../extract_conll.py", line 103, in <module>
    proc(sys.argv[1])						
  File "../../extract_conll.py", line 40, in proc
    prop=props[phrase]
KeyError: 'la lucarne du grenier'
Traceback (most recent call last):
  File "../../extract_conll.py", line 103, in <module>
    proc(sys.argv[1])						
  File "../../extract_conll.py", line 40, in proc
    prop=props[phrase]
KeyError: 'mère'
Traceback (most recent call last):
  File "../../extract_conll.py", line 103, in <module>
    proc(sys.argv[1])						
  File "../../extract_conll.py", line 40, in proc
    prop=props[phrase]
KeyError: 'Sardanapale'



 2038  for i in `ls *conll`; do python ../../extract_conll.py $i ../../phrase.pos.txt > ../ace/$i; done



Après:

#### Convert to TSV

Il faut créer à la main les {train,dev,test}.ids > Je reprends ceux de Bamman

C'est good on obtient nos fichiers d'entraînement {train,dev,test}.data



################################################################################################



# Pour la COREF : 

##### 2 paramètres hyper importants : 

- maxdoc = 2000 dans convert_sacr2conll_2_conll.py - en gros ça coupe les documents en morceaux pour le training - on veut plus que 2000 mais à 10000 on a un cuda out of memory - reéssayer car David lançait un truc aussi, ou essayer 5000 ou voir la batch size lors du training 

et dans sacr2conll.py (de oberle)

choisi la maniere dont on converti le sacr - le meme output que le num est celui avec  WORD tokenization donc on commente whitespace tokenization
        tokenization_mode=sacr_parser.WORD_TOKENIZATION,
- #        tokenization_mode=sacr_parser.WHITESPACE_TOKENIZATION,


On se met dans '/data/booknlp/JEAN_DEV_BOOKNLP_FR/CONVERSION_SCRIPTS/convert_sacr_to_coref/'

Au début je commence avec seulement le dossier annot_noNested du github, je rajoute direct les ids de bamman

Les étapes 

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
python3 ../../sacr2conll.py -o ../conll1/$i.conll $i
done
```

le début se passe bien (attention cependant aux chemins relatifs qui peuvent changer par rapport à Bamman)


ensuite

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

breaking doc car on break les docs tous les 2000 elements -> on pourrait mettre plus, mais cuda out of memory à 10000 -> essayer 5000

Enfin -> 
```
cd ../splits
mkdir ../final
for split in train dev test
do
python3 ../../convert_sacr2conll_2_conll.py $split.conll > ../final/$split.conll
done
```


On obtient des {train,dev,test}.conll, ce qui est bizarre puisque on travaille avec des .num depuis le début ????


Les fichiers ont l'air similaires, je tente un premier training comme ça !

Je les renomme en .num (EDIT -> c'est bien cela)


#######################################

À l'entraînement, Quand on met tokenization_mode=sacr_parser.WHITESPACE_TOKENIZATION, voici l'erreur :

Traceback (most recent call last):
  File "/data/booknlp/JEAN_DEV_BOOKNLP_FR/training/train_coref.py", line 322, in <module>
    all_docs, all_ents, all_named_ents, all_truth, all_max_words, all_max_ents, doc_ids, token_maps, train_quotes, train_ids=model.read_conll(trainData, quotes)
  File "/data/booknlp/JEAN_DEV_BOOKNLP_FR/training/multilingualbooknlp/english/bert_coref_quote_pronouns.py", line 1125, in read_conll
    ents=list(ents.values())
AttributeError: 'list' object has no attribute 'values'


