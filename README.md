# COMBO
COMBO is jointly trained neural tagger, lemmatizer and dependency parser implemented in python 3 using Keras framework. It took part in [*2018 CoNLL Universal Dependency shared task*](http://universaldependencies.org/conll18/) and ranked 3rd/4th in the [*official evaluation*](http://universaldependencies.org/conll18/results.html).

## Paper
The COMBO description can be found here: [*Semi-Supervised Neural System for Tagging, Parsing and Lematization*](http://universaldependencies.org/conll18/proceedings/pdf/K18-2004.pdf).

## Usage
Training your own model:
```
python main.py --mode autotrain --train train_data.conllu --valid valid_data.conllu --embed external_embedding.txt --model model_name.pkl --force_trees
```

Making predictions:
```
python main.py --mode predict --test test_data.conllu --pred output_path.conllu --model model_name.pkl
```

## Trained models
Models trained on UD dataset:

| Language | Treebank | LAS | MLAS | BLEX | Model |
|-|-|-|-|-|-|
| Afrikaans | af_afribooms | 84.72 | 72.91 | 74.98 | [*377 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.af_afribooms.pkl) |
| Ancient Greek | grc_perseus | 74.20 | 53.30 | 54.29 | [*101 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.grc_perseus.pkl) |
| Ancient Greek | grc_proiel | 76.45 | 59.95 | 67.47 | [*101 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.grc_proiel.pkl) |
| Arabic | ar_padt | 71.95 | 62.75 | 64.38 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ar_padt.pkl) |
| Armenian | hy_armtdp | 28.15 | 5.02 | 11.25 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.hy_armtdp.pkl) |
| Basque | eu_bdt | 83.12 | 68.82 | 77.96 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.eu_bdt.pkl) |
| Bulgarian | bg_btb | 89.36 | 81.10 | 79.98 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.bg_btb.pkl) |
| Buryat | bxr_bdt | 15.16 | 1.09 | 1.92 | [*90 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.bxr_bdt.pkl) |
| Catalan | ca_ancora | 90.54 | 83.11 | 85.20 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ca_ancora.pkl) |
| Chinese | zh_gsd | 63.92 | 53.48 | 57.84 | [*744 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.zh_gsd.pkl) |
| Croatian | hr_set | 86.32 | 71.12 | 79.74 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.hr_set.pkl) |
| Czech | cs_cac | 90.72 | 83.27 | 86.69 | [*740 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.cs_cac.pkl) |
| Czech | cs_fictree | 91.83 | 84.23 | 87.81 | [*740 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.cs_fictree.pkl) |
| Czech | cs_pdt | 90.34 | 84.04 | 86.96 | [*740 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.cs_pdt.pkl) |
| Danish | da_ddt | 83.43 | 74.22 | 77.58 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.da_ddt.pkl) |
| Dutch | nl_alpino | 87.15 | 74.93 | 77.06 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.nl_alpino.pkl) |
| Dutch | nl_lassysmall | 84.27 | 72.65 | 75.44 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.nl_lassysmall.pkl) |
| English | en_ewt | 82.31 | 73.33 | 76.52 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.en_ewt.pkl) |
| English | en_gum | 82.82 | 73.24 | 73.57 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.en_gum.pkl) |
| English | en_lines | 80.33 | 72.25 | 74.01 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.en_lines.pkl) |
| Estonian | et_edt | 83.46 | 75.79 | 72.07 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.et_edt.pkl) |
| Finnish | fi_ftb | 86.89 | 78.42 | 81.06 | [*739 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fi_ftb.pkl) |
| Finnish | fi_tdt | 85.93 | 78.65 | 72.39 | [*739 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fi_tdt.pkl) |
| French | fr_gsd | 85.42 | 77.08 | 79.72 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fr_gsd.pkl) |
| French | fr_sequoia | 88.99 | 81.48 | 84.67 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fr_sequoia.pkl) |
| French | fr_spoken | 74.31 | 63.43 | 65.34 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fr_spoken.pkl) |
| Galician | gl_ctg | 81.17 | 68.15 | 73.60 | [*736 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.gl_ctg.pkl) |
| Galician | gl_treegal | 73.21 | 52.88 | 62.86 | [*736 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.gl_treegal.pkl) |
| German | de_gsd | 77.43 | 54.28 | 68.59 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.de_gsd.pkl) |
| Gothic | got_proiel | 65.87 | 50.81 | 59.30 | [*48 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.got_proiel.pkl) |
| Greek | el_gdt | 88.49 | 76.15 | 78.57 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.el_gdt.pkl) |
| Hebrew | he_htb | 63.69 | 50.26 | 53.58 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.he_htb.pkl) |
| Hindi | hi_hdtb | 91.43 | 76.23 | 86.29 | [*593 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.hi_hdtb.pkl) |
| Hungarian | hu_szeged | 79.47 | 66.09 | 72.51 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.hu_szeged.pkl) |
| Indonesian | id_gsd | 78.40 | 67.30 | 75.10 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.id_gsd.pkl) |
| Irish | ga_idt | 69.24 | 37.31 | 47.32 | [*206 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ga_idt.pkl) |
| Italian | it_isdt | 91.03 | 83.18 | 84.76 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.it_isdt.pkl) |
| Italian | it_postwita | 73.99 | 61.14 | 62.98 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.it_postwita.pkl) |
| Japanese | ja_gsd | 73.69 | 57.82 | 60.62 | [*743 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ja_gsd.pkl) |
| Kazakh | kk_ktb | 22.38 | 4.40 | 7.86 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.kk_ktb.pkl) |
| Korean | ko_gsd | 80.66 | 74.49 | 66.13 | [*741 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ko_gsd.pkl) |
| Korean | ko_kaist | 84.88 | 76.92 | 72.40 | [*743 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ko_kaist.pkl) |
| Kurmanji | kmr_mg | 21.95 | 2.26 | 05.01 | [*45 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.kmr_mg.pkl) |
| Latin | la_ittb | 85.54 | 79.84 | 83.51 | [*526 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.la_ittb.pkl) |
| Latin | la_perseus | 68.07 | 49.77 | 52.75 | [*526 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.la_perseus.pkl) |
| Latin | la_proiel | 70.08 | 56.82 | 64.94 | [*526 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.la_proiel.pkl )|
| Latvian | lv_lvtb | 80.71 | 66.22 | 71.80 | [*637 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.lv_lvtb.pkl) |
| North Sámi | sme_giella | 57.16 | 39.66 | 45.03 | [*47 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sme_giella.pkl) |
| Norwegian | no_bokmaal | 89.33 | 79.51 | 84.68 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.no_bokmaal.pkl) |
| Norwegian | no_nynorsk | 88.36 | 79.32 | 82.89 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.no_nynorsk.pkl) |
| Norwegian | no_nynorsklia | 68.26 | 57.51 | 60.98 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.no_nynorsklia.pkl) |
| Old Church Slavonic | cu_proiel | 71.14 | 56.52 | 66.04 | [*48 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.cu_proiel.pkl) |
| Old French | fro_srcmf | 84.81 | 76.75 | 81.20 | [*52 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fro_srcmf.pkl) |
| Persian | fa_seraji | 86.14 | 80.30 | 76.29 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.fa_seraji.pkl) |
| Polish | pl_lfg | 94.62 | 86.44 | 89.31 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.pl_lfg.pkl) |
| Polish | pl_sz | 91.38 | 80.45 | 85.59 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.pl_sz.pkl) |
| Polish | poleval2018 | 86.11 | 76.18 | 79.86 | [*115 MB*](http://mozart.ipipan.waw.pl/~prybak/model_poleval2018/model_A_semi.pkl) |
| Portuguese | pt_bosque | 87.57 | 74.31 | 80.31 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.pt_bosque.pkl) |
| Romanian | ro_rrt | 85.31 | 76.84 | 79.54 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ro_rrt.pkl) |
| Russian | ru_syntagrus | 91.10 | 85.37 | 87.16 | [*741 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ru_syntagrus.pkl) |
| Russian | ru_taiga | 74.24 | 61.59 | 64.36 | [*741 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ru_taiga.pkl) |
| Serbian | sr_set | 87.27 | 73.79 | 79.92 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sr_set.pkl) |
| Slovak | sk_snk | 83.76 | 63.97 | 75.34 | [*54 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sk_snk.pkl) |
| Slovenian | sl_ssj | 85.72 | 75.07 | 81.11 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sl_ssj.pkl) |
| Slovenian | sl_sst | 58.12 | 45.93 | 50.94 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sl_sst.pkl) |
| Spanish | es_ancora | 89.68 | 82.60 | 84.51 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.es_ancora.pkl) |
| Swedish | sv_lines | 81.97 | 66.26 | 77.01 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sv_lines.pkl) |
| Swedish | sv_talbanken | 85.89 | 77.68 | 80.74 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.sv_talbanken.pkl) |
| Turkish | tr_imst | 63.54 | 52.51 | 58.89 | [*737 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.tr_imst.pkl) |
| Ukrainian | uk_iu | 84.71 | 69.88 | 77.97 | [*738 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.uk_iu.pkl) |
| Upper Sorbian | hsb_ufal | 21.30 | 1.45 | 4.53 | [*139 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.hsb_ufal.pkl) |
| Urdu | ur_udtb | 81.53 | 55.70 | 72.49 | [*485 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ur_udtb.pkl) |
| Uyghur | ug_udt | 63.10 | 40.71 | 52.76 | [*165 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.ug_udt.pkl) |
| Vietnamese | vi_vtb | 42.53 | 35.11 | 38.47 | [*736 MB*](http://mozart.ipipan.waw.pl/~prybak/model_conll2018/model.vi_vtb.pkl) |


## License
CC BY-NC-SA 4.0

## Citation

```
@InProceedings{rybak-wrblewska:2018:K18-2,
  author    = {Rybak, Piotr  and  Wr{\'{o}}blewska, Alina},
  title     = {Semi-Supervised Neural System for Tagging, Parsing and Lematization},
  booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {45--54},
  url       = {http://www.aclweb.org/anthology/K18-2004}
}
```

