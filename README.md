

# Instructions

## Install dependencies
Run `pip install -r requirements.txt`

## Download the data
- Go to 'data' directory
- Run `mkdir files; .\gdc-client download -m gdc_manifest.2025-12-07.102659.txt -d files/` 

## Preprocess data

```
python scripts/preprocess_tcga_rna_seq.py 
--dataset-path data/files/ 
--output-folder data/preprocessed 
--common-gene-ids-path data/common_gene_id.txt 
--rna-seq-column tpm_unstranded
```

## Run model

```
python main.py
```