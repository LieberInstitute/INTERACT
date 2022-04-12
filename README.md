#
This repository contains codes for predicting DNA methylation (DNAm) regulatory variants in the human brain from local DNA sequence and for training new DNAm prediction models for any tissues with genome-wide DNAm profile.

## Install
Install PyTorch following instructions from https://pytorch.org/.  Use `pip install -r requirements.txt` to install the other dependencies.

## Usage

Step 1. Compute features for each CpG site in the Illumina HumanMethylationEPIC (epic) array. CpG sites in chromsomes 1-20 are used as training dataset, whereas CpG sites in chromosome 21 and chromosome 22 are used as validation dataset and test dataset, respectively.

```bash
# Example compute features for chromosome 1 for the 84 samples of four different tissues (brain, blood, buccal and saliva).
$python run_feature.py chr1

```

Step 2. Pre-train DNAm prediction model using wgbs data.

#Example
```bash
pre-train a DNAm prediction model using four GPUs

$CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer wgbs_methylation_regression \
	--exp_name wgbs_methylation_regression \
	--batch_size 1024 \
	--learning_rate 0.000176 \
	--fp16 \
	--warmup_steps 10000 \
	--nproc_per_node 4 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/2kb_wgbs \
	--output_dir ./outputs/2kb_wgbs \
	--local_rank 0 \
	--num_train_epochs 500 \
	--model_config_file ./config/pretrain/config.json
```

Step 3. Fine-tune DNAm prediction model using epic array data. We trained one DNAm prediction model for each tissue in our study.  

#Example
```bash
fine-tuning the DNAm prediction model for brain tissue using four GPUs

$CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer array_methylation_regression \
	--exp_name array_methylation_regression \
	--batch_size 512 \
	--learning_rate 0.000176 \
	--fp16 \
	--warmup_steps 10000 \
	--nproc_per_node 4 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/2kb_epic \
	--output_dir ./outputs/2kb_epic \
	--local_rank 0 \
	--num_train_epochs 500 \
	--model_config_file ./config/finetune/config.json \
	--from_pretrained ./outputs/2kb_wgbs
```
Step 4. Compute features for genome-wide CpG sites. read_variant.py computes two feature vectors for each CpG site, one from the DNA sequence with reference allele and the other from the DNA sequence with alternative allele. Due to memory limitation, read_variant.py splits each chromosome into chunks with length of 1000000 bp. Then read_variant.py computes features for all CpG sites in each chunk.
#Example
```bash
compute features for CpG sites in the first chunk of chromosome 1

$python read_variant.py chr1 0

```
Step 5. Predicts DNAm levels of CpG sites from DNA sequence with the reference allele using trained brain-specific model.
#Example
```bash
predict DNAm levels of CpG sites in the chunk 0 of chromosome 1 using the trained brain-specific model

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer array_mQTL_regression \
	--exp_name array_mQTL_regression \
	--batch_size 1024 \
	--learning_rate 0.000176 \
	--fp16 \
	--warmup_steps 20000 \
	--nproc_per_node 4 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/2kb_mqtl/reference \
	--output_dir ./outputs/2kb_genome_cpg/reference \
	--local_rank 0 \
	--num_train_epochs 1 \
	--model_config_file ./config/finetune/config.json \
	--from_pretrained ./outputs/2kb_epic \
	--split chr1_0
```
Step 6. Predicts DNAm levels of CpG sites from DNA sequence with the alternative allele using trained brain-specific model.
#Example
```bash
predict DNAm levels of CpG sites in chunk 0 of chromosome 1 using trained brain-specific model

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch main.py transformer array_mQTL_regression \
	--exp_name array_mQTL_regression \
	--batch_size 1024 \
	--learning_rate 0.000176 \
	--fp16 \
	--warmup_steps 20000 \
	--nproc_per_node 4 \
	--gradient_accumulation_steps 1 \
	--data_dir ./datasets/2kb_mqtl/variation \
	--output_dir ./outputs/2kb_genome_cpg/variation \
	--local_rank 0 \
	--num_train_epochs 1 \
	--model_config_file ./config/finetune/config.json \
	--from_pretrained ./outputs/2kb_epic \
	--split chr1_0
```
Step 7. Predict DNAm regulatory variants. This script uses the predicted DNAm levels of the reference allele and the alternative allele to compute the absolute difference of DNAm levels between the two alleles, and then computes the maximum effect for each snp among all its targeted CpG sites within a window of 2000 bps
#Example
```bash
predicts DNAm regulatory variants for fragment with index being 0 in chromosome 1

$python Fine_mapping.py chr1 chr1_0

combines DNAm regulatory variants across chunks of chromosome

$python Fine_mapping.py chr1

combines DNAm regulatory variants across chromosomes

$python Fine_mapping.py

```

#### Details:
`READ_SIGNAL.py` reads DNAm level for each CpG site in all subjects.  

`run_feature.py` computes features for each CpG site in the Illumina HumanMethylationEPIC (epic) array, CpG sites in chromosomes 1-20 are used as training data, whereas CpG sites in chromosome 21 and chromosome 22 are used as validation data and test data, respectively.

`run_variant.py` computes features for genome-wide CpG sites from DNA sequences with a genetic variation. It computes two feature vectors, one is for the reference allele and the other for the alternative allele

