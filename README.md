#
This repository contains code for predicting DNA methylation(DNAm) regulatory variants in the human brain from sequence and training new sequence-based methylation model for any tissue with methylation profile.

## Install
Install PyTorch following instructions from https://pytorch.org/.  Use `pip install -r requirements.txt` to install the other dependencies.

## Usage

step 1. computes features for each CpG site in the Illumina HumanMethylationEPIC (epic) array, the CpG sites in chromsome 1-20 are used as training dataset, whereas the CpG sites in chromosome 21 and chromosome 22 are used as validation dataset and test dataset, respectively.

#Example
```bash
computes features for chromosome 1 for the 84 subjects in the four tissues
$python run_feature.py chr1

```

step 2. pre-trains methylation prediction model by using wgbs data.

#Example
```bash
pre-trains a methylation prediction model using four GPUs

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

step 3. fine-tunes methylation prediction model by using epic array data. We trained a methylation prediction model for each human tissue. 

#Example
```bash
fine-tunes trains a methylation prediction model for brain using four GPUs

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
step 4. computes features for genome-wide CpG sites. read_variant.py computes two feature vectors for each CpG site, one for the CpG site with snp being reference allel and the othat for the CpG site with snp being alteration allel. Due to the momery limitation, read_variant.py split each chromosome into multiple DNA fragments with length being 1000000 bp. Then read_variant.py computes features for all CpG sites in each DNA fragment.
#Example
```bash
computes features for CpG sites in the first DNA fragment of chromosome 1

$python read_variant.py chr1 0

```
step 5. predicts methylation level for CpG sites with snp being reference allel by loading trained brain model.
#Example
```bash
predicts methylation level for CpG sites in the fragment 0 of chromosome 1 by using the brain model

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
step 6. predicts methylation level for CpG sites with snp being alteration allel by loading trained brain model.
#Example
```bash
predicts methylation level for CpG sites in the fragment 0 of chromosome 1 by using the brain model

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
step 7. Predict methylation regulatory variants. This script uses the predicted methylation level of reference allel and the predicted methylation level of alteration allel to compute the absolute difference of methylation level between reference allel and alteration allel, and then computes the maximum effect for each snp among all the CpG sites within window of 2000 bps
#Example
```bash
predicts mQTL for fragment with index being 0 in chromosome 1

$python Fine_mapping.py chr1 chr1_0

combines mQTL in fragments of chromosome

$python Fine_mapping.py chr1

combines mQTL in chromsomes

$python Fine_mapping.py

```

#### Details:
`read_signal.py` reads methylation level for each CpG site in all subjects. 

`run_feature.py` computes features for each CpG site in the Illumina HumanMethylationEPIC (epic) array, the CpG sites in chromsome 1-21 are used as training data, whereas the CpG sites in chromosome 21 and chromosome 22 are used as validation data and test data, respectively.

`run_variant.py` computes features for genome-wide CpG sites with a common snp. For a CpG site with a common snp, it computes two feature vectors, one for the reference allel and the other for the alteration allel

