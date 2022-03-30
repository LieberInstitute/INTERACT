from collections import OrderedDict
import read_signal
import numpy as np
import random
import pickle
import json
import sys

chrom = sys.argv[1]
batch_index = sys.argv[2]
window = 2001
whole_chroms = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]

def chrom_length():
    chr_len = {}
    chr_len["chr1"] = 249250621
    chr_len["chr2"] = 243199373
    chr_len["chr3"] = 198022430
    chr_len["chr4"] = 191154276
    chr_len["chr5"] = 180915260
    chr_len["chr6"] = 171115067
    chr_len["chr7"] = 159138663
    chr_len["chr8"] = 146364022
    chr_len["chr9"] = 141213431
    chr_len["chr10"] = 135534747
    chr_len["chr11"] = 135006516
    chr_len["chr12"] = 133851895
    chr_len["chr13"] = 115169878
    chr_len["chr14"] = 107349540
    chr_len["chr15"] = 102531392
    chr_len["chr16"] = 90354753
    chr_len["chr17"] = 81195210
    chr_len["chr18"] = 78077248
    chr_len["chr19"] = 59128983
    chr_len["chr20"] = 63025520
    chr_len["chr21"] = 48129895
    chr_len["chr22"] = 51304566
    chr_len["chrX"] = 155270560
    chr_len["chrY"] = 59373566
    return chr_len

def label_sequence(sequence, MAX_SEQ_LEN):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(sequence):
        X[i] = nucleotide_ind[ch]
    return X

def label_complementary(sequence, MAX_SEQ_LEN):
    complementary = {"N":"N","A":"T","T":"A","C":"G","G":"C"}
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(sequence):
        X[MAX_SEQ_LEN-i-1] = nucleotide_ind[complementary[ch]]
    return X

def Get_DNA(cpg_sites,genome):
    X_sample,sample_num = {},0
    length = int(int(window) / 2)
    for cpg_site in cpg_sites:
        cpg_id = cpg_site[0]
        pos = int(cpg_site[1]) - 1
        content = genome[chrom][pos-length:pos+length+1]
        while len(content) < length*2+1: content = content + "N"
        sample = label_sequence(content, window)
        X_sample[cpg_id] = sample
        sample_num = sample_num + 1
    print("the number of CpG is %d" % sample_num)
    return X_sample

def read_SNP():
    sample_num = 0
    chr_len = chrom_length()
    infile = open("./datasets/1KGCEU_qc.bim","r")
    SNPs,batch_size = {},1000000
    start,end = int(batch_index) * batch_size,(int(batch_index)+1)*batch_size
    #start, end = 0, chr_len[chrom]
    for line in infile:
        line = line.strip("\n").split("\t")
        snp_chr,snp_pos,snp_id = "chr"+line[0],int(line[3]),line[1]
        A1,A2 = line[4],line[5]
        if snp_chr != chrom or (snp_pos > end or snp_pos < start): continue
        key = snp_chr + "_" + str(snp_pos)
        element = [snp_chr,snp_pos,snp_id,A1,A2]
        #SNPs.append(element)
        SNPs[key] = element
        sample_num += 1
    print("the number of SNP is %d" % sample_num)
    return SNPs

#search snps in CpG sites out of array
def genome_CpG(genome):
    cpg_sites = []
    batch_size = 1000000
    start,end = int(batch_index) * batch_size,(int(batch_index)+1)*batch_size
    content = genome[chrom][start:end]
    index = content.find("CG",0)
    while index >= 0:
        pos = index + start + 1
        cpg_id = chrom + "_" + str(pos)
        cpg_site = [cpg_id,pos]
        cpg_sites.append(cpg_site)
        index = content.find("CG",index+1)
    cpg_sites = sorted(cpg_sites,key=lambda element: element[1])
    return cpg_sites

#search snps in CpG sites in array
def identify_CpG(snp_pos,batch_CpGs):
    cpg_sites,start,end = [],0,len(batch_CpGs)-1
    middle = int((start+end)/2)
    pre_start,pre_end = start,end
    while start <= end:
        middle = int((start+end)/2)
        cpg_pos = batch_CpGs[middle][1]
        if snp_pos < cpg_pos-int(window/2):
            pre_end = end
            end = middle-1
        elif snp_pos > cpg_pos+int(window/2):
            pre_start = start
            start = middle+1
        else:
             for k in range(start,end+1):
                 cpg_pos = batch_CpGs[k][1]
                 if snp_pos>=cpg_pos-int(window/2) and snp_pos<=cpg_pos+int(window/2):
                     cpg_site = batch_CpGs[k]
                     cpg_sites.append(cpg_site)
             break
    return cpg_sites

#read CpG sites in array
def read_CpG():
    genome_CpGs = {}
    for chrom in whole_chroms:
        genome_CpGs[chrom] = []
    cpg_file = open("./datasets/cpg.annotation.csv")
    line = cpg_file.readline()
    for line in cpg_file:
        line = line.split(",")
        number = line[0][1:-1]
        ID = line[1][1:-1]
        chrom = "chr"+line[3][1:-1]
        pos = int(line[4])
        element = [ID,pos]
        try: genome_CpGs[chrom].append(element)
        except: continue
    for chrom in whole_chroms:
        genome_CpGs[chrom] = sorted(genome_CpGs[chrom],key=lambda element: element[1])
    return genome_CpGs

def read_GWAS_DNA(genome):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    sample_num,length = 0,int(window/2)
    ref_DNA,var_DNA,CPG_POS,VAR_POS = [],[],[],[]
    SNPs = read_SNP()
    batch_CPGs = genome_CpG(genome)
    for snp_site in SNPs:
        snp_chrom,snp_pos = snp_site[0],int(snp_site[1])
        snp_id,A1,A2 = snp_site[2],snp_site[3],snp_site[4]
        REF = genome[snp_chrom][snp_pos-1]
        if REF == A1: ALT = A2
        elif REF == A2: ALT = A1
        cpg_sites = identify_CpG(snp_pos,batch_CPGs)
        for cpg_site in cpg_sites:
            cpg_id,cpg_pos = cpg_site[0],int(cpg_site[1])
            content = genome[snp_chrom][cpg_pos-length-1:cpg_pos+length]
            while len(content) < length*2+1: content = content + "N"
            ref_sample = label_sequence(content,window)
            var_sample = ref_sample.copy()
            if snp_pos >= cpg_pos: index = window//2+(snp_pos-cpg_pos)
            else: index = window//2-(cpg_pos-snp_pos)
            
            var_sample[index] = nucleotide_ind[ALT]
            ref_DNA.append(ref_sample)
            var_DNA.append(var_sample)
            CPG_POS.append(cpg_pos)
            VAR_POS.append(snp_pos) 
            sample_num = sample_num + 1
    print("The number of variants is %d"%sample_num)
    ref_DNA,var_DNA = np.array(ref_DNA,dtype="float32"),np.array(var_DNA,dtype="float32")
    target_file = "./datasets/2kb_genome_cpg/reference/"+chrom + "/" + chrom + "_" + str(batch_index)
    np.savez_compressed(target_file,DNA_data=ref_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)
    target_file = "./datasets/2kb_genome_cpg/variation/"+chrom + "/" + chrom + "_" + str(batch_index)
    np.savez_compressed(target_file,DNA_data=var_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)

def read_mQTL_DNA(genome):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    sample_num,length = 0,int(window/2)
    SNP = read_SNP()
    ref_DNA,var_DNA,CPG_POS,VAR_POS = [],[],[],[]
    infile = open("./datasets/2kb_mQTL.txt","r")
    line = infile.readline()
    cpg_annot = read_CpG()
    for line in infile:
        line = line.strip("\n").split()
        var_chrom,var_pos = "chr"+str(line[3]),int(line[5])
        if var_chrom != chrom: continue
        chunk = line[4].split(":")
        if len(chunk) == 4: var_id,A1,A2 = chunk[0],chunk[2],chunk[3]
        else:
            key = var_chrom + "_" + str(var_pos)
            if key not in SNP.keys(): continue
            var_id,A1,A2 = SNP[key][2],SNP[key][3],SNP[key][4]
        REF = genome[var_chrom][var_pos-1]
        if REF == A1: ALT = A2
        elif REF == A2: ALT = A1
        cpg_sites = identify_CpG(var_pos,cpg_annot[var_chrom])
        for cpg_site in cpg_sites:
            var_id = line[1]
            cpg_id,cpg_pos = cpg_site[0],int(cpg_site[1])
            content = genome[var_chrom][cpg_pos-length-1:cpg_pos+length]
            while len(content) < length*2+1: content = content + "N"
            ref_sample = label_sequence(content,window)
            var_sample = ref_sample.copy()
            if var_pos >= cpg_pos: index = int(window)//2+(var_pos-cpg_pos)
            else: index = int(window)//2-(cpg_pos-var_pos)
            var_sample[index] = nucleotide_ind[ALT]
            ref_DNA.append(ref_sample)
            var_DNA.append(var_sample)
            CPG_POS.append(cpg_pos)
            VAR_POS.append(var_pos)
            sample_num = sample_num + 1
    print("The number of variants is %d"%sample_num)
    ref_DNA,var_DNA = np.array(ref_DNA,dtype="float32"),np.array(var_DNA,dtype="float32")
    target_file = "./datasets/2kb_mqtl/reference/"+chrom
    np.savez_compressed(target_file,DNA_data=ref_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)
    target_file = "./datasets/2kb_mqtl/variation/"+chrom
    np.savez_compressed(target_file,DNA_data=var_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)


def read_ROC_DNA(genome):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    sample_num,length = 0,int(window/2)
    SNP = read_SNP()
    ref_DNA,var_DNA,CPG_POS,VAR_POS = [],[],[],[]
    infile = open("./datasets/sensitivity/susie_ppi_0.5_dist_1kb.txt","r")
    line = infile.readline()
    for line in infile:
        line = line.strip("\n").split()
        var_chrom,var_pos,cpg_pos = "chr"+str(line[0]),int(float(line[1])),int(float(line[2]))
        A1,A2 = line[6],line[7]
        if len(A1) > 1 or len(A2) > 1: continue
        REF = genome[var_chrom][var_pos-1]
        if REF == A1: ALT = A2
        elif REF == A2: ALT = A1

        content = genome[var_chrom][cpg_pos-length-1:cpg_pos+length]
        while len(content) < length*2+1: content = content + "N"
        ref_sample = label_sequence(content,window)
        var_sample = ref_sample.copy()
        if var_pos >= cpg_pos: index = int(window)//2+(var_pos-cpg_pos)
        else: index = int(window)//2-(cpg_pos-var_pos)
        var_sample[index] = nucleotide_ind[ALT]
        ref_DNA.append(ref_sample)
        var_DNA.append(var_sample)
        CPG_POS.append(cpg_pos)
        VAR_POS.append(var_pos)
        sample_num = sample_num + 1
    print("The number of variants is %d"%sample_num)
    ref_DNA,var_DNA = np.array(ref_DNA,dtype="float32"),np.array(var_DNA,dtype="float32")
    target_file = "./datasets/sensitivity/positive/reference"
    np.savez_compressed(target_file,DNA_data=ref_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)
    target_file = "./datasets/sensitivity/positive/variation"
    np.savez_compressed(target_file,DNA_data=var_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)


def read_MAP_DNA(genome):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    sample_num,length = 0,int(window/2)
    ref_DNA,var_DNA,CPG_POS,VAR_POS = [],[],[],[]
    infile = open("./datasets/fine_mapping/assoc_pairs.txt","r")
    line = infile.readline()
    for line in infile:
        line = line.strip("\n").split()
        var_chrom,var_pos,cpg_pos = "chr"+str(line[0]),int(line[1]),int(line[5])
        A1,A2 = line[2],line[3]
        if len(A1) > 1 or len(A2) > 1: continue
        REF = genome[var_chrom][var_pos-1]
        if REF == A1: ALT = A2
        elif REF == A2: ALT = A1

        content = genome[var_chrom][cpg_pos-length-1:cpg_pos+length]
        while len(content) < length*2+1: content = content + "N"
        ref_sample = label_sequence(content,window)
        var_sample = ref_sample.copy()
        if var_pos >= cpg_pos: index = int(window)//2+(var_pos-cpg_pos)
        else: index = int(window)//2-(cpg_pos-var_pos)
        var_sample[index] = nucleotide_ind[ALT]
        ref_DNA.append(ref_sample)
        var_DNA.append(var_sample)
        CPG_POS.append(cpg_pos)
        VAR_POS.append(var_pos)
        sample_num = sample_num + 1
    print("The number of variants is %d"%sample_num)
    ref_DNA,var_DNA = np.array(ref_DNA,dtype="float32"),np.array(var_DNA,dtype="float32")
    target_file = "./datasets/fine_mapping/reference"
    np.savez_compressed(target_file,DNA_data=ref_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)
    target_file = "./datasets/fine_mapping/variation"
    np.savez_compressed(target_file,DNA_data=var_DNA,CPG_POS=CPG_POS,VAR_POS=VAR_POS)


def read_GENOME():
    genome,size = {},{}
    chromo,current_chr = "",""
    DNA_file = open("./data/hg19.fa")
    for line in DNA_file:
        line = line.strip("\t\r\n")
        if ">chr" in line:
            print(line)
            if current_chr == "":
                line = line.split()
                current_chr = line[0][1:]
            else:
                genome[current_chr],size[current_chr] = chromo,len(chromo)
                chromo,line = "",line.split()
                current_chr = line[0][1:]
        elif ">" in line:
            genome[current_chr],size[current_chr] = chromo,len(chromo)
            break
        else: chromo += line
    for i in range(1,23):
        print("the length of chr %d is %d " % (i,size["chr"+str(i)]))
    print("the length of chrX is %d" % (size["chrX"]))
    print("the length of chrY is %d" % (size["chrY"]))
    print("the length of chrM is %d" % (size["chrM"]))
    return genome

if __name__=="__main__":
    genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
    #read_GWAS_DNA(genome)
    #read_mQTL_DNA(genome)
    read_ROC_DNA(genome)
    #read_MAP_DNA(genome)
