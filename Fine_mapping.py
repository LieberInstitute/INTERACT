import sys
import random
import gzip
import numpy as np
import scipy.stats as stats
import math
import os

target_chrom = sys.argv[1]
input_file = sys.argv[2]

Tissue = "BRAIN"
whole_chroms = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
#subjects = ["101","102","103","104","105","106","107","108","109","110","111","112","113","114","115","116","117","118","119","120","121"]

def read_cpg_annot():
    cpg_annot = {}
    cpg_file = open("./datasets/cpg.annotation.csv")
    line = cpg_file.readline()
    for line in cpg_file:
        line = line.split(",")
        number = line[0][1:-1]
        ID = line[1][1:-1]
        chrom = "chr"+line[3][1:-1]
        pos = int(line[4])
        cpg_annot[chrom+"_"+str(pos)] = ID
    return cpg_annot

def get_pos():
    genome = np.load("./datasets/genome.npy",allow_pickle=True).item()
    infile = open("./datasets/1KGCEU_qc.bim","r")
    pos2id = {}
    for idx,line in enumerate(infile):
        line = line.strip("\n").split("\t")
        snp_chr,snp_pos,snp_id = "chr"+line[0],int(line[3]),line[1]
        A1,A2 = line[4],line[5]
        if genome[snp_chr][snp_pos-1] == A1:
            REF,ALT = A1,A2
        else:
            REF,ALT = A2,A1
        pos2id[snp_chr+"_"+str(snp_pos)] = [snp_id,REF,ALT]
    return pos2id

def read_SNP():
    infile = open("./data/snp_map.txt","r")
    SNP = {}
    line = infile.readline()
    for line in infile:
        line = line.strip("\n").split("\t")
        snp_chr,snp_pos,snp_id = "chr"+line[0],int(line[2]),line[1]
        A1,A2 = line[3],line[4]
        element = [snp_chr,snp_pos,snp_id,A1,A2]
        SNP[snp_id] = element
    return SNP

def max_diff(chrom,input_file):
    chrom = target_chrom
    ref_mqtl,var_mqtl = {},{}
    pos2id = get_pos()
    cpg_annot = read_cpg_annot()
    subjects = os.listdir("./outputs/2kb_genome_cpg/epic_aa/difference")
    ref_file = open("./outputs/2kb_genome_cpg/epic_aa/reference/"+chrom+"/"+input_file+".txt","r")
    for ref_line in ref_file:
        line = ref_line.strip("\n").split("\t")
        item = [int(line[0]),int(line[1])] + line[2:]
        ref_mqtl[line[0]+"_"+line[1]] = item
    var_file = open("./outputs/2kb_genome_cpg/epic_aa/variation/"+chrom+"/"+input_file+".txt","r")
    for index,var_line in enumerate(var_file):
        line = var_line.strip("\n").split("\t")
        item = [int(line[0]),int(line[1])] + line[2:]
        var_mqtl[line[0]+"_"+line[1]] = item
    for idx,subject in enumerate(subjects[0:21]):
        print(subject)
        map_mqtl = {}
        for k in ref_mqtl.keys():
            #print(str(ref_mqtl[k][1]) + "\t" + str(var_mqtl[k][1]))
            ref_level,var_level = float(ref_mqtl[k][idx+2]),float(var_mqtl[k][idx+2])
            diff,ab_diff = var_level-ref_level,np.abs(var_level-ref_level)
            cpg_pos,snp_pos = ref_mqtl[k][0],ref_mqtl[k][1] 
            var_id, REF, ALT = pos2id[chrom+"_"+str(snp_pos)]
            cpg_id = chrom+"_"+str(cpg_pos)
            element = [chrom,var_id,snp_pos,REF,ALT,cpg_id,cpg_pos,ref_level,var_level,diff,ab_diff]
            try: 
                if map_mqtl[var_id][10] < ab_diff: map_mqtl[var_id] = element
            except: map_mqtl[var_id] = element
        mqtl = map_mqtl.values()
        mqtl = sorted(mqtl,key=lambda element: element[10],reverse=True)
        outfile = open("./outputs/2kb_genome_cpg/epic_aa/difference/"+str(subject)+"/"+chrom+"/"+input_file+".txt","w")
        for k in range(0,len(mqtl)):
            element = mqtl[k]
            outline = element[0]+"\t"+str(element[6])+"\t"+str(element[2])+"\t"+str(element[3])+"\t"+str(element[7])+"\t"+\
            str(element[4])+"\t"+str(element[8])+"\t"+str(element[10])+"\n"
            outfile.write(outline)
        outfile.close()


def combine_batch():
     chrom = target_chrom
     subjects = os.listdir("./outputs/2kb_genome_cpg/epic_aa/difference")
     for idx,subject in enumerate(subjects[0:21]):
         chrom_snps = []
         input_files = os.listdir("./outputs/2kb_genome_cpg/epic_aa/difference/"+str(subject)+"/"+chrom)
         output_file = open("./outputs/2kb_genome_cpg/epic_aa/difference/"+str(subject)+"/"+chrom+".txt","w")
         for input_file in input_files:
             if ".txt" not in input_file: continue
             infile = open("./outputs/2kb_genome_cpg/epic_aa/difference/"+str(subject)+"/"+chrom+"/"+input_file,"r")
             batch_name = input_file.split(".")[0]
             print(batch_name)
             for line in infile:
                 line = line.strip("\n").split("\t")
                 if float(line[7]) > 0.9: print(line)
                 chrom_snps.append([line[0],int(line[1]),int(line[2]),line[3],float(line[4]),line[5],float(line[6]),float(line[7])])
         chrom_snps = sorted(chrom_snps,key=lambda element: element[7],reverse=True)
         for index in range(0,len(chrom_snps)):
             element = chrom_snps[index]
             line = str(element[0])+"\t"+str(element[1])+"\t"+str(element[2])+"\t"+str(element[3])+"\t"+str(element[4])+"\t"+str(element[5])\
                    +"\t"+str(element[6])+"\t"+str(element[7])+"\n"
             output_file.write(line)
         output_file.close()
             

def combine_chrom():
    subjects = os.listdir("./outputs/2kb_genome_cpg/brain/difference")
    for idx,subject in enumerate(subjects):
        genome_snps = []
        #output_file = open("./outputs/2kb_genome_cpg/brain/difference/"+str(subject)+"/genome.txt","w")
        output_file = open("/dcs04/lieber/statsgen/jiyunzhou/Bert_mQTL/plot/distance/"+str(subject)+".txt","w")
        for chrom in whole_chroms:
            input_file = "./outputs/2kb_genome_cpg/brain/difference/"+str(subject)+"/"+chrom+".txt"
            infile = open(input_file,"r")
            print(chrom)
            for line in infile:
                line = line.strip("\n").split("\t")
                genome_snps.append([line[0],int(line[1]),int(line[2]),line[3],float(line[4]),line[5],float(line[6]),float(line[7])])
        genome_snps = sorted(genome_snps,key=lambda element: element[7],reverse=True)
        #output_file.write("chrom\tcpg_pos\tsnp_pos\tref_allele\tref_value\tvar_allele\tvar_value\tdifference\n")
        for index in range(0,len(genome_snps)):
            element = genome_snps[index]
            cpg_pos, snp_pos, effect = int(element[1]), int(element[2]), float(element[7])
            if snp_pos-cpg_pos > 0 and snp_pos-cpg_pos < 1000:
                line = str((snp_pos-cpg_pos)//100*100)+"-"+str((snp_pos-cpg_pos)//100*100+100)+"\t"+str(snp_pos-cpg_pos)+"\t"+str(effect)+"\n"
            elif snp_pos-cpg_pos == 1000:
                line = str((snp_pos-cpg_pos)//100*100-100)+"-"+str((snp_pos-cpg_pos)//100*100)+"\t"+str(snp_pos-cpg_pos)+"\t"+str(effect)+"\n"
            elif cpg_pos-snp_pos == 1000:
                line = str((cpg_pos-snp_pos)//100*100)+"-"+str((cpg_pos-snp_pos)//100*100-100)+"\t"+str(snp_pos-cpg_pos)+"\t"+str(effect)+"\n"
            else:
                line = str((cpg_pos-snp_pos)//100*100+100)+"-"+str((cpg_pos-snp_pos)//100*100)+"\t"+str(snp_pos-cpg_pos)+"\t"+str(effect)+"\n"
            #line = str(element[0])+"\t"+str(element[1])+"\t"+str(element[2])+"\t"+str(element[3])+"\t"+str(element[4])+"\t"+str(element[5])\
            #       +"\t"+str(element[6])+"\t"+str(element[7])+"\n"
            output_file.write(line)
        output_file.close()

if __name__ == "__main__":
    for 
    #max_diff()
    #combine_batch()
    combine_chrom()
