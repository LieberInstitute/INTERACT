import sys
import os
import random
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr

target_chrom = sys.argv[1]

def read_epic_data():
    id2pos = {}
    infile = open("./raw_data/epic_aa/annotation.csv","r")
    line = infile.readline()
    for line in infile:
        line = line.strip("\n").split(",")
        cpg_id,cpg_chr,cpg_pos = line[0][1:-1],line[1][1:-1],int(line[2])
        id2pos[cpg_id] = [cpg_chr,cpg_pos]
    infile = open("./raw_data/independent/EPIC_DLPFC_control_train_data.txt","r")
    output = open("./raw_data/independent/methylation/chr"+str(target_chrom)+".txt","w")
    pos_file = open("./raw_data/independent/position/chr"+str(target_chrom)+".txt","w")
    line = infile.readline()
    content = line.strip("\n").split()
    line = "position"
    for item in content: line = line + "\t" + item
    output.write(line + "\n")
    pos_file.write("x" + "\n")
    pos_index = 0
    for line in infile:
        content = line.strip("\n").split()
        cpg_id = content[0]
        cpg_chr,cpg_pos = id2pos[cpg_id][0], id2pos[cpg_id][1]
        if cpg_chr != "chr"+str(target_chrom): continue
        line = str(cpg_pos)
        for item in content[1:]: line = line + "\t" + item
        output.write(line + "\n")
        pos_file.write(str(pos_index) + "\t" + str(cpg_pos) + "\n")
        pos_index = pos_index + 1
    output.close()
    pos_file.close()

def read_epic():
    output = open("./raw_data/epic/methylation/chr"+str(target_chrom)+".txt","w")
    methylation, positions = {}, []
    tissues = ["BRAIN","BLOOD","BUCCAL","SALIVA"]
    subjects = ["101","102","103","104","105","106","107","108","109","110","111","112","113","114","115","116","117","118","119","120","121"]
    line = "position"
    for tissue in tissues:
        for subject in subjects:
            line = line + "\t" + tissue + "_" + str(subject)
    output.write(line + "\n")
    infile = open("/dcs04/lieber/statsgen/jiyunzhou/mQTL/raw_data/BRAIN/101/chr"+str(target_chrom)+".txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        positions.append(line[1])
    for tissue in tissues:
        for subject in subjects:
            print(tissue+"_"+str(subject))
            methylation[tissue+"_"+str(subject)] = []
            infile = open("/dcs04/lieber/statsgen/jiyunzhou/mQTL/raw_data/"+tissue+"/"+str(subject)+"/chr"+str(target_chrom)+".txt","r")
            for line in infile:
                line = line.strip("\n").split("\t")
                methylation[tissue+"_"+str(subject)].append(line[2])
    for k in range(0,len(methylation["BRAIN_101"])):
        print(k)
        line = positions[k]
        for tissue in tissues:
            for subject in subjects: line = line + "\t" + str(methylation[tissue+"_"+str(subject)][k])
        line = line + "\n"
        output.write(line)
    output = open("./raw_data/epic/position/chr"+str(target_chrom)+".txt","w")
    output.write("x" + "\n")
    for k in range(0,len(positions)):
        output.write(str(k) + "\t" + str(positions[k]) + "\n")
    output.close()
 
def read_methylation():
    methylation = []
    infile = open("./raw_data/epic/methylation/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    for idx,content in enumerate(infile):
        content = content.strip("\n").split("\t")
        content = np.array(content[1:], dtype=np.float)
        methylation.append(content)
        if idx % 10000 == 0: print("reading array methylation " + str(idx))
    return methylation


def read_inputs():
    smooth_methylation,wgbs_methylation = [],[]
    missing_subjects = ["Br1464","Br1649"]

    infile = open("./raw_data/smooth/methylation/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    for idx,content in enumerate(infile):
        content = content.strip("\n").split("\t")
        content = np.array(content[1:], dtype=np.float)
        smooth_methylation.append(content)
        if idx % 10000 == 0: print("reading smooth methylation " + str(idx))
    
    infile = open("./raw_data/wgbs/methylation/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    wgbs_subjects = line.strip("\n").split("\t")
    wgbs_subjects = [item[1:-1] for item in wgbs_subjects]
    missing_index = [wgbs_subjects.index("Br1464"),wgbs_subjects.index("Br1649")]
    for idx,content in enumerate(infile):
        content = content.replace("NA","-100").strip("\n").split("\t")
        content = np.array(content[1:], dtype=np.float)
        content[content==-100] = smooth_methylation[idx][content==-100]
        wgbs_methylation.append(content)
        if idx % 10000 == 0: print("reading wgbs methylation " + str(idx))
    wgbs_methylation = np.array(wgbs_methylation,dtype=np.float)
    wgbs_methylation = np.delete(wgbs_methylation,missing_index,1)
    return wgbs_methylation


def read_coverage():
    coverage = []
    infile = open("./raw_data/wgbs/coverage/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    wgbs_subjects = line.strip("\n").split("\t")
    wgbs_subjects = [item[1:-1] for item in wgbs_subjects]
    missing_index = [wgbs_subjects.index("Br1464"),wgbs_subjects.index("Br1649")]
    for idx,content in enumerate(infile):
        line = content.strip("\n").split("\t")
        content = np.array(line[1:], dtype=np.float)
        coverage.append(content)
        if idx % 10000 ==  0: print("reading coverage " + str(idx))
    coverage = np.array(coverage,dtype=np.float)
    coverage = np.delete(coverage,missing_index,1)
    return coverage


def read_position():
    wgbs_position,array_position, input_index = [],[],[]
    infile = open("./raw_data/epic/position/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    for idx,line in enumerate(infile):
        line = line.strip("\n").split("\t")
        pos = int(line[1])
        wgbs_position.append(pos)
        if idx % 10000 ==  0: print("reading wgbs position " + str(idx))

    infile = open("./raw_data/epic/position/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    for idx,line in enumerate(infile):
        line = line.strip("\n").split("\t")
        pos = int(line[1])
        array_position.append(pos)
        #pos = wgbs_position.index(pos)
        print(pos)
        input_index.append(pos)
        if idx % 10000 ==  0: print("reading array position " + str(idx))
    array_position = np.array(array_position,dtype=np.long)
    input_index = np.array(input_index,dtype=np.long)
    return array_position,input_index


###read hg38 reference genome
"""
def read_GENOME():
    genome,size = {},{}
    chromo,current_chr = "",""
    DNA_file = open("./datasets/hg38.fa")
    for line in DNA_file:
        line = line.strip("\r\n")
        line = line.split()
        if ">" in line[0] and len(line[0]) <= 3:
            print(line[0])
            if current_chr == "":
                current_chr = "chr" + line[0][1:]
            else:
                genome[current_chr],size[current_chr] = chromo,len(chromo)
                chromo = ""
                current_chr = "chr" + line[0][1:]
        elif ">" in line[0] and len(line[0]) > 3:
            genome[current_chr],size[current_chr] = chromo,len(chromo)
            break
        else: chromo += line[0]
    for i in range(1,23):
        print("the length of chr %d is %d " % (i,size["chr"+str(i)]))
    print("the length of chrX is %d" % (size["chrX"]))
    print("the length of chrY is %d" % (size["chrY"]))
    return genome
"""

def read_GENOME():
    genome,size = {},{}
    chromo,current_chr = "",""
    DNA_file = open("./datasets/hg19.fa")
    for line in DNA_file:
        line = line.strip("\t\r\n")
        if ">chr" in line:
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
        else:
            chromo = chromo + line
    for i in range(1,23):
        print("the length of chr %d is %d " % (i,size["chr"+str(i)]))
    print("the length of chrX is %d" % (size["chrX"]))
    print("the length of chrY is %d" % (size["chrY"]))
    print("the length of chrM is %d" % (size["chrM"]))
    return genome

def label_sequence(sequence, MAX_SEQ_LEN):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(sequence):
        X[i] = nucleotide_ind[ch]
    return X


def compute_corr():
    brain_methylation,epic_methylation = {},{}
    infile = open("./raw_data/epic/methylation/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    for idx,content in enumerate(infile):
        content = content.strip("\n").split("\t")
        position = content[0]
        brain_methylation[position] = float(content[1])
    infile = open("./raw_data/epic_aa/methylation/chr"+str(target_chrom)+".txt","r")
    line = infile.readline()
    for idx,content in enumerate(infile):
        content = content.strip("\n").split("\t")
        position = content[0]
        epic_methylation[position] = float(content[81])
    positions = list(set(brain_methylation.keys()) & set(epic_methylation.keys()))
    brain,epic = [],[]
    for pos in positions:
        brain.append(brain_methylation[pos])
        epic.append(epic_methylation[pos])
    pearson = pearsonr(brain, epic)[0] 
    print("The pearson correlation is " + str(pearson))    


def run_feature():
    window = 2001
    genome = read_GENOME()
    #input_data = read_inputs()
    methylation = read_methylation()
    #coverage = read_coverage()
    #coverage = np.array(coverage,dtype=np.float32)
    position,input_index = read_position()
    DNA_data = []
    for idx in range(0,len(position)):
        start, stop = position[idx]-window//2-1, position[idx]+window//2
        sequence = genome["chr"+str(target_chrom)][start:stop]
        DNA_data.append(label_sequence(sequence,window))
        if idx % 10000 ==  0: print("reading feature " + str(idx))
    DNA_data = np.array(DNA_data,dtype=np.long)
    target_file = "./datasets/epic/chr" + target_chrom
    print("The number of samples for chr" + str(target_chrom) + " is " + str(len(input_index)))
    print("The number of tasks is " + str(len(methylation[0])))
    #np.savez_compressed(target_file,input_data=input_data,input_index=input_index,methylation_data=methylation,\
    #                    coverage_data=coverage,DNA_data=DNA_data,position=position)
    np.savez_compressed(target_file,methylation_data=methylation,DNA_data=DNA_data,position=position)


if __name__ == "__main__":
    #read_epic_data()
    #read_epic()
    #methylation = read_methylation()
    #coverage = read_coverage()
    #position = read_position()
    #input_data = read_inputs()
    run_feature()
    #compute_corr()
