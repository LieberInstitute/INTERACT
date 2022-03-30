from collections import OrderedDict

def read_sample_annot():
    sample_annot = {}
    sample_file = open("./data/sample.annotation.csv")
    line = sample_file.readline()    
    for line in sample_file:
        line = line.split(",")
        ID = line[0]
        tissue = line[1]
        subject = line[2]
        sample_annot[ID] = []
        sample_annot[ID].append(tissue)
        sample_annot[ID].append(subject)
        #print("%s + \t + %s + \t + %s" % (ID,tissue,subject))
    return sample_annot

def read_cpg_annot():
    cpg_annot = {}
    cpg_file = open("./data/cpg.annotation.csv")
    line = cpg_file.readline()
    for line in cpg_file:
        line = line.split(",")
        number = line[0][1:-1]
        ID = line[1][1:-1]
        chrom = line[3][1:-1]
        pos = line[4]
        cpg_annot[ID] = []
        cpg_annot[ID].append(number)
        cpg_annot[ID].append(chrom)
        cpg_annot[ID].append(pos)
        #print("%s + \t + %s + \t + %s + \t + %s" % (ID,number,chrom,pos))
    return cpg_annot

def read_signal():
    signal = OrderedDict()
    signal_file = open("./data/signal.csv")
    line = signal_file.readline()
    sample_id = []
    line = line.strip("\n\r")
    line = line.split(",")
    for i in range(2,len(line)):
        sample_id.append(line[i][1:-1])
    for line in signal_file:
        line = line.strip("\n\r")
        line = line.split(",")
        ID = line[1][1:-1]
        cpg_sample = {}
        for i in range(0,len(sample_id)):
            cpg_sample[sample_id[i]] = float(line[2+i])
        signal[ID] = cpg_sample
    print(len(signal))
    return signal

def format_data():
    chrom_data = {}
    sample_id = "X201502810041_R08C01"
    cpg_annot = read_cpg_annot()
    signal = read_signal()
    for k in range(1,23): chrom_data["chr"+str(k)] = []
    for id in signal.keys():
        if id not in cpg_annot: continue
        chrom, pos = cpg_annot[id][1], cpg_annot[id][2]
        value = signal[id][sample_id]
        if chrom == "X" or chrom == "Y": continue
        chrom_data["chr"+str(chrom)].append([pos,value])
    for i in range(1,23):
        print("chr"+str(i))
        output = open("./raw_data/BUCCAL/121/chr"+str(i)+".txt","w")
        for k in range(0,len(chrom_data["chr"+str(i)])):
            pos, value = chrom_data["chr"+str(i)][k][0], chrom_data["chr"+str(i)][k][1]
            line = "chr" + str(i) + "\t" + str(pos) + "\t" + str(value) + "\n"
            output.write(line)
        output.close()
    return chrom_data

def get_window():
    genome = read_GENOME()
    window_file = open("./data/window.bed","w")
    cpg_annot = read_cpg_annot()
    signal = read_signal()
    length = 1000
    for ID in signal.keys():
        if ID not in cpg_annot.keys():
            continue
        cpg_site = cpg_annot[ID]
        chrom = "chr"+cpg_site[1]
        pos = int(cpg_site[2])
        start_pos = pos - length
        end_pos = pos + length
        sequence = genome[chrom][start_pos:end_pos]
        line = ">" + str(chrom) + "_" + str(start_pos) + "_" + str(end_pos) + "\n"
        window_file.write(line)
        window_file.write(sequence+"\n")
    window_file.close()

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

if __name__ == "__main__":
    #read_sample_annot()
    #cpg_annot = read_cpg_annot()
    #signal = read_signal()
    format_data()
    #get_window()
