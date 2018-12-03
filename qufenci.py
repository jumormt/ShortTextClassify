#coding=utf-8
import csv
import jieba
import pandas as pd

filename = ["resource/sougou_data_all/互联网/10.txt","resource/sougou_data_all/互联网/11.txt",
            "resource/sougou_data_all/旅游/26.txt"]
stopwords_file = "resource/stop_words.txt"
outputfile = "resource/output/"
print(outputfile+filename[0][25:])

f = open(stopwords_file,"r",encoding='utf-8')
result = list()
for line in f.readlines():
    line = line.strip()
    if not len(line):
        continue

    result.append(line)
f.close()

with open("resource/stop_words2.txt","w", encoding='utf-8') as fw:
    for sentence in result:
        sentence.encode('utf-8')
        data=sentence.strip()
        if len(data)!=0:
            fw.write(data)
            fw.write("\n")
print ("end")

#整理停用词 去空行和两边的空格

stop_f = open(stopwords_file,"r",encoding='utf-8')
stop_words = list()
for line in stop_f.readlines():
    line = line.strip()
    if not len(line):
        continue

    stop_words.append(line)
# stopwords_file.close()
for file in filename:
    f=open(file,"r")
    result = list()
    for line in f.readlines():
        line = line.strip()
        if not len(line):
            continue
        outstr = ''
        seg_list = jieba.cut(line,cut_all=False)
        for word in seg_list:
            if word not in stop_words:
                if word != '\t':
                    outstr += word
                    outstr += " "
       # seg_list = " ".join(seg_list)
        result.append(outstr.strip())
    f.close()

    outputer = outputfile+file[25:]
    print(outputer)
    with open(outputer,"w",encoding='utf-8') as fw:
        for sentence in result:
            sentence.encode('utf-8')
            data=sentence.strip()
            if len(data)!=0:
                fw.write(data)
                fw.write("\n")

            fw.write("\n")

print ("end")

#分词、停用词过滤（包括标点）


