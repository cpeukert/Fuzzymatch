import random, codecs, multiprocessing, re
from unidecode import unidecode
from joblib import Parallel, delayed
from time import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
import math
import numpy as np
import statsmodels.api as sm
import pandas as pd

def read_in_list(filename,delim,c0,c1,limit):
    a=[]
    j=0
    with codecs.open(filename, encoding='utf-8',mode='r', errors='ignore') as f:
        for i in f:
            if limit>0:
                j=j+1
                if j>limit: break
            i=unidecode(i.encode('ascii', 'ignore'))
            x=i.lower().replace("\n","").replace("\r","").split(delim)
            try:
                a.append([x[c0],x[c1]])
            except IndexError:
                print x
    return a

def overlap(x,y):
    try:
        z=float(len(x))/float(len(y))
        if z>1:
            return 1/z
        else:
            return z
    except ZeroDivisionError:
        return 0

def soundex(x,y,l):
    if soundex_single(x,l)==soundex_single(y,l):
        return 1
    else:
        return 0

def soundex_single(name, len):
    """ soundex module conforming to Knuth's algorithm
        implementation 2000-12-24 by Gregory Jorgensen
        public domain
    """

    # digits holds the soundex values for the alphabet
    digits = '01230120022455012623010202'
    sndx = ''
    fc = ''

    # translate alpha chars in name to soundex digits
    for c in name.upper():
        if c.isalpha():
            if not fc: fc = c   # remember first letter
            d = digits[ord(c)-ord('A')]
            # duplicate consecutive soundex digits are skipped
            if not sndx or (d != sndx[-1]):
                sndx += d

    # replace first digit with first alpha character
    sndx = fc + sndx[1:]

    # remove all 0s from the soundex code
    sndx = sndx.replace('0','')

    # return soundex code padded to len characters
    return (sndx + (len * '0'))[:len]


def first_letter(x):
    x_set=set()
    try:
        for i in x.split(" "):
            if len(i)>1: x_set.add(i[0]) # ignore 1-character tokens
    except IndexError:
        pass
    return x_set

def bucketize(x):
    bucket={}
    for x_item in x:
        firstletter=first_letter(x_item[1])
        for fl in firstletter:
            try:
                bucket[fl].append([x_item[0],x_item[1]])
            except KeyError:
                bucket[fl]=[]
                bucket[fl].append([x_item[0],x_item[1]])
    return bucket


def compare_to_bucket(x_bucket,y_list,params,outfile):
    result=[]
    count=0
    for y_item in y_list:
        firstletter=first_letter(y_item[1])
        for fl in firstletter:
            try:
                for x_item in x_bucket[fl]:
                    mq=match_quality(x_item,y_item,params)
                    if mq<>None:
                        result.append(mq)
            except KeyError:
                print "no suitable x bucket available..."
        if result<>[]:
            write_to_file(result,outfile,"a")
            result=[]

def train_model(training_data,delim):

    # parameters from a logit regression on manually coded training data
    # -> decision criterion is the 5th percentile of the predicted probability associated with known matches
    #   p5 of (eval_hat if eval==1)
    
    data=pd.read_csv(training_data,delimiter=delim)

    data['const'] = 1
    y = 'eval'
    x = ['const', 'set_ratio', 'sort_ratio', 'ratio','overlap','soundex','reverse']

    model = sm.Logit(data[y], data[x])
    results = model.fit(disp=0)
    params=results.params
    data['yhat'] = results.predict(data[x])
    subset = data[data[y] == 1]
    params['threshold']=np.percentile(subset['yhat'], 5,interpolation='midpoint')

    return params

def match_model(x,y,params):
    set_ratio=fuzz.token_set_ratio(x[1],y[1])
    sort_ratio=fuzz.token_sort_ratio(x[1],y[1])
    ratio=fuzz.ratio(x[1],y[1])
    o=overlap(x[1],y[1])
    s=soundex(x[1],y[1],6)
    reverse=0
    if set_ratio==100 and sort_ratio==100 and o==1 and ratio<100: reverse=1


    if x[1]==y[1]:
        prob=1
    else:
        #log_odds=-34.13555+.2899927*set_ratio+.2172413*sort_ratio-.0935159*ratio-3.585193*o+.125812*s-.9100174*reverse
        log_odds=params['const']+params['set_ratio']*set_ratio+params['sort_ratio']*sort_ratio-params['ratio']*ratio+params['overlap']*o+params['soundex']*s+params['reverse']*reverse

        odds=math.exp(log_odds)
        prob=odds/(1+odds)
    return prob

def match_quality(x,y,params):

    prob=match_model(x,y,params)
    

    if prob>=params['threshold']:
        return [x[0],y[0],round(prob,4),x[1],y[1]]
    else:
        return None

        

def write_to_file(result,filename,m):
    if result<>[]:
        with codecs.open(filename, encoding='utf-8',mode=m, errors='ignore') as f:
            for row in result:
                l=len(row)
                for i,c in enumerate(row):
                    f.write(str(c))
                    if i==l-1:
                        f.write("\n")
                    else:
                        f.write("\t")
   

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def remove_duplicates(filename):
    temp=filename+".temp"
    lines_seen = set()
    outfile = open(temp, "w")
    for line in open(filename, "r"):
        if line not in lines_seen: # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    os.remove(filename)
    os.rename(temp,filename)

################################

def main(limit):
    

    print "...estimating parameters"

    params=train_model("data/training_data.csv",'\t')

    print "...reading x data"

    x=read_in_list("data/input_x.csv",'\t',0,1,limit)


    print "...reading y data"    
    
    y=read_in_list("data/input_y.csv",'\t',0,1,limit)


    outfile="data/output.csv"
    result=[["x_id","y_id","prob","x_name","y_name"]]
    write_to_file(result,outfile,"w")


    print "...creating buckets"

    x_bucket=bucketize(x)

    print "...finding matches"
    tic=float(time())

    # set n_jobs=1 if you don't want to use multicore processing
    # set n_jobs=-1 if you want to use all cores on your machine

    Parallel(n_jobs=1,verbose=5)(delayed(compare_to_bucket)(x_bucket,c,params,outfile) for c in chunks(y,100))

    
    print float(time())-tic

    print "...removing duplicates"
    remove_duplicates(outfile)


    

if __name__ == '__main__':
    main(limit=1000)


