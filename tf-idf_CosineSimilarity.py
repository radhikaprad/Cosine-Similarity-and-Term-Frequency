import os
import nltk
import math
import collections
from math import*
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus_root = 'C:\\presidential_debates'
Pstemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))    
filedictionary = {}
idfdictionary = {}

# TermInfo namedtuple to store term info per term per file
TermInfo = collections.namedtuple("TermInfo", ["tf", "tfidf", "normalizedtfidf"])

def populatedictionary():
    #print("Inside populatedictionary")
    global filedictionary
    if len(filedictionary) == 0:
        #print("Getting file info populatedictionary")
        for filename in os.listdir(corpus_root):
            file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
            doc = file.read()
            file.close() 
            doc = doc.lower()
            wordtoken = tokenizer.tokenize(doc)
            stopw = [w for w in wordtoken if not w in stop_words ]
            stemmedwords = []
            for w in stopw:
                stemmedwords.append(Pstemmer.stem(w))
            stemmedcounter = collections.Counter(stemmedwords)
            totalcount = sum(stemmedcounter.values())
            terminfodictionary = {}
            for key in stemmedcounter.keys():
                tinfo = TermInfo(stemmedcounter[key], 0, 0)
                terminfodictionary[key] = tinfo
            filedictionary[filename.lower()] = terminfodictionary

def gettermidf(term):
    count = getdocfreq(term)
    return (math.log10(len(filedictionary)/float(count)))

def populateidfdictionary():
    global idfdictionary
    global filedictionary
    populatedictionary()
    if len(idfdictionary) == 0:
        for fileinfo in filedictionary.values():
            tfidf = []
            for key in fileinfo.keys():
                if key not in idfdictionary:
                    idfdictionary[key] = gettermidf(key)
                terminfo = fileinfo[key]
                terminfo = terminfo._replace(tfidf = ((1 + math.log10(terminfo.tf)) * idfdictionary[key]))
                tfidf.append(terminfo.tfidf)
                fileinfo[key] = terminfo

            normalizeddenominator = float(square_rooted(tfidf))
            for key in fileinfo.keys():
                terminfo = fileinfo[key]
                terminfo = terminfo._replace(normalizedtfidf = (terminfo.tfidf/float(normalizeddenominator)))
                fileinfo[key] = terminfo                

def getcount(term):
    populatedictionary()
    count = 0
    term = term.lower()
    for fileinfo in filedictionary.values():
        if term in fileinfo:
            terminfo = fileinfo[term]
            count = count + terminfo.tf
    print(count)

def getdocfreq(term):
    populatedictionary()
    docfreq = 0
    for fileinfo in filedictionary.values():
        if term in fileinfo:
            terminfo = fileinfo[term]
            if terminfo.tf > 0:
                docfreq = docfreq + 1
    return docfreq

def getidf(term):
    populateidfdictionary()
    idf = 0
    stemmedterm = Pstemmer.stem(term.lower())
    if stemmedterm in idfdictionary:
        idf = idfdictionary[stemmedterm]
    print("%.12f" % idf)

def square_rooted(x):
   return (sqrt(sum([a*a for a in x])))

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    if denominator != 0:
        return (numerator/float(denominator))
    return 0

def query(queryterms):
    populatedictionary()
    populateidfdictionary()
    highcount = 0
    highcountfilename = "Error: No File Found"
    queryterms = queryterms.lower()
    querytokens = tokenizer.tokenize(queryterms)
    querystopw = [w for w in querytokens if not w in stop_words]
    querystemwords = []
    for w in querystopw:
        querystemwords.append(Pstemmer.stem(w))

    stemmedcounter = collections.Counter(querystemwords)
    termcount = len(stemmedcounter)
    if termcount == 0:
        print ("No query terms specified")
        return

    querytermsinfo = []
    for key in stemmedcounter.keys():
        tfvalue = (1 + math.log10(stemmedcounter[key]))
        querytermsinfo.append(tfvalue)

    print (stemmedcounter)

    normalizedquerytermsinfo = []
    normalizeddenominator = float(square_rooted(querytermsinfo))
    print("NormalizedDenominator Query", normalizeddenominator)
    for tfvalue in querytermsinfo:
        normalizedvalue = tfvalue / float(normalizeddenominator)
        normalizedquerytermsinfo.append(normalizedvalue)

    print ("QueryTerms - Regular and Normalized")
    print (querytermsinfo)
    print (normalizedquerytermsinfo)

    for filename in filedictionary.keys():
        fileinfo = filedictionary[filename]
        filetermsinfo = []
        for term in stemmedcounter.keys():
            tfvalue = 0;
            if term in fileinfo:
                terminfo = fileinfo[term]
                #tfvalue = terminfo.tfidf
                tfvalue = terminfo.normalizedtfidf
            filetermsinfo.append(tfvalue)

        normalizedfiletermsinfo = []
        normalizeddenominator = float(square_rooted(filetermsinfo))
        print("NormalizedDenominator File ", normalizeddenominator)
        for tfvalue in filetermsinfo:
            normalizedvalue = 0
            if normalizeddenominator != 0:
                normalizedvalue = tfvalue / float(normalizeddenominator)
            normalizedfiletermsinfo.append(normalizedvalue)

        print ("FileTerms - Regular and Normalized")
        print (filetermsinfo)
        print (normalizedfiletermsinfo)
        similarityvalue = cosine_similarity(normalizedquerytermsinfo, normalizedfiletermsinfo)
        print (similarityvalue)

        if (similarityvalue > highcount):
            highcount = similarityvalue
            highcountfilename = filename

    print (highcount)
    print (highcountfilename)

def querydocsim(queryterms,filename):
    populatedictionary()
    populateidfdictionary()
    filename = filename.lower()
    if filename not in filedictionary:
        print ("Invalid file name")
        return

    queryterms = queryterms.lower()
    querytokens = tokenizer.tokenize(queryterms)
    querystopw = [w for w in querytokens if not w in stop_words]
    querystemwords = []
    for w in querystopw:
        querystemwords.append(Pstemmer.stem(w))

    stemmedcounter = collections.Counter(querystemwords)
    termcount = len(stemmedcounter)
    if termcount == 0:
        print ("No query terms specified")
        return

    querytermsinfo = []
    for key in stemmedcounter.keys():
        tfvalue = (1 + math.log10(stemmedcounter[key]))
        querytermsinfo.append(tfvalue)

    print (stemmedcounter)
    
    normalizedquerytermsinfo = []
    normalizeddenominator = float(square_rooted(querytermsinfo))
    print("NormalizedDenominator Query", normalizeddenominator)
    for tfvalue in querytermsinfo:
        normalizedvalue = tfvalue / float(normalizeddenominator)
        normalizedquerytermsinfo.append(normalizedvalue)

    print ("QueryTerms - Regular and Normalized")
    print (querytermsinfo)
    print (normalizedquerytermsinfo)

    fileinfo = filedictionary[filename]
    filetermsinfo = []
    for term in stemmedcounter.keys():
        tfvalue = 0;
        if term in fileinfo:
            terminfo = fileinfo[term]
            tfvalue = terminfo.tfidf
            print(term, terminfo)
            #tfvalue = terminfo.normalizedtfidf
        filetermsinfo.append(tfvalue)
    
    normalizedfiletermsinfo = []
    normalizeddenominator = float(square_rooted(filetermsinfo))
    print("NormalizedDenominator File ", normalizeddenominator)
    for tfvalue in filetermsinfo:
        normalizedvalue = 0
        if normalizeddenominator != 0:
            normalizedvalue = tfvalue / float(normalizeddenominator)
        normalizedfiletermsinfo.append(normalizedvalue)

    print ("querydocsim - FileTerms - Regular and Normalized")
    print (filetermsinfo)
    print (normalizedfiletermsinfo)
    similarityvalue = cosine_similarity(normalizedquerytermsinfo, normalizedfiletermsinfo)
    similarityvaluenotnorm = cosine_similarity(normalizedquerytermsinfo, filetermsinfo)
    similarityvaluebothnotnorm = cosine_similarity(querytermsinfo, filetermsinfo)
    print ("querydocsim complete - ", similarityvalue, similarityvaluenotnorm, similarityvaluebothnotnorm)

def docdocsim(filename1,filename2):
    populatedictionary()
    populateidfdictionary()
    file1termsinfo = []
    file2termsinfo = []
    filename1 = filename1.lower()
    filename2 = filename2.lower()

    if filename1 not in filedictionary:
        print ("file 1 name is invalid")
        return

    if filename2 not in filedictionary:
        print ("file 2 name is invalid")
        return

    fileinfo1 = filedictionary[filename1]
    fileinfo2 = filedictionary[filename2]

    for term in sorted(fileinfo1):
        termslist = fileinfo[term]
        if term in termslist:
            tfvalue = termslist[term].tfidf
            filetermsinfo.append(tfvalue/float(termcount))

        print (filetermsinfo)
        similarityvalue = cosine_similarity(querytermsinfo, filetermsinfo)
        print (similarityvalue)

def test():
    populatedictionary()
    populateidfdictionary()
    print(Pstemmer.stem("uninsured"))
    print(Pstemmer.stem("healthcare"))
    key = "health"
    tfidf = getidf(key)
    tfidf = ((1 + math.log10(5)) * idfdictionary[key])
    print(tfidf)
