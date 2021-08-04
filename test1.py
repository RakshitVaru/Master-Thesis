
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
from time import time
from pycaret.arules import *

import pymongo

from flask import send_file

from flask import Flask, flash, redirect, render_template, request,session, abort,send_from_directory,send_file,jsonify, url_for
import pandas as pd
import json
app = Flask(__name__)

sup=0
conf=0


def add(thre_sup, thre_conf):    
    df= pd.read_csv("new_crime.csv")
    df.head()
    global sup,conf
    if (thre_sup is None or thre_sup==''):thre_sup=0.05
    if (thre_conf is None or thre_conf==''):thre_conf=0.5
    sup=thre_sup
    conf=thre_conf
    exp_arules = setup(df, transaction_id = 'zone', item_id = 'rucr_ext_d')
    rules_fp = create_model(threshold = float(conf), min_support=float(sup))
    rules_fp.to_csv('Final_fp.csv')
    inter1="antecedent support"
    rules_fp=rules_fp.sort_values(inter1)
    rules_fp.to_csv('Final_ap.csv')
    row=[]
    for i in rules_fp['antecedents']:
        for j in list(i):
            if j not in row:
                row.append(j)


    for i in rules_fp['consequents']:
        for j in list(i):
            if j not in row:
                row.append(j)
    test=[]
    for i in range(len(rules_fp['antecedents'])): 
        x= (list(rules_fp['antecedents'][i]))
        y= (list(rules_fp['consequents'][i]))
        test.append([x,y])
    name= list(rules_fp.columns)
    
    count=[]
    for i in row:
        c=0
        for j in rules_fp['consequents']:
            if i in list(j):
                c+=1
        count.append(c)
       

    row.sort(key=dict(zip(row, count)).get)
  
    
    inter2="consequent support"
    support= rules_fp[inter2].tolist()
    rules_fp.iloc[:,2:]=rules_fp.iloc[:,2:].round(3)
    return test,row,support, name[2:], rules_fp.iloc[:,2:].to_dict('records'), thre_sup, thre_conf




def market_basket(thre_sup, thre_conf):
    df= pd.read_csv("marketbasket.csv")
    global sup,thre
    
    if (thre_sup is None or thre_sup==''):thre_sup=0.02
    if (thre_conf is None or thre_conf==''):thre_conf=0.7
    sup=thre_sup
    conf=thre_conf
    start= time()
    frequent_itemsets_fp=fpgrowth(df, min_support=float(sup), use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=float(conf))
    end= time()
    print("Time for Fp growth= ", end-start)
    inter1="antecedent support"
    rules_fp=rules_fp.sort_values(inter1)
    rules_fp.to_csv('Final_ap.csv')
    row=[]
    for i in rules_fp['antecedents']:
        for j in list(i):
            if j not in row:
                row.append(j)


    for i in rules_fp['consequents']:
        for j in list(i):
            if j not in row:
                row.append(j)
    test=[]
    for i in range(len(rules_fp['antecedents'])): 
        x= (list(rules_fp['antecedents'][i]))
        y= (list(rules_fp['consequents'][i]))
        test.append([x,y])
    name= list(rules_fp.columns)
    
    count=[]
    for i in row:
        c=0
        for j in rules_fp['consequents']:
            if i in list(j):
                c+=1
        count.append(c)
       

    row.sort(key=dict(zip(row, count)).get)

    inter2="consequent support"
    support= rules_fp[inter2].tolist()
    norm_lift,norm_conviction=[],[]
    for i in range(len(rules_fp["lift"])):
        # norm_lift.append((rules_fp["lift"][i]-1)/((1/max(rules_fp["antecedent support"][i], rules_fp['consequent support'][i]))-1))
        norm_lift.append((rules_fp["lift"][i]-(max(rules_fp["antecedent support"][i]+rules_fp["consequent support"][i]-1, 1/len(rules_fp))))/((1/max(rules_fp["antecedent support"][i], rules_fp['consequent support'][i]))-(max(rules_fp["antecedent support"][i]+rules_fp["consequent support"][i]-1, 1/len(rules_fp)))))
        norm_conviction.append((rules_fp["conviction"][i]-1)/((1/max(rules_fp["antecedent support"][i], rules_fp['consequent support'][i]))-1))    
    normalized_conviction = [float(i)/max(norm_conviction) for i in norm_conviction]
    normalized_lift=[float(i)/max(norm_lift) for i in norm_lift]
    normalized_support = [float(i)/max(rules_fp["support"]) for i in rules_fp["support"]]
    normalized_antesupport = [float(i)/max(rules_fp["antecedent support"]) for i in rules_fp["antecedent support"]]
    normalized_consesupport = [float(i)/max(rules_fp["consequent support"]) for i in rules_fp["consequent support"]]
    rules_fp["normalized_lift"]=normalized_lift 
    rules_fp["lift"]= rules_fp["normalized_lift"]
    rules_fp["normalized_conviction"]=normalized_conviction
    rules_fp["normalized_support"]=normalized_support
    rules_fp["normalized_antecedent support"]=normalized_antesupport
    rules_fp["normalized_consequent support"]=normalized_consesupport
    rules_fp["normalized_confidence"]=rules_fp["confidence"]
    name.remove("leverage")
    name.remove("conviction")
    rules_fp.iloc[:,2:]=rules_fp.iloc[:,2:].round(3)
    return test,row,support, name[2:], rules_fp.iloc[:,2:].to_dict('records'), thre_sup, thre_conf


def heart_as(thre_sup, thre_conf):
    
    if (thre_sup is None or thre_sup==''):thre_sup=0.15
    if (thre_conf is None or thre_conf==''):thre_conf=0.9
    global sup,thre
    sup=thre_sup
    conf=thre_conf
    
    df= pd.read_csv("heart2.csv")
    df=df.iloc[:, 1:]
    frequent_itemsets_fp=fpgrowth(df, min_support=float(sup), use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=float(conf))

    inter1="antecedent support"
    rules_fp=rules_fp.sort_values(inter1)
    rules_fp.to_csv('Final_ap.csv')
    row=[]
    for i in rules_fp['antecedents']:
        for j in list(i):
            if j not in row:
                row.append(j)


    for i in rules_fp['consequents']:
        for j in list(i):
            if j not in row:
                row.append(j)
    test=[]
    for i in range(len(rules_fp['antecedents'])): 
        x= (list(rules_fp['antecedents'][i]))
        y= (list(rules_fp['consequents'][i]))
        test.append([x,y])
    name= list(rules_fp.columns)
    
    count=[]
    for i in row:
        c=0
        for j in rules_fp['consequents']:
            if i in list(j):
                c+=1
        count.append(c)
       

    row.sort(key=dict(zip(row, count)).get)
  
    inter2="consequent support"
    support= rules_fp[inter2].tolist()

    norm_lift,norm_conviction=[],[]
    for i in range(len(rules_fp["lift"])):
        norm_lift.append((rules_fp["lift"][i]-(max(rules_fp["antecedent support"][i]+rules_fp["consequent support"][i]-1, 1/len(rules_fp))))/((1/max(rules_fp["antecedent support"][i], rules_fp['consequent support'][i]))-(max(rules_fp["antecedent support"][i]+rules_fp["consequent support"][i]-1, 1/len(rules_fp)))))
        norm_conviction.append((rules_fp["conviction"][i]-1)/((1/max(rules_fp["antecedent support"][i], rules_fp['consequent support'][i]))-1))    
    normalized_conviction = [float(i)/max(norm_conviction) for i in norm_conviction]
    normalized_lift=[float(i)/max(norm_lift) for i in norm_lift]
    normalized_support = [float(i)/max(rules_fp["support"]) for i in rules_fp["support"]]
    normalized_antesupport = [float(i)/max(rules_fp["antecedent support"]) for i in rules_fp["antecedent support"]]
    normalized_consesupport = [float(i)/max(rules_fp["consequent support"]) for i in rules_fp["consequent support"]]
    rules_fp["normalized_lift"]=normalized_lift 
    rules_fp["lift"]= rules_fp["normalized_lift"]  
    rules_fp["normalized_conviction"]=normalized_conviction
    rules_fp["normalized_support"]=normalized_support
    rules_fp["normalized_antecedent support"]=normalized_antesupport
    rules_fp["normalized_consequent support"]=normalized_consesupport
    rules_fp["normalized_confidence"]=rules_fp["confidence"]
    name.remove("leverage")
    name.remove("conviction")
    rules_fp.iloc[:,2:]=rules_fp.iloc[:,2:].round(3)
    return test,row,support, name[2:], rules_fp.iloc[:,2:].to_dict('records'), thre_sup, thre_conf




@app.route('/heart', methods=["GET","POST"])
def heart():
    inter1= request.form.get("inter1")
    inter2= request.form.get("inter2")
    thre_sup= request.form.get("thre_sup")
    thre_conf= request.form.get("thre_conf")
    columns,row,support,i_measures, val, sup, conf= heart_as(thre_sup, thre_conf)
    # print("columns:"+str(len(columns))+" row: "+str(len(row)))
    columns= json.dumps(columns)
    
    val= json.dumps(val)
    data = {'columns': columns,'row':row,'support':support, 'interesting_measures': i_measures, 'dat': val, 'min_sup':sup, 'min_conf':conf}
    return render_template("new.html", data=data)





@app.route('/market', methods=["GET","POST"])
def market():
    thre_sup= request.form.get("thre_sup")
    thre_conf= request.form.get("thre_conf")
    columns,row,support,i_measures, val, sup, conf= market_basket(thre_sup, thre_conf)
    columns= json.dumps(columns)
    val= json.dumps(val)
    data = {'columns': columns,'row':row,'support':support, 'interesting_measures': i_measures, 'dat': val, 'min_sup':sup, 'min_conf':conf}
    return render_template("new.html", data=data)


@app.route('/testing', methods=["GET","POST"])
def testing():
    thre_sup= request.form.get("thre_sup")
    thre_conf= request.form.get("thre_conf")    
    columns,row,support,i_measures, val, sup, conf= add(thre_sup, thre_conf)
    columns= json.dumps(columns)
    # print("columns:"+str(len(columns))+" row: "+str(len(row)))
    val= json.dumps(val)
    data = {'columns': columns,'row':row,'support':support, 'interesting_measures': i_measures, 'dat': val, 'min_sup':sup, 'min_conf':conf}
    return render_template("new.html", data=data)



@app.route('/new_table', methods = ['POST'])
def SomeFunction():
    share = request.args.get['share']
    return "Nothing"

from pymongo import MongoClient
@app.route('/receiver', methods = ['POST'])
def receiver():
    jso = request.get_json()
    jso["thre_conf"]=float(conf)
    jso["thre_sup"]=float(sup)
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    mydb = myclient['association']
    mycol = mydb["history_data"]
    if(mycol.find({ "name": jso["name"]}).count()>0 or len(jso["name"])==0):
        return jsonify(False)
    else:
        mycol.insert(jso)
    return jsonify(True)



@app.route('/retrieve/<id>', methods = ['GET'])
def retrieve(id):
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    mydb = myclient['association']
    mycol = mydb["history_data"]
    
    result= mycol.find_one({'name':id}, {"_id":False})
    #print(result)
    #output = {'name' : result['name'],"lef_to_rig": result["lef_to_rig"], "color": result["color"],"lis":result["lis"], "new_graphData":result["new_graphData"], "new_da":result["new_da"], "new_vav": result["new_vav"], "graphData": result["graphData"] , "filters": result["filters"], "graphData":result["graphData"], "da": result["da"], "vav":result["vav"], "thre_sup": result["thre_sup"], "thre_conf": result["thre_conf"]},
    return  jsonify({'result' : result})


@app.route('/delete/<id>', methods = ['GET'])
def delete(id):
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    mydb = myclient['association']
    mycol = mydb["history_data"]
    mycol.delete_one({"name":id})
    return jsonify("done")


@app.route('/all_data', methods = ['GET'])
def all_data():
    myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    mydb = myclient['association']
    mycol = mydb["history_data"]
    total_data= list(mycol.find({}, {"_id":0, "name":1}))
    return jsonify(total_data)



@app.route('/csv_save', methods = ['POST'])
def csv_save():
    jso = request.get_json()
    tra=pd.DataFrame.from_dict(jso["data"])

    tra.to_csv(str(jso["name"])+".csv")
    return "hello"



@app.route('/similarity', methods = ['POST'])
def similarity():
    jso = request.get_json()
    hir_data=[]
    for i in jso["new_graphData"]:
        hir_data.append(i[0]+i[1])
    bb = list()
    for i in hir_data:
        bb.append(str(','.join(str(e) for e in [str(e) for e in i])))
    hello = [x.replace(' ', '') for x in bb]
    hello = [x.replace('.', '') for x in hello]
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(hello)
    X.toarray()
    test=X.toarray()
    prox_matrix = pd.DataFrame(squareform(pdist(test, metric='jaccard')))
    testing=1-prox_matrix
    clustering = AgglomerativeClustering(n_clusters=5, affinity='precomputed',linkage = 'complete').fit(prox_matrix)
    clustering
    ls=clustering.labels_
    df = pd.DataFrame({"data":jso["new_graphData"],"one_measure":jso["new_da"],"measures":jso["new_vav"],"ls":ls})
    df=df.sort_values(by=['ls'])
    # df["conviction"].fillna("Infinity", inplace=True)
    result=df.to_dict('list')
    return jsonify({'result':result})

if __name__ == '__main__':
   app.run(debug = True)
