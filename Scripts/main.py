import pandas as pd #0.24.2
from pandas.plotting import scatter_matrix
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn import preprocessing
from category_encoders import *
from loo_encoder.encoder import LeaveOneOutEncoder
import seaborn as sns
import time
import hypertools as hyp
from matplotlib.animation import FuncAnimation
#library to show progress bar in for loops
from tqdm import tqdm

#library to make a 3D plot
# from mpl_toolkits.mplot3d import Axes3D
def read_features():
    input_file = "featuresCat.csv"
    featuresTable = pd.read_csv(input_file,encoding = "ISO-8859-1",names = ['No.','Name', 'Type', 'Description'])
    #print (featuresTable)
    #values = pd.DataFrame(featuresTable.Type.value_counts())
    # count_nominal = 0;
    # for feature in featuresTable.Type:
    #     if feature == "nominal":
    #         count_nominal += 1
    # print ("The number of nominal features is: {}".format(count_nominal))


    # _ = plt.bar(values, height = 10)
    # plt.title("Histogram of Feature Type")
    # plt.show()
    return featuresTable
def read_clean_data(data,rows):
    feat = read_features()
    stime = time.time()
    #may need to add chunksize here to make this more efficiet when testing
    if rows == 0:
        cleanData = pd.read_csv(data)#names = feat.Name)
        print("It took {} to read the data".format((time.time()-stime)))

    else:
        cleanData = pd.read_csv(data, nrows = rows)#, names = feat.Name)
        print("It took {} to read {} rows from the data".format((time.time()-stime),rows))

    #cleanData =  (cleanData.reset_index().drop([0],axis=0))
    return cleanData
    #ip = pd.DataFrame(cleanData.srcip.value_counts())

    # normal = 0
    # abnormal = 0
    # for point in cleanData.attack_cat: 
    #     if point == "Normal":
    #         normal +=1
    #     else:
    #         abnormal += 1
    # clean = "imbalanced/clean_data_2.csv"
    # cleanData = pd.read_csv(clean, names = feat.Name)
   
    # print ("Normal:{}, Abnormal:{}, Normal Percentage: {}".format(normal,abnormal, (normal/(normal+abnormal))*100))

def applyTSNE(data):
    stime = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-stime))

#function to combine the 3 datasets into 1 main dataset
def combine_datasets(data1, data2, data3):
    #read the files
    d1 = pd.read_csv(data1)
    d2 = pd.read_csv(data2)
    d3 = pd.read_csv(data3)
    #drop the description from the pd 
    d1 = d1.drop([0],axis=0)
    d2 = d2.drop([0],axis=0)
    d3 = d3.drop([0],axis=0)
    #combine the 3 sets into one
    dataset = pd.concat([d1,d2,d3])
    dataset.to_csv("imbalanced/all_clean_data.csv")
#combine_datasets("imbalanced/clean_data_1.csv","imbalanced/clean_data_2.csv","imbalanced/clean_data_3.csv")
#read_clean_data("imbalanced/clean_data_1.csv")
# data = read_clean_data("imbalanced/clean_data_1.csv")
# applyTSNE(data)

#function to get and plot the number of attacks in each different attack category
def services(data):

    general = data.shape[0] 
    main = 0
    otherper = 0
    second = 0
    for i in data.service:
        if i=="other":
            otherper += 1
        if i=="other" or i == "dns" or i=="http" or i=="ftp-data"or i=="smtp"or i=="ftp"or i=="ssh":
            main += 1
        if i=="dns" or i=="other":
            second +=1
    print ("The percentage of the other class over all the data is {}%".format((otherper/general)*100))
    print ("The percentage of the 7 main classes over all the data is {}%".format((main/general)*100))
    print ("The percentage of the 2 classes over all the data is {}%".format((second/general)*100))

    #Following is the code to build the bar plot
    # serveLabel = []
    # serveCount = []
    # endrange = ((pd.DataFrame(data.service.value_counts())).shape[0])+1
    # generalCount= data.service.value_counts()
    # for i in range (1,endrange):
    #     serveLabel.append((str(generalCount[i-1:i]).split())[0])
    #     serveCount.append((str(generalCount[i-1:i]).split()[1]))

    # plt.bar(serveLabel,[int(x) for x in serveCount], label = 'Data Point Count')
    # plt.ylabel('Service Count')
    # plt.xlabel('Type of Service')
    # plt.title('Services Bar Plot')
    # plt.show()
def excludeProto(data):
    #prots to look at: unas arp ospf sctp icmp any
    protosdf = pd.DataFrame(
        {
            "proto":["unas", "arp", "ospf", "sctp" ,"icmp","any"],
            "normal":[0,0,0,0,0,0],
            "abnormal":[0,0,0,0,0,0],
            "excluded":[False,False,False,False,False,False]
        }
    )
    for index,row in tqdm(data.iterrows(),total=data.shape[0]):
        if int(row.Label)==0:
            #normal
            protosdf.loc[protosdf.proto==row.proto,'normal']+=1
        else:
            #abnormal
            protosdf.loc[protosdf.proto==row.proto,'abnormal']+=1
            protosdf.loc[protosdf.proto==row.proto,'excluded'] = False
    protosdf.to_csv("ProtocolsExcluded.csv")
def proto(data):

    general = data.shape[0] 
    main = 0
    for i in data.proto:
        if i=="tcp" or i == "udp":
            main += 1
    print ("The percentage of the 2 main classes over all the data is {}%".format((main/general)*100))

    #Following is code for bar plot
    # label = []
    # count = []
    # endrange = ((pd.DataFrame(data.proto.value_counts())).shape[0])+1
    # generalCount= data.proto.value_counts()
    # for i in range (1,endrange):
    #     label.append((str(generalCount[i-1:i]).split())[0])
    #     count.append((str(generalCount[i-1:i]).split()[1]))

    # plt.bar(label,[int(x) for x in count], label = 'Data Point Count')
    # plt.ylabel('Protocol Count')
    # plt.xlabel('Type of Protocol')
    # plt.title('Protocols Bar Plot')
    # plt.show()
def excludeService(data):
    new = data[['service','Label']].copy()
    new = new.astype({'Label': int})
    results = pd.DataFrame(
        {
            "service":["pop3","dhcp","ssl","snmp","radius","irc"],
            "normal":[len(new[(new.service=="http")&(new.Label==0)]),len(new[(new.service=="dhcp")&(new.Label==0)]),len(new[(new.service=="ssl")&(new.Label==0)]),len(new[(new.service=="snmp")&(new.Label==0)]),len(new[(new.service=="radius")&(new.Label==0)]),len(new[(new.service=="irc")&(new.Label==0)])],
            "abnormal":[len(new[(new.service=="http")&(new.Label==1)]),len(new[(new.service=="dhcp")&(new.Label==1)]),len(new[(new.service=="ssl")&(new.Label==1)]),len(new[(new.service=="snmp")&(new.Label==1)]),len(new[(new.service=="radius")&(new.Label==1)]),len(new[(new.service=="irc")&(new.Label==1)])]
            
        }
    )
    results.to_csv("ServicesExcluded.csv")
def excludeState(data):
    new = data[['state','Label']].copy()
    new = new.astype({'Label': int})
    print((len(new[new.state=="URH"])))
    return None
    stateList = data.state.unique()
    results = pd.DataFrame(
        {
            "state":stateList,
            "normal":[len(new[(new.state==state)&(new.Label==0)]) for state in stateList ],
            "abnormal":[len(new[(new.state==state)&(new.Label==1)]) for state in stateList ]
            
        }
    )
    results.to_csv("StatesExcluded.csv")
def states(data):
    
    req = [0,0, "REQ"]
    rst = [0,0, "RST"]
    eco = [0,0, "ECO"]
    clo = [0,0, "CLO"]
    urh = [0,0, "URH"]
    acc = [0,0, "ACC"]
    par = [0,0, "PAR"]
    tst = [0,0, "TST"]
    ecr = [0,0, "ECR"]
    no = [0,0, "no"]
    urn = [0,0, "URN"]
    mas = [0,0, "MAS"]
    txd = [0,0,"TXD"]
    for _, row in data.iterrows():
        if row.state=="REQ":
            if row.Label==0:
                req[0] += 1
            else:
                req[1] += 1
        if row.state=="RST":
            if row.Label==0:
                rst[0] += 1
            else:
                rst[1] += 1
        if row.state=="ECO":
            if row.Label==0:
                eco[0] += 1
            else:
                eco[1] += 1
        if row.state=="CLO":
            if row.Label==0:
                clo[0] += 1
            else:
                clo[1] += 1
        if row.state=="URH":
            if row.Label==0:
                urh[0] += 1
            else:
                urh[1] += 1
        if row.state=="ACC":
            if row.Label==0:
                acc[0] += 1
            else:
                acc[1] += 1
        if row.state=="PAR":
            if row.Label==0:
                par[0] += 1
            else:
                par[1] += 1
        if row.state=="TST":
            if row.Label==0:
                tst[0] += 1
            else:
                tst[1] += 1
        if row.state=="ECR":
            if row.Label==0:
                ecr[0] += 1
            else:
                ecr[1] += 1
        if row.state=="no":
            if row.Label==0:
                no[0] += 1
            else:
                no[1] += 1
        if row.state=="URN":
            if row.Label==0:
                urn[0] += 1
            else:
                urn[1] += 1
        if row.state=="MAS":
            if row.Label==0:
                mas[0] += 1
            else:
                mas[1] += 1
        if row.state=="TXD":
            if row.Label==0:
                txd[0] += 1
            else:
                txd[1] += 1
    final = [req,rst,eco,clo,urh,acc,par,tst,ecr,no,urn,mas,txd]   
    pd.DataFrame(
        {
            "state":[]
        }
    )
    for i in range (len(final)):
        if final[i][1]==0:
            final[i].append("Can be excluded")
        else:
            final[i].append("Cannot be excluded")
    with open("states.txt","w+") as f: 
        f.write("Normal|Abnormal|State\n")
        for state in final:
            state[0] = str(state[0])
            state[1] = str(state[1])
            f.write("|".join(state)+"\n")

    # general = data.shape[0] 
    # main = 0
    # for i in data.state:
    #     if i=="FIN" or i == "CON" or i == "INT":
    #         main += 1
    # print ("The percentage of the 3 main classes over all the data is {}%".format((main/general)*100))



    #Code to produce the bar plot for the states
    
    # label = []
    # count = []
    # endrange = ((pd.DataFrame(data.state.value_counts())).shape[0])+1
    # generalCount= data.state.value_counts()
    # for i in range (1,endrange):
    #     label.append((str(generalCount[i-1:i]).split())[0])
    #     count.append((str(generalCount[i-1:i]).split()[1]))


    # plt.bar(label,[int(x) for x in count], label = 'Data Point Count')
    # plt.ylabel('Count')
    # plt.xlabel('Type of State')
    # plt.title('States Bar Plot')
    # plt.show()


def attackCatPlot(data):
    #row_number = data.shape[0]
    #this step is already done so we can use the csv file to plot 
    # new = data[['attack_cat','Label']].copy()
    # new = new.astype({'Label': int})
    # catList = data.attack_cat.unique()
    # catList = np.delete(catList,0)
    # results = pd.DataFrame(
    #     {
    #         "attack_cat":catList,
    #         "normal":[len(new[(new.attack_cat==attack_cat)&(new.Label==0)]) for attack_cat in catList ],
    #         "abnormal":[len(new[(new.attack_cat==attack_cat)&(new.Label==1)]) for attack_cat in catList ]
            
    #     }
    # )
    # results.to_csv("AttackCat_Counter.csv")


    # catIndices = []
    # catCount = []
    # endRange = ((pd.DataFrame(data.attack_cat.value_counts())).shape[0])+1
    # generalCount= data.attack_cat.value_counts()
    # for i in range (1,endRange):
    #     catIndices.append((str(generalCount[i-1:i]).split())[0])
    #     catCount.append((str(generalCount[i-1:i]).split()[1]))

    attack = pd.read_csv("AttackCat_Counter.csv")
    labels = attack.attack_cat.tolist()
    numbers = attack.abnormal.tolist()
    total = sum(numbers)
    
    
def excludePort(data):

    sports = data.sport.value_counts()
    dsports = data.dsport.value_counts()

    
    excSport = pd.DataFrame(
        {
            "sport":[],
            "normal":[],
            "abnormal":[],
            "excluded":[]
        }
    )
    
    excDsport = pd.DataFrame(
        {
            "dsport":[],
            "normal":[],
            "abnormal":[],
            "excluded":[]
        }
    )
    for index,row in tqdm(sports.iteritems(),total=sports.size):
        df = pd.DataFrame(
                {
                    "sport":[index],
                    "normal":[0],
                    "abnormal":[0],
                    "excluded":[True]
                }
            )
        excSport = excSport.append(df)
    excSport = excSport.reset_index(drop=True)

    for index,row in tqdm(dsports.iteritems(),total=dsports.size):
        df = pd.DataFrame(
                {
                    "dsport":[index],
                    "normal":[0],
                    "abnormal":[0],
                    "excluded":[True]
                }
            )
        excDsport = excDsport.append(df)
    excDsport = excDsport.reset_index(drop=True)

    for index,row in tqdm(data.iterrows(),total=data.shape[0]):
        if int(row.Label)==0:
            #normal
            excSport.loc[excSport.sport==row.sport,'normal']+=1
        else:
            #abnormal
            excSport.loc[excSport.sport==row.sport,'abnormal']+=1
            excSport.loc[excSport.sport==row.sport,'excluded'] = False

    for index,row in tqdm(data.iterrows(),total=data.shape[0]):
        if int(row.Label)==0:
            #normal
            excDsport.loc[excDsport.dsport==row.dsport,'normal']+=1
        else:
            #abnormal
            excDsport.loc[excDsport.dsport==row.dsport,'abnormal']+=1
            excDsport.loc[excDsport.dsport==row.dsport,'excluded'] = False


  
    
    excDsport.to_csv("DestPorts.csv")
    excSport.to_csv("SrcPorts.csv")
    #The pie chart plot for this heavily imbalanced plot does not work


    # colors = ['#003f5c',"#2f4b7c","#665191","#a05195","#bc5090","#d45087","#f95d6a","#ff7c43","#ff6361","#ffa600"]
    # plt.pie(catCount,labels=catIndices,colors=colors,autopct='%1.1f%%',shadow= True,startangle = 140)
    # plt.axis('equal')
    # plt.show()


    #Code for the bar plot

    # plt.bar(catIndices,[int(x) for x in catCount], label = 'Data Point Count')
    # plt.ylabel('Dataset Count')
    # plt.xlabel('Type of Attack')
    # plt.title('Attack Category Bar Plot')
    # plt.show()
    #anablyse the behaviour
# def targetEncodePort(data):
#     tar = pd.Series(data.Label.astype(int))
#     test = pd.DataFrame(data.sport.astype(int))
#     enc = category_encoders.target_encoder.TargetEncoder(verbose = 0, cols=['sport'])

def targetPort(data):
    #enc = LeaveOneOutEncoder(cols=['gender', 'country'], handle_unknown='impute', sigma=0.02, random_state=42)
    #the target is the label 0 or 1 Z|| 0 for normal and 1 for abnormal
    tar = pd.Series(data.Label.astype(int))
    #the test is just the ports
    test = pd.DataFrame([data.sport,data.dsport]).transpose()
    enc = LeaveOneOutEncoder(cols=['sport','dsport'], handle_unknown='impute', sigma=0.00, random_state=0)
    ports = enc.fit_transform(X=test,y=tar)
    return pd.DataFrame([ports.loo_sport,ports.loo_dsport]).transpose()
def transformIP (ip):
    groups = ip.split(".")
    equalize_group_length = "".join( map( lambda group: group.zfill(3), groups ))
    left_pad_with_zeros = list(( equalize_group_length ).zfill( IPV4_LENGTH ))[-3:]
    return left_pad_with_zeros
def oneHotProtocol(data):
    enc = OneHotEncoder()
    encoded = enc.fit_transform(data.proto)
    print (encoded)
    return encoded
def one_hot_ip(data):
    """
    Converts the ipAddress column of pandas DataFrame df, to one-hot
    Also returns the encoder used
    """
    enc = OneHotEncoder()
    #print (data.srcip.apply( lambda ip: transformIP(ip) ))
    ip_df = (data.srcip.apply( lambda ip: transformIP(ip) )).apply( pd.Series ) # creates separate columns for each char in IP
    X_ip = enc.fit_transform( ip_df )
    print (X_ip.shape[1])

    return X_ip
def analyzeIP(data):
    srcip = data.srcip.str.split(".", expand=True).astype(int)
    #dstip = data.dstip.str.split(".",expand = True)
    srcip["cat"] = data.attack_cat
    #dstip["cat"] = data.attack_cat 

    #3D plot not completed
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(srcip[0], [0,1,2,3,4,5,6,7,8], srcip.index.values, c='skyblue', s=60)
    # ax.view_init(30, 185)

    #make the count plot 
    #ax = sns.countplot(y="cat",hue=data.srcip, data=srcip)

    #scatter plot of IPs
    #need to make the hue better
    # ax = sns.catplot(x="cat",y=0, kind = "swarm", data = srcip)
    
    #doesnt work, initial countplot will have to do
    # fig, ax = plt.subplots()
    # ax.plot(srcip.cat, srcip[0].tolist(), label="P1")
    # ax.plot(srcip.cat, srcip[1].tolist(), label="P2")
    # ax.plot(srcip.cat, srcip[2].tolist(), label="P3")
    # ax.plot(srcip.cat, srcip[3].tolist(), label="P4")
    # ax.legend()    

    plt.show()
def encodeIP(data):
    for i, rows in data.iterrows(): 
        data.at[i,'srcip'] = transformIP( data.at[i,'srcip'])   
    return data
#function to normalize the counter data to 0-1
def normalizeCounters(data,cnt):
    
    data = data.astype({cnt: int})

    counter = data[[cnt]].values.astype(int) 
    min_max_scaler = preprocessing.MinMaxScaler()
    counter_scaled = min_max_scaler.fit_transform(counter)
    counter_normalized = pd.DataFrame(counter_scaled)
    return counter_normalized
def plotTime(data):
    temp = pd.to_numeric(data.Stime.value_counts().astype(int))
    print (temp)

    dates = pd.to_datetime(data.Stime,unit='s') 
    data.Stime = dates.dt.minute
    data = data.astype({'Label': int})
    test =  (data.Stime.value_counts())
    test = test.sort_index()
    test.plot(kind='bar')
    plt.show()

    #test.get_xaxis().set_visible(False)

    #funcion to plot the data by arranging the times based on the index (which is the time) 
    # temp = temp.sort_index()
    # ax = plt.axes()
    # ax.plot(temp) 
    # ax.get_xaxis().set_visible(False)
    # plt.show()   

def calcExcluded():

    sports = pd.read_csv("SrcPorts.csv")
    dsports = pd.read_csv("DestPorts.csv")
    
    excsp = sports.excluded.value_counts()
    excdsp = dsports.excluded.value_counts()
    
    with open("excports.txt",'w') as file:
        file.write("The number of source ports that can be excluded are {} of the total {} \n".format(excsp['False'],sports.shape[0]))
        file.write("The number of destination ports that can be excluded are {} of the total {}\n".format(excdsp['False'], dsports.shape[0]))
    
#write the time data to preview
 
def exampleTimeData(data):
    #data = data.head()
    with open("timeinfo.txt","w") as file:
        file.write("Example of sttl (and dsttl) {} \n".format(data.sttl))    
        file.write("Example of Stime (and Ltime) {} \n".format(data.Stime))    
        file.write("Example of Sintpkt (and Dintpkt) {} \n".format(data.Sintpkt))
        file.write("Example of tcprtt  {} \n".format(data.tcprtt))    
        file.write("Example of synack {} \n".format(data.synack))    
        file.write("Example of ackdat  {} \n".format(data.ackdat))

def write(data):
    if ROWS == 0:
        data.to_csv("all_clean_data.csv")
    else:
        print ("Only {} were selected, cancelling operation...".format(ROWS))
def allNormalize(data):

    #the counters srv_dst and ftp_cmd have already been normalized and changed in the dataset
    data.ct_state_ttl = normalizeCounters(data,"ct_state_ttl").values
    data.ct_flw_http_mthd = normalizeCounters(data,"ct_flw_http_mthd").values

    data.ct_srv_src = normalizeCounters(data,"ct_srv_src").values

    data.ct_dst_ltm = normalizeCounters(data,"ct_dst_ltm").values

    data.ct_src_ltm = normalizeCounters(data,"ct_src_ ltm").values

    data.ct_src_dport_ltm = normalizeCounters(data,"ct_src_dport_ltm").values

    data.ct_dst_sport_ltm = normalizeCounters(data,"ct_dst_sport_ltm").values

    data.ct_dst_src_ltm = normalizeCounters(data,"ct_dst_src_ltm").values

def main():
    #read the features and labels and assign them 
    #labels =  read_features().Name

    #read the data 0 to read all of it or use a custom number to read the first n rows of the data
    data = read_clean_data(clean,ROWS)

    df = oneHotProtocol(data)
    data = data.join(df)
    print (df.shape[1])
    write(data)

    #remove the added list of indices if we want (the first column that tags Unnamed)
    #data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


    #data = data.astype({'Label': int})
    #data = data.drop(data[(data.state!="REQ")&~(data.state=="RST")&~(data.state=="FIN")&~(data.state=="CON")&~(data.state=="INT")].index)

    #print(len(data[(data.Stime==4)&(data.Label==1)]))   
    #exampleTimeData(data)
    #call function with which one you want to normalize, doesn't change the data yet
    # new = normalizeCounters(data,'ct_srv_src')
    # print (new)

    #very slow for the whole data
    #data = data.astype({'ct_srv_src': int})
    # data.ct_srv_src.plot(kind='bar')
    # data.get_xaxis().set_visible(False)
    # plt.show()



    #states(data)
    #plotTime(data)
    #applyTSNE(data)
    #services(data)
    #(targetPort(data))
    #newdata = one_hot_ip(data)
    # print (data.sloss.value_counts())
    # print (data.dloss.value_counts())
    #proto(data)
    #states(data)
    #attackCatPlot(data)
    #df = pd.DataFrame({'Attack Category': data.attack_cat},index = (data.attack_cat.value_counts()[0]))
    
if __name__ == '__main__':
    clean = "all_clean_data.csv"
    IPV4_LENGTH =12
    ROWS = 0
    main()
