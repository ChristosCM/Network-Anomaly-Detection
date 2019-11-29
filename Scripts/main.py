import pandas as pd #0.24.2
from pandas.plotting import scatter_matrix
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from category_encoders import *
from loo_encoder.encoder import LeaveOneOutEncoder
import seaborn as sns
import time
import hypertools as hyp
from matplotlib.animation import FuncAnimation
#library to show progress bar in for loops
from tqdm import tqdm

#VARIABLES FOR PANDAS
#pd.set_option('display.max_columns', 500)
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
def applyPCA(data):
    targets = data[['Label','attack_cat']].copy()
    features = data.drop(['Label','attack_cat'],axis=1)
    
    features = StandardScaler().fit_transform(features)


    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(features)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2','pca3'])
    finalDf = pd.concat([principalDf, targets.attack_cat], axis = 1)

    #sns plot 2D
    # plt.figure(figsize=(16,10))
    # sns.scatterplot(
    # x="pca1", y="pca2",
    # hue=finalDf.attack_cat,
    # palette=sns.color_palette("hls", len(targets.attack_cat.unique().tolist())),
    # data=principalDf,
    # legend="full",
    # alpha=0.3)
    # plt.show()

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=principalDf.pca1, 
        ys=principalDf.pca2, 
        zs=principalDf.pca3, 
        c=None, 
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()
    #2-D Version of PCA with pyplot
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    # finalDf = pd.concat([principalDf, targets.attack_cat], axis = 1)
    # fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1) 
    # ax.set_xlabel('Principal Component 1', fontsize = 15)
    # ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_title('2 component PCA', fontsize = 20)
    # targets = data.attack_cat.unique().tolist()
    # colors = ['r', 'b','k','g','y','c','m','#2f968a','#8943b5','#89b543']
    # for target, color in zip(targets,colors):
    #     indicesToKeep = finalDf['attack_cat'] == target
    #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #             , finalDf.loc[indicesToKeep, 'principal component 2']
    #             , c = color
    #             , s = 25)
    # ax.legend(targets)
    # ax.grid()
    # plt.show()
def applyTSNE(data):
    stime = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    features = data.drop(['Label','attack_cat'],axis=1)

    tsne_results = tsne.fit_transform(features)
    target = data.attack_cat
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=target,
        palette=sns.color_palette("hls", len(target.unique().tolist())),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-stime))
    plt.show()
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
def deleteProtos(data):
    df = pd.read_csv("ProtocolsExcluded.csv")
    excluded = df[df.abnormal==0]
    excluded = excluded.prot.tolist()
    data = data.drop(data[(data.proto.isin(excluded))].index)
    data = data.reset_index(drop=True)
    return data
    #write(data)
def excludeProto(data):
    #prots to look at: unas arp ospf sctp icmp any
    new = data[['proto','Label']].copy()
    new = new.astype({'Label': int})
    protoList = data.proto.unique()
    results = pd.DataFrame(
        {
            "prot":protoList,
            "normal":[len(new[(new.proto==proto)&(new.Label==0)]) for proto in protoList ],
            "abnormal":[len(new[(new.proto==proto)&(new.Label==1)]) for proto in protoList ]
            
        }
    )
    results.to_csv("ProtocolsExcluded.csv")

    #old code that is useless
    # protosdf = pd.DataFrame(
    #     {
    #         "proto":["unas", "arp", "ospf", "sctp" ,"icmp","any"],
    #         "normal":[0,0,0,0,0,0],
    #         "abnormal":[0,0,0,0,0,0],
    #         "excluded":[False,False,False,False,False,False]
    #     }
    # )
    # for index,row in tqdm(data.iterrows(),total=data.shape[0]):
    #     if int(row.Label)==0:
    #         #normal
    #         protosdf.loc[protosdf.proto==row.proto,'normal']+=1
    #     else:
    #         #abnormal
    #         protosdf.loc[protosdf.proto==row.proto,'abnormal']+=1
    #         protosdf.loc[protosdf.proto==row.proto,'excluded'] = False
    # protosdf.to_csv("ProtocolsExcluded.csv")
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
    
    stateList = data.state.unique()
    results = pd.DataFrame(
        {
            "state":stateList,
            "normal":[len(new[(new.state==state)&(new.Label==0)]) for state in stateList ],
            "abnormal":[len(new[(new.state==state)&(new.Label==1)]) for state in stateList ]
            
        }
    )
    results.to_csv("StatesExcluded.csv")
def excludeWindow(data):
    new = data[['swin','dwin','Label']].copy()
    new = new.astype({'Label': int})

    swinlist = data.swin.unique()
    dwinlist = data.dwin.unique()

    swinres = pd.DataFrame(
        {
            "swin":swinlist,
            "normal":[len(new[(new.swin==swin)&(new.Label==0)]) for swin in swinlist ],
            "abnormal":[len(new[(new.swin==swin)&(new.Label==1)]) for swin in swinlist]
        }
    )
    dwinres = pd.DataFrame(
        {
            "swin":dwinlist,
            "normal":[len(new[(new.dwin==dwin)&(new.Label==0)]) for dwin in dwinlist ],
            "abnormal":[len(new[(new.dwin==dwin)&(new.Label==1)]) for dwin in dwinlist]
        }
    )
    swinres.to_csv("Outputs/SwinExcluded.csv")
    dwinres.to_csv("Outputs/DwinExcluded.csv")
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
    
def looProto(data):
    
    enc = LeaveOneOutEncoder(cols=['proto'], handle_unknown='impute', sigma=0.00, random_state=0)
    X = pd.DataFrame(data.proto)
    y = data.Label
    
    result = enc.fit_transform(X=X,y=y)

    data = data.drop(['proto'],axis=1)
    data = data.join(result.loo_proto)
    return data
    #write(data)

def looIP(data,ipname):
    ip = data[ipname].str.split(".",expand=True).astype(int)
    names = [ipname+"net1",ipname+"net2",ipname+"host1",ipname+"host2"]
    ip = ip.rename(columns={0:names[0],1:names[1],2:names[2],3:names[3]})
    #dstip = data.dstip.str.split(".",expand=True).astype(int)

    enc = LeaveOneOutEncoder(cols=[ipname+'net1',ipname+"host2"],handle_unknown='impute', sigma=0.0,random_state=0 )
    X = pd.DataFrame([ip[ipname+"net1"],ip[ipname+"host2"]]).transpose()
    y=data.Label

    result = enc.fit_transform(X=X,y=y)
    data.drop([ipname],axis=1, inplace= True)
    new = pd.DataFrame([result["loo_"+ipname+"net1"],result["loo_"+ipname+"host2"]]).transpose()
    data = data.join(new)
    return data
    #write(data)
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
def oneHotServices(data):

    df = pd.get_dummies(data.service,prefix="service")

    data = data.drop(['service'],axis=1 )

    data = data.join(df)
    return data
    #write(data)
def oneHotStates(data):
    df = pd.get_dummies(data.state,prefix="state")

    data = data.drop(['state'],axis=1 )

    data = data.join(df)
    return data
    #write(data)
def removePorts(data):

    dfs = pd.read_csv("SrcPorts.csv")
    dfd = pd.read_csv("DestPorts.csv")
    excludedSrc = dfs[dfs.abnormal==0]
    excludedSrc = excludedSrc.sport.tolist()
    data = data.drop(data[(data.sport.isin(excludedSrc))].index)
    data = data.reset_index(drop=True)

    excludedDst = dfd[dfd.abnormal==0]
    excludedDst = excludedDst.dsport.tolist()
    data = data.drop(data[(data.sport.isin(excludedSrc))].index)
    data = data.reset_index(drop=True)

    return (data)
    #write(data)    
def targetPort(data):
    #enc = LeaveOneOutEncoder(cols=['gender', 'country'], handle_unknown='impute', sigma=0.02, random_state=42)
    #the target is the label 0 or 1 Z|| 0 for normal and 1 for abnormal
    tar = pd.Series(data.Label.astype(int))
    #the test is just the ports
    test = pd.DataFrame([data.sport,data.dsport]).transpose()
    enc = LeaveOneOutEncoder(cols=['sport','dsport'], handle_unknown='impute', sigma=0.00, random_state=0)
    ports = enc.fit_transform(X=test,y=tar)

    new = pd.DataFrame([ports.loo_sport,ports.loo_dsport]).transpose()

    data = data.drop(['sport'],axis=1 )
    data = data.drop(['dsport'],axis=1)
    data = data.join(new)
    return data
def transformIP (ip):
    groups = ip.split(".")
    equalize_group_length = "".join( map( lambda group: group.zfill(3), groups ))
    left_pad_with_zeros = list(( equalize_group_length ).zfill( IPV4_LENGTH ))[-3:]
    return left_pad_with_zeros

def one_hot_ip(data):
    """
    Converts the ipAddress column of pandas DataFrame df, to one-hot
    Also returns the encoder used
    """
    enc = OneHotEncoder()
    #print (data.srcip.apply( lambda ip: transformIP(ip) ))
    ip_df = (data.srcip.apply( lambda ip: transformIP(ip) )).apply( pd.Series ) # creates separate columns for each char in IP
    X_ip = enc.fit_transform( ip_df )
    print (X_ip)

    return X_ip

def onehotIP(data):
    srcip = data.srcip.str.split(".",expand=True).astype(int)
    dstip = data.dstip.str.split(".",expand=True).astype(int)

    df = pd.get_dummies(srcip[3],prefix="sip")
    df2= pd.get_dummies(dstip[3],prefix="dip")

    data = data.drop(['srcip'],axis=1 )
    data = data.drop(['dstip'],axis=1)

    data = data.join(df2)
    data = data.join(df)
    write(data)
    # label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(srcip[3])

    # print (integer_encoded)

    # onehot_encoder = OneHotEncoder()
    # #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print (onehot_encoded)

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
    min_max_scaler = MinMaxScaler()
    counter_scaled = min_max_scaler.fit_transform(counter)
    counter_normalized = pd.DataFrame(counter_scaled)
    return counter_normalized
def plotTime(data):





    #the following is the code for just plotting hour of day
    # temp = pd.to_numeric(data.Stime.value_counts().astype(int))
    # print (temp)
    new = data[['Stime','attack_cat']].copy()
    attackList = data.attack_cat.unique()
    
    dates = pd.to_datetime(data.Stime,unit='s') 
    new.Stime = dates.dt.hour
    
    # pivot_df = new.pivot(index='Stime', columns='attack_cat', values='Stime.value_counts()')
    # print(pivot_df)
    #test5 = pd.crosstab(index=new['Stime'], columns=new['attack_cat'])  
    #new = new.set_index('Stime')
    #new = new.groupby('attack_cat')['Stime'].count()
    #test5.plot(kind='bar', stacked=True)
    #plt.show()
    # dates = pd.to_datetime(data.Stime,unit='s') 
    # data.Stime = dates.dt.minute

    # test =  (data.Stime.value_counts())
    # test = test.sort_index()
    # test.plot(kind='bar')
    # plt.show()


    #test.get_xaxis().set_visible(False)

    #funcion to plot the data by arranging the times based on the index (which is the time) 
    # temp = temp.sort_index()
    # ax = plt.axes()
    # ax.plot(temp) 
    # ax.get_xaxis().set_visible(False)
    # plt.show()   
def convertToMin(data,time):
    dates = pd.to_datetime(data[time],unit='s') 
    data[time] = dates.dt.minute
    
    enc = LeaveOneOutEncoder(cols=[time], handle_unknown='impute', sigma=0.00, random_state=0)
    X = pd.DataFrame(data[time])
    y = data.Label
    
    result = enc.fit_transform(X=X,y=y)

    data = data.drop([time],axis=1)
    data = data.join(result["loo_"+time])
    return data
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
        data.to_csv("all_clean_data.csv",index=False)
    else:
        print ("Only {} were selected, cancelling operation...".format(ROWS))
def allNormalize(data):

    #the counters srv_dst and ftp_cmd have already been normalized and changed in the dataset
    data.ct_state_ttl = normalizeCounters(data,"ct_state_ttl").values
    data.ct_flw_http_mthd = normalizeCounters(data,"ct_flw_http_mthd").values

    data.ct_srv_src = normalizeCounters(data,"ct_srv_src").values

    data.ct_dst_ltm = normalizeCounters(data,"ct_dst_ltm").values

    #this one was wrong might need to redo
    data['ct_src_ ltm'] = normalizeCounters(data,"ct_src_ ltm").values

    data.ct_src_dport_ltm = normalizeCounters(data,"ct_src_dport_ltm").values

    data.ct_dst_sport_ltm = normalizeCounters(data,"ct_dst_sport_ltm").values

    data.ct_dst_src_ltm = normalizeCounters(data,"ct_dst_src_ltm").values
    return data
    #write(data)
def checkTime(data):
    new = data[['Stime','Label']]
    dates = pd.to_datetime(data.Stime,unit='s') 
    new.Stime = dates.dt.hour
    final = new.drop(new[(new.Label==0)].index)
    print (final.Stime.value_counts())
# def convertTime(data):
#     new = data[['Stime','attack_cat']].copy()
#     dates = pd.to_datetime(data.Stime,unit='s') 
#     new.Stime = dates.dt.hour
def conversion(data):

    data = data.drop(data[(data.state!="REQ")&~(data.state=="RST")&~(data.state=="FIN")&~(data.state=="CON")&~(data.state=="INT")].index)

    data = removePorts(data)
    data = deleteProtos(data)
    #LeaveOneOut Encoding of the IP
    data = looIP(data,"srcip")
    data = looIP(data,"dstip")

    data = oneHotServices(data)
    data = oneHotStates(data)
    #LeaveOneOut encode of the protocol
    data = looProto(data)

    #encode the ports with LOO
    data = targetPort(data)
    #normalize the counters
    #already done
    #data = allNormalize(data)
    
    #convert the timestamps of Stime and Ltime into Minutes of hour
    data = convertToMin(data,"Stime")
    data = convertToMin(data,"Ltime")

    #remove added indices columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    print (data)
    #write(data)

def main():
    #read the features and labels and assign them 
    #labels =  read_features().Name

    #read the data 0 to read all of it or use a custom number to read the first n rows of the data
    data = read_clean_data(clean,ROWS)
    
    excludeWindow(data)

    #conversion(data)
    #data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    #print (data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).value_counts())
    #looProto(data)

    #allNormalize(data)
    # df = oneHotProtocol(data)
    # data = data.join(df)
    # print (df.shape[1])
    # write(data)

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
