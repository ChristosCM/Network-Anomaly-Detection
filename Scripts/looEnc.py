from loo_encoder.encoder import LeaveOneOutEncoder
import pandas as pd
import numpy as np


enc = LeaveOneOutEncoder(cols=['ip'], handle_unknown='impute', sigma=0.00, random_state=0)

X = pd.DataFrame(
    {
        "ip": ["3361", "1464", "3593","0","0", "49664"],
    }
)

y = pd.Series([0, 0, 0, 0,1,1] ,name="orders")

df_train = enc.fit_transform(X=X, y=y)

# X_val = pd.DataFrame(
#     {
#         "gender": ["unknown", "male", "female", "male"],
#         "country": ["Germany", "USA", "Germany", "Japan"]
#     }
# )

# df_test = enc.transform(X=X_val)
print (df_train.loo_ip)


#deleted from all data.csv
#,srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sloss,dloss,service,Sload,Dload,Spkts,Dpkts,swin,dwin,stcpb,dtcpb,smeansz,dmeansz,trans_depth,res_bdy_len,Sjit,Djit,Stime,Ltime,Sintpkt,Dintpkt,tcprtt,synack,ackdat,is_sm_ips_ports,ct_state_ttl,ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,ct_srv_src,ct_srv_dst,ct_dst_ltm,ct_src_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,attack_cat,Label
