"""
Projeto: Gerenciamento inteligente de manutenção com NASA C-MAPSS (dataset nº 6)
Pipeline: dados -> RUL -> normalização por condição -> seleção de sensores -> PCA/HI
          -> estados -> transições empíricas -> MDP -> Value Iteration -> Q-Learning.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COLS=["unit","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
SENS=[f"s{i}" for i in range(1,22)]
ACTIONS=["operate","reduced_load","maintenance"]
STATE_NAMES=["healthy","moderate","severe","critical","failure"]

def load_fd(folder, k):
    folder=Path(folder)
    train=pd.read_csv(folder/f"train_FD00{k}.txt",sep=r"\s+",header=None,names=COLS)
    # aceita tanto test_FD001.txt quanto test_FD001(1).txt
    p=folder/f"test_FD00{k}.txt"
    if not p.exists(): p=folder/f"test_FD00{k}(1).txt"
    test=pd.read_csv(p,sep=r"\s+",header=None,names=COLS)
    rul=np.loadtxt(folder/f"RUL_FD00{k}.txt")
    return train,test,rul

def prepare_train(df,n_conditions):
    d=df.copy()
    d["RUL"]=d.groupby("unit")["cycle"].transform("max")-d["cycle"]
    if n_conditions==1:
        d["condition"]=0
    else:
        ops=StandardScaler().fit_transform(d[["op1","op2","op3"]])
        d["condition"]=KMeans(n_clusters=n_conditions,random_state=42,n_init=10).fit_predict(ops)

    Z=pd.DataFrame(index=d.index)
    for s in SENS:
        z=np.zeros(len(d))
        for _,idx in d.groupby("condition").groups.items():
            v=d.loc[idx,s]
            sd=v.std()
            z[idx]=0.0 if sd < 1e-8 else (v-v.mean())/sd
        Z[s]=z

    corr={s:spearmanr(Z[s],d["RUL"]).statistic for s in SENS if Z[s].std()>1e-8}
    top=sorted(corr,key=lambda s:abs(corr[s]),reverse=True)[:10]
    pca=PCA(n_components=1).fit(Z[top])
    pc=pca.transform(Z[top]).ravel()
    if spearmanr(pc,d["RUL"]).statistic<0: pc=-pc
    lo,hi=np.percentile(pc,[1,99])
    d["HI_raw"]=np.clip((pc-lo)/(hi-lo),0,1)
    d["HI"]=d.groupby("unit")["HI_raw"].transform(lambda x:x.rolling(5,min_periods=1).mean())
    # 0 saudável, 1 moderado, 2 severo, 3 crítico
    d["state"]=pd.cut(d["HI"],[-np.inf,.25,.50,.75,np.inf],
                      labels=[3,2,1,0]).astype(int)
    return d,top,corr,pca.explained_variance_ratio_[0]

def transition_matrix(d):
    C=np.zeros((5,5),dtype=int)
    for _,g in d.groupby("unit"):
        st=g.sort_values("cycle")["state"].to_numpy()
        for a,b in zip(st[:-1],st[1:]): C[a,b]+=1
        C[st[-1],4]+=1
    P=np.zeros((5,5),float)
    for s in range(4):
        if C[s].sum(): P[s]=C[s]/C[s].sum()
    P[4,4]=1
    # Dano latente é progressivo; reversões do HI são tratadas como ruído.
    for s in range(4):
        P[s,s]+=P[s,:s].sum()
        P[s,:s]=0
    return C,P

def build_mdp(Pnat,degrade_factor=.5,r_oper=10,r_reduced=6,c_maint=25,c_fail=100):
    P=np.zeros((5,3,5)); R=np.zeros((5,3,5))
    # Falha: reparo corretivo e retorno ao saudável.
    for a in range(3):
        P[4,a,0]=1; R[4,a,0]=-c_fail
    for s in range(4):
        P[s,0]=Pnat[s]
        R[s,0,:]=r_oper; R[s,0,4]=r_oper-c_fail

        row=Pnat[s].copy()
        worse=list(range(s+1,5))
        moved=row[worse].sum()*(1-degrade_factor)
        row[worse]*=degrade_factor
        row[s]+=moved
        P[s,1]=row
        R[s,1,:]=r_reduced; R[s,1,4]=r_reduced-c_fail

        P[s,2,0]=1
        R[s,2,0]=-c_maint
    return P,R

def value_iteration(P,R,gamma=.95,tol=1e-10):
    V=np.zeros(5)
    for _ in range(10000):
        Q=(P*(R+gamma*V[None,None,:])).sum(axis=2)
        V2=Q.max(axis=1)
        if np.max(np.abs(V2-V))<tol: break
        V=V2
    Q=(P*(R+gamma*V[None,None,:])).sum(axis=2)
    return V,Q,Q.argmax(axis=1)

def sample_next(row,rng):
    x=rng.random(); acc=0.0
    for i,p in enumerate(row):
        acc+=p
        if x<=acc: return i
    return len(row)-1

def q_learning(P,R,gamma=.95,episodes=30000,steps=60,seed=42):
    rng=random.Random(seed)
    Q=[[0.0]*3 for _ in range(5)]
    N=[[0]*3 for _ in range(5)]
    eps=1.0
    for _ in range(episodes):
        s=rng.randrange(5)  # exploring starts
        for _ in range(steps):
            a=rng.randrange(3) if rng.random()<eps else max(range(3),key=lambda j:Q[s][j])
            sp=sample_next(P[s,a],rng)
            r=float(R[s,a,sp])
            N[s][a]+=1
            alpha=1/(N[s][a]**0.6)
            Q[s][a]+=alpha*(r+gamma*max(Q[sp])-Q[s][a])
            s=sp
        eps=max(.02,eps*.9997)
    Q=np.asarray(Q)
    return Q,Q.argmax(axis=1)

def policy_value(P,R,policy,gamma=.95):
    Ppi=np.array([P[s,policy[s]] for s in range(5)])
    rpi=np.array([(P[s,policy[s]]*R[s,policy[s]]).sum() for s in range(5)])
    return np.linalg.solve(np.eye(5)-gamma*Ppi,rpi)

def run(folder="."):
    rows=[]
    for k in range(1,5):
        train,test,rul=load_fd(folder,k)
        ncond=1 if k in (1,3) else 6
        d,top,corr,pca_var=prepare_train(train,ncond)
        C,Pnat=transition_matrix(d)
        P,R=build_mdp(Pnat)
        V,Qvi,pvi=value_iteration(P,R)
        Qql,pql=q_learning(P,R)
        baseline=np.array([0,0,2,2,0])
        print(f"\nFD00{k}")
        print("Top sensores:",top)
        print("Spearman HI-RUL:",d["HI"].corr(d["RUL"],method="spearman"))
        print("VI:",[ACTIONS[a] for a in pvi[:4]])
        print("QL:",[ACTIONS[a] for a in pql[:4]])
        print("Retorno baseline:",policy_value(P,R,baseline)[0])
        print("Retorno VI:",policy_value(P,R,pvi)[0])
        pd.DataFrame(Pnat,index=STATE_NAMES,columns=STATE_NAMES).to_csv(
            Path(folder)/f"transitions_FD00{k}.csv")
        rows.append([f"FD00{k}",len(train),train.unit.nunique(),
                     d["HI"].corr(d["RUL"],method="spearman"),pca_var,
                     ",".join(top),policy_value(P,R,baseline)[0],
                     policy_value(P,R,pvi)[0],
                     "/".join(ACTIONS[a] for a in pvi[:4]),
                     "/".join(ACTIONS[a] for a in pql[:4])])
    pd.DataFrame(rows,columns=["dataset","rows","units","HI_RUL","PC1_var","top_sensors",
                               "baseline_return","VI_return","VI_policy","QL_policy"]).to_csv(
                                   Path(folder)/"project_results.csv",index=False)

if __name__=="__main__":
    run(".")
