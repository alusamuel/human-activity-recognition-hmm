"""Human activity recognition from Sensor Logger CSV files.
Only independent, separate sessions qualify as unseen evaluation recordings.
Gaussian diagonal emissions with log-space forward/backward and Viterbi.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ACTIVITIES=['standing','walking','jumping','still']
FS=50.0; WINDOW_SEC=1.0; HOP_SEC=0.5
FEATURES=['acc_std','acc_rms_dynamic','acc_sma_dynamic','gyro_mean','gyro_std','acc_dominant_hz','acc_spectral_energy','gyro_spectral_energy']

def load_session(path,fs=FS):
    path=Path(path); frames=[]
    for sensor,prefix in [('Accelerometer','a'),('Gyroscope','g')]:
        df=pd.read_csv(path/f'{sensor}.csv')
        needed=['seconds_elapsed','x','y','z']
        if not set(needed).issubset(df.columns):raise ValueError(f'Missing fields in {path}/{sensor}.csv')
        df=df[needed].apply(pd.to_numeric,errors='coerce').dropna().sort_values('seconds_elapsed').drop_duplicates('seconds_elapsed')
        df=df.rename(columns={axis:f'{prefix}{axis}' for axis in 'xyz'});frames.append(df)
    lo=max(df.seconds_elapsed.min() for df in frames);hi=min(df.seconds_elapsed.max() for df in frames)
    if hi-lo<WINDOW_SEC:raise ValueError(f'Too short: {path}')
    grid=lo+np.arange(int((hi-lo)*fs)+1)/fs
    data={'t':grid-lo}
    for df in frames:
        for axis in 'xyz':
            col= ('a' if 'a'+axis in df.columns else 'g')+axis
            data[col]=np.interp(grid,df.seconds_elapsed,df[col])
    return pd.DataFrame(data)

def extract_features(df,fs=FS):
    n=int(round(fs*WINDOW_SEC));h=int(round(fs*HOP_SEC));rows=[]
    for start in range(0,len(df)-n+1,h):
        w=df.iloc[start:start+n]; a=w[['ax','ay','az']].to_numpy();g=w[['gx','gy','gz']].to_numpy()
        amag=np.linalg.norm(a,axis=1);gmag=np.linalg.norm(g,axis=1)
        dynamic=amag-amag.mean(); freqs=np.fft.rfftfreq(n,1/fs)
        apower=np.abs(np.fft.rfft(dynamic*np.hanning(n)))**2/n**2
        gp=np.abs(np.fft.rfft(detrend(gmag)*np.hanning(n)))**2/n**2
        usable=freqs>=0.5
        dominant=float(freqs[usable][np.argmax(apower[usable])]) if usable.any() else 0.
        rows.append([np.std(amag),np.sqrt(np.mean(dynamic**2)),np.mean(np.sum(np.abs(a-a.mean(axis=0)),axis=1)),np.mean(gmag),np.std(gmag),dominant,float(apower[usable].sum()),float(gp[usable].sum())])
    return pd.DataFrame(rows,columns=FEATURES)

def gaussian_logpdf(X,mu,var):
    return -.5*(np.sum(np.log(2*np.pi*var),axis=1)[None,:]+np.sum((X[:,None,:]-mu[None,:,:])**2/var[None,:,:],axis=2))

class GaussianHMM:
    def __init__(self,n_states=4,max_iter=70,tol=1e-5,random_state=42):
        self.K=n_states;self.max_iter=max_iter;self.tol=tol;self.random_state=random_state
    def _fb(self,X):
        B=gaussian_logpdf(X,self.mu,self.var);T=len(X);la=np.zeros((T,self.K));lb=np.zeros_like(la)
        la[0]=np.log(self.pi)+B[0]
        for t in range(1,T):la[t]=B[t]+logsumexp(la[t-1,:,None]+np.log(self.A),axis=0)
        ll=logsumexp(la[-1])
        for t in range(T-2,-1,-1):lb[t]=logsumexp(np.log(self.A)+B[t+1][None,:]+lb[t+1][None,:],axis=1)
        gamma=np.exp(la+lb-ll)
        xi=np.zeros((self.K,self.K))
        for t in range(T-1):xi+=np.exp(la[t,:,None]+np.log(self.A)+B[t+1][None,:]+lb[t+1][None,:]-ll)
        return gamma,xi,ll
    def fit(self,sequences,init_labels=None):
        rng=np.random.default_rng(self.random_state);pool=np.vstack(sequences);D=pool.shape[1]
        self.pi=np.ones(self.K)/self.K
        self.A=np.full((self.K,self.K),.05/(self.K-1));np.fill_diagonal(self.A,.95)
        if init_labels is not None:
            self.mu=np.vstack([np.mean(np.vstack([x for x,y in zip(sequences,init_labels) if y==k]),axis=0) if any(y==k for y in init_labels) else pool[rng.integers(len(pool))] for k in range(self.K)])
        else:self.mu=pool[rng.choice(len(pool),self.K,replace=len(pool)<self.K)]
        self.var=np.repeat(np.maximum(pool.var(axis=0)[None,:],.05),self.K,axis=0)
        self.history=[];previous=-np.inf
        for iteration in range(self.max_iter):
            accum_pi=np.zeros(self.K);accum_xi=np.zeros((self.K,self.K));gcount=np.zeros(self.K);first=np.zeros((self.K,D));second=np.zeros((self.K,D));total=0
            for x in sequences:
                g,xi,ll=self._fb(x);total+=ll;accum_pi+=g[0];accum_xi+=xi;gcount+=g.sum(axis=0);first+=g.T@x;second+=g.T@(x*x)
            self.history.append(float(total))
            if iteration and abs(total-previous)<self.tol*(1+abs(previous)):break
            previous=total
            self.pi=np.maximum(accum_pi,1e-7);self.pi/=self.pi.sum()
            self.A=np.maximum(accum_xi+1e-3,1e-8);self.A/=self.A.sum(axis=1,keepdims=True)
            count=np.maximum(gcount[:,None],1e-10)
            mu=first/count;var=np.maximum(second/count-mu**2,1e-3)
            self.mu=mu;self.var=var
        return self
    def predict(self,X):
        B=gaussian_logpdf(X,self.mu,self.var);T=len(X);delta=np.empty((T,self.K));psi=np.zeros((T,self.K),int)
        delta[0]=np.log(self.pi)+B[0]
        for t in range(1,T):
            scores=delta[t-1,:,None]+np.log(self.A)
            psi[t]=np.argmax(scores,axis=0);delta[t]=np.max(scores,axis=0)+B[t]
        path=np.zeros(T,int);path[-1]=np.argmax(delta[-1])
        for t in range(T-2,-1,-1):path[t]=psi[t+1,path[t+1]]
        return path

def sessions(root):
    found=[]
    for label in ACTIVITIES:
        for folder in sorted((root/'data'/'raw'/label).glob('session_*')):
            if (folder/'Accelerometer.csv').exists() and (folder/'Gyroscope.csv').exists():found.append((label,folder))
    return found

def heatmap(matrix,labels,title,path,xlabels=None):
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(matrix,aspect='auto',cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set_yticks(np.arange(len(labels)),labels=labels);ax.set_xticks(np.arange(matrix.shape[1]),labels=xlabels or labels,rotation=40,ha='right')
    ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def main(root):
    root=Path(root);(root/'figures').mkdir(exist_ok=True);(root/'results').mkdir(exist_ok=True);(root/'data'/'processed').mkdir(exist_ok=True)
    found=sessions(root);seq=[];audit=[]
    for label,path in found:
        df=load_session(path);f=extract_features(df);f.insert(0,'activity',label);f.insert(1,'session',path.name)
        f.to_csv(root/'data'/'processed'/f'{label}_{path.name}_features.csv',index=False)
        original=pd.read_csv(path/'Accelerometer.csv')
        times=original.seconds_elapsed.to_numpy();dt=np.diff(times);dt=dt[dt>0]
        audit.append(dict(activity=label,session=path.name,duration_sec=round(float(times[-1]-times[0]),3),median_sample_hz=round(float(1/np.median(dt)),2),windows=len(f)))
        seq.append((label,path,f[FEATURES].to_numpy(),df))
    pd.DataFrame(audit).to_csv(root/'results'/'data_audit.csv',index=False)
    if not seq:raise RuntimeError('No data sessions found')
    representatives=[next(q for q in seq if q[0]==activity) for activity in ACTIVITIES]
    fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=False)
    for ax,(label,path,_,df) in zip(axes,representatives):
        ax.plot(df.t,np.linalg.norm(df[['ax','ay','az']],axis=1),lw=.75)
        ax.set_title(f'{label}: {path.name}');ax.set_ylabel('Acceleration magnitude')
    axes[-1].set_xlabel('Seconds');fig.tight_layout();fig.savefig(root/'figures'/'raw_signals.png',dpi=180);plt.close(fig)
    # Strict split by recording session, never windows from the same session.
    train=[];test=[]
    for label in ACTIVITIES:
        items=[q for q in seq if q[0]==label]
        if len(items)>=4:train.extend(items[:-2]);test.extend(items[-2:])
        elif len(items)>=2:train.extend(items[:-1]);test.append(items[-1])
        else:train.extend(items)
    true_unseen=(len(test)>0 and set(q[0] for q in test)==set(ACTIVITIES))
    scaler=StandardScaler().fit(np.vstack([s[2] for s in train]))
    X=[scaler.transform(s[2]) for s in train]
    labels=[ACTIVITIES.index(s[0]) for s in train]
    model=GaussianHMM().fit(X,labels)
    # Gaussian mixture HMM states are permutation-invariant; align components using training-label emission log scores.
    from scipy.optimize import linear_sum_assignment
    costs=np.zeros((4,4))
    for label,x in zip(labels,X):costs[label]-=gaussian_logpdf(x,model.mu,model.var).mean(axis=0)
    rr,cc=linear_sum_assignment(costs);mapping=np.empty(4,dtype=int)
    for r,c in zip(rr,cc):mapping[c]=r
    heatmap(model.A,[ACTIVITIES[mapping[i]] for i in range(4)],'Learned HMM transition probabilities',root/'figures'/'transition_heatmap.png')
    # Emissions: mean standardized feature values (Gaussian emissions are densities, NOT discrete probability table).
    heatmap(model.mu,[ACTIVITIES[mapping[i]] for i in range(4)],'Gaussian emission means (standardized features)',root/'figures'/'emission_means.png',FEATURES)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(model.history,marker='o',ms=2);ax.set(xlabel='EM iteration',ylabel='Training log-likelihood',title='Baum–Welch convergence');fig.tight_layout();fig.savefig(root/'figures'/'convergence.png',dpi=180);plt.close(fig)
    evaluation=test if test else train;truth=[];pred=[];rows=[]
    for label,path,f,df in evaluation:
        v=model.predict(scaler.transform(f));p=mapping[v];actual=np.full(len(v),ACTIVITIES.index(label));truth.extend(actual.tolist());pred.extend(p.tolist())
        rows.extend([{'session':path.name,'true_activity':label,'predicted_activity':ACTIVITIES[int(i)],'window':j} for j,i in enumerate(p)])
    pd.DataFrame(rows).to_csv(root/'results'/'decoded_windows.csv',index=False)
    cm=confusion_matrix(truth,pred,labels=np.arange(4));heatmap(cm,ACTIVITIES,'Confusion matrix ('+('held-out sessions' if test else 'TRAINING ONLY')+')',root/'figures'/'confusion_matrix.png')
    metric=[];n=cm.sum()
    for i,label in enumerate(ACTIVITIES):
        tp=cm[i,i];fn=cm[i,:].sum()-tp;fp=cm[:,i].sum()-tp;tn=n-tp-fn-fp
        metric.append({'Activity':label,'Number of Samples':int(cm[i].sum()),'Sensitivity':float(tp/(tp+fn)) if tp+fn else None,'Specificity':float(tn/(tn+fp)) if tn+fp else None,'Overall Accuracy':float((tp+tn)/n) if n else None})
    pd.DataFrame(metric).to_csv(root/'results'/'evaluation_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,4));ax.step(np.arange(len(pred))*.5,pred,where='post',label='Decoded',lw=1.4);ax.step(np.arange(len(truth))*.5,truth,where='post',label='Reference',alpha=.6)
    ax.set(yticks=range(4),yticklabels=ACTIVITIES,xlabel='Window index × 0.5 seconds (session concatenation)',title='Viterbi decoded activity states');ax.legend();fig.tight_layout();fig.savefig(root/'figures'/'decoded_states.png',dpi=180);plt.close(fig)
    status={'unseen_evaluation':bool(true_unseen),'test_sessions':len(test),'training_sessions':len(train),'evaluation_type':'independent held-out sessions' if true_unseen else ('PARTIAL independent session coverage' if test else 'TRAINING DATA ONLY — not valid unseen evaluation'),'training_loglikelihood':model.history[-1],'baum_welch_iterations':len(model.history),'overall_window_accuracy':float(np.trace(cm)/n)}
    (root/'results'/'run_status.json').write_text(json.dumps(status,indent=2))
    np.savez(root/'results'/'model.npz',pi=model.pi,A=model.A,mu=model.mu,var=model.var,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,state_mapping=mapping,feature_names=FEATURES)
    print(json.dumps(status,indent=2));print(pd.DataFrame(metric).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);main(p.parse_args().root)
