kya b nai...

1. ImplementA*Searchalgorithm.
SourceCode:
importsys
inf=99999
g=[
[0,1,inf,inf,inf,10],
[1,0,2,1,inf,inf],
[inf,2,0,inf,5,inf],
[inf,1,inf,0,3,4],
[inf,inf,3,0,2],
[10,inf,inf,4,2,0],
]
h=[5,3,4,2,6,0]
src=0
goal=5
classobj:
def __init__(self,cost,path):
self.cost=cost
self.path=path
arr=[]
new_item=obj(h[src],[src])
arr.append(new_item)
#a*algorithm
whilearr:
cur_item=arr[0]
cur_node=cur_item.path[-1]
cur_cost=cur_item.cost
cur_path=cur_item.path
foriinrange(0,len(h)):
ifg[cur_node][i]!=infandg[cur_node][i]!=0:
new_cost=cur_cost- h[cur_node] +h[i]+g[cur_node][i]
new_path=cur_path.copy()
new_path.append(i)
ifi==goal:
print(new_cost)
print(new_path)
# sys.exit()
new_item=obj(new_cost,new_path)
arr.append(new_item)
arr.pop(0)
arr=sorted(arr,key=lambdaitem:item.cost)
OUTPUT
17
[0,2,3,4,6]
18
[0,2,4,6]
21
[0,1,4,6]
25
[0,1,5,6]





2. ImplementAO*Searchalgorithm

importtime
importos
defget_node(mark_road,extended):
temp=[0]
i=0
while1:
current=temp[i]
ifcurrentnotinextended:
returncurrent
else:
forchildinmark_road[current]:
ifchildnotintemp:
temp.append(child)
i+=1
defget_current(s,nodes_tree):
iflen(s)==1:
returns[0]
fornodeins:
flag=True
foredgeinnodes_tree(node):
forchild_nodeinedge:
ifchild_nodeins:
flag=False
ifflag:
returnnode
defget_pre(current,pre,pre_list):
ifcurrent==0:
return
forpre_nodeinpre[current]:
ifpre_nodenotinpre_list:
pre_list.append(pre_node)
get_pre(pre_node,pre,pre_list)
return
defans_print(mark_rode,node_tree):
print("Thefinalconnectionisasfollow:")
temp=[0]
whiletemp:
time.sleep(1)
print(f"[{temp[0]}]----->{mark_rode[temp[0]]}")
forchildinmark_rode[temp[0]]:
ifnode_tree[child]!=[[child]]:
temp.append(child)
temp.pop(0)
time.sleep(5)
os.system('cls')
return
defAOstar(node_tree,h_val):
futility=0xfff
extended=[]
choice=[]
mark_rode={0:None}
solved={}
pre={0:[]}
foriinrange(1,9):
pre[i]=[]
foriinrange(len(nodes_tree)):
solved[i]=False
os.system('cls')
print("Theconnectionprocessisasfollows")
time.sleep(1)
whilenotsolved[0]andh_val[0]<futility:
node=get_node(mark_rode,extended)
extended.append(node)
ifnodes_tree[node]isNone:
h_val[node]=futility
continue
forsuc_edgeinnodes_tree[node]:
forsuc_nodeinsuc_edge:
ifnodes_tree[suc_node]==[[suc_node]]:
solved[suc_node]=True
s=[node]
whiles:
current=get_current(s,nodes_tree)
s.remove(current)
origen_h=h_val[current]
origen_s=solved[current]
min_h=0xfff
foredgeinnodes_tree[current]:
edge_h=0
fornodeinedge:
edge_h+=h_val[node]+1
ifedge_h<min_h:
min_h=edge_h
h_val[current]=min_h
mark_rode[current]=edge
ifmark_rode[current]notinchoice:
choice.append(mark_rode[current])
print(f"[{current}]-----{mark_rode[current]}")
time.sleep(1)
forchild_nodeinmark_rode[current]:
pre[child_node].append(current)
solved[current]=True
fornodeinmark_rode[current]:
solved[current]=solved[current]andsolved[node]
iforigen_s!=solved[current]ororigen_h!=h_val[current]:
pre_list=[]
ifcurrent!=0:
get_pre(current,pre,pre_list)
s.extend(pre_list)
ifnotsolved[0]:
print("Thequeryfailed,thepathcouldnotbefound!")
else:
ans_print(mark_rode,nodes_tree)
return
if__name__=="__main__":
nodes_tree={}
nodes_tree[0]=[[1],[4,5]]
nodes_tree[1]=[[2],[3]]
nodes_tree[2]=[[3],[2,5]]
nodes_tree[3]=[[5,6]]
nodes_tree[4]=[[5],[8]]
nodes_tree[5]=[[6],[7,8]]
nodes_tree[6]=[[7,8]]
nodes_tree[7]=[[7]]
nodes_tree[8]=[[8]]
h_val=[3,2,4,4,1,1,2,0,0]
AOstar(nodes_tree,h_val)
OUTPUT:
Theconnectionprocessisasfollows
[0]-----[1]
[1]-----[2]
[0]-----[4,5]
[4]-----[8]
[5]-----[7,8]
Thefinalconnectionisasfollow:
[0]----->[4,5]
[4]----->[8]
[5]----->[7,8]




Program:3.CANDIDATEELIMINATIONALGORITHM

importcsv
a=[]
csvfile=open('1.csv','r')
reader=csv.reader(csvfile)for
rowin reader:
a.append(row)
print(row)
num_attributes=len(a[0])-1
print("Initial hypothesisis")
S=['0']*num_attributes
G=['?']*num_attributes
print("Themost specific: ",S)
print("Themost general :",G)
for jin range(0,num_attributes):
S[j]=a[0][j]
print("Thecandidatealgorithm \n")
temp=[]
for i in range(0,len(a)):
if(a[i][num_attributes]=='Yes'):
for jin range(0,num_attributes):
if(a[i][j]!=S[j]):
S[j]='?'
for jin range(0,num_attributes):for
k in range(1,len(temp)):
if temp[k][j]!='?' andtemp[k][j]!=S[j]:del
temp[k]
print("Forinstance{0} thehypothesisisS{0}".format(i+1),S)
if(len(temp)==0):
print("Forinstance{0}thehypothesisisG{0}".format(i+1),G)else:
print("Forinstance{0} thehypothesisisS{0}".format(i+1),temp)
if(a[i][num_attributes]=='No'):
for jin range(0,num_attributes):
if(S[j]!=a[i][j]andS[j]!='?'):
G[j]=S[j]
temp.append(G)
G=['?']*num_attributes
print("Forinstance{0} thehypothesisisS{0}".format(i+1),S)print("For
instance{0}thehypothesisisG{0}".format(i+1),temp)
output:
['Sunny','Warm','Normal','Strong','Warm','Same','Yes']
['Sunny','Warm','High','Strong','Warm','Same','Yes']
['Rainy','Cold','High','Strong','Warm','Change','No']
['Sunny','Warm','High','Strong','Cool','Change','Yes']Initial
hypothesisis
Themostspecific: ['0','0','0','0','0','0']
Themostgeneral : ['?','?','?', '?','?','?']The
candidate algorithm
For instance1 the hypothesisisS1 ['Sunny','Warm','Normal','Strong','Warm','Same']
For instance 1thehypothesisisG1['?','?','?','?','?','?']
For instance 2the hypothesisisS2['Sunny','Warm','?','Strong','Warm','Same']For
instance2thehypothesisisG2['?','?','?','?','?','?']
For instance 3the hypothesisisS3['Sunny','Warm','?','Strong','Warm','Same']
For instance3 the hypothesisisG3 [['Sunny','?','?','?','?','?'],['?','Warm','?','?','?','?'],['?','?','?','?','?','Same']]
For instance 4thehypothesisisS4['Sunny','Warm','?','Strong','?','?']
For instance 4thehypothesisisS4[['Sunny','?','?','?','?','?'],['?','Warm','?','?','?','?']]






Program:4.ID3 ALGORITHM
importpandasaspd
fromcollectionsimportCounter
importmath
tennis=pd.read_csv('playtennis.csv')
print("\nGiven PlayTennisData Set:\n\n",tennis)
defentropy(alist):
c= Counter(x for xin alist)
instances=len(alist)
prob = [x /instancesforx in c.values()]return
sum( [-p*math.log(p,2)forp in prob])
def information_gain(d, split,target):
splitting= d.groupby(split)
n= len(d.index)
agent =splitting.agg({target:[entropy,lambdax:len(x)/n]})[target]#aggregatingagent.columns=
['Entropy','observations']
newentropy= sum( agent['Entropy']* agent['observations'])
oldentropy=entropy(d[target])
returnoldentropy-newentropy
def id3(sub,target,a):
count = Counter(x for xin sub[target])#classof YES /NOif
len(count)== 1:
return next(iter(count)) #nextinput dataset,or raisesStopIterationwhen EOFishit
else:
gain =[information_gain(sub,attr,target)forattr ina]
print("Gain=",gain)
maximum = gain.index(max(gain))
best= a[maximum]
print("BestAttribute:",best)
tree= {best:{}}
remaining= [i for i in a ifi != best]
for val,subset insub.groupby(best):
subtree =id3(subset,target,remaining)
tree[best][val]= subtree
return tree
names=list(tennis.columns)print("List
of Attributes:", names)
names.remove('PlayTennis')
print("PredictingAttributes:",names)
tree= id3(tennis,'PlayTennis',names)
print("\n\nTheResultantDecision Tree is:\n")
print(tree)





Program:5.BACKPROPOGATION
SourceCode

importnumpyasnp
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100
def sigmoid(x):
return1/(1+np.exp(-x))
def derivatives_sigmoid(x):
returnx*(1-x)
epoch=7000
lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for iinrange(epoch):
hinp1=np.dot(X,wh)
hinp=hinp1+bh
hlayer_act=sigmoid(hinp)
outinp1=np.dot(hlayer_act,wout)
outinp=outinp1+bout
output=sigmoid(outinp)
E0=y-output
outgrad=derivatives_sigmoid(output)
d_output=E0*outgrad
EH=d_output.dot(wout.T)
hiddengrad=derivatives_sigmoid(hlayer_act)
d_hiddenlayer=EH*hiddengrad
wout+=hlayer_act.T.dot(d_output)*lr
print("Input:\n"+str(X))
print("ActualOutput:\n"+str(y))
print("Predicted Output:\n",output)
output
Input:
[[0.66666667 1. ]
[0.33333333 0.55555556]
[1. 0.66666667]]
Actual Output:
[[0.92]
[0.86]
[0.89]]
Predicted Output:
[[0.89282584]
[0.87763012]
[0.89905218]]



Program:6.NAÏVEBAYESIANCLASSIFIER
importcsv
importmath
importrandom
importstatistics
defcal_probability(x,mean,stdev):
exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
return(1/(math.sqrt(2*math.pi)*stdev))*exponent
dataset=[]
dataset_size=0
with open('lab5.csv')ascsvfile:
lines=csv.reader(csvfile)
for rowin lines:
dataset.append([float(attr)for attr in row])
dataset_size=len(dataset)
print("Sizeof dataset is: ",dataset_size)
train_size=int(0.7*dataset_size)
print(train_size)
X_train=[]
X_test=dataset.copy()
training_indexes=random.sample(range(dataset_size),train_size)
for i in training_indexes:
X_train.append(dataset[i])
X_test.remove(dataset[i])
classes={}
for samples in X_train:
last=int(samples[-1])
if lastnotin classes:
classes[last]=[]
classes[last].append(samples)
print(classes)
summaries={}
for classValue,training_data in classes.items():
summary=[(statistics.mean(attribute),statistics.stdev(attribute))forattributein
zip(*training_data)]
del summary[-1]
summaries[classValue]=summary
print(summaries)
X_prediction=[]
for i in X_test:
probabilities={}
for classValue,classSummaryin summaries.items():
probabilities[classValue]=1
for index,attrin enumerate(classSummary):
probabilities[classValue]*=cal_probability(i[index],attr[0],attr[1])
best_label,best_prob=None,-1
for classValue,probabilityin probabilities.items():if
best_labelisNoneor probability>best_prob:
best_prob=probability
best_label=classValue
X_prediction.append(best_label)
correct=0
for index,keyin enumerate(X_test):
if X_test[index][-1]==X_prediction[index]:
correct+=1
print("Accuracy:",correct/(float(len(X_test)))*100)




Program:7.EMALGORITHM

importnumpyasnp
importpandasaspd
frommatplotlibimport pyplotasplt
from sklearn.mixtureimport GaussianMixture
fromsklearn.clusterimport KMeans
data = pd.read_csv('lab8.csv')
print("Input Data and Shape")
print(data.shape)
data.head()
f1= data['V1'].values
f2= data['V2'].values
X= np.array(list(zip(f1,f2)))
print("X ",X)
print('Graph for whole dataset')
plt.scatter(f1, f2, c='black', s=7)
plt.show()
kmeans=KMeans(20, random_state=0)
labels= kmeans.fit(X).predict(X)
print("labels ",labels)
centroids=kmeans.cluster_centers_
print("centroids ",centroids)
plt.scatter(X[:,0],X[:,1],c=labels,s=40,cmap='viridis');print('Graph using
KmeansAlgorithm')
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='#050505')
plt.show()
gmm= GaussianMixture(n_components=3).fit(X)
labels= gmm.predict(X)
probs=gmm.predict_proba(X)size
= 10* probs.max(1)** 3 print('Graph
usingEMAlgorithm')
plt.scatter(X[:,0],X[:,1],c=labels,s=size,cmap='viridis');
plt.show()





Program:8.K-NEARESTNEIGHBOUR

importnumpyasnp
from sklearn.datasetsimportload_iris
iris=load_iris()
x=iris.data
y=iris.target
print(x[:5],y[:5])
from sklearn.model_selection import train_test_splitxtrain,xtest,ytrain,ytest
=train_test_split(x,y,test_size=0.4,random_state=1)print(iris.data.shape)
print(len(xtrain))
print(len(ytest))
from sklearn.neighborsimportKNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)
from sklearn import metrics
print("Accuracy",metrics.accuracy_score(ytest,pred))
print(iris.target_names[2])ytestn=[iris.target_names[i]
for i in ytest]predn=[iris.target_names[i]fori inpred]
print(" predicted Actual")
for iin range(len(pred)):
print(i," ",predn[i]," ",ytestn[i])





Program:9.LOCALLYWEIGHTEDREGRESSIONALGORITHM

importnumpyasnp
importmatplotlib.pyplot asplt
importpandasaspd
tou = 0.5
data=pd.read_csv("lab10.csv")
X_train = np.array(data.total_bill)
print(X_train)
X_train =X_train[:,np.newaxis]
print(len(X_train))
y_train = np.array(data.tip)
X_test =np.array([i/10 for iinrange(500)])X_test
=X_test[:,np.newaxis]
y_test = []
count= 0
for rinrange(len(X_test)):
wts= np.exp(-np.sum((X_train-X_test[r])** 2, axis=1)/ (2* tou ** 2))W =
np.diag(wts)
factor1 =np.linalg.inv(X_train.T.dot(W).dot(X_train))
parameters= factor1.dot(X_train.T).dot(W).dot(y_train)
prediction = X_test[r].dot(parameters)
y_test.append(prediction)
count+=1
print(len(y_test))
y_test = np.array(y_test)
plt.plot(X_train.squeeze(),y_train,'o')
plt.plot(X_test.squeeze(),y_test,'o')plt.
show()







