# Caltech101 image classification
소프트웨어학과 17011660 김민주

## 이미지 불러오기
```python
caltech_dir = "/content/train/"
lists = paths.list_images(caltech_dir)

X = []
y = []

for l in lists:
  path = os.path.dirname(l)

  X.append(cv2.imread(l))
  y.append(os.path.basename(path))

print(len(X))
```
train 파일을 불러와 이미지는 X라는 배열에, label은 y라는 배열에 저장합니다.

```python
caltech_dir = "/content/testAll_v2/"
lists = paths.list_images(caltech_dir)

X_final = []
img_name = []
for l in lists:
  path = os.path.dirname(l)
  X_final.append(cv2.imread(l))
  img_name.append(l[-14:])
  
X_final = np.array(X_final)
img_name = np.array(img_name)
```
test 파일은 이미지의 이름이 제출 시 들어가게되므로 이미지는 X_final이라는 배열에, 이미지 이름은 img_name이라는 배열에 저장하였습니다.

## denseSIFT descriptor구하기
```python
def computeSIFT(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        step_size = 8
        kp = [cv2.KeyPoint(x, y, step_size)
              for x in range(0, img_gray.shape[0], step_size) 
                for y in range(0, img_gray.shape[1], step_size)]
        dense_feat = sift.compute(img_gray, kp)
        x.append(dense_feat[1])
        
    return x
    
X_sift = computeSIFT(X)
all_train_desc = np.vstack((descriptor for descriptor in X_sift))
```
step size가 8인 dense sift를 구해주는 함수인 computeSIFT입니다.<br>
train 이미지를 이 함수를 이용하여 descriptor를 구합니다.

## codebook 계산하기
```python
def clusterFeatures(all_train_desc, k):
    seeding = kmc2.kmc2(all_train_desc, k)
    model = MiniBatchKMeans(k, init=seeding).fit(all_train_desc)
    return model
    
kmeans = clusterFeatures(all_train_desc, 600)
```
아까 구한 descriptor에 kmeans를 이용하여 600 size의 codebook를 구합니다.


## SPM으로 histogram 구하기
```python
def getImageFeaturesSPM(L, img, kmeans, k):
    W = img.shape[1]
    H = img.shape[0]   
    h = []
    for l in range(L+1):
        w_step = math.floor(W/(2**l))
        h_step = math.floor(H/(2**l))
        x, y = 0, 0
        for i in range(1,2**l + 1):
            x = 0
            for j in range(1, 2**l + 1):                
                desc = computeSIFT(img[y:y+h_step, x:x+w_step])                
                predict = kmeans.predict(desc)
                histo = np.bincount(predict, minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                h.append(weight*histo)
                x = x + w_step
            y = y + h_step
            
    hist = np.array(h).ravel()
    if L == 0:
        hist = hist.astype('float64')
    dev = np.std(hist)
    hist -= np.mean(hist)
    hist /= dev
    return hist
    
def getHistogramSPM(L, data, kmeans, k):    
    x = []
    for i in range(len(data)):        
        hist = getImageFeaturesSPM(L, data[i], kmeans, k)        
        x.append(hist)
    return np.array(x)
    

train_histo = getHistogramSPM(2, X, kmeans, 600)
```
getImageFeaturesSPM에서는 이미지를 레벨에 맞춰 자른 뒤 자른 이미지의 descriptor를 구한 뒤
histogram을 구합니다.

## SVM 학습
```python
C_range = 10.0 ** np.arange(-3, 3)
param_grid = dict(C=C_range.tolist())
clf = GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-2)
clf.fit(train_histo, y)
```
구한 histogram을 LinearSVC와 GridSearch를 이용하여 학습시켰습니다.

## TEST
```python
X_histo = getHistogramSPM(2, X_final, kmeans, 600)
y_predict = clf.predict(X_histo)
```
test 이미지를 위와 같은 방법으로 histogram을 만들어준 뒤 학습시킨 모델로 예측값을 얻습니다.

## submit file 만들기
```python
df = pd.read_csv('/content/Label2Names.csv')
df = np.array(df)
result = []

for i in range(len(y_predict)):
  flag = 0
  if y_predict[i] == 'BACKGROUND_Google':
    flag = 1
    result.append(102)
  else:
    for j in range(len(df)):
        if y_predict[i] == df[j][1]:
          result.append(df[j][0])
          flag = 1
  
  if flag != 1:
    result.append(1)
    
result = np.array(result)
img_name = np.array(img_name)
result = result.reshape(-1, 1)
img_name = img_name.reshape(-1, 1)
total_result = np.hstack([img_name, result])

df = pd.DataFrame(total_result, columns=['Id', 'Category'])
df.to_csv('results-mjkim-v4.csv', index=False, header=True)

! kaggle competitions submit -c 2019-ml-finalproject -f results-mjkim-v4.csv -m "mjkim-20191123"
```
label과 이름이 적힌 csv 파일을 불러온 뒤 이를 이용해 predict값을 label로 바꿔줍니다.<br>
(numpy array로 변환하니 1번 label이 잘려서 따로 처리해주었습니다..)<br>
그 후 csv 파일로 변환하여 제출하였습니다.

## 
| level| codebook size | stepsize | accuracy |
|:---:|:---:|:---:|:---:|
|0|200|4|32|
|1|200|4|48|
|2|200|4|57|
|0|600|8|46|
|1|600|8|51|
|2|600|8|57|
