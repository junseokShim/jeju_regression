import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import BertConfig, BertModel
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())

mps_device = torch.device("mps")

# 데이터 불러오기 (시계열 가격 데이터)
data = pd.read_csv('./processed_data/train.csv')
data = data.drop(columns=['Unnamed: 0'])


# 입력 데이터에 Min-Max 스케일링 적용
scaler = MinMaxScaler()

X = data.drop(columns=['price(원/kg)']).values
X = scaler.fit_transform(X)
#X = X.reshape(X.shape[0], 1, X.shape[1])

# 입력 텍스트와 레이블 생성
Y = data['price(원/kg)'].values
Y = Y.reshape(Y.shape[0], 1)

# 데이터를 훈련 세트와 테스트 세트로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 모델 정의 트랜스포머
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(TimeSeriesTransformer, self).__init__()
        
        # Transformer 모델 불러오기
        transformer_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.transformer = BertModel(transformer_config)
        
        # Fully Connected Layer 추가
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, 1)
    
    def forward(self, x, attention_mask=None):
        outputs = self.transformer(x, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(1)  # 각 시퀀스의 평균을 사용
        out = self.fc1(pooled_output)
        out = self.fc2(out)
        out = self.fc3(out)
        return out




# 데이터를 PyTorch 텐서로 변환합니다.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_train_tensor = torch.tensor(X_train, dtype=torch.long)  # 정수 데이터 유형으로 변환

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# 모델 생성 및 훈련
input_size = X_train.shape[1]
hidden_size = 64
num_layers = 2
num_heads = 8
model = TimeSeriesTransformer(input_size, hidden_size, num_layers, num_heads)
model.to(mps_device) 

# 모델 훈련 설정
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = torch.nn.MSELoss()

# Validation 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Early stopping 관련 설정
best_val_loss = float('inf')
patience = 10  # 일정 횟수 동안 검증 손실이 향상되지 않을 때 조기 종료
counter = 0

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation 손실 확인
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    print(f"epoch : {epoch} / Train loss : {math.sqrt(loss)} / Validation loss: {math.sqrt(val_loss)}")

        # 검증 손실이 이전 최고 손실보다 낮으면 모델 가중치 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './weights/best_model.pth')
        counter = 0
    else:
        counter += 1
    
    # 검증 손실이 일정 횟수 동안 향상되지 않으면 조기 종료
    if counter >= patience:
        print("Early stopping.")
        break



# 테스트 데이터로 평가
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    mse = mean_squared_error(y_test_tensor, test_outputs.numpy())

print("평균 제곱 오차 (MSE):", math.sqrt(mse))